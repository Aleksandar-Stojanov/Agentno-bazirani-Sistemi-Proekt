import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
import optuna


def plot_rewards(episodes, rewards, label):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, episodes + 1), rewards, label=label)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title(f"Rewards per Episode ({episodes} Episodes)")
    plt.axhline(y=np.mean(rewards), color='r', linestyle='--', label='Average Reward')
    plt.legend()
    plt.grid()
    plt.savefig("DDPG-StableBaselines-Pendulum-results.png")
    plt.show()


def optimize_ddpg(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [100000, 500000, 1000000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    tau = trial.suggest_float("tau", 0.001, 0.01)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)

    env = gym.make("Pendulum-v1")

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        verbose=0,
    )

    total_timesteps = 300000
    model.learn(total_timesteps=total_timesteps)

    num_episodes = 10
    reward_list = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        reward_list.append(total_reward)

    mean_reward = np.mean(reward_list)
    env.close()
    return mean_reward


study = optuna.create_study(direction="maximize")
study.optimize(optimize_ddpg, n_trials=40)

print("Best hyperparameters:", study.best_params)

env = gym.make("Pendulum-v1")
best_params = study.best_params

ddpg_model = DDPG(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=best_params["learning_rate"],
    buffer_size=best_params["buffer_size"],
    batch_size=best_params["batch_size"],
    tau=best_params["tau"],
    gamma=best_params["gamma"],
)

total_timesteps = 300000
ddpg_model.learn(total_timesteps=total_timesteps)

num_episodes = 100
reward_list_ddpg = []
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    truncated = False
    while not done and not truncated:
        action, _ = ddpg_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    reward_list_ddpg.append(total_reward)

    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(reward_list_ddpg[-10:])
        print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Average Reward (last 10): {avg_reward:.2f}")

plot_rewards(num_episodes, reward_list_ddpg, label="DDPG Rewards (Pendulum-v1)")
print("Max reward:", max(reward_list_ddpg))
print("Average reward:", np.mean(reward_list_ddpg))
env.close()
