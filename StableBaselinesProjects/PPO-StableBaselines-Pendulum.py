import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
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
    plt.savefig("PPO-StableBaselines-Pendulum-results.png")
    plt.show()


def optimize_ppo(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024])
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1)

    env = gym.make("Pendulum-v1")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
    )

    total_timesteps = 300000
    model.learn(total_timesteps=total_timesteps)

    num_episodes = 10
    reward_list = []
    for episode in range(num_episodes):
        obs = env.reset()[0]
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
study.optimize(optimize_ppo, n_trials=40)

print("Best hyperparameters:", study.best_params)

env = gym.make("Pendulum-v1")
best_params = study.best_params

ppo_model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=best_params["learning_rate"],
    batch_size=best_params["batch_size"],
    n_steps=best_params["n_steps"],
    gamma=best_params["gamma"],
    ent_coef=best_params["ent_coef"],
)

total_timesteps = 300000
ppo_model.learn(total_timesteps=total_timesteps)

num_episodes = 100
reward_list_ppo = []
for episode in range(num_episodes):
    obs = env.reset()[0]
    done = False
    total_reward = 0
    truncated = False
    while not done and not truncated:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    reward_list_ppo.append(total_reward)

    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(reward_list_ppo[-10:])
        print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Average Reward (last 10): {avg_reward:.2f}")

plot_rewards(num_episodes, reward_list_ppo, label="PPO Rewards (Pendulum-v1)")
print("Max reward:", max(reward_list_ppo))
print("Average reward:", np.mean(reward_list_ppo))
env.close()
