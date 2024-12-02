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
    plt.savefig("PPO-StableBaselines-CartPole-results.png")
    plt.show()


def optimize_ppo(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_steps = trial.suggest_int("n_steps", 512, 4096, step=512)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)

    env = gym.make("CartPole-v1")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        verbose=0,
    )

    model.learn(total_timesteps=300000, progress_bar=False)

    total_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    return np.mean(total_rewards)


study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=40)

print("Best hyperparameters:", study.best_params)

env = gym.make("CartPole-v1")
best_params = study.best_params

final_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=best_params["learning_rate"],
    batch_size=best_params["batch_size"],
    n_steps=best_params["n_steps"],
    gamma=best_params["gamma"],
    verbose=1,
)

final_model.learn(total_timesteps=300000, progress_bar=True)

num_episodes = 100
reward_list_ppo = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    done, truncated = False, False
    total_reward = 0
    while not done and not truncated:
        action, _states = final_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    reward_list_ppo.append(total_reward)

    # if (episode + 1) % 10 == 0:
    #     avg_reward = np.mean(reward_list_ppo[-10:])
    #     print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Average Reward (last 10): {avg_reward:.2f}")

plot_rewards(num_episodes, reward_list_ppo, label="PPO Rewards (100 Episodes)")

print("Max reward:", max(reward_list_ppo))
print("Average reward:", np.mean(reward_list_ppo))
env.close()
