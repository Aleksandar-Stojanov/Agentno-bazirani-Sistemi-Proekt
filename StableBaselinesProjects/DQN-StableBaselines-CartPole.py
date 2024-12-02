import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
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
    plt.savefig("DQN-StableBaselines-CartPole-results.png")
    plt.show()

def optimize_dqn(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000])
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.05, 0.3)
    net_arch = trial.suggest_categorical("net_arch", [[64, 64], [128, 128], [256, 256]])

    env = gym.make("CartPole-v1")

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": net_arch},
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        verbose=0,
    )

    model.learn(total_timesteps=300000, progress_bar=False)

    total_rewards = []
    for _ in range(10):
        obs = env.reset()
        obs = obs[0]
        done, truncated = False, False
        total_reward = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    return np.mean(total_rewards)

study = optuna.create_study(direction="maximize")
study.optimize(optimize_dqn, n_trials=40)

print("Best hyperparameters:", study.best_params)

env = gym.make("CartPole-v1")
best_params = study.best_params

final_model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs={"net_arch": best_params["net_arch"]},
    learning_rate=best_params["learning_rate"],
    buffer_size=best_params["buffer_size"],
    exploration_fraction=best_params["exploration_fraction"],
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    verbose=1,
)

final_model.learn(total_timesteps=300000, progress_bar=True)


reward_list_dqn = []
for episode in range(100):
    obs = env.reset()
    obs = obs[0]
    done, truncated = False, False
    total_reward = 0
    while not done and not truncated:
        action, _ = final_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    reward_list_dqn.append(total_reward)

plot_rewards(100, reward_list_dqn, label="DQN Rewards (100 Episodes)")
print("Max reward:", max(reward_list_dqn))
print("Average reward:", np.mean(reward_list_dqn))
env.close()
