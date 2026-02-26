import gymnasium as gym
from collections import defaultdict
import numpy as np

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n)) 
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # epsilon-greedy: epsilon for ramdom, otherwise take max action
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_diff = target - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_diff
        
        self.training_error.append(temporal_diff)

learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

def test_agent(agent, env, num_episodes=1000):
    total_rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
    
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3%}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


from tqdm import tqdm


for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

    if episode % 10_000 == 0:
        test_agent(agent, env)

# from matplotlib import pyplot as plt
#
# def get_moving_avgs(arr, window, convolution_mode):
#     """Compute moving average to smooth noisy data."""
#     return np.convolve(
#         np.array(arr).flatten(),
#         np.ones(window),
#         mode=convolution_mode
#     ) / window
#
# # Smooth over a 500-episode window
# rolling_length = 5000
# fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
#
# # Episode rewards (win/loss performance)
# axs[0].set_title("Episode rewards")
# reward_moving_average = get_moving_avgs(
#     env.return_queue,
#     rolling_length,
#     "valid"
# )
# axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
# axs[0].set_ylabel("Average Reward")
# axs[0].set_xlabel("Episode")
#
# # Episode lengths (how many actions per hand)
# axs[1].set_title("Episode lengths")
# length_moving_average = get_moving_avgs(
#     env.length_queue,
#     rolling_length,
#     "valid"
# )
# axs[1].plot(range(len(length_moving_average)), length_moving_average)
# axs[1].set_ylabel("Average Episode Length")
# axs[1].set_xlabel("Episode")
#
# # Training error (how much we're still learning)
# axs[2].set_title("Training Error")
# training_error_moving_average = get_moving_avgs(
#     agent.training_error,
#     rolling_length,
#     "same"
# )
# axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
# axs[2].set_ylabel("Temporal Difference Error")
# axs[2].set_xlabel("Step")
#
# plt.tight_layout()
# plt.show()

