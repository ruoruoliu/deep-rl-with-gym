import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name, episode_trigger=lambda x: True)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon: 
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "_episode" in infos:
            for i, done in enumerate(infos["_episode"]):
                if done:
                    episodic_return = float(infos["episode"]["r"][i])
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")
                    episodic_returns.append(episodic_return)
        obs = next_obs

    return episodic_returns

if __name__ == "__main__":
    pass
