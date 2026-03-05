import os 
import time
import random
from dataclasses import dataclass

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

from utils.atari_wrappers import (
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
)
from utils.buffers import ReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    capture_video: bool = True 
    """whether to capture video of the agent performance (adds rendering overhead)"""
    save_model: bool = True 
    """whether to save model into the `runs/{run_name}` folder"""

    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiment"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100_000
    """the replay memory buffer size (reduced for 16GB RAM; 1M needs ~53GB)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the replay memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timestamps` it takes from start_e to go end_e"""
    learning_starts: int = 80_000 
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    resume: bool = False
    """whether to resume training from a checkpoint"""
    checkpoint_path: str = ""
    """path to checkpoint file to resume from"""
    checkpoint_frequency: int = 100_000
    """save checkpoint every N timesteps"""

def make_env(env_id, seed, idx, capture_video, run_name, episode_trigger=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            record_kwargs = {"video_folder": f"logs/breakout_with_dqn_runs/videos/{run_name}"}
            if episode_trigger is not None:
                record_kwargs["episode_trigger"] = episode_trigger
            env = gym.wrappers.RecordVideo(env, **record_kwargs)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x.float() / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.env_id}_{args.exp_name}__{args.seed}__{int(time.time())}"
    print(run_name)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_episode_trigger = lambda x: x % 500 == 0  # record every 500 episodes during training
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, episode_trigger=train_episode_trigger) for i in range(args.num_envs)]
    )

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # --- Resume from checkpoint ---
    start_step = 0
    if args.resume and args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        q_network.load_state_dict(checkpoint["q_network"])
        target_network.load_state_dict(checkpoint["target_network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Allow overriding LR on resume via --learning-rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        start_step = checkpoint["global_step"]
        print(f"Resumed from checkpoint at step {start_step}, lr={args.learning_rate}")

    checkpoint_dir = f"logs/breakout_with_dqn_runs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(start_step, args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # buffer must have enough data to sample; on resume buffer refills from scratch
        buffer_ready = rb.size() >= args.batch_size
        if buffer_ready and global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
                
                if global_step % 10000 == 0:
                    print(f"loss at {global_step} is {loss}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        # --- Periodic checkpoint save ---
        if global_step > 0 and global_step % args.checkpoint_frequency == 0:
            ckpt_path = f"{checkpoint_dir}/{run_name}_step_{global_step}.pt"
            torch.save({
                "q_network": q_network.state_dict(),
                "target_network": target_network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "args": vars(args),
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    if args.save_model:
        model_path = f"logs/breakout_with_dqn_runs/{run_name}_{args.exp_name}.model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
        from utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )

    envs.close()

