"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
import rlkit.torch.pytorch_util as U

from rlkit.envs.mujoco_manip_env import MujocoManipEnv

# Sets the GPU mode.
USE_GPU = True
U.set_gpu_mode(USE_GPU)

EXPERIMENT_NAME = "cans-50-50-reward-scale-1"
#EXPERIMENT_NAME = "pegs-50-50-reward-scale-0.1"
#EXPERIMENT_NAME = "lift-lr-1e-4"
HORIZON = 250
UPDATES_PER_STEP = 1
REWARD_SCALE = 1

# DEMO_PATH = None
DEMO_PATH = "/home/robot/Downloads/test_extraction/bins-Can0-sars.pkl"

ACTION_SKIP = 1
LR = 3E-4

def experiment(variant):
    # env = NormalizedBoxEnv(HalfCheetahEnv())
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v2'))
    # env = gym.make('HalfCheetah-v2')

    env = MujocoManipEnv("SawyerBinsCanEnv") # wrap as a gym env
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_skip=ACTION_SKIP,
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100000, 
            num_steps_per_epoch=HORIZON, # number of env steps per epoch (just make this an episode?)
            num_steps_per_eval=HORIZON, # number of steps in eval?
            batch_size=128,
            max_path_length=HORIZON - 1, # TODO: is this off by one? 
            num_updates_per_env_step=UPDATES_PER_STEP, # batch learning steps per train step
            discount=0.99,
            soft_target_tau=0.001,
            policy_lr=LR,
            qf_lr=LR,
            vf_lr=LR,
            demo_path=DEMO_PATH, # path to demos
            action_skip=ACTION_SKIP, # number of env steps per policy action
            experiment_name=EXPERIMENT_NAME,
            batch_reward_scale=REWARD_SCALE,
        ),
        net_size=300,
    )

    # # noinspection PyTypeChecker
    # variant = dict(
    #     algo_params=dict(
    #         num_epochs=1000, 
    #         num_steps_per_epoch=1000, # number of env steps per epoch (just make this an episode?)
    #         num_steps_per_eval=1000, # number of steps in eval?
    #         batch_size=128,
    #         max_path_length=999, # TODO: is this off by one? 
    #         discount=0.99,

    #         soft_target_tau=0.001,
    #         policy_lr=3E-4,
    #         qf_lr=3E-4,
    #         vf_lr=3E-4,
    #     ),
    #     net_size=300,
    # )

    setup_logger(EXPERIMENT_NAME, variant=variant)
    experiment(variant)
