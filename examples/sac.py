import argparse
import h5py
from gym.envs.mujoco import HalfCheetahEnv

import torch
import robosuite

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.data_management.demo_buffer import DemoBuffer

import batchRL
from batchRL.envs.residual import RobosuiteEnv, ResidualRobosuiteEnv, SequenceRobosuiteEnv

def create_robosuite_env(data_path, horizon=1000, residual_algo='', residual_agent='', goal_seq=False, eval_env=False):
    f = h5py.File(data_path, "r")  
    env_name = f["data"].attrs["env"]
    f.close()

    # use visual versions of environments for training by default
    env_name = env_name[:6] + "Visual" + env_name[6:]

    reward_shaping = True
    if goal_seq and eval_env:
        reward_shaping = False

    env = robosuite.make(
        env_name,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_object_obs=True,
        use_camera_obs=False,
        camera_height=84,
        camera_width=84,
        camera_name="agentview",
        gripper_visualization=False,
        reward_shaping=reward_shaping,
        control_freq=100,
        horizon=horizon,
    )

    if len(residual_agent):
        env = ResidualRobosuiteEnv(env, algo=residual_algo, batch=data_path, agent=residual_agent)
    elif goal_seq:
        env = SequenceRobosuiteEnv(env, hdf5_path=data_path, seq_length=30, imitation=eval_env)
    else:
        env = RobosuiteEnv(env)
    return env

def experiment(variant):

    demo_path = variant["batch"]
    horizon = variant["horizon"]
    eval_horizon = variant["eval_horizon"]
    residual_algo = variant["residual_algo"]
    residual_agent = variant["residual_agent"]
    goal_seq = variant["goal_seq"]

    expl_env = create_robosuite_env(
        data_path=demo_path, 
        horizon=horizon, 
        residual_algo=residual_algo, 
        residual_agent=residual_agent, 
        goal_seq=goal_seq,
        eval_env=False,
    )
    eval_env = create_robosuite_env(
        data_path=demo_path, 
        horizon=eval_horizon, 
        residual_algo=residual_algo, 
        residual_agent=residual_agent, 
        goal_seq=goal_seq,
        eval_env=True,
        # eval_env=False,
    )
    # expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    if len(residual_agent):
        demo_buffer = None
    else:
        # make demonstration buffer to provide training data
        demo_buffer = None
        # demo_buffer = DemoBuffer(size=100000000)
        # demo_buffer.load_from_hdf5(demo_path)

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        mix_data=variant['mix_data'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
    )
    parser.add_argument(
        "--batch",
        type=str,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--eval_horizon",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--eval_rollouts",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--mix",
        action='store_true',
    )
    parser.add_argument(
        "--seq",
        action='store_true', # for goal-reaching training
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="gl", # residual RL: algo type
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="", # residual RL: agent path
    )
    args = parser.parse_args()

    # cuda support
    ptu.set_gpu_mode(torch.cuda.is_available())

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=(args.eval_rollouts * args.eval_horizon), #5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=args.horizon, #1000,
            max_eval_path_length=args.eval_horizon, #1000
            batch_size=256,
            experiment_name=args.name,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        batch=args.batch,
        horizon=args.horizon,
        eval_horizon=args.eval_horizon,
        mix_data=args.mix,
        residual_algo=args.algo,
        residual_agent=args.agent,
        goal_seq=args.seq,
    )
    setup_logger(args.name, variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
