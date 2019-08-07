from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import h5py
import argparse
import torch
import uuid
import robosuite
from rlkit.core import logger

from rlkit.envs.robosuite_env import RobosuiteEnv

filename = str(uuid.uuid4())

def create_robosuite_env(data_path):
    f = h5py.File(data_path, "r")  
    env_name = f["data"].attrs["env"]
    f.close()

    env = robosuite.make(
        env_name,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_object_obs=True,
        use_camera_obs=False,
        camera_height=84,
        camera_width=84,
        camera_name="agentview",
        gripper_visualization=False,
        reward_shaping=True,
        control_freq=100,
        horizon=1000,
    )
    env = RobosuiteEnv(env)
    return env

def simulate_policy(args, env=None):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    if env is None:
        env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        # policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batch', type=str,
                        help='path to the dataset')
    args = parser.parse_args()

    env = create_robosuite_env(args.batch)
    simulate_policy(args, env=env)
