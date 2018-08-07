import numpy as np
import MujocoManip as MM
from gym import spaces

class MujocoManipEnv(object):
    """
    General wrapper to make the environment look like a gym environment.
    """
    def __init__(self, name, keys=None):
        self.env = MM.make(name, 
                           has_renderer=False, 
                           use_camera_obs=False,
                           ignore_done=True, 
                           reward_shaping=True,
                           has_offscreen_renderer=False)
        if keys is None:
            keys = ['proprio', 'low-level', 
                    # 'cube_pos','gripper_to_cube','cyl_to_hole','angle', 't', 'd', 'hole_pos'
                    'SquareNut0_pos', 'SquareNut0_quat', 'SquareNut0_to_eef_pos', 'SquareNut0_to_eef_quat',
                    'SquareNut1_pos', 'SquareNut1_quat', 'SquareNut1_to_eef_pos', 'SquareNut1_to_eef_quat',
                    'RoundNut0_pos', 'RoundNut0_quat', 'RoundNut0_to_eef_pos', 'RoundNut0_to_eef_quat',
                    'RoundNut1_pos', 'RoundNut1_quat', 'RoundNut1_to_eef_pos', 'RoundNut1_to_eef_quat',
                    'Milk0_pos', 'Milk0_quat', 'Milk0_to_eef_pos', 'Milk0_to_eef_quat',
                    'Bread0_pos', 'Bread0_quat', 'Bread0_to_eef_pos', 'Bread0_to_eef_quat',
                    'Cereal0_pos', 'Cereal0_quat', 'Cereal0_to_eef_pos', 'Cereal0_to_eef_quat',
                    'Can0_pos', 'Can0_quat', 'Can0_to_eef_pos', 'Can0_to_eef_quat']
        self.keys = keys

        # set up observation and action spaces
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec()
        self.action_space = spaces.Box(low=low, high=high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filter keys of interest out and concatenate the information.

        :param obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        return np.concatenate(ob_lst)

    def reset(self):
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info


