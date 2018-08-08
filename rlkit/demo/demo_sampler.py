"""
A class that is responsible for sampling batches from demonstrations 
as if they came from a replay buffer.
"""

import random
import pickle
import numpy as np

class DemoSampler(object):
    def __init__(self, demo_path, observation_dim, action_dim, preload=False):
        """
        :param demo_path: Path to a demo pkl file. Should also have a corresponding
                          bkl file located there.
        :param observation_dim: size of observations
        :param action_dim: size of actions
        :param preload: If true, preload the huge bkl file instead of lazily loading.
        """

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.preload = preload

        with open(demo_path, "rb") as small_pkl:
            self.offsets = pickle.load(small_pkl)[:-1] # offsets have one too many values
        assert(self.offsets[0] == 0)

        self.demo_big_file = open(demo_path.replace('.pkl', '.bkl'), "rb")
        self.demo_data = None
        if preload:
            data = []
            for ofs in self.offsets:
                self.demo_big_file.seek(ofs)
                data.extend(pickle.load(self.demo_big_file))
            self.demo_data = data # stores all data in a huuuge buffer
            self.demo_length = len(data)
            self.demo_big_file.close()
            self.demo_big_file = None
            print("Demo length is {}".format(self.demo_length))


    ### TODO: time how long it takes to get a batch ###
    def _uniform_sample(self):
        # NOTE: this is not truly uniform... 
        if self.preload:
            return random.choice(self.demo_data)
        else:
            # first sample an episode, then a point in time
            episode_choice = random.choice(self.offsets) 
            self.demo_big_file.seek(episode_choice)
            episode = pickle.load(self.demo_big_file)
            return random.choice(episode)

    def get_batch(self, batch_size):
        """
        Returns a dictionary of batched (s, a, r, s', done) arrays.
        """

        if self.preload:
            indices = np.random.randint(0, self.demo_length, batch_size)
            choices = [self.demo_data[ind] for ind in indices]
        else:
            choices = [self._uniform_sample() for _ in range(batch_size)]

        obs = np.zeros((batch_size, self.observation_dim))
        acs = np.zeros((batch_size, self.action_dim))
        next_obs = np.zeros((batch_size, self.observation_dim))
        rewards = np.zeros((batch_size, 1))
        terminals = np.zeros((batch_size, 1), dtype='uint8')

        for i in range(batch_size):
            o, a, r, no, t = choices[i]
            obs[i] = o
            acs[i] = a[:self.action_dim] # fix this after fixing postproc
            next_obs[i] = no
            rewards[i] = r
            terminals[i] = int(t)

        return dict(
                    observations=obs,
                    actions=acs,
                    rewards=rewards,
                    terminals=terminals,
                    next_observations=next_obs,
                )