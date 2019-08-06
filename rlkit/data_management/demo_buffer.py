"""
A batch memory for storing transitions.
Adapted from OpenAI baselines.
"""

import numpy as np
import random
import sys
import h5py

class DemoBuffer:
    def __init__(self, size):
        """
        Create a batch memory to store transitions.

        Args:
            size (int): maximum number of transitions that can
                be stored in memory before old transitions are
                overwritten
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._done_idxes = {} # stores terminal transition indices

        self._disable_new_data = False

    def __len__(self):
        return len(self._storage)

    def load_from_hdf5(self, hdf5_path):
        """
        Loads transitions from an hdf5 file.

        Args:
            hdf5_path (str): path to the hdf5 file
                to load the transitions from

            skip_every_n_steps (int): if provided, the transitions
                in the hdf5 will be subsampled by this factor

            split1 (float): if provided, it determines the left boundary of a
                subsequence of the demonstrations to keep

            split2 (float): if provided, it determines the right boundary of a
                subsequence of the demonstrations to keep
        """
        assert(len(self) == 0)

        f = h5py.File(hdf5_path, "r")  
        demos = list(f["data"].keys())
        total_transitions = f["data"].attrs["total"]
        print("Loading {} transitions from {}...".format(total_transitions, hdf5_path))

        sample_count = 0
        for i in range(len(demos)):
            ep = demos[i]
            obs = f["data/{}/obs".format(ep)][()]
            actions = f["data/{}/actions".format(ep)][()]
            rewards = f["data/{}/rewards".format(ep)][()]
            next_obs = f["data/{}/next_obs".format(ep)][()]
            dones = f["data/{}/dones".format(ep)][()]
            if "data/{}/states".format(ep) in f:
                states = f["data/{}/states".format(ep)][()]
            else:
                states = [None] * len(obs)

            ### important: this is action clipping! ###
            actions = np.clip(actions, -1., 1.)

            zipped = zip(obs, actions, rewards, next_obs, dones, states)
            for item in zipped:
                ob, ac, rew, next_ob, done, mjstate = item
                self.add(ob, ac, rew, next_ob, done, mjstate, ep)
                sample_count += 1
        f.close()

        self._disable_new_data = True

    def add(self, obs_t, action, reward, obs_tp1, done, mjstate, ep=None):
        """
        Adds a transition to the memory.
        """
        if self._disable_new_data:
            raise Exception("BatchMemory: tried to add new data when adding data is disabled.")
        data = (obs_t, action, reward, obs_tp1, done, mjstate, ep)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._done_idxes.pop(self._next_idx, None) # remove entry in done indices if it exists
            self._storage[self._next_idx] = data
        if done:
            self._done_idxes[self._next_idx] = 1 # record index of terminal transition
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """
        Internal function to index into the memory and collect
        tuples into numpy arrays.
        """
        obses_t, actions, rewards, obses_tp1, dones, mjstates, eps = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, mjstate, ep = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            mjstates.append(mjstate)
            eps.append(ep)

        return np.array(obses_t), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(obses_tp1), np.array(dones).reshape(-1, 1), np.array(mjstates), list(eps)

    def random_batch(self, batch_size):
        """
        Sample a batch of experiences.

        Args:
            batch_size (int): number of transitions to sample
            terminal_fraction (float): if provided, should be a value in [0, 1].
                this allows for sampling a guaranteed fraction of terminal transitions
                from the buffer.

        Returns:
            obs_batch (np.array): batch of next observations

            act_batch (np.array): batch of actions 

            rew_batch (np.array): batch of rewards

            next_obs_batch (np.array): batch of next observations

            done_mask (np.array): done_mask[i] = 1 if the
                transition is terminal
        """
        if len(self._storage) < batch_size:
            return None
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        obs, acts, rews, obs_next, dones, _, _ = self._encode_sample(idxes)
        batch = dict(
            observations=obs,
            actions=acts,
            rewards=rews,
            terminals=dones,
            next_observations=obs_next,
        )
        return batch

