"""
Script for extacting (s, a, r, s', done) pairs
from demo files.

It will output two pickle files.
"""

import os
import pickle
import numpy as np
from MujocoManip.miscellaneous.utils import postprocess_model_xml
from rlkit.envs.mujoco_manip_env import MujocoManipEnv

TASK = "SawyerBinsBreadEnv"
DEMO_FILE = "/home/robot/Downloads/test_extraction/bins-Bread0.pkl"
OUT_FILE = "/home/robot/Downloads/test_extraction/bins-Bread0-sars.pkl"

if __name__ == "__main__":
    env = MujocoManipEnv(TASK)
    # env = make(TASK,
    #            ignore_done=True,
    #            has_renderer=True,
    #            use_camera_obs=False,
    #            gripper_visualization=False,
    #            show_gripper_visualization=False,
    #            render_visual_mesh=True,
    #            reward_shaping=True)

    # load small pickle file (offsets) and big pickle file (data)
    with open(DEMO_FILE,'rb') as f:
        d = pickle.load(f)
        # a hacky way to tell if this is a small pickle files with indices into 
        # a bigger one, or if this is a pickle file with demos in it.
        if d[0] == 0:
            big = open(DEMO_FILE.replace('.pkl','.bkl'), 'rb')
        else:
            big = None

    # We'll maintain a big pickle file of all data and a small offsets array that
    # will be stored in a smaller pickle file.
    fname = OUT_FILE.replace('.pkl', '.bkl')
    big_pkl = open(fname, 'wb')
    offsets = [0]
    current_offset = 0
    for i, t in enumerate(d):

        if type(t) == int:
            big.seek(t)
            t = pickle.load(big)

        try:
            # env.env.reset()
            env.env.reset()
            env.env.reset_from_xml_string(postprocess_model_xml(t['model.xml']))

            # list of (s, a, r, s', done) tuples
            sarsd = []
            s = None
            a = None
            r = None
            ns = None

            for mj_state, ad in zip(t['states'], t['additional_data']):

                # force simulation state to recorded state
                env.env.sim.set_state_from_flattened(mj_state)
                env.env.sim.forward()

                # extract state and record (s, a, r, s') for previous timestep
                ob_dict = env.env._get_observation()
                ns = env._flatten_obs(ob_dict)
                if s is not None:
                    sarsd.append((s, a, r, ns, 0))
                s = np.array(ns)

                ### TODO: fix this here and in ccr... ###
                ### TODO: 1 is open, 0/-1 is closed ###
                ### TODO: if gripper cmd, then close, otherwise open, playback a traj to verify ###

                ### TODO: in ccr, mujoco sawyer env gripper commands are useless... ###
                ### set it such that true closes gripper, false opens it.. ###

                # extract action and reward
                gripper_pos = [0., 0.] if ad['gripper_cmd'] else [1., -1.]
                a = np.array(ad['controller_info']['joint_vel'].tolist() + gripper_pos)
                r = env.env.reward(None)

                # print(r)
                # env.env.render()

            print("final reward for demo {}: {}".format(i, r))
            # last state is treated as terminal
            r += 1. # boost artificially since sparse reward did not work?
            sarsd.append((s, a, r, ns, 1))

            # write pickle object to big pickle file, and keep track of offsets
            raw = pickle.dumps(sarsd)
            delta_offset = big_pkl.write(raw)
            current_offset += delta_offset
            offsets.append(current_offset)

        except Exception as e:
            print("ERROR on demo {}".format(i))
            print("ERROR: {}".format(e))

    # finally, write all of the offsets to the small pickle file
    small_pkl = open(OUT_FILE, 'wb')
    pickle.dump(offsets[:-1], small_pkl) # all but last, since thats the EOF
    big_pkl.close()
    small_pkl.close()



