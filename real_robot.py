import os
import math
import cv2
import json
import torch
from gym_grasping.envs.iiwa_env import IIWAEnv
#from agent_surface import TQC
from agent import TQC
from helper import FrameStack
import argparse
import numpy as np
from robot_io.input_devices.space_mouse import SpaceMouse
from datetime import datetime
print(torch.__version__)




def main(args):
    joint_states =  (0, 25, 0, -80, 0, 75, -45) # (-90, 25, 0, -80, 0, 75, -45)
    cartesian_pose = (0, -0.56, 0.2, math.pi, 0, math.pi / 2)
    env = IIWAEnv(act_type='continuous', freq=20, obs_type='image', dv=0.01,
                  drot=0.2, use_impedance=True, max_steps=200, gripper_delay=0,
                  reset_pose=joint_states)  # (-90, 30, 0, -90, 0, 60, 0)
    env = FrameStack(env, args)
    # snapshot = "/tmp/models/from_snapshot/aug_loss_weight_0.01_aug_dataset_batch_size_128/2020-07-26_18-02_seed-80/save/ppo/stackVel_brown_no_dr-v0_975.pt"
    state_dim = 512
    state_dim = 200
    action_dim = 5
    max_action = float(1)
    model = TQC(state_dim, action_dim, args)
    size = args['size']
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    p = "results/" + str(dt_string)
    if not os.path.exists(p):
        os.makedirs(p)
    #directory = "20.03_32/pytorch_models/"   # only sim model (predicter)
    #directory = "08_05_seed_3/pytorch_models/"
    #directory = "10_05_seed_4/pytorch_models/"
    directory = "pytorch_models/"   # sim and real model (predicter)
    #directory = "model_to_test_juli/seed_4/model_seed_4/"  # different predicter
    #directory = "seed_3/model_seed_3/"
    #directory = "working_predicter_models/model_seed_1/"
    # juli 13
    directory = "experiments_juli/test_models_working_predicter/model_seed_3/"
    directory = "experiments_juli/test_sim_data_model/model_seed_0/"
    directory = "experiments_juli/test_real_data_model/model_seed_0/"
    directory = "experiments_juli/test_sim_real_predicter_juli/model_seed_3/"
    directory = "test_gray_model/model_seed_2/"

    #filename = 'kuka_block_grasping-v0-10472reward_-0.90'   # only sim model (predicter)
    #filename = 'kuka_block_grasping-v0-5007reward_-1.29'   # sim and real model (predicter)
    filename = 'kuka_block_grasping-v0-4058reward_-1.16'  # sim and real model working
    #filename = 'kuka_block_grasping-v0-4031reward_-1.69' # seed 4 different surface
    #filename = "kuka_block_grasping-v0-4002reward_-1.69" # seed 3 differet surface
    #filename = "kuka_block_grasping-v0-4017reward_-1.63" # seed 1 working
    #filename = "kuka_block_grasping-v0-3400reward_-1.14" # not gripping
    #filename = "kuka_block_grasping-v0-3300reward_-1.28"
    #filename = 'kuka_block_grasping-v0-6144reward_-1.08'
    filename = "kuka_block_grasping-v0-3300reward_-1.28" # seed 0 working not grasping
    filename = "kuka_block_grasping-v0-3300reward_-1.32" # working pred seed 3 works
    filename = "kuka_block_grasping-v0-4000reward_-1.10" #  working predseed 3 works
    filename = "kuka_block_grasping-v0-4100reward_-3.63"
    filename = "kuka_block_grasping-v0-4000reward_-1.08"

    filename = directory + filename
    model.load(filename)
    print("...weights agent loaded")
    t=0
    obs = env.reset()
    print(obs.shape)
    #sys.exit()
    #obs = preprocess_image(obs)
    #obs, state_buffer = stacked_frames(obs, size, args, model)

    while True:

        #print('obs', obs.shape)

        action = model.select_action(obs)
        #print(action)
        obs, reward, done, info = env.step(action)
        #obs = preprocess_image(obs)
        #frame = cv2.imwrite("wi{}.png".format(t), np.array(obs))

        img = obs.copy()
        img = np.array(img)[:, :, [2, 1, 0]]
        frame = cv2.imwrite(p + "/wi{}.png".format(t), cv2.resize(img, (300,300)))
        cv2.imshow("win2", cv2.resize(img, (300,300)))
        cv2.waitKey(1)
        t += 1
        #print("step ", t)
        if t == 150:
            env.reset()
            break

def test():
    img = cv2.imread('wi2.png')
    #img = cv2.resize(img, (500, 500))
    #cv2.imshow("win2", img.transpose(1, 2, 0))
    img = preprocess_image(img)
    while True:
        cv2.imshow("win2", img)
        cv2.waitKey(100)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="experiments/kuka", type=str)
    args= parser.parse_args()
    path = args.locexp
    # experiment_name = args.experiment_name
    res_path = os.path.join(path, "results")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    dir_model = os.path.join(path, "pytorch_models")
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    print("Created model dir {} ".format(dir_model))
    with open(args.param, "r") as f:
        param = json.load(f)
    action_dim = 5
    print("action_dim ", action_dim)

    param["target_entropy"] = -np.prod(action_dim)
    print('call policy')
    main(param)
    #main(arg)




















