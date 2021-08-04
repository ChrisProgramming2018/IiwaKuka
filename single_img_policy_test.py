import argparse

import cv2
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.widgets import Slider

from gym_grasping.envs.robot_sim_env import RobotSimEnv
# from robot_io.input_devices.space_mouse import SpaceMouse
from robot_io.cams.realsenseSR300_librs2 import RealsenseSR300
from agent import TQC, stacked_frames, create_next_obs


class VisionPolicyTest:
    def __init__(self, args):
        self.cam = RealsenseSR300()
        state_dim = 200
        action_dim = 5
        max_action = float(1)
        min_action = float(-1)
        model = TQC(state_dim, action_dim, max_action, args)
        self.size = args.size
        self.args = args
        directory = "pretrained/"
        filename = "kuka_block_grasping-v0-97133reward_-1.05-agentTCQ"
        filename = directory + filename
        print("Load ", filename)
        model.load(filename)
        model.actor.training = False
        self.model = model

    # def run_3dmouse(self):
    #     mouse = SpaceMouse()
    #     prev_gripper_ac = 1
    #     while 1:
    #         action = mouse.handle_mouse_events()
    #         mouse.clear_events()
    #         obs, _, _, _ = env.step([action])
    #         if action[4] == -1 and prev_gripper_ac == 1:
    #             self.plot(obs[0])
    #         prev_gripper_ac = action[4]

    def main(self):
        rgb, _ = self.cam.get_image(flip_image=True, crop=True)
        rgb = cv2.resize(rgb, (84, 84))
        #rgb =cv2.imread('wi1.png')
        self.plot(rgb)



    def play_trajectory(self, filename):
        data = np.load(filename)
        initial_configuration = data["arr_0"]
        actions = data["arr_1"]
        state_obs = data["arr_2"]
        img_obs = data["arr_3"]
        i = 0
        while 1:
            img = img_obs[i]
            resized = cv2.resize(img[:, :, ::-1], (500, 500))
            cv2.imshow("window", resized)
            k = cv2.waitKey(0)
            if (k % 256 == 97 or k % 256 == 65) and i >= 0:
                i -= 1
            elif (k % 256 == 100 or k % 256 == 68) and i < len(img_obs) - 1:
                i += 1
            elif k % 256 == 87 or k % 256 == 119:
                self.plot(img)

    def policy_act(self, obs):
        obs, state_buffer = stacked_frames(obs, self.size, self.args, self.model)
        action = self.model.select_action(obs)
        print('action', action)
        #action = np.array([0.3,-0.5,0.8,-0.1,-1])
        action = np.clip(action, -1, 1)
        return action

    def visualize_action(self, action):
        h, w = 84, 84
        c_x, c_y = w // 2, h // 2

        def draw_vert_bar(img, start, end, c_x, c_y, width=2):
            img = img.copy()
            if end < start:
                start, end = end, start
            img[start: end, c_x - width // 2:c_x + width // 2] = 0
            return img

        def draw_hor_bar(img, start, end, c_x, c_y, width=2):
            img = img.copy()
            if end < start:
                start, end = end, start
            img[c_y - width // 2:c_y + width // 2, start:end] = 0
            return img

        x_y = np.ones((h, w, 3))
        x_y = draw_vert_bar(x_y, c_y, int(c_x - action[0] * h // 2), c_x, c_y)
        x_y = draw_hor_bar(x_y, int(c_y - action[1] * w // 2), c_x, c_x, c_y)

        z_rot = np.ones((h, w, 3))
        z_rot = draw_vert_bar(z_rot, c_y, int(c_x - action[2] * h // 2), c_x, c_y)
        z_rot = draw_hor_bar(z_rot, int(c_y + action[3] * w // 2), c_x, c_x, c_y)

        action[4] = (((np.clip(action[4], 0,
                               1) - 0) /
                      (1 - 0)) * 2) - 1
        op = np.ones((h, w, 3))
        op = draw_hor_bar(op, c_x - int((action[4] + 1) / 2 * (w / 2)),
                          c_x + int((action[4] + 1) / 2 * (w / 2)), c_x, c_y)

        return x_y, z_rot, op

    def plot(self, img):
        # rgb = rgb.resize((84, 84))
        original_img = img.copy()
        img = Image.fromarray(original_img)
        fig = plt.figure()
        ax1 = fig.add_subplot(241)
        ax2 = fig.add_subplot(242)
        ax3 = fig.add_subplot(243)
        ax4 = fig.add_subplot(244)
        fig.subplots_adjust(left=0, bottom=0)

        im1 = ax1.imshow(img)
        im2 = ax2.imshow(self.visualize_action(np.zeros(5))[0])
        im3 = ax3.imshow(self.visualize_action(np.zeros(5))[0])
        im4 = ax4.imshow(self.visualize_action(np.zeros(5))[0])

        bright = fig.add_axes([0.25, 0, 0.65, 0.03])
        con = fig.add_axes([0.25, 0.05, 0.65, 0.03])
        col = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        sharp = fig.add_axes([0.25, 0.15, 0.65, 0.03])
        zoom = fig.add_axes([0.25, 0.2, 0.65, 0.03])
        blur = fig.add_axes([0.25, 0.25, 0.65, 0.03])
        hue = fig.add_axes([0.25, 0.3, 0.65, 0.03])

        sbright = Slider(bright, 'bright', 0, 2, valinit=1)
        scon = Slider(con, 'con', 0, 2, valinit=1)
        scolor = Slider(col, 'color', 0, 2, valinit=1)
        ssharp = Slider(sharp, 'sharp', 0, 2, valinit=1)
        szoom = Slider(zoom, 'zoom', 0, 50, valinit=0)
        sblur = Slider(blur, 'blur', 0, 20, valinit=0)
        shue = Slider(hue, 'hue', -0.2, 0.2, valinit=0)

        def update(val):
            # rgb = original_img.copy()
            image = Image.fromarray(original_img)
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(scon.val)
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(sbright.val)
            color = ImageEnhance.Color(image)
            image = color.enhance(scolor.val)
            print('value brightnes', sbright.val)

            obs = np.array(image , dtype=np.uint8
                           )
            action = self.policy_act(obs)
            print('change',action)

            x_y, z_rot, grip = self.visualize_action(action)
            im2.set_array(x_y)
            im3.set_array(z_rot)
            im4.set_array(grip)

            fig.canvas.draw()

        sbright.on_changed(update)
        scon.on_changed(update)
        scolor.on_changed(update)
        ssharp.on_changed(update)
        szoom.on_changed(update)
        sblur.on_changed(update)
        shue.on_changed(update)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="kuka_block_grasping-v0", type=str,
                        help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=True, type=bool, help='use different seed for each episode')
    parser.add_argument('--epi', default=25, type=int)
    parser.add_argument('--max_episode_steps', default=50, type=int)
    parser.add_argument('--eval_freq', default=10000,
                        type=int)  # How often the evaluation step is performed (after how many timesteps)
    parser.add_argument('--repeat', default=1, type=int)  # every nth episode write in to tensorboard
    parser.add_argument('--max_timesteps', default=2e6, type=int)  # Total number of iterations/timesteps
    parser.add_argument('--lr-critic', default=0.0005, type=int)  # Total number of iterations/timesteps
    parser.add_argument('--lr-actor', default=0.0005, type=int)  # Total number of iterations/timesteps
    parser.add_argument('--lr_alpha', default=3e-4, type=float)
    parser.add_argument('--lr_decoder', default=1e-4, type=float)  # Divide by 5
    parser.add_argument('--save_model', default=True,
                        type=bool)  # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--expl_noise', default=0.1,
                        type=float)  # Exploration noise - STD value of exploration Gaussian noise
    parser.add_argument('--batch_size', default=256, type=int)  # Size of the batch
    parser.add_argument('--discount', default=0.99,
                        type=float)  # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type=float)  # Target network update rate
    parser.add_argument('--policy_freq', default=2,
                        type=int)  # Number of iterations to wait before the policy network (Actor model) is updated
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--target_update_freq', default=50, type=int)
    parser.add_argument('--num_q_target', default=4, type=int)  # amount of qtarget nets
    parser.add_argument('--train_every_step', default=True, type=bool)  # amount of qtarget nets
    parser.add_argument('--tensorboard_freq', default=5000, type=int)  # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)  # amount of qtarget nets
    parser.add_argument('--run', default=1, type=int)  # every nth episode write in to tensorboard
    parser.add_argument('--agent', default=None, type=str)  # load the weights saved after the given number
    parser.add_argument('--reward_scalling', default=1, type=int)  # amount
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--image_pad', default=4, type=int)  #
    parser.add_argument('--actor_clip_gradient', default=1., type=float)  # Maximum value of the Gaussian noise added to
    parser.add_argument('--locexp', type=str)  # Maximum value
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--no_render', type=bool)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--buffer_size', default=3e5, type=int)
    args = parser.parse_args()
    VIS = VisionPolicyTest(args)
    VIS.main()
    # vis.run_3dmouse()
    # VIS.play_trajectory(
    #     "/home/kuka/lang/robot/gym_grasping/gym_grasping/recordings/data/episode_3.npz")

