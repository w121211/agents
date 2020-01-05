r"""模擬一個極簡單的圖像操作環境，用PIL實現圖像操作功能
* 實驗：
"""

import gin
import numpy as np
from PIL import Image, ImageDraw

import gym
from gym import error, spaces
import tensorflow as tf
import tensorflow_probability as tfp


from tf_agents.networks import network
from tf_agents.specs import distribution_spec, tensor_spec
from tf_agents.networks import normal_projection_network, encoding_network, categorical_projection_network
from tf_agents.networks import utils as network_utils
from tf_agents.utils import nest_utils


CANVAS_WIDTH = 64
OBJ_WIDTH = 10
N_ITEMS = 2


class CanvasEnv(gym.Env):
    def __init__(self):
        self.max_step = 10
        self.cur_step = 0
        self.width = CANVAS_WIDTH
        self.obj_wh = np.array([OBJ_WIDTH, OBJ_WIDTH],
                               dtype=np.float32) / self.width
        self.n_items = N_ITEMS

        # self.observation_space = spaces.Dict(
        #     {
        #         "target_im": spaces.Box(
        #             low=0, high=1, shape=(self.width, self.width, 1)
        #         ),  # (H, W, C)
        #         "cur_im": spaces.Box(
        #             low=0, high=1, shape=(self.width, self.width, 1)
        #         ),  # (H, W, C)
        #         "cur_coord": spaces.Box(low=-10, high=10, shape=(self.n_items, 4)),
        #     }
        # )
        self.observation_space = spaces.Dict(
            {
                # (H, W, C)
                "im": spaces.Box(low=0, high=1, shape=(self.width, self.width, 1)),
                "coord": spaces.Box(low=-10, high=10, shape=(self.n_items, 4)),
            }
        )
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(self.n_items),  # choosed item
                spaces.Discrete(5),  # dx
                spaces.Discrete(5),  # dy
                # spaces.Box(low=0, high=self.width, shape=(2,))
            ]
        )
        # self.observation_space = spaces.Box(
        #     low=-10, high=10, shape=(self.n_items * 4,))
        # self.action_space = spaces.Discrete(5)

        self.action_map = np.array(
            [-8, 8, -1, 1, 0], dtype=np.float32) / self.width

        self.target_im = None
        self.target_coord = None  # (n_obj, 4=(x0, y0, x1, y1))
        self.cur_im = None
        self.cur_coord = None  # (n_obj, 4=(x0, y0, x1, y1))
        self.viewer = None

    def reset(self):
        self.cur_step = 0
        xy0 = np.random.rand(self.n_items, 2)
        self.target_coord = np.concatenate(
            [xy0, xy0 + np.tile(self.obj_wh, (self.n_items, 1))], axis=1
        )
        self.cur_coord = np.tile(
            np.array([0, 0, *tuple(self.obj_wh)],
                     dtype=np.float32), (self.n_items, 1)
        )
        self.target_im = self._render(self.target_coord)
        self.cur_im = self._render(self.cur_coord)
        return self._obs()

    def step(self, action):
        """
        Args:
            action: [int, int]
        Return:
            obs: target_im (H, W, C), cur_im (H, W, C), field_info (x0, y0)
        """
        reward = 1

        self.cur_step += 1
        done = self.cur_step >= self.max_step

        return self._obs(), reward, done, {}

    def _step(self, action):
        """
        Args:
            action: [int, int]
        Return:
            obs: target_im (H, W, C), cur_im (H, W, C), field_info (x0, y0)
        """
        print(action)
        # print(self.action_map[action[0]], self.action_map[action[1]])
        idx = action[0]
        dmove = np.array(
            [self.action_map[action[1]], self.action_map[action[2]]], dtype=np.float32
        )
        xy0 = self.cur_coord[idx, :2] + dmove
        self.cur_coord[idx] = np.concatenate([xy0, xy0 + self.obj_wh], axis=0)
        self.cur_im = self._render(self.cur_coord)

        reward = self._reward(self.cur_coord, self.target_coord)

        self.cur_step += 1
        done = self.cur_step >= self.max_step

        return self._obs(), reward, done, {}

    def _render(self, coord: np.array):
        """
        Args: coord: (n_items, 4)
        """
        coord = (coord * self.width).astype(np.int16)
        im = Image.new("L", (self.width, self.width))
        draw = ImageDraw.Draw(im)
        for i, c in enumerate(coord):
            if i == 0:
                draw.rectangle(tuple(c), fill=255)
            else:
                draw.ellipse(tuple(c), fill=255)
        x = np.array(im, dtype=np.float32) / 255.0  # normalize
        x = np.expand_dims(x, axis=-1)  # (H, W, C=1)
        return x

    def _obs(self):
        # return {
        #     "target_im": self.target_im,
        #     "cur_im": self.cur_im,
        #     "cur_coord": self.cur_coord,
        # }
        return self.cur_coord.flatten()

    def _reward(self, xy_a: np.array, xy_b: np.array):
        dist = np.linalg.norm(xy_a - xy_b, axis=1)
        r = -1 * dist / 2 + 1
        r = np.clip(r, -1, None)
        r = np.sum(r)
        #     elif r > 0:
        #         r *= 0.05 ** self.cur_step  # 衰退因子
        return r

    def _denorm(self, a: np.array):
        return (a * self.width).astype(np.int16)

    def _regression_step(self, action):
        """@Deprecated
        Args:
            action: list[obj_id: int, coord: np.array]
        Return:
            obs: target_im (H, W, C), cur_im (H, W, C), field_info (x0, y0)
        """
        obj_id, coord = action
        coord *= self.width
        x0, y0 = coord
        self.obj_status = (
            np.array([[x0, y0, (x0 + self.obj_w), (y0 + self.obj_w)]],
                     dtype=np.float32)
            / self.width
        )
        self.cur_im = self._render(x0, y0)
        reward = -(
            ((x0 - self.target_coords[0, 0]) ** 2)
            + ((y0 - self.target_coords[0, 1]) ** 2)
        )
        self.cur_step += 1
        done = self.cur_step >= self.max_step
        return self._obs(), reward, done, {"episode": {"r": reward}}

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass
