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


CANVAS_WIDTH = 128
# RECT_WIDTH = 10
RECTS_WH = [[10, 10], [20, 10], [10, 20]]


class RectEnv(gym.Env):

    def __init__(self):
        self.max_step = 10
        self.cur_step = 0
        self.width = CANVAS_WIDTH
        self.rects_wh = np.array(RECTS_WH, dtype=np.int32)
        self.n_items = len(RECTS_WH)
        self.action_map = np.array(
            [-8, 8, -1, 1, 0], dtype=np.float) / self.width

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
        # self.observation_space = spaces.Dict(
        #     {
        #         # (H, W, C)
        #         "im": spaces.Box(low=0, high=1, shape=(self.width, self.width, 1)),
        #         "coord": spaces.Box(low=-10, high=10, shape=(self.n_items, 4)),
        #     }
        # )
        self.observation_space = spaces.Dict({
            # target image, (H, W, C=1)
            "target": spaces.Box(low=0, high=1, shape=(self.width, self.width, 1)),
            # current canvas, (H, W, C=1)
            "canvas": spaces.Box(low=0, high=1, shape=(self.width, self.width, 1)),
            # current rectangle coords, (N, 4)
            "coord": spaces.Box(low=-10, high=10, shape=(self.n_items, 4)),
        })

        self.action_space = spaces.Tuple([
            spaces.Discrete(self.n_items),  # i_item
            spaces.Discrete(len(self.action_map)),  # i_dx
            spaces.Discrete(len(self.action_map)),  # i_dy
            # spaces.Box(low=0, high=self.width, shape=(2,))
        ])

        self.target_coord = None  # (n_obj, 4=(x0, y0, x1, y1))
        self.cur_coord = None  # (n_obj, 4=(x0, y0, x1, y1))
        self.viewer = None

    def reset(self):
        self.cur_step = 0

        cxy = np.random.randint(0, self.width, self.n_items)
        self.target_coord = np.concatenate(
            [cxy-wh/2, cxy+wh/2], axis=1).astype(np.int32)
        self.cur_coord = np.array(
            [np.concatenate([[0, 0], wh]) for wh in self.rects_wh], dtype=np.int32)

        return self._obs()

    def step(self, action):
        """
        Args:
            action, (i_rect, i_dx, i_dy)
        Return:
            obs: target_im (H, W, C), cur_im (H, W, C), field_info (x0, y0)
        """
        # print(action)
        # print(self.action_map[action[0]], self.action_map[action[1]])
        i = action[0]
        dxy = np.array(
            [self.action_map[action[1]], self.action_map[action[2]]], dtype=np.float
        )
        xy0 = self.cur_coord[i, :2] + dxy
        self.cur_coord[i] = np.concatenate(
            [xy0, xy0 + self.rects_wh[i]], axis=0)

        reward = self._reward(self.cur_coord / self.width,
                              self.target_coord / self.width)
        done = self.cur_step >= self.max_step
        self.cur_step += 1

        return self._obs(), reward, done, {}

    def _render(self, coord: np.ndarray) -> np.ndarray:
        """Args: coord: (n_items, 4)"""
        im = Image.new("L", (self.width, self.width))
        draw = ImageDraw.Draw(im)

        for _, c in enumerate(coord):
            draw.rectangle(tuple(c), fill=255)

        # normalize & transform image
        x = np.array(im, dtype=np.float) / 255.0
        x = np.expand_dims(x, axis=-1)  # (H, W, C=1)
        return x

    def _obs(self):
        return {
            "target": self._render(self.target_coord),
            "canvas": self._render(self.cur_coord),
            "coord": self.cur_coord / self.width
        }

    def _reward(self, a_xy: np.array, b_xy: np.array):
        """越遠離target_coord (ie, L2-norm)，獎勵越低（負獎勵）"""
        d = np.linalg.norm(a_xy - b_xy, axis=1)
        r = -1 * d / 2 + 1
        r = np.clip(r, -1, None)
        r = np.sum(r)
        #     elif r > 0:
        #         r *= 0.05 ** self.cur_step  # 衰退因子
        return r

    def _denorm(self, a: np.array):
        return (a * self.width).astype(np.int16)

    def _step(self, action):
        """@Debug"""
        reward = 1

        self.cur_step += 1
        done = self.cur_step >= self.max_step

        return self._obs(), reward, done, {}

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
                     dtype=np.float)
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
