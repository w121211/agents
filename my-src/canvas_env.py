from tf_agents.specs import tensor_spec
from tf_agents.networks import normal_projection_network
from tf_agents.networks import encoding_network
from tf_agents.networks import categorical_projection_network
import numpy as np
from PIL import Image, ImageDraw

import gym
from gym import error, spaces

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.specs import distribution_spec, tensor_spec
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
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(self.n_items),  # choosed item
                spaces.Discrete(5),  # dx
                spaces.Discrete(5),  # dy
                # spaces.Box(low=0, high=self.width, shape=(2,))
            ]
        )
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.n_items * 4,))
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


class ActorNet(network.DistributionNetwork):
    def __init__(self, input_spec, action_spec, name=None):
        output_spec = self._get_normal_distribution_spec(action_spec)
        super(DummyActorNet, self).__init__(
            input_spec, (), output_spec=output_spec, name='DummyActorNet')
        self._action_spec = action_spec
        self._flat_action_spec = tf.nest.flatten(self._action_spec)[0]

        self._layers.append(
            tf.keras.layers.Dense(
                self._flat_action_spec.shape.num_elements() * 2,
                kernel_initializer=tf.compat.v1.initializers.constant([[2.0, 1.0],
                                                                       [1.0, 1.0]]),
                bias_initializer=tf.compat.v1.initializers.constant([
                                                                    5.0, 5.0]),
                activation=None,
            ))

    def _get_normal_distribution_spec(self, sample_spec):
        input_param_shapes = tfp.distributions.Normal.param_static_shapes(
            sample_spec.shape)
        input_param_spec = tf.nest.map_structure(
            lambda tensor_shape: tensor_spec.TensorSpec(  # pylint: disable=g-long-lambda
                shape=tensor_shape,
                dtype=sample_spec.dtype),
            input_param_shapes)

        return distribution_spec.DistributionSpec(
            tfp.distributions.Normal, input_param_spec, sample_spec=sample_spec)

    def call(self, inputs, step_type=None, network_state=()):
        del step_type
        hidden_state = tf.cast(tf.nest.flatten(inputs), tf.float32)[0]

        # Calls coming from agent.train() has a time dimension. Direct loss calls
        # may not have a time dimension. It order to make BatchSquash work, we need
        # to specify the outer dimension properly.
        has_time_dim = nest_utils.get_outer_rank(inputs,
                                                 self.input_tensor_spec) == 2
        outer_rank = 2 if has_time_dim else 1
        batch_squash = network_utils.BatchSquash(outer_rank)
        hidden_state = batch_squash.flatten(hidden_state)

        for layer in self.layers:
            hidden_state = layer(hidden_state)

        actions, stdevs = tf.split(hidden_state, 2, axis=1)
        actions = batch_squash.unflatten(actions)
        stdevs = batch_squash.unflatten(stdevs)
        actions = tf.nest.pack_sequence_as(self._action_spec, [actions])
        stdevs = tf.nest.pack_sequence_as(self._action_spec, [stdevs])

        return self.output_spec.build_distribution(
            loc=actions, scale=stdevs), network_state


def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
    return categorical_projection_network.CategoricalProjectionNetwork(
        action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):
    std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        init_means_output_factor=init_means_output_factor,
        std_bias_initializer_value=std_bias_initializer_value,
        scale_distribution=False)


@gin.configurable
class ActorDistributionNetwork(network.DistributionNetwork):
    """Creates an actor producing either Normal or Categorical distribution.

    Note: By default, this network uses `NormalProjectionNetwork` for continuous
    projection which by default uses `tanh_squash_to_spec` to normalize its
    output. Due to the nature of the `tanh` function, values near the spec bounds
    cannot be returned.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(200, 100),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 discrete_projection_net=_categorical_projection_net,
                 continuous_projection_net=_normal_projection_net,
                 name='ActorDistributionNetwork'):
        """Creates an instance of `ActorDistributionNetwork`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input.
          output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
            the output.
          preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations.
            All of these layers must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them. Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
            This layer must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size,
            stride).
          fc_layer_params: Optional list of fully_connected parameters, where each
            item is the number of units in the layer.
          dropout_layer_params: Optional list of dropout layer parameters, each item
            is the fraction of input units to drop or a dictionary of parameters
            according to the keras.Dropout documentation. The additional parameter
            `permanent', if set to True, allows to apply dropout at inference for
            approximated Bayesian inference. The dropout layers are interleaved with
            the fully connected layers; there is a dropout layer after each fully
            connected layer, except if the entry in the list is None. This list must
            have the same length of fc_layer_params, or be None.
          activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
          kernel_initializer: Initializer to use for the kernels of the conv and
            dense layers. If none is provided a default glorot_uniform
          batch_squash: If True the outer_ranks of the observation are squashed into
            the batch dimension. This allow encoding networks to be used with
            observations with shape [BxTx...].
          dtype: The dtype to use by the convolution and fully connected layers.
          discrete_projection_net: Callable that generates a discrete projection
            network to be called with some hidden state and the outer_rank of the
            state.
          continuous_projection_net: Callable that generates a continuous projection
            network to be called with some hidden state and the outer_rank of the
            state.
          name: A string representing name of the network.

        Raises:
          ValueError: If `input_tensor_spec` contains more than one observation.
        """

        if not kernel_initializer:
            kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

        encoder = encoding_network.EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype)

        def map_proj(spec):
            if tensor_spec.is_discrete(spec):
                return discrete_projection_net(spec)
            else:
                return continuous_projection_net(spec)

        projection_networks = tf.nest.map_structure(
            map_proj, output_tensor_spec)
        output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                            projection_networks)

        super(ActorDistributionNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            output_spec=output_spec,
            name=name)

        self._encoder = encoder
        self._projection_networks = projection_networks
        self._output_tensor_spec = output_tensor_spec

    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec

    def call(self, observations, step_type, network_state, training=False):
        state, network_state = self._encoder(
            observations,
            step_type=step_type,
            network_state=network_state,
            training=training)
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        output_actions = tf.nest.map_structure(
            lambda proj_net: proj_net(state, outer_rank), self._projection_networks)
        return output_actions, network_state
