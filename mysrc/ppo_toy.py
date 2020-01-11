import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network, encoding_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common

from envs.toy import RectEnv

tf.compat.v1.enable_v2_behavior()


def train():
    num_parallel_environments = 2
    collect_episodes_per_iteration = 2  # 30

    tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
        [lambda: tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(RectEnv()))] * num_parallel_environments))

    print(tf_env.time_step_spec())
    print(tf_env.action_spec())
    print(tf_env.observation_spec())

    preprocessing_layers = {
        'target': tf.keras.models.Sequential([
            # tf.keras.applications.MobileNetV2(
            #     input_shape=(64, 64, 1), include_top=False, weights=None),
            tf.keras.layers.Conv2D(1, 6),
            tf.keras.layers.Flatten()]),
        'canvas':  tf.keras.models.Sequential([
            # tf.keras.applications.MobileNetV2(
            #     input_shape=(64, 64, 1), include_top=False, weights=None),
            tf.keras.layers.Conv2D(1, 6),
            tf.keras.layers.Flatten()]),
        'coord': tf.keras.models.Sequential([tf.keras.layers.Dense(5),
                                             tf.keras.layers.Flatten()])
    }
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner)
    value_net = value_network.ValueNetwork(
        tf_env.observation_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner)

    tf_agent = ppo_agent.PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=actor_net,
        value_net=value_net,
        normalize_observations=False,
        use_gae=False)
    tf_agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=num_parallel_environments)
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        tf_agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=collect_episodes_per_iteration)
    # print(tf_agent.collect_data_spec)

    def train_step():
        trajectories = replay_buffer.gather_all()
        return tf_agent.train(experience=trajectories)

    collect_driver.run = common.function(collect_driver.run, autograph=False)
    # tf_agent.train = common.function(tf_agent.train, autograph=False)
    # train_step = common.function(train_step)

    # for _ in range(10):
    collect_driver.run()
    # total_loss, _ = train_step()
    # replay_buffer.clear()

    # collect_driver.run()
    # trajectories = replay_buffer.gather_all()
    # # # print(trajectories)
    # loss, _ = tf_agent.train(trajectories)
