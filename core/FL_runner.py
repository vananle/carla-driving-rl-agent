import os
import time
import random

import numpy as np

import carla

from rl import CARLACollectWrapper, utils
from rl.environments.carla import env_utils as carla_utils
from rl import *

from core import CARLAEnv, CARLAgent

from typing import Union
import copy


class FL_Stage:
    """A Curriculum Learning FL_Stage"""

    def __init__(self, agent: dict, environment: dict, learning: dict, representation: dict = None,
                 collect: dict = None, imitation: dict = None, name='FL_Stage'):
        assert isinstance(agent, dict)
        assert isinstance(environment, dict)
        assert isinstance(learning, dict)

        # Agent
        self.agent_class = agent.pop('class', agent.pop('class_'))
        self.agent_args = agent
        self.agent = None

        assert isinstance(learning['agent'], dict)

        # Environment
        self.env_class = environment.pop('class', environment.pop('class_'))
        self.env_args = environment
        print(self.env_args)
        self.env = None

        # Representation
        if isinstance(representation, dict):
            self.should_do_repr_lear = True
            self.repr_args = representation
        else:
            self.should_do_repr_lear = False

        # Collect
        if isinstance(collect, dict):
            self.should_collect = True
            self.collect_args = collect

            assert isinstance(learning['collect'], dict)
        else:
            self.should_collect = False

        # Supervised Imitation:
        if isinstance(imitation, dict):
            self.should_imitate = True
            self.imitation_args = imitation
        else:
            self.should_imitate = False

        self.learn_args = learning
        self.name = name

    def init(self):
        if self.env is None:
            self.env = self.env_class(**self.env_args)
            self.agent = self.agent_class(self.env, **self.agent_args)

    def run(self, epochs: int, collect: Union[bool, int] = True, representation=True):
        assert epochs > 0
        self.init()

        if (collect is False) or (not self.should_collect):
            collect = 0
        elif collect is True:
            collect = epochs + 1

        for epoch in range(epochs):
            t0 = time.time()

            # collect -> representation learning -> rl
            if collect > 0:
                self.collect()
                collect -= 1

            if self.should_do_repr_lear and representation:
                self.representation_learning()

            self.reinforcement_learning()
            print(f'[Stage] Epoch {epoch + 1}/{epochs} took {round(time.time() - t0, 3)}s.')

        self.cleanup()

    def run2(self, epochs: int, copy_weights=True, epoch_offset=0) -> 'FL_Stage':
        assert epochs > 0
        self.init()

        for epoch in range(epochs):
            t0 = time.time()

            if self.should_imitate:
                self.imitation_learning()

            self.reinforcement_learning()
            print(f'[{self.name}] Epoch {epoch + 1}/{epochs} took {round(time.time() - t0, 3)}s.')

            if copy_weights:
                utils.copy_folder(src=self.agent.base_path, dst=f'{self.agent.base_path}-{epoch + epoch_offset}')

        self.cleanup()
        return self

    def evaluate(self, **kwargs) -> 'FL_Stage':
        self.init()
        self.agent.evaluate(**kwargs)
        return self

    def record(self, **kwargs) -> 'FL_Stage':
        self.init()
        self.agent.record(**kwargs)
        return self

    def collect(self):
        wrapper = CARLACollectWrapper(env=self.env, **self.collect_args)
        wrapper.collect(**self.learn_args['collect'])

    def representation_learning(self):
        self.agent.learn_representation(**self.repr_args)

    def imitation_learning(self):
        self.agent.imitation_learning(**self.imitation_args)

    def reinforcement_learning(self):
        self.agent.learn(**self.learn_args['agent'])
        return self.agent.get_weights()

    def update_weights(self, weights):
        self.agent.update_weights(weights)

    def cleanup(self):
        self.env.close()
        self.env = None
        self.agent = None


def define_agent(class_=CARLAgent, batch_size=128, consider_obs_every=4, load=False, **kwargs) -> dict:
    return dict(class_=class_, batch_size=batch_size, consider_obs_every=consider_obs_every, skip_data=1, load=load,
                **kwargs)


def define_env(image_shape=(90, 120, 3), render=True, town: Union[None, str] = 'Town01', window_size=(1080, 270),
               debug=False, port=2000, **kwargs) -> dict:
    return dict(class_=CARLAEnv, debug=debug, window_size=window_size, render=render, town=town,
                image_shape=image_shape, port=port, **kwargs)


def calculate_global_weights(client_weights, n_trained_clients):
    """
        Calculating the avg weights
    """
    global_weights = {}
    for client_idx, weights in enumerate(client_weights):
        for k, v in weights.items():
            if k not in global_weights.keys():
                global_weights[k] = [param / n_trained_clients for param in v]
            else:
                assert len(v) == len(global_weights[k])
                for idx, param in enumerate(v):
                    global_weights[k][idx] += param / n_trained_clients

    return global_weights


class FL_Learning:
    def __init__(self, n_clients, n_train_round):

        self.n_clients = n_clients  # number of clients

        self.n_train_round = n_train_round

        self.clients = []  # list of stages or clients

    def random_stage(self, episodes: int, timesteps: int, batch_size: int, save_every=None, seed=42,
                     stage_name='stage-random', load=False, env_port=2000, town='Town01', **kwargs):
        """random-stage: town with dense traffic (random vehicles and random pedestrians) + random light weather
        + data-aug"""
        policy_lr = kwargs.pop('policy_lr', 3e-4)
        value_lr = kwargs.pop('value_lr', 3e-4)
        clip_ratio = kwargs.pop('clip_ratio', 0.2)
        entropy = kwargs.pop('entropy_regularization', 0.1)
        dynamics_lr = kwargs.pop('dynamics_lr', 3e-4)

        agent_dict = define_agent(
            class_=CARLAgent, **kwargs,
            batch_size=batch_size, name=stage_name, traces_dir=None, load=load, seed=seed,
            advantage_scale=2.0, load_full=True,
            policy_lr=policy_lr,
            value_lr=value_lr,
            dynamics_lr=dynamics_lr,
            entropy_regularization=entropy, shuffle_batches=False, drop_batch_remainder=True, shuffle=True,
            clip_ratio=clip_ratio, consider_obs_every=1, clip_norm=1.0, update_dynamics=True)

        light_weathers = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.SoftRainNoon,
            carla.WeatherParameters.SoftRainSunset,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.WetSunset
        ]

        env_dict = define_env(town=town, debug=True, throttle_as_desired_speed=True,
                              image_shape=(90, 120, 3),
                              random_weathers=light_weathers,
                              spawn=dict(vehicles=50, pedestrians=50),
                              info_every=kwargs.get('repeat_action', 1),
                              disable_reverse=True, window_size=(900, 245),
                              port=env_port)

        return FL_Stage(agent=agent_dict, environment=env_dict,
                        learning=dict(
                            agent=dict(episodes=episodes, timesteps=timesteps, render_every=False, close=False,
                                       save_every=save_every)))

    def init_clients(self, env_ports, towns, timesteps=512):
        for i in range(self.n_clients):
            if i < 2:
                log_mode = 'summary'
            else:
                log_mode = 'log'

            self.clients.append(
                self.random_stage(stage_name=f'stage-random-client-{i}', episodes=1, timesteps=timesteps,
                                  batch_size=64, gamma=0.9999, lambda_=0.999, save_every='end',
                                  update_frequency=1, policy_lr=3e-5, value_lr=3e-5, dynamics_lr=3e-4,
                                  clip_ratio=0.125, entropy_regularization=1.0,
                                  seed_regularization=True,
                                  seed=i, polyak=1.0, aug_intensity=0.0, repeat_action=1,
                                  log_mode=log_mode, load=False, env_port=env_ports[i],
                                  town=towns[i]))

        for idx, client in enumerate(self.clients):
            print(f'|--- Init client {idx}')
            client.init()

    def train_clients(self):
        client_idx = np.arange(self.n_clients)

        global_weights = None

        for round_idx in range(self.n_train_round):
            print(f'|--- Start training round {round_idx}')

            if self.n_clients <= 3:
                random_client_idx = client_idx
            else:
                random_client_idx = np.random.choice(client_idx, size=int(0.6 * self.n_clients), replace=False)

            client_weights = []

            for client_id in random_client_idx:
                client = self.clients[client_id]

                print(f'|--- Start training client {client_id}')
                client_weight = client.reinforcement_learning()
                client_weights.append(copy.deepcopy(client_weight))
                print(f'|--- Finish training client {client_id}')

            global_weights = calculate_global_weights(client_weights,
                                                      n_trained_clients=len(random_client_idx))
            for client_id in client_idx:
                client = self.clients[client_id]
                client.update_weights(global_weights)
