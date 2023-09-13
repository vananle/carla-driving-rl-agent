import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pygame
import logging
import tensorflow as tf

tf.test.is_gpu_available()

from core.FL_runner import FL_Learning
from core import learning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from args import get_args


def record(amount: int, steps: int, town: str, weights: str, seed=42):
    """Utils for saving images of the agent"""
    import carla
    weather = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.HardRainSunset,
        carla.WeatherParameters.MidRainSunset,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.WetCloudySunset
    ]

    s = learning.stage_s5(episodes=5, timesteps=512, batch_size=64, gamma=0.9999, lambda_=0.999, save_every='end',
                          update_frequency=1, policy_lr=1e-5, value_lr=1e-5, dynamics_lr=1e-5, stage_name=weights,
                          clip_ratio=0.1, entropy_regularization=1.0, seed_regularization=True, traffic='regular',
                          seed=42, polyak=1.0, aug_intensity=0.0, repeat_action=1, town=town, weather=weather)

    s.record(name=weights, timesteps=steps, trials=amount, seed=seed, close=True)


if __name__ == '__main__':

    args = get_args()

    timesteps = args.timesteps

    scenario_name = args.scenario_name
    n_train_round = args.n_train_round
    env_port = args.env_port
    traffic_manager_port = args.traffic_manager_port
    if scenario_name != 'fl' and env_port < 0:
        raise ValueError('Must specify the port of carla server!')

    if scenario_name == 's1':

        # CURRICULUM LEARNING:
        # -- STAGE-1 --
        stage1 = learning.stage_s1(episodes=1, timesteps=timesteps, batch_size=64, gamma=0.9999, lambda_=0.999,
                                   save_every='end',
                                   update_frequency=1, policy_lr=3e-4, value_lr=3e-4, dynamics_lr=3e-4,
                                   clip_ratio=0.2, entropy_regularization=1.0, seed_regularization=True, load=False,
                                   seed=51, polyak=1.0, aug_intensity=0.0, repeat_action=1, load_full=False,
                                   port=env_port, traffic_manager_port=traffic_manager_port)

        stage1.run2(epochs=n_train_round, epoch_offset=0)
    elif scenario_name == 's2':

        # -- STAGE-2 --
        stage2 = learning.stage_s2(episodes=1, timesteps=timesteps, batch_size=64, gamma=0.9999, lambda_=0.999,
                                   save_every='end',
                                   update_frequency=1, policy_lr=3e-5, value_lr=3e-5, dynamics_lr=3e-4,
                                   clip_ratio=0.15, entropy_regularization=2.0, seed_regularization=True,
                                   seed=51, polyak=1.0, aug_intensity=0.0, repeat_action=1,
                                   port=env_port, traffic_manager_port=traffic_manager_port)

        stage2.run2(epochs=n_train_round, epoch_offset=0)
    elif scenario_name == 's3':

        # -- STAGE-3 --

        stage3 = learning.stage_s3(episodes=1, timesteps=timesteps, batch_size=258, gamma=0.9999, lambda_=0.999,
                                   save_every='end',
                                   update_frequency=1, policy_lr=3e-5, value_lr=3e-5, dynamics_lr=3e-4,
                                   clip_ratio=0.125, entropy_regularization=1.0, seed_regularization=True,
                                   seed=51, polyak=1.0, aug_intensity=0.0, repeat_action=1, town='Town01',
                                   port=env_port, traffic_manager_port=traffic_manager_port)

        stage3.run2(epochs=n_train_round, epoch_offset=0)
        stage3.record(name='stage-s3', timesteps=5000, trials=1, seed=51, close=True)
    elif scenario_name == 's4':

        # -- STAGE-4 --
        stage4 = learning.stage_s4(episodes=1, timesteps=timesteps, batch_size=64, gamma=0.9999, lambda_=0.999,
                                   save_every='end',
                                   update_frequency=1, policy_lr=1e-5, value_lr=1e-5, dynamics_lr=3e-5,
                                   clip_ratio=0.1, entropy_regularization=1.0, seed_regularization=True,
                                   seed=51, polyak=1.0, aug_intensity=1.0, repeat_action=1,
                                   port=env_port, traffic_manager_port=traffic_manager_port)

        stage4.run2(epochs=n_train_round, epoch_offset=0)
    elif scenario_name == 's5':
        # -- STAGE-5 --
        stage5 = learning.stage_s5(episodes=1, timesteps=timesteps, batch_size=64, gamma=0.9999, lambda_=0.999,
                                   save_every='end',
                                   update_frequency=1, policy_lr=1e-5, value_lr=1e-5, dynamics_lr=1e-5,
                                   clip_ratio=0.1, entropy_regularization=1.0, seed_regularization=True,
                                   seed=51, polyak=1.0, aug_intensity=0.8, repeat_action=1, town='Town03',
                                   port=env_port, traffic_manager_port=traffic_manager_port)

        stage5.run2(epochs=n_train_round, epoch_offset=0)

    elif scenario_name == 'fl':
        towns = ['Town01', 'Town02', 'Town03']

        n_client = args.n_client
        if n_client > len(towns):
            n_client = len(towns)

        # FL training
        env_ports = np.arange(20000, 20000 + n_client)

        fl_train = FL_Learning(n_clients=len(env_ports), n_train_round=n_train_round)
        fl_train.init_clients(env_ports=env_ports, towns=towns, timesteps=timesteps)
        fl_train.train_clients()
    else:
        # EVALUATION:
        towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']

        for mode in ['train', 'test']:
            for town in towns:
                for traffic in ['no', 'regular', 'dense']:
                    for num_steps in [timesteps]:
                        print(f'Evaluating [mode={mode}, town={town}, traffic={traffic}, steps={num_steps}]')
                        learning.evaluate(mode, town=town, steps=num_steps, seeds=[42], traffic=traffic)
    pygame.quit()
