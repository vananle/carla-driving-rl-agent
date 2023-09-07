import argparse


def get_args():
    # create argument parser
    parser = argparse.ArgumentParser(
        description='RL-DRIVING', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario_name", type=str, default="s1",
                        choices=['s1', 's2', 's3', 's4', 's5', 'fl', 'all'],
                        help="an identifier to distinguish different experiment.")
    parser.add_argument('--n_client', type=int, help='number of clients (3)', default=3)
    parser.add_argument('--timesteps', type=int, help='number of timesteps per episode (512)', default=512)
    parser.add_argument('--n_train_round', type=int, help='number of training round (100)', default=100)
    parser.add_argument('--env_port', type=int, help='env_port (20000)', default=-1)

    args = parser.parse_args()

    return args
