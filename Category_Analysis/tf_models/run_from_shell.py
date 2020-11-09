# Arguments
import argparse
from configs.config_utils import get_config

from train import train


if __name__ == '__main__':

    ### Load arguments ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/default.yml')

    args = parser.parse_args()
    
    train(config_path=args.config_path)