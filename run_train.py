import sys
import os
import hydra
from omegaconf import DictConfig
from train import train_lumina


@hydra.main(
    config_path="./", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    train_lumina(cfg)


if __name__ == "__main__":
    main()
 