import sys
import os
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.adapter.trainer import train_adapter


@hydra.main(
    config_path="../configs", config_name="lora", version_base=None
)
def main(cfg: DictConfig):
    train_adapter(cfg)


if __name__ == "__main__":
    main()