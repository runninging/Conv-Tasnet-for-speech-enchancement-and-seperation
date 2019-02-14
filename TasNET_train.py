import pdb
import argparse
import time
from TasNET_model import TasNET
from trainer import TasNET_trainer
from dataset import TasDataset
from torch.utils.data import DataLoader
from utils import parse_yaml


def train(args):
    config_dict = parse_yaml(args.config)

    loader_config = config_dict["dataloader"]
    train_config = config_dict["trainer"]
    temp = config_dict["temp"]

    train_dataset = TasDataset(loader_config["train_path_npz"])
    valid_dataset = TasDataset(loader_config["valid_path_npz"])

    train_loader = DataLoader(train_dataset, batch_size=loader_config["batch_size"], shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=loader_config["batch_size"], shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)

    tasnet = TasNET()
    trainer = TasNET_trainer(tasnet, **train_config)
    trainer.run(train_loader, valid_loader)
    #trainer.rerun(train_loader, valid_loader, temp["model_path"], temp["epoch_done"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Speech Enhancement Neural Network by PyTorch ")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    args = parser.parse_args()
    train(args)
