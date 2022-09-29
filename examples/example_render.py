import argparse

from src.aimotive_dataset import AiMotiveDataset
from src.renderer import Renderer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example script for visualizing aiMotive Multimodal Dataset.')

    parser.add_argument("--root-dir", default="data",
                        type=str, help="Root dir of aiMotive Multimodal Dataset.")
    parser.add_argument("--split", default="train",
                        type=str, help="Data split. Options: [train, val]")
    args = parser.parse_args()

    train_dataset = AiMotiveDataset(args.root_dir, args.split)
    renderer = Renderer()
    for data in train_dataset:
        renderer.render(data)
