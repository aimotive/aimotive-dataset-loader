import os

from typing import List

from src.data_loader import DataLoader, DataItem
from src.sequence import Sequence


class AiMotiveDataset:
    """
    Multimodal Autonomous Driving dataset.
    The dataset consists of four cameras, two radars, one lidar sensor, and corresponding
    3D bounding box annotations of dynamic objects.

    Attributes:
        dataset_index: a list of keyframe paths
        data_loader: a DataLoader class for loading multimodal sensor data.
    """
    def __init__(self, root_dir: str, split: str = 'train'):
        """
        Args:
            root_dir: path to the dataset
            split: data split, either train or val
        """
        self.dataset_index = self.get_frames(root_dir, split)
        self.data_loader = DataLoader(self.dataset_index)

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index: int) -> DataItem:
        return self.data_loader[self.dataset_index[index]]

    def get_frames(self, path: str, split: str = 'train') -> List[str]:
        """
        Collects the keyframe paths.

        Args:
            path: path to the dataset
            split: data split, either train or val

        Returns:
            data_paths: a list of keyframe paths

        """
        data_paths = []
        odd_path = os.path.join(path, split)
        for odd in os.listdir(odd_path):
            for seq in os.listdir(os.path.join(odd_path, odd)):
                seq_path = os.path.join(odd_path, odd, seq)
                sequence = Sequence(seq_path)
                data_paths.extend(sequence.get_frames())

        return data_paths


if __name__ == '__main__':
    root_directory = "/media/tamas.matuszka/shares/hackathondata/temp/aimotive_dataset"
    train_dataset = AiMotiveDataset(root_directory, split='val')
    for data in train_dataset:
        print(data['annotations'].path)
