from typing import List, Dict, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, SequentialSampler

from src.aimotive_dataset import AiMotiveDataset
from src.data_loader import DataItem
from src.loaders.camera_loader import CameraData
from src.loaders.lidar_loader import LidarData
from src.loaders.radar_loader import RadarData

CATEGORY_MAPPING = {'CAR': 0, 'Size_vehicle_m': 0,
                    'TRUCK': 1, 'BUS': 1, 'TRUCK/BUS': 1, 'TRAIN': 1, 'Size_vehicle_xl': 1, 'VAN': 1,
                    'PICKUP': 1,
                    'MOTORCYCLE': 2, 'RIDER': 2, 'BICYCLE': 2, 'BIKE': 2, 'Two_wheel_without_rider': 2,
                    'Rider': 2,
                    'OTHER_RIDEABLE': 2, 'OTHER-RIDEABLE': 2,
                    'PEDESTRIAN': 3, 'BABY_CARRIAGE': 3
                    }

crop = T.CenterCrop(size=(704, 1280))
to_tensor = T.ToTensor()


class AiMotiveTorchDataset(Dataset):
    """
    PyTorch Dataset for loading data to PyTorch framework.
    """
    def __init__(self, root_dir: str, train: bool = True, max_objects: int = 30, max_lidar_points: int = 150_000,
                 max_radar_targets: int = 100):
        data_split = 'train' if train else 'val'
        self.dataset = AiMotiveDataset(root_dir, data_split)
        self.max_objects = max_objects
        self.max_lidar_points = max_lidar_points
        self.max_radar_targets = max_radar_targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset.data_loader[self.dataset.dataset_index[index]]
        sensor_data = self.get_sensor_data(data_item)
        annotations = self.get_targets(data_item.annotations.objects, CATEGORY_MAPPING)
        annotations = self.prepare_annotations(annotations)

        return sensor_data, annotations

    def get_sensor_data(self, data_item: DataItem) -> List:
        lidar_data, radar_data, camera_data = data_item.lidar_data, data_item.radar_data, data_item.camera_data

        lidar_data = self.prepare_lidar_data(lidar_data)
        front_radar_data, back_radar_data = self.prepare_radar_data(radar_data)
        front_cam, back_cam, left_cam, right_cam = self.prepare_camera_data(camera_data)

        sensor_data = [lidar_data, [front_radar_data, back_radar_data], [front_cam, back_cam, left_cam, right_cam]]

        return sensor_data

    def get_targets(self, annotations: List[Dict], category_mapping: Dict[str, int]):
        # Generate your custom target representation here.
        targets = []
        for obj in annotations:
            x, y, z = [obj[f'BoundingBox3D Origin {ax}'] for ax in ['X', 'Y', 'Z']]
            l, w, h = [obj[f'BoundingBox3D Extent {ax}'] for ax in ['X', 'Y', 'Z']]
            vel_x, vel_y, vel_z = [obj[f'Relative Velocity {ax}'] for ax in ['X', 'Y', 'Z']]
            q_w, q_x, q_y, q_z = [obj[f'BoundingBox3D Orientation Quat {ax}'] for ax in ['W', 'X', 'Y', 'Z']]
            cat = category_mapping[obj['ObjectType']]

            targets.append(torch.tensor([[cat, x, y, z, l, w, h, q_w, q_x, q_y, q_z, vel_x, vel_y, vel_z]]))

        return targets

    def prepare_lidar_data(self, lidar_data: LidarData) -> torch.tensor:
        lidar_data = torch.from_numpy(lidar_data.top_lidar.point_cloud)
        lidar_data = self.pad_data(lidar_data, self.max_lidar_points, lidar_data.shape[1])

        return lidar_data

    def prepare_radar_data(self, radar_data: RadarData) -> Tuple[torch.tensor, torch.tensor]:
        front_radar_data = torch.from_numpy(radar_data.front_radar.point_cloud)
        back_radar_data = torch.from_numpy(radar_data.back_radar.point_cloud)

        front_radar_data = self.pad_data(front_radar_data, self.max_radar_targets, front_radar_data.shape[1])
        back_radar_data = self.pad_data(back_radar_data, self.max_radar_targets, back_radar_data.shape[1])

        return front_radar_data, back_radar_data

    def prepare_camera_data(self, camera_data: CameraData) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        front_cam = crop(to_tensor(camera_data.front_camera.image))
        back_cam = crop(to_tensor(camera_data.back_camera.image))
        left_cam = crop(to_tensor(camera_data.left_camera.image))
        right_cam = crop(to_tensor(camera_data.right_camera.image))

        return front_cam, back_cam, left_cam, right_cam

    def pad_data(self, data: torch.tensor, max_items: int, attributes: int) -> torch.tensor:
        if len(data) > max_items:
            data = data[:max_items]
            return data
        else:
            padded_data = torch.zeros([max_items, attributes])
            padded_data[:data.shape[0], :] = data

        return padded_data

    def prepare_annotations(self, annotations: torch.tensor) -> torch.tensor:
        if len(annotations) > self.max_objects:
            annotations = annotations[:self.max_objects]
        else:
            pad = self.max_objects - len(annotations)
            for i in range(pad):
                annotations.append(torch.zeros(14))

        annotations = torch.vstack(annotations)

        return annotations


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    root_directory = 'data'
    train_dataset = AiMotiveTorchDataset(root_directory, train=True)
    train_sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        num_workers=8,
    )


    def to_device(d, dev):
        if isinstance(d, (list, tuple)):
            return [to_device(x, dev) for x in d]
        else:
            return d.to(dev)


    step = 0
    for data in train_loader:
        step += 1
        sensor_data, annotation = data
        sensor_data, annotation = to_device(sensor_data, device), to_device(annotation, device)
        print('Iter ', step, annotation[0][0])

