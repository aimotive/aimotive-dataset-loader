import os

from src.annotation import Annotation
from src.loaders.camera_loader import load_camera_data, CameraData
from src.loaders.lidar_loader import load_lidar_data, LidarData
from src.loaders.radar_loader import load_radar_data, RadarData


class DataItem:
    """
    Data structure for storing data for a given frame id.

    Attributes:
        annotations: Annotation instance
        lidar_data: LidarData instance
        radar_data: RadarData instance
        camera_data: CameraData instance
    """
    def __init__(self, annotations: Annotation, lidar_data: LidarData, radar_data: RadarData, camera_data: CameraData):
        self.annotations = annotations
        self.lidar_data = lidar_data
        self.radar_data = radar_data
        self.camera_data = camera_data


class DataLoader:
    """
    Loads sensor data for a given frame id.

    Attributes:
        data_paths: a list of keyframe paths
    """
    def __init__(self, data_paths):
        """
        Args:
            data_paths: a list of keyframe paths
        """
        self.data_paths = data_paths

    def __getitem__(self, path: str) -> DataItem:
        """
        Returns sensor data for a given keyframe.

        Args:
            path: path of the keyframe's annotation file

        Returns:
            a DataItem with annotations and sensor data
        """

        data_folder = self.get_directory(path)
        frame_id = self.get_frame_id(path)
        annotations = Annotation(path)
        lidar_data = load_lidar_data(data_folder, frame_id)
        radar_data = load_radar_data(data_folder, frame_id)
        camera_data = load_camera_data(data_folder, frame_id)

        return DataItem(annotations, lidar_data, radar_data, camera_data)

    def get_directory(self, path: str) -> str:
        """
        Returns with the sequence directory of a given keyframe.

        Args:
            path: path of the keyframe's annotation file

        Returns:
            directory_path: path to the directory of the sequence which contains the given keyframe
        """
        directory_path = os.path.normpath(path).split(os.path.sep)[:-4]
        directory_path = os.path.sep.join(directory_path)

        return directory_path

    def get_frame_id(self, path: str) -> str:
        """
        Parses the frame id form a given path.

        Args:
            path: a path to a frame.

        Returns:
            frame_id: the parsed frame id
        """
        frame_id = os.path.normpath(path).split(os.path.sep)[-1]
        frame_id = os.path.splitext(frame_id)[0]
        frame_id = frame_id.split('_')[1]

        return frame_id


