import os

import laspy
import numpy as np


class LidarData:
    """ Stores the LidarDataItem for each lidar.

    Attributes
        top_lidar: LidarDataItem for top lidar
    """

    def __init__(self, point_cloud: np.array):
        """
        Args:
            point_cloud: numpy array (shape: [N, 5]), format: [x, y, z, intensity, timestamp]
        """
        self.top_lidar = LidarDataItem('top_lidar', point_cloud)


class LidarDataItem:
    """
    Stores the point cloud for a lidar with a given frame id.
    Coordinate system: x -> forward, y -> left, z -> top
    x, y, z coordinates are stored in meters.

    Attributes
        name: sensor name
        point_cloud: numpy array (shape: [N, 5]), format: [x, y, z, intensity, timestamp]
    """

    def __init__(self, name: str, point_cloud: np.array):
        """
        Args:
            name: sensor name
            point_cloud: numpy array (shape: [N, 5]), format: [x, y, z, intensity, timestamp]
        """
        self.name = name
        self.point_cloud = point_cloud


def load_lidar_data(data_folder: str, frame_id: str) -> LidarData:
    """
    Loads data for each lidar with a given frame id. Current dataset has only one lidar.

    Args:
        data_folder: the path of the sequence from where data is loaded
        frame_id: id if the loadable camera data, e.g. 0033532

    Returns:
        lidar_data: a LidarData instance with a top lidar point cloud.
    """
    lidar_path = os.path.join(data_folder, 'dynamic', 'raw-revolutions', 'frame_' + frame_id + '.laz')

    if lidar_path is None:
        return LidarData(np.zeros((0, 5)))

    if os.path.getsize(lidar_path) > 226:  # Check whether it has at least the LAS header
        with laspy.open(lidar_path) as fh:
            las = fh.read()
            lidar_pcd = np.array([las.x, las.y, las.z, las.intensity, las.gps_time])
            lidar_pcd = lidar_pcd.T
            lidar_data = LidarData(lidar_pcd)
    else:
        return LidarData(np.zeros((0, 5)))

    return lidar_data
