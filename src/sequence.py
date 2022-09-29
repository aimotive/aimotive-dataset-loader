import os
from typing import List


class Sequence:
    """
    This class represents a sequence (a 15 sec long annotated recording).

    Attributes:
        path: path to the sequence
        keyframes: a list of keyframe paths
    """
    def __init__(self, path: str) :
        """
        Args:
            path: path to the sequence
        """
        self.path = path
        self.keyframes = sorted(os.listdir(os.path.join(path, 'dynamic', 'box', '3d_body')))

    def get_frames(self) -> List[str]:
        """
        Collects the paths of the keyframes corresponding to the sequence.

        Returns:
            a list of keyframe paths
        """
        return [os.path.join(self.path, 'dynamic', 'box', '3d_body', keyframe) for keyframe in self.keyframes]

