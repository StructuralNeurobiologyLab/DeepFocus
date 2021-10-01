from typing import Tuple
from abc import ABCMeta, abstractmethod


class EM(metaclass=ABCMeta):
    """
    EM base class for the acquisition of images and controlling machine parameters.
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_wd(self) -> float:
        """

        Returns:
            Working distance (m).
        """
        return -1

    @abstractmethod
    def set_wd(self, wd: float) -> bool:
        """

        Args:
            wd: Target working distance (m).

        Returns:
            True if success, else False.
        """
        return False

    @abstractmethod
    def get_stig_xy(self) -> Tuple[float, float]:
        """

        Returns:
            Stigmation in x, stigmation in y.
        """
        pass

    @abstractmethod
    def set_stig_xy(self, stigx: float, stigy: float) -> bool:
        """

        Args:
            stigx: Target stigmation in x.
            stigy: Target stigmation in y.

        Returns:
            True if success, else False.
        """
        pass

    @abstractmethod
    def acquire_frame(self, fname: str, delay: float) -> bool:
        """
        Acquire current frame of the microscope.

        Args:
            fname: Filename.
            delay: Extra delay (s).

        Returns:
            True if success, else False.
        """
        return False

    @abstractmethod
    def refresh_frame(self):
        pass

    @abstractmethod
    def apply_frame_settings(self, frame_size: int, pixel_size: float, dwell_time: float) -> bool:
        """

        Args:
            frame_size: Frame size indicator.
            pixel_size: Pixel size (nm).
            dwell_time: Pixel dwell time (µs).

        Returns:
            True if success, else False.
        """
        return False

    @abstractmethod
    def move_stage_to_xy(self, coordinates: Tuple[float, float]) -> float:
        """

        Args:
            coordinates:

        Returns:
            True if success, else False.
        """
        return False

    @abstractmethod
    def get_stage_xy(self) -> Tuple[float, float]:
        """

        Returns:
            Stage coordinates in x, y.
        """

    @abstractmethod
    def set_scan_rotation(self, angle: float) -> bool:
        """

        Args:
            angle:

        Returns:
            True if success, else False.
        """
        return False
