from typing import Tuple, Optional, Sequence
from abc import ABCMeta, abstractmethod

import numpy as np

from em_utils import EM


class AutoFocusBase(metaclass=ABCMeta):
    """Interface template that defines the minimum class requirements to use the experiment methods in
    'experiments_base.py'.
    """

    def __init__(self, pixel_size: float = 10.03, dwell_time: float = 0.1, frame_settings: int = 0):
        """

        Args:
            pixel_size: Image pixel size in nm, must match the training data.
            dwell_time: Pixel dwell time in us. Value should match used for the acquisition of the training data.
            frame_settings: Indicator, i.e. 0: 1024 x 768, 2: 2048 x 1536
        """
        # accumulate time for processing (images -> focus corrections)
        self.pixel_size = pixel_size
        self.dwell_time = dwell_time
        self.frame_settings = frame_settings
        self.images = None
        self.dt = {'init': [], 'proc': [], 'acq': [], 'proc_pre': [], 'proc_gpu': []}
        self.pred_details = dict()

    @abstractmethod
    def run_on_em(self,  em: EM, apply_result: bool = True,
                  refresh_after_apply: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            em: Electron microscope object.
            apply_result: Apply focus result.
            refresh_after_apply: Refresh microscope frame after applying focus results.

        Returns:
            Correction values for working distance (in m), stigmatism in X and Y (in percentage) and corresponding
            uncertainty measurement (e.g. standard deviations): (wd, stigx, stigy), (wd_sd, stigx_sd, stigy_sd).
        """
        pass

    def __repr__(self):
        """
        String that contains all important model parameters.

        Returns:
            String representation of autofocus model.
        """
        return f'{self.__dict__}'


class RandomMult:
    """
    """
    def __init__(self, fact: float, p: float = 0.5):
        """
        Inplace.

        Args:
.           p: Probability to apply factor multiplication..
            fact: Factor multiplied with input
        """
        self.p = p
        self.fact = fact

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if np.random.uniform(0, 1, 1)[0] < self.p:
            inp *= self.fact
        return inp, target


class RandomCrop:
    """
    Adapted from elektronn3:
    https://github.com/ELEKTRONN/elektronn3/blob/56ad4e806dcafb826180c22
    1ebc82f298d968f0c/elektronn3/data/transforms/transforms.py#L711

    MIT License

    Copyright (c) 2017 - now, ELEKTRONN team

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, crop_shape: Sequence[int], deterministic: bool = True, independent_xy: bool = False):
        """

        Args:
            crop_shape: Patch shape that will be cropped out.
            deterministic: Always use seed 0 when drawing the crop locations.
            independent_xy: If True, ignores z extent in crop_shape and draws xy offsets
                independently for every z slice.
        """
        self.crop_shape = np.array(crop_shape)
        self.deterministic = deterministic
        self.independent_xy = independent_xy
        self._seed = 0

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None,  # returned without modifications
            fixed_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """

        Args:
            inp: Input images.
            target: Target tensor, not modified.
            fixed_seed: If not None, use it as random seed. Only applied if ``self.independent_xy`` is False.

        Returns:

        """
        ndim_spatial = len(self.crop_shape)  # Number of spatial axes E.g. 3 for (C,D,H.W)
        img_shape = np.array(inp.shape[-ndim_spatial:])
        # Number of nonspatial axes (like the C axis). Usually this is one
        ndim_nonspatial = inp.ndim - ndim_spatial
        if any(self.crop_shape > img_shape):
            raise ValueError(
                f'crop shape {self.crop_shape} can\'t be larger than image shape {img_shape}.'
            )
        if not self.independent_xy:
            # reseed
            if self.deterministic and fixed_seed is None:
                np.random.seed(self._seed)
                self._seed = np.random.randint(0, 2 ** 32 - 1, dtype=np.uint32)
            elif fixed_seed is not None:
                np.random.seed(fixed_seed)
            # Calculate the "lower" corner coordinate of the slice
            coords_lo = np.array([
                np.random.randint(0, img_shape[i] - self.crop_shape[i] + 1)
                for i in range(ndim_spatial)
            ])
            coords_hi = coords_lo + self.crop_shape  # Upper ("high") corner coordinate.
            # Calculate necessary slice indices for reading the file
            nonspatial_slice = [  # Slicing all available content in these dims.
                slice(0, inp.shape[i]) for i in range(ndim_nonspatial)
            ]
            spatial_slice = [  # Slice only the content within the coordinate bounds
                slice(coords_lo[i], coords_hi[i]) for i in range(ndim_spatial)
            ]
            full_slice = tuple(nonspatial_slice + spatial_slice)
            inp_cropped = inp[full_slice]
        else:
            # Calculate necessary slice indices for reading the file
            nonspatial_slice = [  # Slicing all available content in these dims.
                slice(0, inp.shape[i]) for i in range(ndim_nonspatial)
            ]
            # use ndim_nonspatial shape plus z extent of input (inpt) and combine it with the target crop shape for xy.
            inp_cropped = np.zeros(list(inp.shape[:(ndim_nonspatial+1)]) + list(self.crop_shape[-2:]), dtype=inp.dtype)
            # iterate over z slices (inp shape: [C, Z, Y, X])
            #                                       ^
            for z_ix in range(inp.shape[-3]):
                # Calculate the "lower" corner coordinate of the slice
                coords_lo = np.array([
                    np.random.randint(0, img_shape[i] - self.crop_shape[i] + 1)
                    for i in range(ndim_spatial)
                ])
                coords_hi = coords_lo + self.crop_shape  # Upper ("high") corner coordinate.
                # reseed
                if self.deterministic:
                    np.random.seed(self._seed)
                    self._seed = np.random.randint(0, 2 ** 32 - 1, dtype=np.uint32)
                spatial_slice = [  # Slice only the content within the coordinate bounds except z
                    slice(coords_lo[i], coords_hi[i]) for i in range(1, ndim_spatial)
                ]
                full_slice = tuple(nonspatial_slice + [slice(z_ix, z_ix+1)] + spatial_slice)
                # 'ndim_nonspatial' is the location of the z axis, e.g. ndim_nonspatial=1 for (C, Z, Y, X)
                if ndim_nonspatial == 0:
                    inp_cropped[z_ix] = inp[full_slice]
                elif ndim_nonspatial == 1:
                    inp_cropped[:, z_ix] = inp[full_slice]
                else:
                    raise ValueError('Incompatible input shape.')
        if target is None:
            return inp_cropped, target

        if target.ndim == inp.ndim - 1:  # inp: (C, [D,], H, W), target: ([D,], H, W)
            full_slice = full_slice[1:]  # Remove C axis from slice because target doesn't have it
        target_cropped = target[full_slice]
        return inp_cropped, target_cropped
