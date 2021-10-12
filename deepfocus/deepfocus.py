import logging
import tempfile
import time
from typing import Optional, Tuple

import imageio
import numpy as np
import scipy.ndimage
import torch

from .em_utils import EM
from .af_utils import AutoFocusBase, RandomCrop


class DeepFocus(AutoFocusBase):
    def __init__(self, mpath: str, perturbation_list: list, n_consensus: int = 5,
                 deterministic_consensus: bool = True, use_fix_locations: Optional[int] = 0,
                 device: str = None, crop_edge_length: int = 256,
                 normalization: Tuple[float, float] = (128., 256.),
                 model_pixel_size: float = 10.03, rot_input: Optional[float] = None,
                 independent_xy: bool = False, **kwargs):
        """
        Minimum init. parameter is the path to the torch model. Depending on the training paradigm, the model requires
        2 symmetrically perturbed images (only defocus, no stigmatism). The :func:`~apply` method consumes
        ``2*n_perturbations`` images and returns a triplet of correction terms (working distance, stigm. X, stigm. Y).
        Prediction performance benefits a lot from ``n_consensus>1``, but the improvement levels off quickly for
        ``n_consensus>>10``.

        Args:
            mpath: Path to pytorch script file (.pts).
            perturbation_list: Perturbations (in m) used during focus routine.
            n_consensus: Number of patches extracted from input images during :func:`~apply`.
            deterministic_consensus: Use deterministic reseeding of crop locations.
            use_fix_locations: Use fixed random seed for the crop locations.
            device: Torch device.
            crop_edge_length: Crop size used for generating model inputs.
            normalization: Offset and scale for normalizing input images. Must match the values used during training.
            model_pixel_size: Pixel size used during model training.
            rot_input: Rotation angle to use as extra input for predictions.
            independent_xy: Use different crop locations in the perturbed image pair.
        """
        super().__init__(**kwargs)
        self.dt = {'init': [], 'proc': [], 'acq': [], 'proc_pre': [], 'proc_gpu': []}
        self._mpath = mpath
        self.rot_input = rot_input
        self.perturbation_list = perturbation_list
        self.n_perturbations = len(perturbation_list)
        self.n_consensus = n_consensus
        self.use_fix_locations = use_fix_locations
        self.deterministic_consensus = deterministic_consensus
        if device is None:
            device = 'cuda'
        self.device = device
        self.normalization = normalization
        start = time.time()
        self.model = torch.jit.load(mpath, map_location=device)
        self.model.eval()
        if hasattr(self.model, 'n_consensus'):
            self.model.n_consensus = n_consensus
        self._model_pixel_size = model_pixel_size
        self.crop_edge_length = crop_edge_length
        self.independent_xy = independent_xy
        self.dt['init'] = time.time() - start
        # init here so the deterministic random seed is used for each DeepFocus
        # instance and not repeated within every "apply" call
        self._rc = RandomCrop((2*self.n_perturbations, crop_edge_length, crop_edge_length),
                              deterministic=self.deterministic_consensus, independent_xy=independent_xy)

    def __repr__(self):
        return (f'DeepFocus model at "{self._mpath}" with n_consensus={self.n_consensus}, '
                f'dwell_time={self.dwell_time} µs, device={self.device}, '
                f'model_pixel_size={self._model_pixel_size}, pixel_size={self.pixel_size}, '
                f'crop_edge_length={self.crop_edge_length}, normalization={self.normalization}, '
                f'deterministic_consensus={self.deterministic_consensus}, independent_xy={self.independent_xy}, '
                f'frame_settings={self.frame_settings}, use_fix_locations={self.use_fix_locations}\n')

    def apply(self, imgs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply deep focus model on input images `imgs`.

        Args:
            imgs: Input images, requires same ordering as used during training. Shape: (2*n_perturbations, Y, X).

        Returns:
            Consensus correction values for working distance (in m), stigmatism in X and Y (in percentage) and standard
            deviation (if ensemble model consensus is learned weighted average; if normal model use mean estimator):
            ``(consensus_wd, consensus_stigx, consensus_stigy), (std_wd, std_stigx, std_stigy)``.
        """
        start = time.time()
        if (self.n_perturbations is not None) and (imgs.shape[0] != 2 * self.n_perturbations):
            raise ValueError('Invalid input shape. First axis must match 2 times the number of perturbations.')
        # add channel axis: (Z, Y, X) -> (C, Z, Y, X)
        inp = (imgs[None] - self.normalization[0]) / self.normalization[1]
        if not self.use_fix_locations:
            inp = np.array([self._rc(inp, None)[0] for _ in range(self.n_consensus)], dtype=np.float32)
        else:
            inp = np.array([self._rc(inp, None, ii)[0] for ii in range(self.n_consensus)], dtype=np.float32)
        inp = torch.from_numpy(inp).to(self.device)
        self.dt['proc_pre'].append(time.time() - start)
        start2 = time.time()
        if self.rot_input is None:
            preds = self.model(inp)
        else:
            angle_inp = torch.from_numpy(
                np.array([self.rot_input] * self.n_consensus, dtype=np.float32)).to(self.device)
            preds = self.model(inp, angle_inp.unsqueeze(1))
        # model predicts the defocus corretion in um
        if isinstance(preds, tuple):
            pred_final = preds[0].squeeze().detach().cpu().numpy()
            if preds[1].ndim == 4:  # dense score prediction model
                sd_final = np.std(pred_final, axis=0) / 1e6  # m -> um
                if pred_final.ndim > 1:
                    pred_final = np.mean(pred_final, axis=0)
                pred_final[0] /= 1e6  # m -> um
                orig_preds = preds[1].view(preds[1].size()[0], 4, -1).detach().cpu().numpy()
                scores = orig_preds[:, 3]
                if hasattr(self.model, 'avgpool'):
                    scores = scores.reshape(scores.shape[0], int(inp.shape[-1]), int(inp.shape[-1]))
                    scores = self.model.avgpool(
                        torch.from_numpy(scores).to(self.device)).detach().cpu().numpy().reshape(scores.shape[0], -1)
                # calc softmax
                scores = np.exp(scores)
                scores /= np.sum(scores, axis=1)[..., None]
                scores = scores.reshape((int(inp.shape[0]), int(inp.shape[-1]), int(inp.shape[-1])))
                self.pred_details.update(dict(scores=scores, consensus_res=pred_final, inp=inp.detach().cpu().numpy()))
            else:
                pred_final[0] /= 1e6  # um -> m
                orig_preds = preds[1].detach().cpu().numpy()
                scores = orig_preds[:, 3]
                # calc softmax
                scores = np.exp(scores)
                scores /= np.sum(scores)
                preds = orig_preds[:, :3]
                preds[:, 0] /= 1e6  # um -> m
                self.pred_details.update(dict(mean_pred_res=(np.mean(preds, axis=0), np.std(preds, axis=0)),
                                         scores=scores, consensus_res=pred_final, inp=inp.detach().cpu().numpy()))
                sd_final = np.std(preds, axis=0)
        else:
            preds = preds.detach().cpu().numpy()
            preds[:, 0] = preds[:, 0] / 1e6  # um -> m
            pred_final = np.mean(preds, axis=0)
            sd_final = np.std(preds, axis=0)
        self.dt['proc_gpu'].append(time.time() - start2)
        self.dt['proc'].append(time.time() - start)
        return pred_final, sd_final

    def acquire_images_em(self, em):
        """
        Acquire images and store them in :attr:`~images`.

        Args:
            em: Electron microscope object.

        """
        em.apply_frame_settings(self.frame_settings, self.pixel_size, self.dwell_time)
        wd = em.get_wd()
        images = []
        start = time.time()
        with tempfile.TemporaryDirectory() as temp_dir:
            # this is the order in data loader used for training and validation, e.g. [-500, -5000, 500, 5000]
            for sign in [-1, 1]:
                for defocus in self.perturbation_list:
                    em.set_wd(wd + sign * defocus)  # perturbations conversion from nm to m
                    # acquire defocused image
                    fname = f'{temp_dir}\\def{int(sign * defocus * 1e9)}.png'
                    em.acquire_frame(save_path_filename=fname, delay=0.1)
                    if self.pixel_size == self._model_pixel_size:
                        images.append(imageio.imread(fname))
                    else:
                        logging.warning(f'Upsampling acquired images from pixel size {self.pixel_size} nm to '
                                        f'model pixel size {self._model_pixel_size} nm.')
                        images.append(scipy.ndimage.zoom(
                            imageio.imread(fname), self.pixel_size / self._model_pixel_size))
        self.dt['acq'].append(time.time() - start)
        self.images = np.array(images, dtype='f4')
        em.set_wd(wd)  # reset working distance

    def run_on_em(self, em: EM, apply_result: bool = True, refresh_after_apply: bool = True)\
            -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            apply_result: Apply focus result.
            em: Electron microscope object.
            refresh_after_apply: Refresh microscope frame after applying focus results.

        Returns:
            Consensus correction values for working distance (in m), stigmatism in X and Y (in percentage) and standard
            deviation (if ensemble model consensus is learned weighted average; if normal model use mean estimator):
            ``(consensus_wd, consensus_stigx, consensus_stigy), (std_wd, std_stigx, std_stigy)``.
        """
        wd = em.get_wd()
        stigx, stigy = em.get_stig_xy()
        # stored in self.images
        self.acquire_images_em(em)
        means, stds = self.apply(self.images)
        if apply_result:
            em.set_wd(wd + means[0])
            em.set_stig_xy(stigx + means[1], stigy + means[2])
            time.sleep(0.25)
            if refresh_after_apply:
                em.refresh_frame()
        return means, stds
