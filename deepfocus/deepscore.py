import tempfile
import time
from typing import Tuple, Optional

import imageio
import numpy as np
import scipy.ndimage
import torch

from .em_utils import EM
from .af_utils import AutoFocusBase, RandomCrop
from .nelder_mead import nelder_mead


class DeepFocusScore(AutoFocusBase):
    def __init__(self, mpath: str, n_consensus: int = 5, deterministic_consensus: bool = True, device: str = 'cuda',
                 crop_edge_length: int = 256, normalization: Tuple[float, float] = (128., 128.),
                 pixel_size: float = 10.03, dwell_time: float = 0.1, model_pixel_size: float = 10.03,
                 frame_settings: int = 0, nelder_mead_kwargs: Optional[dict] = None):
        """
        Minimum init. parameter is the path to the torch model. The :func:`~apply` method consumes
        1 image and returns a mean score and standard deviation array (N is equals to `n_consensus`).
        Prediction performance benefits from ``n_consensus>1``, but the improvement is expected to level off
        quickly for ``n_consensus>>10``.

        Args:
            mpath: Path to pytorch script file (.pts).
            n_consensus: Number of patches extracted from input images during :func:`~apply`.
            deterministic_consensus:
            device: Torch device.
            crop_edge_length: Crop size used for generating model inputs.
            normalization: Offset and scale for normalizing input images. Must match the values used during training.
            pixel_size: Image pixel size in nm, must match the training data.
            dwell_time: Pixel dwell time in us. Value should match used for the acquisition of the training data.
            model_pixel_size: Pixel size used during model training.
            frame_settings: 0: 1024 x 768, 2: 2048 x 1536
        """
        super().__init__()
        self.dt = {'init': [], 'proc': [], 'acq': []}
        self._mpath = mpath
        self.n_consensus = n_consensus
        self.deterministic_consensus = deterministic_consensus
        self.device = device
        self.frame_settings = frame_settings
        self.pixel_size = pixel_size
        self.dwell_time = dwell_time
        self.normalization = normalization
        start = time.time()
        self.model = torch.jit.load(mpath, map_location=device)
        self.model.eval()
        self._model_pixel_size = model_pixel_size
        self.crop_edge_length = crop_edge_length
        self.dt['init'].append(time.time() - start)
        self.nelder_mead_kwargs = dict(step=np.array([5e-6, 2, 2]), no_improve_thr=10e-3,
                                       no_improv_break=25, max_iter=100)
        if nelder_mead_kwargs is not None:
            self.nelder_mead_kwargs.update(nelder_mead_kwargs)

        self.pred_details = dict(niters=[], score=[], sd=[])
        # init here so the deterministic random seed is used for each DeepFocus
        # instance and not repeated within every "apply" call
        self._rc = RandomCrop((1, crop_edge_length, crop_edge_length),
                              deterministic=self.deterministic_consensus)

    def __repr__(self) -> str:
        return (f'DeepFocus model at "{self._mpath}" with n_consensus={self.n_consensus}, '
                f'dwell_time={self.dwell_time} µs, '
                f'model_pixel_size={self._model_pixel_size}, pixel_size={self.pixel_size}, '
                f'crop_edge_length={self.crop_edge_length}, normalization={self.normalization}, '
                f'deterministic_consensus={self.deterministic_consensus},'
                f'frame_settings={self.frame_settings}, nelder_mead_kwargs={self.nelder_mead_kwargs}')

    def apply(self, imgs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply deep focus model on input `imgs`.

        Args:
            imgs: Input images, requires same ordering as used during training. Shape: (1, Y, X)

        Returns:
            Two channel mean score of image sharpness (lower is better) for working distance and astigmatism
            with their respective standard deviations.
        """
        # add channel axis: (Z, Y, X) -> (C, Z, Y, X)
        inp = (imgs[None] - self.normalization[0]) / self.normalization[1]
        inp = torch.from_numpy(np.array([self._rc(inp, None)[0] for _ in range(self.n_consensus)], dtype=np.float32))
        preds = self.model(inp.to(self.device))
        preds = preds.detach().cpu().numpy()
        # model predicts score equal to L1-norm of defocus and astig-x and astig-y
        return np.sum(np.min(preds, axis=0)), np.sum(np.std(preds, axis=0))

    def apply_on_em(self, em: EM) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            em: Electron microscope object.

        Returns:
            Two channel mean score of image sharpness (lower is better) for working distance and astigmatism
            with their respective standard deviations.
        """
        em.apply_frame_settings(self.frame_settings, self.pixel_size, self.dwell_time)  # 0: Image res of 1024x768
        start = time.time()
        with tempfile.TemporaryDirectory() as temp_dir:
            fname = f'{temp_dir}\\dfs_input.png'
            em.acquire_frame(fname, delay=0.1)
            if self.pixel_size == self._model_pixel_size:
                image = imageio.imread(fname)
            else:
                image = scipy.ndimage.zoom(
                    imageio.imread(fname), self.pixel_size / self._model_pixel_size)
        self.dt['acq'].append(time.time() - start)
        start = time.time()
        images = np.array(image, dtype='f4')
        score, std = self.apply(images[None])
        self.dt['proc'].append(time.time() - start)
        return score, std

    def run_on_em(self, em: EM, max_iter: int = 1, score_tol: float = 0.2,
                  apply_result: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find correction term for focus parameters.

        Notes:
            * This method resets lists in :attr:`~.pred_details`.

        Args:
            em: Electron microscope object.
            max_iter: Maximum number of Nelder-Mead repetitions.
            score_tol: Score tolerance used as stopping criterion.
            apply_result: Apply focus result.

        Returns:
            Corrections for (wd, stigx, stigy) and standard deviation of the final vertices of Nelder-Mead optimization.
        """
        wd = em.get_wd()
        stigx, stigy = em.get_stig_xy()
        dfs_fun = DeepFocusScoreFunction(self, em)
        n_iter = 0
        score = np.inf
        self.pred_details['score'] = []
        self.pred_details['sds'] = []
        self.pred_details['pred'] = []
        self.pred_details['func_call_cnt'] = 0
        while True:
            if n_iter == max_iter:
                break
            wd_curr = em.get_wd()
            stigx_curr, stigy_curr = em.get_stig_xy()
            progress_traj = nelder_mead(
                dfs_fun, x_start=np.array([wd_curr, stigx_curr, stigy_curr]),
                track_progress=True, score_tol=score_tol, pred_details=self.pred_details,
                **self.nelder_mead_kwargs)

            self.pred_details['score'].extend([el[1] for el in progress_traj])
            self.pred_details['sds'].extend([el[2] for el in progress_traj])
            self.pred_details['pred'].extend([el[0] for el in progress_traj])
            # final scores
            x, score, sds = progress_traj[-1]
            em.set_wd(x[0])
            if len(x) > 1:
                em.set_stig_xy(x[1], x[2])
            if score < score_tol:
                break
            n_iter += 1
        if score >= score_tol or not apply_result:
            em.set_wd(wd)
            em.set_stig_xy(stigx, stigy)
        if score >= score_tol:
            raise RuntimeError('Did not converge!')
        fp_updates = np.array([x[0] - wd, x[1] - stigx, x[2] - stigy])
        return fp_updates, sds


class DeepFocusScoreFunction:
    def __init__(self, dfs, em):
        self.em = em
        self.dfs = dfs

    def __call__(self, x):
        self.em.set_wd(x[0])
        if len(x) > 1:
            self.em.set_stig_xy(x[1], x[2])
        return self.dfs.apply_on_em(self.em)[0]
