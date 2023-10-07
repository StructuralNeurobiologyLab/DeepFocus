"""Implementation of the data loader class for model trainings."""
import glob
import os
from functools import lru_cache
from typing import Callable
from typing import Optional, Union, Dict

import imageio
import numpy as np
import pandas as pd
import torch
from elektronn3.data.transforms import Identity
from torch.utils.data import Dataset


class DeepFocusData(Dataset):

    def __init__(
            self,
            source_dir: str,
            focus_perturbations: list,
            n_entries_per_sample: Union[int, Dict[str, int]] = 12,  # number of csv entries per sample
            magnitude_only_target: bool = False,
            only_defoc: bool = False,
            train=True,
            prepare_aux_score: Optional[int] = None,
            transform: Callable = Identity(),
    ):
        self.magnitude_only_target = magnitude_only_target
        self.only_defoc = only_defoc
        self.prepare_aux_score = prepare_aux_score
        self.train = train
        # fix random seed.
        np.random.seed(0)
        super().__init__()
        self.source_dir = source_dir
        self.focus_perturbations = focus_perturbations
        if isinstance(n_entries_per_sample, int):
            self.n_entries_per_sample = {}
        else:
            self.n_entries_per_sample = n_entries_per_sample
        csv_files = glob.glob(source_dir + '/*/*.csv', recursive=True)
        self.df_dict = {}
        self.valid_dict = {}
        self.n_samples = 0
        for fname in csv_files:
            df = pd.read_csv(fname)
            self.df_dict[os.path.split(fname)[0]] = df
            if isinstance(n_entries_per_sample, int):
                self.n_entries_per_sample[os.path.split(fname)[0]] = n_entries_per_sample
            # get only the "base" entries in csv, per sample the csv contains n_entries_per_sample
            n_samples_df = len(df['fname']) // self.n_entries_per_sample[os.path.split(fname)[0]]
            self.valid_dict[os.path.split(fname)[0]] = np.random.choice(n_samples_df, int(n_samples_df * 0.2),
                                                                        replace=False)
            self.n_samples += int(n_samples_df * 0.8) if self.train else int(n_samples_df * 0.2)
        assert self.n_samples > 0
        self.transform = transform

    def __getitem__(self, index):
        df_ix = np.random.randint(0, len(self.df_dict))
        df_key = list(self.df_dict.keys())[df_ix]
        df = self.df_dict[df_key]
        while True:
            index = np.random.randint(0, len(df['fname'])) // self.n_entries_per_sample[df_key]
            if index * self.n_entries_per_sample[df_key] + 1 == len(df):
                continue
            if self.train:
                if index not in self.valid_dict[df_key]:
                    break
            else:
                if index in self.valid_dict[df_key]:
                    break
        return self._get_item(df_key, index)

    def _get_item(self, df_key: str, index: int) -> dict:
        """
        Retrieve dataframe item at `index`.

        Args:
            df_key: DataFrame key in :attr:`~df_dict`.
            index: Index

        Returns:

        """
        df = self.df_dict[df_key]
        # convert to index for csv entry instead of sample index
        index = index * self.n_entries_per_sample[df_key] + 1  # index to "base" image with introduced aberrations
        # build image file name; df_key is the directory path to the corresponding csv file
        fnames = tuple([df_key + '/' + df['fname'][index].replace('base', f'pert{sign * pert}')
                        for sign in [-1, 1] for pert in self.focus_perturbations])
        # inp: (N, X, Y), target: (1, 3)
        inp = self._get_inputs(fnames)

        # scale defocus to um so astigmatism magnitude and defocus are in a similar range
        delta_wd = df['target_wd'][index] * 1e6 - df['wd'][index] * 1e6
        delta_stigx = df['target_astigx'][index] - df['astigmx'][index]
        delta_stigy = df['target_astigy'][index] - df['astigmy'][index]
        target = np.array([delta_wd, delta_stigx, delta_stigy])
        if self.only_defoc:
            target = target[:1]
        if self.magnitude_only_target:
            target = np.abs(target)
        if self.prepare_aux_score is None:
            if isinstance(inp, tuple):  # contains rotation angle
                inp, scalar = inp
                scalar = np.array([scalar], dtype=np.float32)
                return {'inp': torch.from_numpy(inp), 'target': torch.from_numpy(target),
                        'scalar': torch.from_numpy(scalar)}
            inp, _ = self.transform(inp, None)
            return {'inp': torch.from_numpy(inp), 'target': torch.from_numpy(target)}
        else:
            if isinstance(inp, tuple):  # contains rotation angle
                inp, scalar = inp
                scalar = np.array([scalar], dtype=np.float32)
                return {'inp': torch.from_numpy(inp), 'target': torch.from_numpy(target),
                        'scalar': torch.from_numpy(scalar)}
            inp = np.array([self.transform(inp, None)[0] for _ in range(self.prepare_aux_score)], dtype=np.float32)
            return {'inp': torch.from_numpy(inp), 'target': torch.from_numpy(target)}

    @lru_cache
    def _get_inputs(self, fnames: tuple):
        return np.concatenate([imageio.imread(fname)[None,] for fname in fnames], axis=0).squeeze()[None].astype(
            np.float32)

    def __len__(self):
        """Determines epoch size(s)"""
        if self.train:
            return min(self.n_samples * 400, 80000)
        else:
            return min(self.n_samples * 400, 20000)


class DeepFocusScoreData(Dataset):

    def __init__(
            self,
            source_dir: str,
            only_defoc: bool = False,
            train=True,
            transform: Callable = Identity(),
            use_l1_norm: bool = False,
            split_defocus_astig: bool = False,
    ):
        self.only_defoc = only_defoc
        self.train = train
        self.use_l1_norm = use_l1_norm
        self.split_defocus_astig = split_defocus_astig
        # fix random seed.
        np.random.seed(0)
        super().__init__()
        self.source_dir = source_dir
        self.n_entries_per_sample = 12
        csv_files = glob.glob(source_dir + '/*/*.csv', recursive=True)
        self.df_dict = {}
        self.valid_dict = {}
        self.n_samples = 0
        for fname in csv_files:
            df = pd.read_csv(fname)
            self.df_dict[os.path.split(fname)[0]] = df
            # get only the "base" entries in csv, per sample the csv contains n_entries_per_sample
            n_samples_df = len(df['fname']) // self.n_entries_per_sample
            self.valid_dict[os.path.split(fname)[0]] = np.random.choice(n_samples_df, int(n_samples_df * 0.2),
                                                                        replace=False)
            self.n_samples += int(n_samples_df * 0.8) if self.train else int(n_samples_df * 0.2)
        assert self.n_samples > 0
        self.transform = transform

    def __getitem__(self, index):
        df_ix = np.random.randint(0, len(self.df_dict))
        df_key = list(self.df_dict.keys())[df_ix]
        df = self.df_dict[df_key]
        while True:
            index = np.random.randint(0, len(df['fname'])) // self.n_entries_per_sample
            if index * self.n_entries_per_sample + 1 == len(df):
                continue
            if self.train:
                if index not in self.valid_dict[df_key]:
                    break
            else:
                if index in self.valid_dict[df_key]:
                    break
        return self._get_item(df_key, index)

    def _get_item(self, df_key: str, index: int, draw_index_offset: bool = True) -> dict:
        """
        Retrieve dataframe item at `index`.

        Args:
            df_key: DataFrame key in :attr:`~df_dict`.
            index: Data entry index.
            draw_index_offset: Use `index` "as is".

        Returns:

        """
        df = self.df_dict[df_key]
        # build image file name; df_key is the directory path to the corresponding csv file
        if draw_index_offset:
            # convert to index for csv entry instead of sample index
            index = index * self.n_entries_per_sample + 1  # index to "base" image with introduced aberrations
            rnd_read = np.random.randint(0, self.n_entries_per_sample)
            index += rnd_read
        fname = df_key + '/' + df['fname'][index]
        # inp: (1, Y, X), target: (1, 3)
        inp = imageio.imread(fname)[None, None].astype(np.float32)
        inp, _ = self.transform(inp, None)

        # scale defocus to um so astigmatism magnitude and defocus are in a similar range
        delta_wd = df['target_wd'][index] * 1e6 - df['wd'][index] * 1e6
        delta_stigx = df['target_astigx'][index] - df['astigmx'][index]
        delta_stigy = df['target_astigy'][index] - df['astigmy'][index]
        target = np.array([delta_wd, delta_stigx, delta_stigy])
        if self.only_defoc:
            target = target[:1]
        elif self.split_defocus_astig:
            target = np.array([np.abs(target[0]), np.linalg.norm(target[1:], ord=1 if self.use_l1_norm else None)],
                              np.float32)
        else:
            target = np.array([np.linalg.norm(target, ord=1 if self.use_l1_norm else None)], np.float32)
        return {'inp': torch.from_numpy(inp), 'target': torch.from_numpy(target)}

    def __len__(self):
        """Determines epoch size(s)"""
        return self.n_samples * 40 * self.n_entries_per_sample
