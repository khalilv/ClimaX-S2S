# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional
import glob 
import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from climax.utils.data_utils import collate_fn
from climax.dataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)

class GlobalForecastDataModule(LightningDataModule):
    """DataModule for global forecast data.

    Args:
        root_dir (str): Root directory for sharded data.
        in_variables (list): List of input variables.
        buffer_size (int): Buffer size for shuffling.
        out_variables (list, optional): List of output variables.
        predict_range (int, optional): Predict range.
        hrs_each_step (int, optional): Hours each step.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
    """

    def __init__(
        self,
        root_dir,
        in_variables,
        buffer_size,
        out_variables=None,
        predict_range: int = 6,
        history_range: int = 1,
        hrs_each_step: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        if num_workers > 1:
            raise NotImplementedError(
                "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
            )

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if isinstance(out_variables, str):
            out_variables = [out_variables]
            self.out_variables = out_variables
        
        self.root_dir = root_dir
        self.in_variables = in_variables
        self.out_variables = out_variables
        self.buffer_size = buffer_size
        self.predict_range = predict_range
        self.history_range = history_range
        self.hrs_each_step = hrs_each_step
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.lister_train = list(dp.iter.FileLister(os.path.join(self.root_dir, "train")))
        self.lister_val = list(dp.iter.FileLister(os.path.join(self.root_dir, "val")))
        self.lister_test = list(dp.iter.FileLister(os.path.join(self.root_dir, "test")))
        
        self.in_transforms = self.get_normalize(self.in_variables)
        self.output_transforms = self.get_normalize(self.out_variables)
        
        self.val_clim, self.val_clim_timestamps = self.get_climatology("val", self.out_variables)
        self.test_clim, self.test_clim_timestamps = self.get_climatology("test", self.out_variables)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None
        
    def get_normalize(self, variables):
        normalize_mean = dict(np.load(os.path.join(self.root_dir, "normalize_mean.npz")))
        normalize_mean = np.array([normalize_mean[var] for var in variables])
        normalize_std = dict(np.load(os.path.join(self.root_dir, "normalize_std.npz")))
        normalize_std = np.array([normalize_std[var] for var in variables])
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.root_dir, "lon.npy"))
        return lat, lon

    def get_climatology(self, partition, variables):
        files = glob.glob(os.path.join(self.root_dir, partition, "*climatology*.npz"))
        assert len(files) == 1, f"Expected exactly one file in {partition} directory, but found {len(files)}"
        path = files[0]
        clim_dict = np.load(path)
        clim = np.concatenate([clim_dict[var] for var in variables])
        clim = torch.from_numpy(clim)
        timestamps = clim_dict['timestamps']
        return clim, timestamps
    
    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=self.lister_train,
                            start_idx=0,
                            end_idx=1,
                            in_variables=self.in_variables,
                            out_variables=self.out_variables,
                            shuffle=False,
                            multi_dataset_training=False,
                            max_predict_range=self.predict_range,
                            history_range=self.history_range,
                            hrs_each_step=self.hrs_each_step,
                        ),
                        random_lead_time=False,
                    ),
                    in_transforms=self.in_transforms,
                    output_transforms=self.output_transforms,
                ),
                buffer_size=self.buffer_size,
            )

            self.data_val = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_val,
                        start_idx=0,
                        end_idx=1,
                        in_variables=self.in_variables,
                        out_variables=self.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                        max_predict_range=self.predict_range,
                        history_range=self.history_range,
                        hrs_each_step=self.hrs_each_step,
                    ),
                    random_lead_time=False,
                ),
                in_transforms=self.in_transforms,
                output_transforms=self.output_transforms,
            )

            self.data_test = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_test,
                        start_idx=0,
                        end_idx=1,
                        in_variables=self.in_variables,
                        out_variables=self.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                        max_predict_range=self.predict_range,
                        history_range=self.history_range,
                        hrs_each_step=self.hrs_each_step,
                    ),
                    random_lead_time=False,
                ),
                in_transforms=self.in_transforms,
                output_transforms=self.output_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
