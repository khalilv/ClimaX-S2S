# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from climax.arch import ClimaX
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from climax.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_rmse,
    lat_weighted_acc_spatial_map,
    lat_weighted_rmse_spatial_map
)
from climax.utils.pos_embed import interpolate_pos_embed
from climax.utils.data_utils import plot_spatial_map
#3) Global forecast module - abstraction for training/validation/testing steps. setup for the module including hyperparameters is included here

class GlobalForecastModule(LightningModule):
    """Lightning module for global forecasting with the ClimaX model.

    Args:
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        img_size: list = [32, 64],
        patch_size: int = 16,
        embed_dim: int = 64,
        depth: int = 8,
        decoder_depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        parallel_patch_embed: bool = False
    ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.drop_rate = drop_rate
        self.parallel_patch_embed = parallel_patch_embed
        
        self.save_hyperparameters(logger=False, ignore=["net"])      

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    
    def init_metrics(self):
        self.train_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat)
        self.val_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, self.denormalization)
        self.val_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, self.denormalization)
        self.val_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, self.val_clim, self.val_clim_timestamps, self.denormalization)
        self.test_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, self.denormalization)        
        self.test_lat_weighted_rmse_spatial_map = lat_weighted_rmse_spatial_map(self.out_variables, self.lat, self.denormalization)
        self.test_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, self.denormalization)
        self.test_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, self.test_clim, self.test_clim_timestamps, self.denormalization)
        self.test_lat_weighted_acc_spatial_map = lat_weighted_acc_spatial_map(self.out_variables, self.lat, self.test_clim, self.test_clim_timestamps, self.denormalization)

    def init_network(self, in_vars: list):
        self.net = ClimaX(in_vars=self.in_variables, 
                          img_size=self.img_size, 
                          patch_size=self.patch_size, 
                          embed_dim=self.embed_dim, 
                          depth=self.depth, 
                          decoder_depth=self.decoder_depth, 
                          num_heads=self.num_heads, 
                          mlp_ratio=self.mlp_ratio, 
                          drop_path=self.drop_path, 
                          drop_rate=self.drop_rate, 
                          parallel_patch_embed=self.parallel_patch_embed)
        if len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_variables(self, in_variables, out_variables):
        self.in_variables = in_variables
        self.out_variables = out_variables

    def set_val_clim(self, clim, timestamps):
        self.val_clim = clim
        self.val_clim_timestamps = timestamps

    def set_test_clim(self, clim, timestamps):
        self.test_clim = clim
        self.test_clim_timestamps = timestamps

    # can add multi-step training here - add k rollouts then average the loss before returning
    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, input_timestamps, output_timestamps = batch #spread batch data 
 
        if x.shape[1] > 1:
            raise NotImplementedError("history_range > 1 is not supported yet.")
        x = x.squeeze() #squeeze history dimension
   
        preds = self.net.forward(x, lead_times, variables, out_variables)
        
        #cast to float
        preds = preds.float()
        y = y.float()
        
        batch_loss = self.train_lat_weighted_mse(preds, y)
        for var in batch_loss.keys():
            self.log(
                "train/" + var,
                batch_loss[var],
                prog_bar=True,
            )
        self.train_lat_weighted_mse.reset()
        return batch_loss['w_mse']
    
    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, input_timestamps, output_timestamps = batch

        if x.shape[1] > 1:
            raise NotImplementedError("history_range > 1 is not supported yet.")
        x = x.squeeze() #squeeze history dimension

        preds = self.net.forward(x, lead_times, variables, out_variables)
        
        #cast to float
        preds = preds.float()
        y = y.float()

        self.val_lat_weighted_mse.update(preds, y)
        self.val_lat_weighted_rmse.update(preds, y)
        self.val_lat_weighted_acc.update(preds, y, output_timestamps)
        
    def on_validation_epoch_end(self):
        w_mse = self.val_lat_weighted_mse.compute()
        w_rmse = self.val_lat_weighted_rmse.compute()
        w_acc = self.val_lat_weighted_acc.compute()
       
        #scalar metrics
        loss_dict = {**w_mse, **w_rmse, **w_acc}
        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                prog_bar=False,
                sync_dist=True
            )
        self.val_lat_weighted_mse.reset()
        self.val_lat_weighted_rmse.reset()
        self.val_lat_weighted_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, input_timestamps, output_timestamps = batch

        if x.shape[1] > 1:
            raise NotImplementedError("history_range > 1 is not supported yet.")
        x = x.squeeze() #squeeze history dimension

        preds = self.net.forward(x, lead_times, variables, out_variables)

        #cast to float
        preds = preds.float()
        y = y.float()
        
        self.test_lat_weighted_mse.update(preds, y)
        self.test_lat_weighted_rmse.update(preds, y)
        self.test_lat_weighted_rmse_spatial_map.update(preds, y)
        self.test_lat_weighted_acc.update(preds, y, output_timestamps)
        self.test_lat_weighted_acc_spatial_map.update(preds, y, output_timestamps)

    def on_test_epoch_end(self):
        w_mse = self.test_lat_weighted_mse.compute()
        w_rmse = self.test_lat_weighted_rmse.compute()
        w_acc = self.test_lat_weighted_acc.compute()
        w_rmse_spatial_maps = self.test_lat_weighted_rmse_spatial_map.compute()
        w_acc_spatial_maps = self.test_lat_weighted_acc_spatial_map.compute()

        #scalar metrics
        loss_dict = {**w_mse, **w_rmse, **w_acc}
        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                prog_bar=False,
                sync_dist=True
            )

        #spatial maps
        for var in w_rmse_spatial_maps.keys():
            plot_spatial_map(np.flipud(w_rmse_spatial_maps[var].cpu().numpy()), title=var, filename=f"{self.logger.log_dir}/test_{var}.png")
        for var in w_acc_spatial_maps.keys():
            plot_spatial_map(np.flipud(w_acc_spatial_maps[var].cpu().numpy()), title=var, filename=f"{self.logger.log_dir}/test_{var}.png")

        self.test_lat_weighted_mse.reset()
        self.test_lat_weighted_rmse.reset()
        self.test_lat_weighted_acc.reset()
        self.test_lat_weighted_acc_spatial_map.reset()
        self.test_lat_weighted_rmse_spatial_map.reset()

    #optimizer definition - will be used to optimize the network based
    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.lr,
                    "betas": (self.beta_1, self.beta_2),
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.lr,
                    "betas": (self.beta_1, self.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.warmup_epochs,
            self.max_epochs,
            self.warmup_start_lr,
            self.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
