# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from climax.global_forecast.datamodule import GlobalForecastDataModule
from climax.global_forecast.module import GlobalForecastModule
from pytorch_lightning.cli import LightningCLI

# 1) entry point high-level class for training climaX. 

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=GlobalForecastModule,
        datamodule_class=GlobalForecastDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_variables(cli.datamodule.in_variables, cli.datamodule.out_variables)
    cli.model.set_val_clim(cli.datamodule.val_clim, cli.datamodule.val_clim_timestamps)
    cli.model.set_test_clim(cli.datamodule.test_clim, cli.datamodule.val_clim_timestamps)
    cli.model.init_metrics()
    cli.model.init_network(cli.datamodule.in_variables)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
