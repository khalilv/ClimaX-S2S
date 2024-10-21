# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

from climax.utils.data_utils import DEFAULT_PRESSURE_LEVELS, HRS_PER_LEAP_YEAR, leap_year_data_adjustment, leap_year_time_adjustment

def zarr2np_climatology(path, variables, years, save_dir, partition, hrs_per_step):
    assert HRS_PER_LEAP_YEAR % hrs_per_step == 0
    if not any(year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) for year in years):
        print("WARNING: No leap year present in climatology years. This may result in issues when calculating ACC during evaluation as there will be no climatology for Feb 29th.")
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)
    zarr_ds = xr.open_zarr(path)
    climatology = {}

    for year in tqdm(years):
        np_vars = {}

        # non-constant fields
        for var in variables:

            if len(zarr_ds[var].shape) == 3:  # surface level variables
                yearly_data = zarr_ds.sel(time=str(year))[var].expand_dims("val", axis=1)
                np_var = yearly_data.to_numpy()
                np_var = np.transpose(np_var (0,1,3,2)) #transpose to T x 1 x H x W

                if var not in climatology:
                    climatology[var] = [leap_year_data_adjustment(np_var, hrs_per_step)]
                else:
                    climatology[var].append(leap_year_data_adjustment(np_var, hrs_per_step))

            else:  # multiple-level variables, only use a subset
                assert len(zarr_ds[var].shape) == 4
                all_levels = zarr_ds["level"][:].to_numpy()
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                for level in all_levels:
                    ds_level = zarr_ds.sel(level=[level], time=str(year))
                    level = int(level)
                    np_vars[f"{var}_{level}"] = ds_level[var].to_numpy()
                    np_vars[f"{var}_{level}"] = np.transpose(np_vars[f"{var}_{level}"], (0,1,3,2)) #transpose to T x 1 x H x W

                    if f"{var}_{level}" not in climatology:
                        climatology[f"{var}_{level}"] = [leap_year_data_adjustment(np_vars[f"{var}_{level}"])]
                    else:
                        climatology[f"{var}_{level}"].append(leap_year_data_adjustment(np_vars[f"{var}_{level}"]))

    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    
    #save timestamps
    timestamps = zarr_ds.sel(time=str(year)).time.to_numpy()
    timestamps = [np.datetime_as_string(datetime, unit='m')[5:] for datetime in timestamps]
    climatology['timestamps'] = leap_year_time_adjustment(np.array(timestamps), hrs_per_step)
    
    np.savez(
        os.path.join(save_dir, partition, f"climatology_{years[0]}_{years[-1]}.npz" if len(years) > 1 else f"climatology_{years[0]}.npz"),
        **climatology,
    )

def zarr2np(path, variables, years, save_dir, partition, num_shards_per_year, grid_size, hrs_per_step):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)
    zarr_ds = xr.open_zarr(path)
    zarr_ds['orography'] = zarr_ds['geopotential_at_surface']/9.80665

    if partition == "train":
        normalize_mean = {}
        normalize_std = {}

    constant_fields = ["land_sea_mask", "orography", "lattitude"]
    constant_values = {}
    for f in constant_fields:
        if f == 'lattitude':
            constant_field = zarr_ds.get('latitude', zarr_ds.get('lattitude')).to_numpy()
            constant_field = np.tile(constant_field, (grid_size[1], 1))
        else:
            constant_field = zarr_ds[f].to_numpy()
        
        constant_field = constant_field.T
                   
        constant_values[f] = np.expand_dims(constant_field, axis=(0, 1)).repeat(
            HRS_PER_LEAP_YEAR // hrs_per_step, axis=0
        )
        if partition == "train":
            normalize_mean[f] = np.atleast_1d(constant_field.mean(axis=(0,1))).astype('float32')
            normalize_std[f] = np.atleast_1d(constant_field.std(axis=(0,1))).astype('float32')


    for year in tqdm(years):
        np_vars = {}

        # constant variables
        for f in constant_fields:
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                np_vars[f] = constant_values[f]
            else:
                np_vars[f] = constant_values[f][:(HRS_PER_LEAP_YEAR-24)//hrs_per_step]

        # non-constant fields
        for var in variables:

            if len(zarr_ds[var].shape) == 3:  # surface level variables
                yearly_data = zarr_ds.sel(time=str(year))[var].expand_dims("val", axis=1)
                # remove the last 24 hours if this year has 366 days
                np_vars[var] = yearly_data.to_numpy()
                np_vars[var] = np.transpose(np_vars[var], (0,1,3,2)) #transpose to T x 1 x H x W
                N = np_vars[var].shape[0]

                if partition == "train":  # compute mean and std of each var in each year
                    var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3)).item()
                    var_std_yearly = np_vars[var].std(axis=(0, 2, 3)).item()
                    if var not in normalize_mean:
                        normalize_mean[var] = [[var_mean_yearly, N]]
                        normalize_std[var] = [[var_std_yearly, N]]
                    else:
                        normalize_mean[var].append([var_mean_yearly, N])
                        normalize_std[var].append([var_std_yearly, N])

            else:  # multiple-level variables, only use a subset
                assert len(zarr_ds[var].shape) == 4
                all_levels = zarr_ds["level"][:].to_numpy()
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                for level in all_levels:
                    ds_level = zarr_ds.sel(level=[level], time=str(year))
                    level = int(level)
                    # remove the last 24 hours if this year has 366 days
                    np_vars[f"{var}_{level}"] = ds_level[var].to_numpy()
                    np_vars[f"{var}_{level}"] = np.transpose(np_vars[f"{var}_{level}"], (0,1,3,2)) #transpose to T x 1 x H x W
                    N = np_vars[f"{var}_{level}"].shape[0]

                    if partition == "train":  # compute mean and std of each var in each year
                        var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3)).item()
                        var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3)).item()
                        if var not in normalize_mean:
                            normalize_mean[f"{var}_{level}"] = [[var_mean_yearly, N]]
                            normalize_std[f"{var}_{level}"] = [[var_std_yearly, N]]
                        else:
                            normalize_mean[f"{var}_{level}"].append([var_mean_yearly, N])
                            normalize_std[f"{var}_{level}"].append([var_std_yearly, N])

        #save timestamps
        timestamps = zarr_ds.sel(time=str(year)).time.to_numpy()
        timestamps = [np.datetime_as_string(datetime, unit='m')[5:] for datetime in timestamps]
        np_vars['timestamps'] = np.array(timestamps)

        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            assert np_vars['timestamps'].shape[0] == (HRS_PER_LEAP_YEAR // hrs_per_step)
            num_steps_per_shard = (HRS_PER_LEAP_YEAR // hrs_per_step) // num_shards_per_year
        else:
            assert np_vars['timestamps'].shape[0] == ((HRS_PER_LEAP_YEAR - 24) // hrs_per_step)
            num_steps_per_shard = ((HRS_PER_LEAP_YEAR - 24) // hrs_per_step) // num_shards_per_year

        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_steps_per_shard
            if shard_id == num_shards_per_year - 1:
                sharded_data = {k: np_vars[k][start_id:] for k in np_vars.keys()}
            else:
                end_id = start_id + num_steps_per_shard
                sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
            )

    if partition == "train":
        for var in normalize_mean.keys():
            if var not in constant_fields: #dont need to aggregate for static variables
                mean = np.stack(normalize_mean[var], axis=0)
                std = np.stack(normalize_std[var], axis=0)
                agg_mean = np.sum(mean[:,1] * mean[:,0]) / np.sum(mean[:,1])
                sum_w_var = np.sum((std[:,1] - 1) * (std[:,0]**2))
                sum_group_var = np.sum((std[:,1]) * (mean[:,0] - agg_mean)**2)
                agg_std = np.sqrt((sum_w_var  + sum_group_var)/(np.sum(std[:,1]) - 1))
                normalize_mean[var] = np.atleast_1d(agg_mean).astype('float32')
                normalize_std[var] = np.atleast_1d(agg_std).astype('float32')

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)


@click.command()
@click.option("--root_dir", type=click.Path(exists=True))
@click.option("--save_dir", type=str)
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        # "toa_incident_solar_radiation",
        "total_precipitation_6hr",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "relative_humidity",
        "specific_humidity",
    ],
)
@click.option("--start_train_year", type=int, default=1979)
@click.option("--start_val_year", type=int, default=2016)
@click.option("--start_test_year", type=int, default=2017)
@click.option("--end_year", type=int, default=2019)
@click.option("--num_shards", type=int, default=8)
@click.option("--grid_size", type=(int, int), default=(32, 64))
@click.option("--hrs_per_step", type=int, default=1)
def main(
    root_dir,
    save_dir,
    variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
    grid_size,
    hrs_per_step
):
    assert start_val_year > start_train_year and start_test_year > start_val_year and end_year > start_test_year
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    os.makedirs(save_dir, exist_ok=True)

    zarr2np(root_dir, variables, train_years, save_dir, "train", num_shards, grid_size, hrs_per_step)
    zarr2np(root_dir, variables, val_years, save_dir, "val", num_shards, grid_size, hrs_per_step)
    zarr2np(root_dir, variables, test_years, save_dir, "test", num_shards, grid_size, hrs_per_step)

    climatology_val_years = range(1990, start_val_year)
    climatology_test_years = range(1990, start_test_year)
    zarr2np_climatology(root_dir, variables, climatology_val_years, save_dir, "val", hrs_per_step)
    zarr2np_climatology(root_dir, variables, climatology_test_years, save_dir, "test", hrs_per_step)

    # save lat and lon data
    zarr_ds = xr.open_zarr(root_dir)
    lat = zarr_ds["latitude"].to_numpy()
    lon = zarr_ds["longitude"].to_numpy()
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)


if __name__ == "__main__":
    main()