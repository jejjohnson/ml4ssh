from typing import List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from inr4ssh._src.metrics.psd import (
    psd_spacetime,
    psd_spacetime_score,
    wavelength_resolved_spacetime,
)
from inr4ssh._src.preprocess.coords import (
    correct_coordinate_labels,
    correct_longitude_domain,
)
from inr4ssh._src.preprocess.regrid import oi_regrid
from dataclasses import dataclass
from tqdm import tqdm
from inr4ssh._src.viz.psd.spacetime import plot_psd_spacetime_wavelength


@dataclass
class Results:
    url: str
    variable: str
    name: str


import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)
import ml_collections
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Evaluation:
    config: ml_collections.ConfigDict
    ds_field: Optional[xr.Dataset] = None
    model_ref: str = None
    variables: List[str] = field(default_factory=lambda: [])
    models: List[str] = field(default_factory=lambda: [])

    def load_reference(self, config: Optional = None):
        if config is None:
            config = self.config.reference
        self.ds_field = get_data_reference(config)
        self.ds_field = self.ds_field.rename({"ssh": "ssh_natl60"})
        self.models = [config.model_name]
        self.variables = ["ssh"]
        self.model_ref = config.model_name

        self.add_units()

    def load_data_from_config(self):
        self.ds_field = get_data_evaluation(self.config)

    def add_model(self, config: Optional = None):
        if config is None:
            config = self.config.study
        ds_study = get_data_study(config)
        var_name = "ssh_" + config.model_name
        var_ref = "ssh_" + self.model_ref

        self.ds_field[var_name] = oi_regrid(
            da_source=ds_study["ssh"], da_target=self.ds_field[var_ref]
        )
        self.models.append(config.model_name)
        self.add_units()

    def add_units(self):

        for ivar in [i for i in self.ds_field.keys()]:
            self.ds_field[ivar].attrs["units"] = "meters"

        self.ds_field.longitude.attrs["units"] = "degrees"
        self.ds_field.longitude.attrs["units_bnd"] = [-180, 180]
        self.ds_field.latitude.attrs["units"] = "degrees"
        self.ds_field.latitude.attrs["units_bnd"] = [-180, 180]
        self.ds_field.time.attrs["units"] = "datetime"

    def add_ke(self):

        ds_variables = [i for i in self.ds_field.data_vars]

        models = [i for i in self.models]

        for imodel in tqdm(models):

            if imodel in ds_variables:
                continue

            # calculate UV components
            u, v = ssh2uv_da_2dt(self.ds_field[f"ssh_{imodel}"])

            self.ds_field[f"u_{imodel}"] = (("time", "latitude", "longitude"), u)
            self.ds_field[f"v_{imodel}"] = (("time", "latitude", "longitude"), v)

            ke = kinetic_energy(u, v)

            self.ds_field[f"ke_{imodel}"] = (("time", "latitude", "longitude"), ke)

        if "ke" not in self.variables:
            self.variables.append("ke")
        if "u" not in self.variables:
            self.variables.append("u")
        if "v" not in self.variables:
            self.variables.append("v")

    def add_rv(self):

        ds_variables = [i for i in self.ds_field.data_vars]

        models = [i for i in self.models]

        for imodel in tqdm(models):

            if imodel in ds_variables:
                continue

            rv = ssh2rv_da_2dt(self.ds_field[f"ssh_{imodel}"])

            self.ds_field[f"rv_{imodel}"] = (("time", "latitude", "longitude"), rv)

        if "rv" not in self.variables:
            self.variables.append("rv")


@dataclass
class PSDSTScoreEval:
    config: ml_collections.ConfigDict
    ds_field: Optional[xr.Dataset] = None
    dict_psd: Dict = field(default_factory=lambda: {})
    model_ref: str = None
    variables: List[str] = field(default_factory=lambda: [])
    models: List[str] = field(default_factory=lambda: [])

    def __init__(self, eval_obj):
        self.config = eval_obj.config
        self.ds_field = eval_obj.ds_field.copy()
        self.model_ref = eval_obj.model_ref
        self.variables = eval_obj.variables
        self.models = eval_obj.models
        self.dict_psd = {}

    def standardize_coords(self, config: Optional = None):
        if config is None:
            config = self.config.psd
        self.ds_field = standardize_field(self.ds_field, config)

    def calculate_psd_score(self):

        # Time-Longitude (Lat avg) PSD Score
        self.ds_field = self.ds_field.chunk(
            {
                "time": 1,
                "longitude": self.ds_field["longitude"].size,
                "latitude": self.ds_field["latitude"].size,
            }
        ).compute()

        variables = [i for i in self.ds_field.data_vars]

        models = list(filter(lambda ivar: self.model_ref not in ivar, self.models))
        self.models = models

        for imodel in tqdm(models):
            for ivariable in tqdm(self.variables):
                ivar_name_study = ivariable + "_" + imodel
                ivar_name_ref = ivariable + "_" + self.model_ref
                self.dict_psd[ivar_name_study] = psd_spacetime_score(
                    self.ds_field[ivar_name_study], self.ds_field[ivar_name_ref]
                )

    def plot(self, variable):

        from inr4ssh._src.viz.psd.spacetime import plot_psd_spacetime_score_wavelength

        factor = 1e3 if self.config.psd.units == "meters" else 1.0

        fig, ax, cbar = plot_psd_spacetime_score_wavelength(
            self.dict_psd[variable].freq_longitude * factor,
            self.dict_psd[variable].freq_time,
            self.dict_psd[variable],
        )

        x_units = "km" if self.config.psd.units == "meters" else "degrees"
        ax.set_xlabel(f"Wavelength [{x_units}]")
        units = get_variable_units(variable)
        cbar.ax.set_ylabel("PSD [" + units + "]")

        plt.tight_layout()
        save_name = f"dc20a_psd_score_spacetime_{self.config.figure.save_name}_{variable}_{self.config.psd.units}.png"
        save_path = Path(self.config.figure.save_path).joinpath(save_name)
        fig.savefig(save_path)
        plt.show()

    def plot_all(self):

        for ivariable in self.variables:
            self.plot(ivariable)

        return None

    def stats(self, variable, model):

        ds_vars = [i for i in self.ds_field.data_vars]

        ds_vars = list(filter(lambda ivar: model in ivar.split("_"), ds_vars))
        ds_vars = list(filter(lambda ivar: variable in ivar.split("_"), ds_vars))

        assert len(ds_vars) == 1

        ds_vars = ds_vars[0]

        units = "km" if self.config.psd.units == "meters" else "degrees"

        column_names = [
            "Algorithm",
            "Variable",
            f"Spatial Scale ({units})",
            "Temporal Scale (days)",
        ]

        spatial_resolved, time_resolved = wavelength_resolved_spacetime(
            self.dict_psd[ds_vars]
        )

        factor = 1e-3 if self.config.psd.units == "meters" else 1.0

        psd_stats = pd.DataFrame(
            data=[
                [
                    self.config.study.model_name.capitalize(),
                    ds_vars,
                    spatial_resolved * factor,
                    time_resolved,
                ]
            ],
            columns=column_names,
        )

        return psd_stats

    def stats_all(self):

        psd_stats = pd.DataFrame()

        for ivariable in self.variables:
            for imodel in self.models:
                psd_stats = pd.concat(
                    [psd_stats, self.stats(ivariable, imodel)], axis=0
                )

        return psd_stats


# class STPSDScorePlotter:
#     def __init__(self, results, config):
#         self.results = results
#         self.config = config
#
#     @property
#     def variables(self):
#         return list(self.results.keys())
#
#     def plot(self, variable):
#
#         factor = 1e3 if config.psd.units == "meters" else 1.0
#
#         fig, ax, cbar = plot_psd_spacetime_score_wavelength(
#             self.results[variable].freq_longitude * factor,
#             self.results[variable].freq_time,
#             self.results[variable],
#         )
#
#         x_units = "km" if config.psd.units == "meters" else "degrees"
#         ax.set_xlabel(f"Wavelength [{x_units}]")
#         units = get_variable_units(variable)
#         cbar.ax.set_ylabel("PSD [" + units + "]")
#
#         plt.tight_layout()
#         save_name = f"dc20a_psd_score_spacetime_{config.figure.save_name}_{variable}_{config.psd.units}.png"
#         save_path = Path(config.figure.save_path).joinpath(save_name)
#         fig.savefig(save_path)
#         plt.show()
#
#     def plot_all(self):
#
#         for ivariable in self.variables:
#             self.plot(ivariable)
#
#         return None
#
#     def stats(self, variable):
#
#         units = "km" if self.config.psd.units == "meters" else "degrees"
#
#         column_names = ["Algorithm", "Variable", f"Spatial Scale ({units})", "Temporal Scale (days)"]
#
#         spatial_resolved, time_resolved = wavelength_resolved_spacetime(self.results[variable])
#
#         factor = 1e-3 if config.psd.units == "meters" else 1.0
#
#         psd_stats = pd.DataFrame(
#                     data=[[config.study.model_name.capitalize(), variable, spatial_resolved*factor, time_resolved]],
#                     columns=column_names
#                 )
#
#         return psd_stats
#
#     def stats_all(self):
#
#         psd_stats = pd.DataFrame()
#
#         for ivariable in self.variables:
#             psd_stats = pd.concat([psd_stats, self.stats(ivariable)], axis=0)
#
#         return psd_stats
#


def get_data_reference(config):

    ds_field = xr.open_mfdataset(config.path, preprocess=preprocess)
    ds_field = post_process(ds_field, config.var_name)

    return ds_field


def get_data_study(config):

    ds = xr.open_dataset(config.path)

    ds = post_process(ds, config.var_name)

    return ds


def get_data_evaluation(config):
    logger.info("Loading reference data...")
    ds_field = get_data_reference(config.reference)

    logger.info("Loading study data...")
    ds_study = get_data_study(config.study)

    var_name = "ssh_" + config.study.model_name

    logger.info("Regridding: Study to Reference...")
    ds_field[var_name] = oi_regrid(ds_study["ssh"], ds_field["ssh"])

    # add attributes
    ds_field = ds_field.rename({"ssh": "ssh_natl60"})
    ds_field.ssh_natl60.attrs["units"] = "meters"
    ds_field[var_name].attrs["units"] = "meters"
    ds_field.longitude.attrs["units"] = "degrees"
    ds_field.longitude.attrs["units_bnd"] = [-180, 180]
    ds_field.latitude.attrs["units"] = "degrees"
    ds_field.latitude.attrs["units_bnd"] = [-180, 180]
    ds_field.time.attrs["units"] = "datetime"

    logger.info("Calculating KE and RV...")
    ds_field = calculate_ke_pv(ds_field)

    return ds_field


def preprocess(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.resample(time="1D").mean()
    return ds


def post_process(ds: xr.Dataset, variable) -> xr.Dataset:

    ds = ds.sel(time=slice(np.datetime64("2012-10-22"), np.datetime64("2012-12-02")))

    ds = correct_coordinate_labels(ds)

    ds = ds.rename({variable: "ssh"})

    ds = correct_longitude_domain(ds)

    ds = ds.transpose("time", "latitude", "longitude")

    return ds


def load_ds_study(ds_ref, path, variable, name):

    ds = xr.open_dataset(path)

    ds = post_process(ds, variable)

    return ds_ref


def load_results_ds(path: str, variable: str) -> xr.Dataset:

    ds = xr.open_dataset(path)

    ds = post_process(ds, variable)

    return ds


def aggregate_results(ds_ref, results: List[Results]):

    for iresult in tqdm(results):
        ds_study = load_results_ds(iresult.url, iresult.variable)

        var_name = "ssh_" + iresult.name

        ds_ref[var_name] = oi_regrid(ds_study["ssh"], ds_ref["ssh"])

    return ds_ref


from inr4ssh._src.operators.ssh import ssh2uv_da_2dt, kinetic_energy, ssh2rv_da_2dt


def calculate_ke_pv(ds_ref):

    variables = [i for i in ds_ref.data_vars]

    for ivar in tqdm(variables):

        imodel = ivar.split("_")[1]

        # calculate UV components
        u, v = ssh2uv_da_2dt(ds_ref[ivar])

        ds_ref[f"u_{imodel}"] = (("time", "latitude", "longitude"), u)
        ds_ref[f"v_{imodel}"] = (("time", "latitude", "longitude"), v)

        ke = kinetic_energy(u, v)

        ds_ref[f"ke_{imodel}"] = (("time", "latitude", "longitude"), ke)

        rv = ssh2rv_da_2dt(ds_ref[ivar])

        ds_ref[f"rv_{imodel}"] = (("time", "latitude", "longitude"), rv)

    return ds_ref


def calculate_psd(ds_ref):

    logger.info("Calculating SpaceTime PSD...")
    # Time-Longitude (Lat avg) PSD Score
    ds_ref = ds_ref.chunk(
        {
            "time": 1,
            "longitude": ds_ref["longitude"].size,
            "latitude": ds_ref["latitude"].size,
        }
    ).compute()

    variables = [i for i in ds_ref.data_vars]

    results = {}

    for ivar in tqdm(variables):
        results[ivar] = psd_spacetime(ds_ref[ivar])

    return results


def calculate_psd_score(ds_ref):

    # Time-Longitude (Lat avg) PSD Score
    ds_ref = ds_ref.chunk(
        {
            "time": 1,
            "longitude": ds_ref["longitude"].size,
            "latitude": ds_ref["latitude"].size,
        }
    ).compute()

    ds_psd_res = {}

    variables = [i for i in ds_ref.data_vars]

    vars_ref = list(filter(lambda ivar: "natl60" in ivar, variables))
    vars_study = list(filter(lambda ivar: "natl60" not in ivar, variables))

    for ivar_study, ivar_ref in tqdm(list(zip(vars_study, vars_ref))):
        ivar_name = ivar_study.split("_")[0]
        ds_psd_res[ivar_name] = psd_spacetime_score(
            ds_ref[ivar_study], ds_ref[ivar_ref]
        )

    return ds_psd_res


def standardize_field(ds_field, config):
    from evaluation import (
        standardize_time,
        transform_coordinates,
        standardize_coordinates,
    )

    logger.info("Standardizing Time...")
    factor_time = np.timedelta64(*config.factor_time)
    ds_field["time"] = standardize_time(ds_field.time, factor=factor_time)

    logger.info("Transforming Lat/Lon...")
    ds_field["longitude"] = transform_coordinates(
        ds_field["longitude"], units=config.units
    )
    ds_field["latitude"] = transform_coordinates(
        ds_field["latitude"], units=config.units
    )

    logger.info("Standardizing lat/lon...")
    factor_space = config.factor_space
    ds_field["longitude"] = standardize_coordinates(
        ds_field["longitude"], factor=factor_space
    )
    ds_field["latitude"] = standardize_coordinates(
        ds_field["latitude"], factor=factor_space
    )

    return ds_field


def standardize_time(ds, factor=np.timedelta64(1, "D")):

    if ds.attrs["units"] == "normalized":
        return ds

    time_min = ds[0]

    ds = (ds - time_min) / factor

    ds.attrs["units"] = "normalized"
    ds.attrs["factor"] = factor
    ds.attrs["min"] = str(time_min.data)

    return ds


def transform_coordinates(ds, units: str = "degrees"):

    if units == "degrees":

        if ds.attrs["units"] == "degrees":
            ds.attrs["unit_bounds"] = [float(ds.min().data), float(ds.max().data)]
            return ds
        else:
            ds = ds / 111e3
            ds.attrs["units"] = "degrees"
            ds.attrs["unit_bounds"] = [float(ds.min().data), float(ds.max().data)]
            return ds

    elif units == "meters":
        if ds.attrs["units"] == "meters":
            ds.attrs["unit_bounds"] = [float(ds.min().data), float(ds.max().data)]
            return ds
        else:
            ds = ds * 111e3
            ds.attrs["units"] = "meters"
            ds.attrs["unit_bounds"] = [float(ds.min().data), float(ds.max().data)]
            return ds

    else:
        raise ValueError(f"Unrecongized unit: {units}")


def standardize_coordinates(ds, factor=1):

    if ds.attrs["units"] == "standardized":
        return ds

    old_units = ds.attrs["units"]
    old_bounds = ds.attrs["unit_bounds"]
    coord_min = ds.min()
    ds = (ds - coord_min) / factor

    ds.attrs["units"] = "standardized"
    ds.attrs["coord_min"] = float(coord_min.data)
    ds.attrs["old_units"] = old_units
    ds.attrs["olds_bounds"] = old_bounds
    return ds


def get_variable_units(variable, unit: str = "meters"):

    if unit in ["m", "meters"]:
        unit = "m"
    elif unit in ["deg", "degrees"]:
        unit = "degrees"

    if variable == "ssh":
        return r"m$^2$/cyles/m"
    elif variable == "ke":
        return r"m$^4$s$^{-2}$/cyles/m"
    elif variable == "u" or variable == "v":
        return r"m$^4$s$^{-2}$/cyles/m"
    elif variable == "rv":
        return r"s$^{-1}$/cyles/m"


def plot_psd_spacetime(dict_psd, config, variable: str = "ssh"):

    data = [
        f"{variable}_natl60",
        f"{variable}_{config.study.model_name}",
    ]
    names = [
        "natl60",
        config.study.model_name,
    ]

    for idata, iname in zip(data, names):

        factor = 1e3 if config.psd.units == "meters" else 1.0

        fig, ax, cbar = plot_psd_spacetime_wavelength(
            dict_psd[idata].freq_longitude * factor,
            dict_psd[idata].freq_time,
            dict_psd[idata],
        )

        x_units = "km" if config.psd.units == "meters" else "degrees"
        ax.set_xlabel(f"Wavelength [{x_units}]")
        units = get_variable_units(variable)
        cbar.ax.set_ylabel("PSD [" + units + "]")

        plt.tight_layout()
        save_name = f"dc20a_psd_spacetime_{config.figure.save_name}_{variable}_{config.psd.units}.png"
        save_path = Path(config.figure.save_path).joinpath(save_name)
        fig.savefig(save_path)
        plt.show()

    return None


def plot_psd_spacetime_all(dict_psd, config):

    plot_psd_spacetime(dict_psd, config, "ssh")
    plot_psd_spacetime(dict_psd, config, "u")
    plot_psd_spacetime(dict_psd, config, "v")
    plot_psd_spacetime(dict_psd, config, "ke")
    plot_psd_spacetime(dict_psd, config, "rv")

    return None
