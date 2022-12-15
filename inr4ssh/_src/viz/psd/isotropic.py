import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_psd_isotropic_wavenumber(freq, psd, ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    else:
        fig = plt.gcf()

    ax.plot(freq, psd, **kwargs)

    ax.set(
        yscale="log",
        xscale="log",
        xlabel="Wavenumber [cycles/km]",
        ylabel="PSD [m$^{2}$/cycles/m]",
    )

    ax.legend()
    ax.grid(which="both", alpha=0.5)

    return fig, ax


def plot_psd_isotropic_wavelength(freq, psd, ax=None, **kwargs):

    fig, ax = plot_psd_isotropic_wavenumber(1.0 / freq, psd, ax=ax, **kwargs)

    ax.set(
        yscale="log",
        xscale="log",
        xlabel="Wavelength [km]",
        ylabel="PSD [m$^{2}$/cycles/m]",
    )

    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.invert_xaxis()

    return fig, ax


def plot_psd_isotropic(freq, psd, ax=None, **kwargs):

    fig, ax = plot_psd_isotropic_wavenumber(freq, psd, ax=ax, **kwargs)

    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: 1 / (x + 1e-20), lambda x: 1 / (x + 1e-20))
    )
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.set(xlabel="Wavelength [km]")

    return fig, ax, secax


def plot_psd_score_isotropic_wavenumber(freq, psd_ref, psd_study, **kwargs):

    return None


def plot_psd_score_isotropic_wavelength(freq, psd_ref, psd_study, **kwargs):

    return None


def plot_psd_score_isotropic(freq, psd_ref, psd_study, **kwargs):

    return None
