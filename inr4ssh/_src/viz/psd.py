from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_psd_spectrum(psd_study, psd_ref, wavenumber):

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.invert_xaxis()

    # plot the reference PSD
    ax.plot((1.0 / wavenumber), psd_ref, label="Reference", color="k")
    # plot reconstruction PSD
    ax.plot((1.0 / wavenumber), psd_study, label="Reconstruction", color="lime")

    ax.set(
        xlabel="Wavelength [km]",
        ylabel="Power Spectral Density [m$^{2}$/cy/km]",
        xscale="log",
        yscale="log",
        xlim=(10e3, 10e0),
    )
    plt.legend(loc="best", fontsize=12)
    plt.grid(which="both", alpha=0.5)
    plt.tight_layout()
    # fig.savefig("./psd_spectrum.png")
    plt.show()

    return fig, ax


def plot_psd_score(psd_diff, psd_ref, wavenumber, resolved_scale, xmin: float = 10e6):

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.invert_xaxis()

    # plot normalized difference psd
    n_psd_diff = 1.0 - psd_diff / psd_ref
    ax.plot((1.0 / wavenumber), n_psd_diff, label="Difference", color="k")

    # plot threshold for score
    ax.hlines(
        y=0.5,
        xmin=np.ma.min(np.ma.masked_invalid(1.0 / wavenumber)),
        xmax=np.ma.max(np.ma.masked_invalid(1.0 / wavenumber)),
        color="red",
        lw=1.2,
        ls="--",
    )
    ax.vlines(
        x=resolved_scale,
        ymin=0.0,
        ymax=1.0,
        lw=1.2,
        color="green",
    )
    ax.fill_betweenx(
        n_psd_diff,
        resolved_scale,
        np.ma.max(np.ma.masked_invalid(1.0 / wavenumber)),
        color="green",
        alpha=0.3,
        label=f"Resolved Scales \n $\lambda$ > {int(resolved_scale)} km",
    )

    ax.set(
        xlabel="Wavelength [km]",
        ylabel="PSD Score [1. - PSD$_{err}$/PSD$_{ref}$]",
        xlim=(10e3, 10e0),
        xscale="log",
    )
    plt.legend(loc="best", fontsize=12)
    plt.grid(which="both", alpha=0.5)
    plt.tight_layout()
    # fig.savefig("./psd_effres.png")
    plt.show()

    return fig, ax
