import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_psd_spacetime_wavenumber(freq_x, freq_y, psd):

    fig, ax = plt.subplots()

    locator = ticker.LogLocator()
    norm = colors.LogNorm()

    pts = ax.contourf(
        freq_x, freq_y, psd, norm=norm, locator=locator, cmap="RdYlGn", extend="both"
    )

    ax.set(
        yscale="log",
        xscale="log",
        xlabel="Wavenumber [cycles/km]",
        ylabel="Wavenumber [cycles/days]",
    )
    # colorbar
    fmt = ticker.LogFormatterMathtext(base=10)
    cbar = fig.colorbar(
        pts,
        pad=0.02,
        format=fmt,
    )
    cbar.ax.set_ylabel(r"PSD [m$^{2}$/cycles/m]")

    plt.grid(which="both", linestyle="--", linewidth=1, color="black", alpha=0.2)

    return fig, ax, cbar


def plot_psd_spacetime_wavelength(freq_x, freq_y, psd):

    fig, ax, cbar = plot_psd_spacetime_wavenumber(1 / freq_x, 1 / freq_y, psd)
    ax.set(
        yscale="log", xscale="log", xlabel="Wavelength [km]", ylabel="Wavelength [days]"
    )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.yaxis.set_major_formatter("{x:.0f}")

    return fig, ax, cbar
