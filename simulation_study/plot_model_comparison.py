from lnl_computer.cosmic_integration.mcz_grid import McZGrid
import sys
from jeff_data import JeffRateMatrix
import matplotlib.pyplot as plt
import numpy as np
import os



jeff_matrix = JeffRateMatrix.load()


def plot(ax, rate2d, z_edges, mc_edges):
    ax.pcolormesh(
        z_edges,
        mc_edges,
        rate2d,
        cmap='inferno',
        norm="linear",
    )
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Chirp mass ($M_{\odot}$)")
    ax.set_facecolor("black")

    n_events_per_year = np.nansum(rate2d)
    ax.annotate(
        f"Grid: {rate2d.T.shape}\nN det: {n_events_per_year:.2f}/yr",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="white",
    )


grid_fname = 'mcz_grid.h5'
if os.path.exists(grid_fname):
    print(f"Loading McZ grid from {grid_fname}")
    grid = McZGrid.from_h5(grid_fname)
else:
    COMPAS_H5 = sys.argv[1]
    grid  = McZGrid.from_compas_output(
        compas_path=COMPAS_H5,
        cosmological_parameters={
            "aSF": 0.012,
            "dSF": 4.253,
            "mu_z": -0.325,
            "sigma_0": 0.213,
        },
    )
    grid.save("mcz_grid.h5")

grid.bin_data(
    mc_bins=jeff_matrix.Mc_edges,
    z_bins=jeff_matrix.z_edges,
)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plot(ax[0], grid.rate_matrix, grid.redshift_bins, grid.chirp_mass_bins)
plot(ax[1], jeff_matrix.data, jeff_matrix.z_edges, jeff_matrix.Mc_edges)
ax[0].set_title("Avi")
ax[1].set_title("Jeff")
plt.tight_layout()
plt.show()




