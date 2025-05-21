from lnl_computer.cosmic_integration.mcz_grid import McZGrid
import sys

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
grid.bin_data()
fig = grid.plot()
fig.savefig("mcz_grid.png")
