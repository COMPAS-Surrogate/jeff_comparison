"""
Lets compute the LnL at Jeff's max-posterior point mine, and plot LnL as a function of aSF.

Lets also plot the model.


python 1d_study.py -c /home/avaj040/Documents/projects/data/COMPAS_data/jeff_data/h5out_32M.h5 -o out -n 10

"""

from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import (
    DEFAULT_DICT,
    draw_star_formation_samples
)
from lnl_computer.observation.mock_observation import MockObservation
from compas_python_utils.cosmic_integration.binned_cosmic_integrator.bbh_population import \
    generate_mock_bbh_population_file
from lnl_computer.observation import load_observation
from collections import namedtuple
import os
import warnings
from collections import namedtuple
from typing import List

import matplotlib.pyplot as plt
import click

JEFF_PARMS = dict(mu_z=-0.154, sigma_0=0.546, aSF=0.001, dSF=4.76)
AVI_PARMS = dict(mu_z=-0.16, sigma_0=0.56, aSF=0.01, dSF=4.70)

DATA = namedtuple("DATA", ["lnl", "aSF", "n_det", "mcz"])


def compute_lnls_for_param(compas_h5_path, obs, param='aSF', n_samples=10, outdir: str = "out",
                           base_params=JEFF_PARMS) -> List[DATA]:
    data = []
    lnl_kwgs = dict(
        mcz_obs=obs,
        compas_h5_path=compas_h5_path,
        save_plots=True,
        outdir=outdir,
    )

    print(f"Generating McZ grid: {base_params}, obs: {obs}")
    samples = draw_star_formation_samples(n_samples, parameters=param, grid=True, as_list=True)
    for sf_sample in samples:
        theta = base_params.copy()
        theta[param] = float(sf_sample[param])

        model = McZGrid.generate_n_save(
            sf_sample=sf_sample,
            **lnl_kwgs,
        )
        lnl, unc = model.get_lnl(mcz_obs=obs)
        n_det = model.n_detections(obs.duration)
        data.append(DATA(lnl=lnl, aSF=theta[param], n_det=n_det, mcz=model.rate_matrix))

    return data


@click.command()
@click.option("-c", "--compas-h5-path", type=str, required=True)
@click.option("-o", "--outdir", type=str, default="out")
@click.option("-n", type=int, default=10)
def main(compas_h5_path, outdir, n=10):
    obs = load_observation("lvk")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data = compute_lnls_for_param(compas_h5_path, obs, param='aSF', n_samples=n, outdir=outdir, base_params=JEFF_PARMS)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot([d.aSF for d in data], [d.lnl for d in data], color="tab:red", label='LnL')
    ax2.plot([d.aSF for d in data], [d.n_det for d in data], color="tab:blue", label='n_det')
    ax.set_xlabel("aSF")
    ax.set_ylabel("LnL", color="tab:red")
    ax2.set_ylabel("n_det", color="tab:blue")
    fig.savefig(os.path.join(outdir, "lnl_vs_aSF.png"))
    fig.clear()
    plt.close(fig)


if __name__ == "__main__":
    main()
