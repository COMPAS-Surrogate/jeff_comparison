import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


def _parse_float_list(line):
    return np.array([
        float(x) if x.lower() != "inf" else np.inf
        for x in line.split(",")
    ])


@dataclass
class JeffRateMatrix:
    Mc_edges: np.ndarray
    z_edges: np.ndarray
    data: np.ndarray

    @classmethod
    def load(cls):
        """
        First 5 rows are header rows:
        #Mc bins, #z bins
        Mc bin right edges (last bin right edge is inf)
        Mc bin widths (last bin right edge is inf)
        z bin right edges
        z bin widths
        Each subsequent row is detection rate for Mc bins for (increasing) z bin
        """

        # ——— LOAD ———
        with open("jeff_data/binned_rates_alpha-0.325_sigma0.213_asf0.012_dsf4.253.csv", "r") as f:
            lines = f.read().splitlines()

        # Parse header
        num_Mc_bins, num_z_bins = map(int, lines[0].split(","))

        # Mc bin right edges
        Mc_right_edges = _parse_float_list(lines[1])
        Mc_widths = _parse_float_list(lines[2])
        Mc_left_edges = Mc_right_edges - Mc_widths
        Mc_left_edges[-1] = Mc_right_edges[-1]
        Mc_edges = np.concatenate((Mc_left_edges, Mc_right_edges[-1:]))

        # z bin right edges
        z_right_edges = _parse_float_list(lines[3])
        z_widths = _parse_float_list(lines[4])
        z_left_edges = z_right_edges - z_widths[0]
        z_edges = np.concatenate((z_left_edges, z_right_edges[-1:]))

        # Print bin edges
        print(f"Mc bin edges: {Mc_edges.shape} vs {num_Mc_bins + 1}")
        print(f"z bin edges: {z_edges.shape} vs {num_z_bins + 1}")

        # LOAD DATA BLOCK
        data_lines = lines[5:5 + num_z_bins]
        data = np.array([
            _parse_float_list(line)
            for line in data_lines
        ])
        print(f"Data shape: {data.shape} (should be {num_z_bins}, {num_Mc_bins})")

        # drop the inf bin (last Mc bin)
        data = data[:, :-2]
        Mc_edges = Mc_edges[:-2]
        return cls(
            Mc_edges=Mc_edges,
            z_edges=z_edges,
            data=data.T,
        )

    def plot(self, ax=None):
        Mc_edges = self.Mc_edges
        z_edges = self.z_edges
        data = self.data

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        pc = ax.pcolormesh(z_edges, Mc_edges, data, shading='auto', cmap='viridis')
        fig.colorbar(pc, ax=ax, label="Detection rate")
        ax.set_xlabel(r"Redshift $z$")
        ax.set_ylabel(r"Chirp mass $M_c$")
        ax.set_title("Detection Rate vs. $M_c$ and $z$ bins")
        ax.set_yscale("log")
        ax.set_ylim(Mc_edges[1], Mc_edges[-1])
        plt.tight_layout()
        return fig


def load_posterior():
    """
    Load the posterior samples from the COMPAS h5 file.
    :return: pandas dataframe with the posterior samples
    """
    # Load the posterior samples from the COMPAS h5 file
    df = pd.read_feather("jeff_data/jeff_posterior.feather")
    return df
