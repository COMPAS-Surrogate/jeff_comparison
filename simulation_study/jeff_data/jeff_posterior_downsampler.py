import pandas as pd
import pygtc


def load_posterior_data(file_path: str) -> pd.DataFrame:
    """
    Load posterior data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the posterior data.
    """
    # Read the CSV file into a DataFrame
    # headers = [muz, sigma_0, aSF, dSF, lnl]
    # drop lnl

    df = pd.read_csv(file_path, header=None)
    df.columns = ["muz", "sigma_0", "aSF", "dSF", "lnl"]
    # Drop the 'lnl' column
    df = df.drop(columns=["lnl"])

    # check i can convert to flaot32 without losing precision
    df2 = df.astype("float32")
    df2 = df2.sample(frac=0.5, random_state=42)
    df2.reset_index(drop=True, inplace=True)
    df2.to_feather("jeff_posterior.feather", compression="zstd", compression_level=10)
    df2 = pd.read_feather("jeff_posterior.feather")

    fig = pygtc.plotGTC(chains=[df.values, df2.values],
                        paramNames=["muz", "sigma_0", "aSF", "dSF"],
                        chainLabels=['64', '32'],
                        plotName="posterior.pdf",
                        )

    # save df2 as hdf5 file (better compression)

    return df


load_posterior_data("COMPAS_mcmc_out.csv")