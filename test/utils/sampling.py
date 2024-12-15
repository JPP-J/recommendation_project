import pandas as pd
import numpy as np


def sample_dataframe(df, method='random', **kwargs):
    """
    Sample data from a DataFrame using different methods.

    Parameters:
    df (pandas.DataFrame): Input DataFrame
    method (str): Sampling method ('random', 'systematic', 'stratified')
    **kwargs:
        - n (int): Number of samples (for random and systematic)
        - frac (float): Fraction of data to sample (alternative to n)
        - strata_col (str): Column name for stratified sampling
        - random_state (int): Random seed for reproducibility

    Returns:
    pandas.DataFrame: Sampled DataFrame
    """

    # Set random seed if provided
    if 'random_state' in kwargs:
        np.random.seed(kwargs['random_state'])

    if method == 'random':
        # Random sampling
        if 'n' in kwargs:
            return df.sample(n=kwargs['n'])
        elif 'frac' in kwargs:
            return df.sample(frac=kwargs['frac'])
        else:
            raise ValueError("Either 'n' or 'frac' must be provided for random sampling")

    elif method == 'systematic':
        # Systematic sampling
        if 'n' in kwargs:
            k = len(df) // kwargs['n']  # Calculate step size
            return df.iloc[::k].head(kwargs['n'])
        elif 'frac' in kwargs:
            n = int(len(df) * kwargs['frac'])
            k = len(df) // n  # Calculate step size
            return df.iloc[::k].head(n)
        else:
            raise ValueError("Either 'n' or 'frac' must be provided for systematic sampling")

    elif method == 'stratified':
        # Stratified sampling
        if 'strata_col' not in kwargs:
            raise ValueError("'strata_col' must be provided for stratified sampling")

        if 'n' in kwargs:
            return df.groupby(kwargs['strata_col'], group_keys=False).apply(
                lambda x: x.sample(n=int(kwargs['n'] / len(df[kwargs['strata_col']].unique())))
            )
        elif 'frac' in kwargs:
            return df.groupby(kwargs['strata_col'], group_keys=False).sample(frac=kwargs['frac'])
        else:
            raise ValueError("Either 'n' or 'frac' must be provided for stratified sampling")

    else:
        raise ValueError("Invalid sampling method. Choose 'random', 'systematic', or 'stratified'")