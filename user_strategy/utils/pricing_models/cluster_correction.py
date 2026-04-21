import pandas as pd


def calculate_cluster_correction(
    cluster_betas: pd.DataFrame, threshold: float = 0.5
) -> pd.Series:
    """Calculate the correction factor for each subcluster.

    Args:
        cluster_betas: Square beta DataFrame (instruments x instruments).
        threshold: Threshold fraction for counting significant cluster members.

    Returns:
        pd.Series of correction factors indexed by instrument.
    """
    if cluster_betas.empty:
        return pd.Series(dtype=float)

    cluster_betas = cluster_betas.sort_index(axis=1).sort_index(axis=0).copy()

    for label in cluster_betas.index:
        if label in cluster_betas.columns:
            cluster_betas.loc[label, label] = 0

    non_zero_counts = (cluster_betas != 0).sum(axis=1)

    cluster_threshold = pd.Series(index=cluster_betas.index, dtype=float)
    cluster_threshold[non_zero_counts > 0] = threshold / non_zero_counts[non_zero_counts > 0]
    cluster_threshold[non_zero_counts == 0] = 0

    cluster_sizes = cluster_betas.gt(cluster_threshold, axis=0).sum(axis=1) + 1
    return cluster_sizes.where(cluster_sizes == 1, (cluster_sizes - 1) / cluster_sizes)
