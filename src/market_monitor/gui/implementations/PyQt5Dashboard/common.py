import pandas as pd


def safe_concat(dfs, **kwargs):
    dfs = [
        df for df in dfs
        if df is not None
        and not df.empty
        and not df.isna().all(axis=None)
    ]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, **kwargs)
