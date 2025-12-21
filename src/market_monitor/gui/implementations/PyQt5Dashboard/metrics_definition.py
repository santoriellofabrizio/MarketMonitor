METRIC_DEFINITIONS = {
    "total_trades": {
        "label": "Total Trades",
        "compute": lambda df: len(df)
    },
    "own_trades": {
        "label": "Own Trades",
        "compute": lambda df: len(df[df["own_trade"] == True])
        if "own_trade" in df.columns else 0
    },
    "market_trades": {
        "label": "Market Trades",
        "compute": lambda df: len(df) -
        (len(df[df["own_trade"] == True]) if "own_trade" in df.columns else 0)
    },

    # --- Spread P&L ---
    "spread_pl_sum": {
        "label": "Total Spread P&L",
        "compute": lambda df: df["spread_pl"].sum()
        if "spread_pl" in df.columns else 0.0,
        "format": "${:,.2f}",
        "colorize": True
    },
    "spread_pl_mean": {
        "label": "Avg Spread P&L",
        "compute": lambda df: df["spread_pl"].mean()
        if "spread_pl" in df.columns else 0.0,
        "format": "${:,.2f}"
    },

    # --- CTV ---
    "ctv_sum": {
        "label": "Total CTV",
        "compute": lambda df: df["ctv"].sum()
        if "ctv" in df.columns else 0.0,
        "format": "${:,.2f}"
    },

    # =========================
    # Marginality (spread_pl)
    # =========================
    "pl_marginality": {
        "label": "PL Marginality",
        "compute": lambda df: (
            df["spread_pl"].sum() / df["ctv"].sum()
            if {"spread_pl", "ctv"}.issubset(df.columns) and df["ctv"].sum() != 0
            else 0.0
        ),
        "format": "{:.4%}"
    },
    "my_pl_marginality": {
        "label": "My PL Marginality",
        "compute": lambda df: (
            df.loc[df["own_trade"] == True, "spread_pl"].sum() /
            df.loc[df["own_trade"] == True, "ctv"].sum()
            if {"spread_pl", "ctv", "own_trade"}.issubset(df.columns)
            and df.loc[df["own_trade"] == True, "ctv"].sum() != 0
            else 0.0
        ),
        "format": "{:.4%}"
    },
    "average_pl_marginality": {
        "label": "Avg PL Marginality",
        "compute": lambda df: (
            (df["spread_pl"] / df["ctv"]).mean()
            if {"spread_pl", "ctv"}.issubset(df.columns)
            else 0.0
        ),
        "format": "{:.4%}"
    },
    "my_average_pl_marginality": {
        "label": "My Avg PL Marginality",
        "compute": lambda df: (
            (df.loc[df["own_trade"] == True, "spread_pl"] /
             df.loc[df["own_trade"] == True, "ctv"]).mean()
            if {"spread_pl", "ctv", "own_trade"}.issubset(df.columns)
            else 0.0
        ),
        "format": "{:.4%}"
    },

    # =========================
    # Marginality (lagged_pl)
    # =========================
    "lagged_pl_marginality": {
        "label": "Lagged PL Marginality",
        "compute": lambda df: (
            df["lagged_pl"].sum() / df["ctv"].sum()
            if {"lagged_pl", "ctv"}.issubset(df.columns) and df["ctv"].sum() != 0
            else 0.0
        ),
        "format": "{:.4%}"
    },
    "my_lagged_pl_marginality": {
        "label": "My Lagged PL Marginality",
        "compute": lambda df: (
            df.loc[df["own_trade"] == True, "lagged_pl"].sum() /
            df.loc[df["own_trade"] == True, "ctv"].sum()
            if {"lagged_pl", "ctv", "own_trade"}.issubset(df.columns)
            and df.loc[df["own_trade"] == True, "ctv"].sum() != 0
            else 0.0
        ),
        "format": "{:.4%}"
    },
    "average_lagged_pl_marginality": {
        "label": "Avg Lagged PL Marginality",
        "compute": lambda df: (
            (df["lagged_pl"] / df["ctv"]).mean()
            if {"lagged_pl", "ctv"}.issubset(df.columns)
            else 0.0
        ),
        "format": "{:.4%}"
    },
    "my_average_lagged_pl_marginality": {
        "label": "My Avg Lagged PL Marginality",
        "compute": lambda df: (
            (df.loc[df["own_trade"] == True, "lagged_pl"] /
             df.loc[df["own_trade"] == True, "ctv"]).mean()
            if {"lagged_pl", "ctv", "own_trade"}.issubset(df.columns)
            else 0.0
        ),
        "format": "{:.4%}"
    },

    # =========================
    # Marginality (model_pl)
    # =========================
    "model_pl_marginality": {
        "label": "Model PL Marginality",
        "compute": lambda df: (
            df["model_pl"].sum() / df["ctv"].sum()
            if {"model_pl", "ctv"}.issubset(df.columns) and df["ctv"].sum() != 0
            else 0.0
        ),
        "format": "{:.4%}"
    },
    "my_model_pl_marginality": {
        "label": "My Model PL Marginality",
        "compute": lambda df: (
            df.loc[df["own_trade"] == True, "model_pl"].sum() /
            df.loc[df["own_trade"] == True, "ctv"].sum()
            if {"model_pl", "ctv", "own_trade"}.issubset(df.columns)
            and df.loc[df["own_trade"] == True, "ctv"].sum() != 0
            else 0.0
        ),
        "format": "{:.4%}"
    },
    "average_model_pl_marginality": {
        "label": "Avg Model PL Marginality",
        "compute": lambda df: (
            (df["model_pl"] / df["ctv"]).mean()
            if {"model_pl", "ctv"}.issubset(df.columns)
            else 0.0
        ),
        "format": "{:.4%}"
    },
    "my_average_model_pl_marginality": {
        "label": "My Avg Model PL Marginality",
        "compute": lambda df: (
            (df.loc[df["own_trade"] == True, "model_pl"] /
             df.loc[df["own_trade"] == True, "ctv"]).mean()
            if {"model_pl", "ctv", "own_trade"}.issubset(df.columns)
            else 0.0
        ),
        "format": "{:.4%}"
    }
}
