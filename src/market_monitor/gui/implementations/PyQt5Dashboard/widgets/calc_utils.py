"""
Shared helpers for calculated-field evaluation in Pivot and GroupBy widgets.
"""
import numpy as np
import pandas as pd


def _str_op(x, fn_str, fn_scalar):
    """Apply a string operation to a Series or scalar."""
    if isinstance(x, pd.Series):
        return fn_str(x.astype(str).str)
    return fn_scalar(str(x))


def build_calc_namespace(df: pd.DataFrame) -> dict:
    """
    Build the eval() namespace for a calculated-field expression.
    Exposes all DataFrame columns plus numeric and string helper functions.

    Numeric:  abs, round, sqrt, log, exp, np.where, np
    String:   upper, lower, strip, len, str_col,
              replace(x, old, new), contains(x, pat),
              startswith(x, pat), endswith(x, pat),
              substr(x, start[, end]), concat(sep, col1, col2, ...)
    """
    namespace = {str(col): df[col] for col in df.columns}

    namespace.update({
        # ── Numeric ──────────────────────────────────────────────────
        'abs':   np.abs,
        'round': np.round,
        'sqrt':  np.sqrt,
        'log':   np.log,
        'exp':   np.exp,
        'np':    np,

        # ── String ───────────────────────────────────────────────────
        'upper': lambda x: _str_op(x,
                                   lambda s: s.upper(),
                                   lambda s: s.upper()),
        'lower': lambda x: _str_op(x,
                                   lambda s: s.lower(),
                                   lambda s: s.lower()),
        'strip': lambda x: _str_op(x,
                                   lambda s: s.strip(),
                                   lambda s: s.strip()),
        'len':   lambda x: (x.str.len() if isinstance(x, pd.Series)
                            else len(str(x))),
        'str_col': lambda x: x.astype(str) if isinstance(x, pd.Series)
                             else str(x),

        'replace': lambda x, old, new: (
            x.astype(str).str.replace(old, new, regex=False)
            if isinstance(x, pd.Series)
            else str(x).replace(old, new)
        ),
        'contains': lambda x, pat: (
            x.astype(str).str.contains(pat, na=False, regex=False)
            if isinstance(x, pd.Series)
            else pat in str(x)
        ),
        'startswith': lambda x, pat: (
            x.astype(str).str.startswith(pat)
            if isinstance(x, pd.Series)
            else str(x).startswith(pat)
        ),
        'endswith': lambda x, pat: (
            x.astype(str).str.endswith(pat)
            if isinstance(x, pd.Series)
            else str(x).endswith(pat)
        ),
        'substr': lambda x, start, end=None: (
            x.astype(str).str[start:end]
            if isinstance(x, pd.Series)
            else str(x)[start:end]
        ),

        # concat(sep, col1, col2, ...) → "val1<sep>val2<sep>..."
        'concat': lambda sep, *cols: (
            cols[0].astype(str).str.cat(
                [c.astype(str) for c in cols[1:]], sep=sep
            ) if cols else pd.Series([], dtype=str)
        ),
    })
    return namespace


# Human-readable hint for UI dialogs
CALC_OPS_HINT = (
    "Numeric: abs · round · sqrt · log · exp · np.where(cond, a, b)  |  "
    "String: upper(x) · lower(x) · strip(x) · len(x) · str_col(x) · "
    "replace(x, old, new) · contains(x, pat) · startswith(x, pat) · "
    "endswith(x, pat) · substr(x, start[, end]) · concat(sep, a, b, …)"
)
