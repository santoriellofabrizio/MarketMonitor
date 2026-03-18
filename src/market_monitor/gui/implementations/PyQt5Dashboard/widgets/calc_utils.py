"""
Shared helpers for calculated-field evaluation in Pivot, GroupBy and Dashboard widgets.
"""
import numpy as np
import pandas as pd


def _str_op(x, fn_str, fn_scalar):
    """Apply a string operation to a Series or scalar."""
    if isinstance(x, pd.Series):
        return fn_str(x.astype(str).str)
    return fn_scalar(str(x))


def _to_str(x, fmt=None):
    """Convert a Series to string, formatting timestamps if needed."""
    if isinstance(x, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(x):
            return x.dt.strftime(fmt or '%Y-%m-%d %H:%M:%S')
        return x.astype(str)
    if hasattr(x, 'strftime'):          # scalar datetime / Timestamp
        return x.strftime(fmt or '%Y-%m-%d %H:%M:%S')
    return str(x)


def build_calc_namespace(df: pd.DataFrame) -> dict:
    """
    Build the eval() namespace for a calculated-field expression.
    Exposes all DataFrame columns plus numeric, string and date helpers.

    Numeric:  abs, round, sqrt, log, exp, np.where, np
    String:   upper, lower, strip, len, str_col,
              replace(x, old, new), contains(x, pat),
              startswith(x, pat), endswith(x, pat),
              substr(x, start[, end]), concat(sep, col1, col2, ...)
    Date:     format_date(x[, fmt])   — fmt defaults to '%Y-%m-%d'
              format_ts(x[, fmt])     — fmt defaults to '%Y-%m-%d %H:%M:%S'
              date_part(x, part)      — part: 'year','month','day','hour','minute','second'
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

        # str_col: converts to string, auto-formats timestamps
        'str_col': lambda x: _to_str(x),

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

        # ── Date / Timestamp ─────────────────────────────────────────
        # format_date(ts_col)              → "2024-01-15"
        # format_date(ts_col, '%d/%m/%Y')  → "15/01/2024"
        'format_date': lambda x, fmt='%Y-%m-%d': _to_str(x, fmt),

        # format_ts(ts_col)                → "2024-01-15 10:30:00"
        # format_ts(ts_col, '%H:%M')       → "10:30"
        'format_ts': lambda x, fmt='%Y-%m-%d %H:%M:%S': _to_str(x, fmt),

        # date_part(ts_col, 'year'|'month'|'day'|'hour'|'minute'|'second')
        'date_part': lambda x, part: (
            getattr(x.dt, part) if isinstance(x, pd.Series) and hasattr(x.dt, part)
            else getattr(x, part) if hasattr(x, part)
            else None
        ),
    })
    return namespace


# Human-readable hint for UI dialogs
CALC_OPS_HINT = (
    "Numeric: abs · round · sqrt · log · exp · np.where(cond, a, b)  |  "
    "String: upper(x) · lower(x) · strip(x) · len(x) · str_col(x) · "
    "replace(x, old, new) · contains(x, pat) · startswith/endswith(x, pat) · "
    "substr(x, start[, end]) · concat(sep, a, b, …)  |  "
    "Date: format_date(x[, fmt]) · format_ts(x[, fmt]) · date_part(x, 'year'|'month'|'day'|…)"
)
