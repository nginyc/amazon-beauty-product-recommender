"""Compatibility shims that MUST be imported before recbole.

RecBole 1.2 was written against NumPy < 2.0 and SciPy < 1.12. The project runs
on newer versions, so we patch the handful of removed aliases / methods exactly
as the training notebooks do. Import this module first, before any recbole import.
"""
import numpy as np

# NumPy 2.0 removed these aliases that RecBole 1.2 still references.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

# SciPy 1.12+ removed dok_matrix._update, which LightGCN's adjacency build uses.
import scipy.sparse as sp  # noqa: E402

if not hasattr(sp.dok_matrix, "_update"):
    sp.dok_matrix._update = sp.dok_matrix.update
