"""Frozen PCA follow-up to the completed Tulu3/PrOntoQA TwoNN experiment."""
import os

# Applied before NumPy/SciPy imports in the command-line entry point.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "4"
