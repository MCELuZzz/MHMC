"""
Selection module for active learning.

Provides:
  - soap_features.py    : SOAP descriptor computation
  - kpca_reducer.py     : Kernel-PCA dimensionality reduction
  - stratified_fps.py   : Stratified Farthest Point Sampling

Reference: Section 2.3 of the manuscript.
"""

from .soap_features import compute_soap_descriptors
from .kpca_reducer import KPCAReducer
from .stratified_fps import stratified_fps_selection

__all__ = [
    "compute_soap_descriptors",
    "KPCAReducer",
    "stratified_fps_selection",
]