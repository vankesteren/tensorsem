# -*- coding: utf-8 -*-
# Options interface for tensorsem module initialization
from typing import NamedTuple, List, Dict
import pickle


class SemOptions(NamedTuple):
    delta_start: List[float]  # The parameter vector starting values
    delta_free: List[float]  # Which elements of the vector are freely estimated
    delta_value: List[float]  # What values do the elements get if they are constrained
    delta_sizes: List[int]  # How many elements of delta for  psi, b_0, lam and tht mats
    psi_shape: List[int]  # The size of Psi [r, c]
    b_0_shape: List[int]  # The size of B_0 [r, c]
    lam_shape: List[int]  # The size of Lambda [r, c]
    tht_shape: List[int]  # The size of Theta [r, c]
    ov_names: List[str]  # The names of the observed variables in the expected order

    @staticmethod
    def from_dict(x: Dict):
        return SemOptions(
            x["delta_start"],
            x["delta_free"],
            x["delta_value"],
            x["delta_sizes"],
            x["psi_shape"],
            x["b_0_shape"],
            x["lam_shape"],
            x["tht_shape"],
            x["ov_names"]
        )

    @staticmethod
    def from_file(x: str):
        return SemOptions.from_dict(pickle.load(open(x, "rb")))
