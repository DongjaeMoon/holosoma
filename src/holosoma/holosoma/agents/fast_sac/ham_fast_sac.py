"""HAM banks use the same shared-router SAC update as biHAM."""
from .bilateral_fast_sac import BilateralFastSAC
from holosoma.agents.modules.bilateral.ham_adapter import build_ham_sac


class HAMFastSAC(BilateralFastSAC):
    network_builder = staticmethod(build_ham_sac)
