import dataclasses
from typing import List

import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.misc import derivative

@dataclasses.dataclass
class Mmg4DofInWavesBasicParams:
    
    L_pp: float
    B: float
    d: float
    m: float
    x_G: float
    C_b: float
    GM: float
    KM: float
    D_p: float
    Tφ: float
    A_R: float
    H_R: float
    x_R: float
    B_R: float
    I_zG: float
    η: float
    J_z: float
    f_α: float
    ϵ: float
    t_R: float
    a_H: float
    x_H: float
    γ_R_minus: float
    γ_R_plus: float
    l_R: float
    κ: float
    t_P: float
    w_P0: float
    x_P: float
    I_xx: float
    J_xx: float
    
    
@dataclasses.dataclass
class Mmg4DofInWavesManeuveringParams:
    
    k_0: float
    k_1: float
    k_2: float
    R_0_dash: float
    X_vv_dash: float
    X_vr_dash: float
    X_rr_dash: float
    X_vvvv_dash: float
    X_vφ_dash: float
    X_rφ_dash: float
    X_φφ_dash: float
    Y_v_dash: float
    Y_r_dash: float
    Y_vvv_dash: float
    Y_vvr_dash: float
    Y_vrr_dash: float
    Y_rrr_dash: float
    Y_φ_dash: float
    Y_vvφ_dash: float
    Y_vφφ_dash: float
    Y_rφφ_dash: float
    N_v_dash: float
    N_r_dash: float
    N_vvv_dash: float
    N_vvr_dash: float
    N_vrr_dash: float
    N_rrr_dash: float
    N_φ_dash: float
    N_vvφ_dash: float
    N_vφφ_dash: float
    N_rφφ_dash: float
    
    
def simulate_mmg_4dof_in_waves(
    basic_params: Mmg4DofInWavesBasicParams,
    maneuvering_params: Mmg4DofInWavesManeuveringParams,
    time_list: List[float],
    δ_list: List[float],
    nps_list: List[float],
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    p0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    ψ0: float = 0.0,
    φ0: float = 0.0,
    ρ: float = 1025.0,
    method: str = "RK45",
    t_eval=None,
    events=None,
    vectorized=False,
    **options
):
    return simulate(
        L_pp=basic_params.L_pp,
        B=basic_params.B,
        d=basic_params.d,
        
    )
