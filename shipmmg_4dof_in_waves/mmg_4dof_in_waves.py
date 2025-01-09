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
    y_G: float
    z_G: float
    m_x: float
    m_y: float
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
        m=basic_params.m,
        x_G=basic_params.x_G,
        y_G=basic_params.y_G,
        z_G=basic_params.z_G,
        m_x=basic_params.m_x,
        m_y=basic_params.m_y,
        C_b=basic_params.C_b,
        GM=basic_params.GM,
        KM=basic_params.KM,
        D_p=basic_params.D_p,
        Tφ=basic_params.Tφ,
        A_R=basic_params.A_R,
        H_R=basic_params.H_R,
        x_R=basic_params.x_R,
        B_R=basic_params.B_R,
        I_zG=basic_params.I_zG,
        η=basic_params.η,
        J_z=basic_params.J_z,
        f_α=basic_params.f_α,
        ϵ=basic_params.ϵ,
        t_R=basic_params.t_R,
        a_H=basic_params.a_H,
        x_H=basic_params.x_H,
        γ_R_minus=basic_params.γ_R_minus,
        γ_R_plus=basic_params.γ_R_plus,
        l_R=basic_params.l_R,
        κ=basic_params.κ,
        t_P=basic_params.t_P,
        w_P0=basic_params.w_P0,
        x_P=basic_params.x_P,
        I_xx=basic_params.I_xx,
        J_xx=basic_params.J_xx,
        k_0=maneuvering_params.k_0,
        k_1=maneuvering_params.k_1,
        k_2=maneuvering_params.k_2,
        R_0_dash=maneuvering_params.R_0_dash,
        X_vv_dash=maneuvering_params.X_vv_dash,
        X_vr_dash=maneuvering_params.X_vr_dash,
        X_rr_dash=maneuvering_params.X_rr_dash,
        X_vvvv_dash=maneuvering_params.X_vvvv_dash,
        X_vφ_dash=maneuvering_params.X_vφ_dash,
        X_rφ_dash=maneuvering_params.X_rφ_dash,
        X_φφ_dash=maneuvering_params.X_φφ_dash,
        Y_v_dash=maneuvering_params.Y_v_dash,
        Y_r_dash=maneuvering_params.Y_r_dash,
        Y_vvv_dash=maneuvering_params.Y_vvv_dash,
        Y_vvr_dash=maneuvering_params.Y_vvr_dash,
        Y_vrr_dash=maneuvering_params.Y_vrr_dash,
        Y_rrr_dash=maneuvering_params.Y_rrr_dash,
        Y_φ_dash=maneuvering_params.Y_φ_dash,
        Y_vvφ_dash=maneuvering_params.Y_vvφ_dash,
        Y_vφφ_dash=maneuvering_params.Y_vφφ_dash,
        Y_rφφ_dash=maneuvering_params.Y_rφφ_dash,
        N_v_dash=maneuvering_params.N_v_dash,
        N_r_dash=maneuvering_params.N_r_dash,
        N_vvv_dash=maneuvering_params.N_vvv_dash,
        N_vvr_dash=maneuvering_params.N_vvr_dash,
        N_vrr_dash=maneuvering_params.N_vrr_dash,
        N_rrr_dash=maneuvering_params.N_rrr_dash,
        N_φ_dash=maneuvering_params.N_φ_dash,
        N_vvφ_dash=maneuvering_params.N_vvφ_dash,
        N_vφφ_dash=maneuvering_params.N_vφφ_dash,
        N_rφφ_dash=maneuvering_params.N_rφφ_dash,
        time_list=time_list,
        δ_list=δ_list,
        nps_list=nps_list,
        u0=u0,
        v0=v0,
        r0=r0,
        p0=p0,
        x0=x0,
        y0=y0,
        ψ0=ψ0,
        φ0=φ0,
        ρ=ρ,
        method=method,
        t_eval=t_eval,
        events=events,
        vectorized=vectorized,
        **options
    )
    

def simulate(
    L_pp: float,
    B: float,
    d: float,
    m: float,
    x_G: float,
    y_G: float,
    z_G: float,
    m_x: float,
    m_y: float,
    C_b: float,
    GM: float,
    KM: float,
    D_p: float,
    Tφ: float,
    A_R: float,
    H_R: float,
    x_R: float,
    B_R: float,
    I_zG: float,
    η: float,
    J_z: float,
    f_α: float,
    ϵ: float,
    t_R: float,
    a_H: float,
    x_H: float,
    γ_R_minus: float,
    γ_R_plus: float,
    l_R: float,
    κ: float,
    t_P: float,
    w_P0: float,
    x_P: float,
    I_xx: float,
    J_xx: float,
    k_0: float,
    k_1: float,
    k_2: float,
    R_0_dash: float,
    X_vv_dash: float,
    X_vr_dash: float,
    X_rr_dash: float,
    X_vvvv_dash: float,
    X_vφ_dash: float,
    X_rφ_dash: float,
    X_φφ_dash: float,
    Y_v_dash: float,
    Y_r_dash: float,
    Y_vvv_dash: float,
    Y_vvr_dash: float,
    Y_vrr_dash: float,
    Y_rrr_dash: float,
    Y_φ_dash: float,
    Y_vvφ_dash: float,
    Y_vφφ_dash: float,
    Y_rφφ_dash: float,
    N_v_dash: float,
    N_r_dash: float,
    N_vvv_dash: float,
    N_vvr_dash: float,
    N_vrr_dash: float,
    N_rrr_dash: float,
    N_φ_dash: float,
    N_vvφ_dash: float,
    N_vφφ_dash: float,
    N_rφφ_dash: float,
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
    
    spl_δ = interp1d(time_list, δ_list, "cubic", fill_value="extrapolate")
    spl_nps = interp1d(time_list, nps_list, "cubic", fill_value="extrapolate")
    
    def mmg_4dof_in_waves_eom_solve_ivp(t, X):
        
        u, v, r, p, x, y, ψ, φ, δ, nps = X
        
        v_m = v - x_G * r + z_G * p
        β = 0.0 if u == 0.0 else np.arctan2(-v_m / u)
        
        
