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
    g: float
    m: float
    x_G: float
    z_G: float
    z_H: float
    m_x: float
    m_y: float
    m_x_dash: float
    m_y_dash: float
    GM: float
    D_p: float
    A_R: float
    x_R: float
    I_zz: float
    η: float
    f_α: float
    ϵ: float
    t_R: float
    a_H: float
    x_H: float
    γ_R0: float
    l_R_dash: float
    z_R: float
    κ: float
    t_P: float
    w_P0: float
    x_P: float
    I_xx: float
    J_xx: float
    J_zz: float
    c_x0: float
    c_xββ: float
    c_xrr: float
    c_nβ: float
    c_yβ: float
    c_nr: float
    c_yr: float
    c_γ: float
    a: float
    
    
    
@dataclasses.dataclass
class Mmg4DofInWavesManeuveringParams:
    
    k_0: float
    k_1: float
    k_2: float
    X_0_dash: float
    X_vv_dash: float
    X_vr_dash: float
    X_rr_dash: float
    X_vvvv_dash: float
    X_rφ_dash: float
    Y_v_dash: float
    Y_r_dash: float
    Y_vvv_dash: float
    Y_vvr_dash: float
    Y_vrr_dash: float
    Y_rrr_dash: float
    Y_φ_dash: float
    Y_vvφ_dash: float
    Y_vrφ_dash: float
    Y_rrφ_dash: float
    N_v_dash: float
    N_r_dash: float
    N_vvv_dash: float
    N_vvr_dash: float
    N_vrr_dash: float
    N_rrr_dash: float
    N_φ_dash: float
    N_vvφ_dash: float
    N_vrφ_dash: float
    N_rrφ_dash: float
    K_φ_dash: float
    K_v_dash: float
    K_r_dash: float
    K_vvφ_dash: float
    K_vrφ_dash: float
    K_rrφ_dash: float
    K_vvv_dash: float
    K_vvr_dash: float
    K_vrr_dash: float
    K_rrr_dash: float
    
    
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
        g=basic_params.g,
        m=basic_params.m,
        x_G=basic_params.x_G,
        z_G=basic_params.z_G,
        z_H=basic_params.z_H,
        m_x=basic_params.m_x,
        m_y=basic_params.m_y,
        m_x_dash=basic_params.m_x_dash,
        m_y_dash=basic_params.m_y_dash,
        GM=basic_params.GM,
        D_p=basic_params.D_p,
        A_R=basic_params.A_R,
        x_R=basic_params.x_R,
        I_zz=basic_params.I_zz,
        η=basic_params.η,
        f_α=basic_params.f_α,
        ϵ=basic_params.ϵ,
        t_R=basic_params.t_R,
        a_H=basic_params.a_H,
        x_H=basic_params.x_H,
        γ_R0=basic_params.γ_R0,
        l_R_dash=basic_params.l_R_dash,
        z_R=basic_params.z_R,
        κ=basic_params.κ,
        t_P=basic_params.t_P,
        w_P0=basic_params.w_P0,
        x_P=basic_params.x_P,
        I_xx=basic_params.I_xx,
        J_xx=basic_params.J_xx,
        J_zz=basic_params.J_zz,
        c_x0=basic_params.c_x0,
        c_xββ=basic_params.c_xββ,
        c_xrr=basic_params.c_xrr,
        c_nβ=basic_params.c_nβ,
        c_yβ=basic_params.c_yβ,
        c_nr=basic_params.c_nr,
        c_yr=basic_params.c_yr,
        c_γ=basic_params.c_γ,
        a=basic_params.a,
        k_0=maneuvering_params.k_0,
        k_1=maneuvering_params.k_1,
        k_2=maneuvering_params.k_2,
        X_0_dash=maneuvering_params.X_0_dash,
        X_vv_dash=maneuvering_params.X_vv_dash,
        X_vr_dash=maneuvering_params.X_vr_dash,
        X_rr_dash=maneuvering_params.X_rr_dash,
        X_vvvv_dash=maneuvering_params.X_vvvv_dash,
        X_rφ_dash=maneuvering_params.X_rφ_dash,
        Y_v_dash=maneuvering_params.Y_v_dash,
        Y_r_dash=maneuvering_params.Y_r_dash,
        Y_vvv_dash=maneuvering_params.Y_vvv_dash,
        Y_vvr_dash=maneuvering_params.Y_vvr_dash,
        Y_vrr_dash=maneuvering_params.Y_vrr_dash,
        Y_rrr_dash=maneuvering_params.Y_rrr_dash,
        Y_φ_dash=maneuvering_params.Y_φ_dash,
        Y_vvφ_dash=maneuvering_params.Y_vvφ_dash,
        Y_vrφ_dash=maneuvering_params.Y_vrφ_dash,
        Y_rrφ_dash=maneuvering_params.Y_rrφ_dash,
        N_v_dash=maneuvering_params.N_v_dash,
        N_r_dash=maneuvering_params.N_r_dash,
        N_vvv_dash=maneuvering_params.N_vvv_dash,
        N_vvr_dash=maneuvering_params.N_vvr_dash,
        N_vrr_dash=maneuvering_params.N_vrr_dash,
        N_rrr_dash=maneuvering_params.N_rrr_dash,
        N_φ_dash=maneuvering_params.N_φ_dash,
        N_vvφ_dash=maneuvering_params.N_vvφ_dash,
        N_vrφ_dash=maneuvering_params.N_vrφ_dash,
        N_rrφ_dash=maneuvering_params.N_rrφ_dash,
        K_φ_dash=maneuvering_params.K_φ_dash,
        K_v_dash=maneuvering_params.K_v_dash,
        K_r_dash=maneuvering_params.K_r_dash,
        K_vvφ_dash=maneuvering_params.K_vvφ_dash,
        K_vrφ_dash=maneuvering_params.K_vrφ_dash,
        K_rrφ_dash=maneuvering_params.K_rrφ_dash,
        K_vvv_dash=maneuvering_params.K_vvv_dash,
        K_vvr_dash=maneuvering_params.K_vvr_dash,
        K_vrr_dash=maneuvering_params.K_vrr_dash,
        K_rrr_dash=maneuvering_params.K_rrr_dash,
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
    g: float,
    m: float,
    x_G: float,
    z_G: float,
    z_H: float,
    m_x: float,
    m_y: float,
    m_x_dash: float,
    m_y_dash: float,
    GM: float,
    D_p: float,
    A_R: float,
    x_R: float,
    I_zz: float,
    η: float,
    f_α: float,
    ϵ: float,
    t_R: float,
    a_H: float,
    x_H: float,
    γ_R0: float,
    l_R_dash: float,
    z_R: float,
    κ: float,
    t_P: float,
    w_P0: float,
    x_P: float,
    I_xx: float,
    J_xx: float,
    J_zz: float,
    c_x0: float,
    c_xββ: float,
    c_xrr: float,
    c_nβ: float,
    c_yβ: float,
    c_nr: float,
    c_yr: float,
    c_γ: float,
    a: float,
    k_0: float,
    k_1: float,
    k_2: float,
    X_0_dash: float,
    X_vv_dash: float,
    X_vr_dash: float,
    X_rr_dash: float,
    X_vvvv_dash: float,
    X_rφ_dash: float,
    Y_v_dash: float,
    Y_r_dash: float,
    Y_vvv_dash: float,
    Y_vvr_dash: float,
    Y_vrr_dash: float,
    Y_rrr_dash: float,
    Y_φ_dash: float,
    Y_vvφ_dash: float,
    Y_vrφ_dash: float,
    Y_rrφ_dash: float,
    N_v_dash: float,
    N_r_dash: float,
    N_vvv_dash: float,
    N_vvr_dash: float,
    N_vrr_dash: float,
    N_rrr_dash: float,
    N_φ_dash: float,
    N_vvφ_dash: float,
    N_vrφ_dash: float,
    N_rrφ_dash: float,
    K_φ_dash: float,
    K_v_dash: float,
    K_r_dash: float,
    K_vvφ_dash: float,
    K_vrφ_dash: float,
    K_rrφ_dash: float,
    K_vvv_dash: float,
    K_vvr_dash: float,
    K_vrr_dash: float,
    K_rrr_dash: float,
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
        
        U = np.sqrt(u**2 + (v - r * x_G) ** 2)
        
        β = 0.0 if U == 0.0 else np.arcsin(-(v - r * x_G) / U)
        v_dash = 0.0 if U == 0.0 else v / U
        r_dash = 0.0 if U == 0.0 else r * L_pp / U
        
        w_P = w_P0 * np.exp(-4.0 * (β - x_P * r_dash) ** 2)
        
        J = 0.0 if nps == 0.0 else u * (1 - w_P) / (nps * D_p)
        K_T = k_0 + k_1 * J + k_2 * J**2
        β_R = β - l_R_dash * r_dash
        γ_R = γ_R0 * (1 + c_γ * np.abs(φ))
        v_R = 0.0 if U == 0.0 else -U * γ_R * (β - l_R_dash * r_dash + (p * (z_R - z_G) / U))
        u_R = (
            np.sqrt(η * (κ * ϵ * 8.0 * k_0 * nps**2 * D_p**4 / np.pi) ** 2)
            if J == 0.0
            else u
            * (1 - w_P)
            * ϵ
            * np.sqrt(
                η * (1.0 + κ * (np.sqrt(1.0 + 8.0 * K_T / (np.pi * J**2)) - 1)) ** 2
                + (1 - η)
            )
        )
        U_R = np.sqrt(u_R**2 + v_R**2)
        α_R = δ - np.arctan2(v_R, u_R)
        F_N = 0.5 * A_R * ρ * f_α * (U_R**2) * np.sin(α_R)
        
        
        X_P = (1 - t_P) * ρ * nps**2 * D_p**4 * K_T
        X_R = -(1 - t_R) * F_N * np.sin(δ) * np.cos(φ)
        X_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U**2)
            * (
                X_0_dash * (1 + c_x0 * np.abs(φ))
                + X_rφ_dash * r_dash * φ
                + X_vv_dash * (1 + c_xββ * np.abs(φ)) * (v_dash**2)
                + (X_vr_dash - m_y_dash) * v_dash * r_dash
                + X_rr_dash * (1 + c_xrr * np.abs(φ)) * (r_dash**2)
                + X_vvvv_dash * (v_dash**4)
            )
        )
        
        Y_R = -(1 + a_H) * F_N * np.cos(δ) * np.cos(φ)
        Y_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U**2)
            * (
                Y_φ_dash * φ
                + Y_v_dash * (1 + c_yβ * np.abs(φ)) * v_dash 
                + (Y_r_dash - m_x_dash) * (1 + c_yr * np.abs(φ)) * r_dash
                + Y_vvφ_dash * (v_dash**2) * φ
                + Y_vrφ_dash * v_dash * r_dash * φ
                + Y_rrφ_dash * (r_dash**2) * φ
                + Y_vvv_dash * (v_dash**3)
                + Y_vvr_dash * (v_dash**2) * r_dash
                + Y_vrr_dash * v_dash * (r_dash**2)
                + Y_rrr_dash * (r_dash**3)
            )
        )
        
        N_R = -(x_R + a_H * x_H) * F_N * np.cos(δ) * np.cos(φ)
        N_H = (
            0.5
            * ρ
            * (L_pp**2)
            * d
            * (U**2)
            * (
                N_φ_dash * φ
                + N_v_dash * (1 + c_nβ * np.abs(φ)) * v_dash
                + N_r_dash * (1 + c_nr * np.abs(φ)) * r_dash
                + N_vvφ_dash * (v_dash**2) * φ
                + N_vrφ_dash * v_dash * r_dash * φ
                + N_rrφ_dash * (r_dash**2) * φ
                + N_vvv_dash * (v_dash**3)
                + N_vvr_dash * (v_dash**2) * r_dash
                + N_vrr_dash * v_dash * (r_dash**2)
                + N_rrr_dash * (r_dash**3)
            )
        )
        
        K_H = (
            -0.5
            * ρ
            * L_pp
            * (d**2)
            * (U**2)
            * (
                K_φ_dash * φ
                + K_v_dash * v_dash
                + K_r_dash * r_dash
                + K_vvφ_dash * (v_dash**2) * φ
                + K_vrφ_dash * v_dash * r_dash * φ
                + K_rrφ_dash * (r_dash**2) * φ
                + K_vvv_dash * (v_dash**3)
                + K_vvr_dash * (v_dash**2) * r_dash
                + K_vrr_dash * v_dash * (r_dash**2)
                + K_rrr_dash * (r_dash**3)
            )
        )
        
        B_44 = 2 * a / np.pi * np.sqrt(g * m * GM * (I_xx + J_xx))
        C_44 = g * m * GM
        
        d_u = ((X_H + X_R + X_P) + (m + m_y) * v * r) / (m + m_x)
        d_v = ((Y_H + Y_R) - (m + m_x) * u * r) / (m + m_y)
        d_r = (N_H + N_R - x_G * (Y_H + Y_R)) / (I_zz + J_zz)
        d_p = -(K_H + z_G * Y_H + B_44 * p + C_44 * φ - (z_R - z_G) * Y_R + (z_H - z_G) * (m_y * d_v + m_x * u * r))
        # d_p = (K_H + z_G * Y_H - B_44 * p - C_44 * φ - (z_R - z_G) * Y_R + (z_H - z_G) * (m_y * d_v + m_x * u * r))
        
        d_x = u * np.cos(ψ) - v * np.sin(ψ)
        d_y = u * np.sin(ψ) + v * np.cos(ψ)
        d_ψ = r
        d_φ = p
        d_δ = derivative(spl_δ, t)
        d_nps = derivative(spl_nps, t)
        
        return [d_u, d_v, d_r, d_p, d_x, d_y, d_ψ, d_φ, d_δ, d_nps]
    
    sol = solve_ivp(
        mmg_4dof_in_waves_eom_solve_ivp,
        [time_list[0], time_list[-1]],
        [u0, v0, r0, p0, x0, y0, ψ0, φ0, δ_list[0], nps_list[0]],
        dense_output=True,
        method=method,
        t_eval=t_eval,
        events=events,
        vectorized=vectorized,
        **options
    )
    return sol
        
        
        
