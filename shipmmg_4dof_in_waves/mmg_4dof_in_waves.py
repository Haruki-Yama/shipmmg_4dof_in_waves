import dataclasses
from typing import List

import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.misc import derivative

from wave_induced_steady_force_coefficients import si_C_XW_index, si_C_YW, si_C_NW, U_values

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
    γ_R_plus: float
    γ_R_minus: float
    l_P_dash: float
    l_R_dash: float
    z_P_dash: float
    z_R_dash: float
    κ: float
    t_P: float
    w_P0: float
    I_xx: float
    J_xx: float
    J_zz: float
    a: float
    b: float
    α_z: float
    z_R: float
    h_a: float
    z_W: float
    
    
    
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
    Y_rrφ_dash: float
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
    N_rrφ_dash: float
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
    χ0: float = 0.0,
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
        γ_R_plus=basic_params.γ_R_plus,
        γ_R_minus=basic_params.γ_R_minus,
        l_P_dash=basic_params.l_P_dash,
        l_R_dash=basic_params.l_R_dash,
        z_P_dash=basic_params.z_P_dash,
        z_R_dash=basic_params.z_R_dash,
        κ=basic_params.κ,
        t_P=basic_params.t_P,
        w_P0=basic_params.w_P0,
        I_xx=basic_params.I_xx,
        J_xx=basic_params.J_xx,
        J_zz=basic_params.J_zz,
        a=basic_params.a,
        b=basic_params.b,
        α_z=basic_params.α_z,
        z_R=basic_params.z_R,
        h_a=basic_params.h_a,
        z_W=basic_params.z_W,
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
        Y_rrφ_dash=maneuvering_params.Y_rrφ_dash,
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
        N_rrφ_dash=maneuvering_params.N_rrφ_dash,
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
    g: float,
    m: float,
    x_G: float,
    z_G: float,
    z_H: float,
    m_x: float,
    m_y: float,
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
    γ_R_plus: float,
    γ_R_minus: float,
    l_P_dash: float,
    l_R_dash: float,
    z_P_dash: float,
    z_R_dash: float,
    κ: float,
    t_P: float,
    w_P0: float,
    I_xx: float,
    J_xx: float,
    J_zz: float,
    a: float,
    b: float,
    α_z: float,
    z_R: float,
    h_a: float,
    z_W: float,
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
    Y_rrφ_dash: float,
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
    N_rrφ_dash: float,
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
    χ0: float = 0.0,
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
        
        U = np.sqrt(u**2 + v_m** 2)
        
        β = 0.0 if u == 0.0 else np.arctan2(-v_m, u)
        v_dash = 0.0 if U == 0.0 else v / U
        r_dash = 0.0 if U == 0.0 else r * L_pp / U
        p_dash = 0.0 if U == 0.0 else p * B / U
        
        β_P = β - l_P_dash * r_dash + z_P_dash * p_dash
        
        w_P = w_P0 * (1 - (1 - np.cos(β_P)**2) * (1 - np.abs(β_P)))
        
        J = 0.0 if nps == 0.0 else u * (1 - w_P) / (nps * D_p)
        K_T = k_0 + k_1 * J + k_2 * J**2
        β_R = β - l_R_dash * r_dash + z_R_dash * p_dash
        γ_R = γ_R_minus if β_R < 0.0 else γ_R_plus
        v_R = U * γ_R * β_R
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
                -R_0_dash
                + X_vv_dash * v_dash**2
                + X_vr_dash * v_dash * r_dash
                + X_rr_dash * r_dash**2
                + X_vvvv_dash * v_dash**4
                + X_vφ_dash * v_dash * φ
                + X_rφ_dash * r_dash * φ
                + X_φφ_dash * φ**2
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
                Y_v_dash * v_dash
                + Y_r_dash * r_dash
                + Y_vvv_dash * v_dash**3
                + Y_vvr_dash * v_dash**2 * r_dash
                + Y_vrr_dash * v_dash * r_dash**2
                + Y_rrr_dash * r_dash**3
                + Y_φ_dash * φ
                + Y_vvφ_dash * v_dash**2 * φ
                + Y_vφφ_dash * v_dash * φ**2
                + Y_rrφ_dash * r_dash**2 * φ
                + Y_rφφ_dash * r_dash * φ**2
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
                N_v_dash * v_dash
                + N_r_dash * r_dash
                + N_vvv_dash * v_dash**3
                + N_vvr_dash * v_dash**2 * r_dash
                + N_vrr_dash * v_dash * r_dash**2
                + N_rrr_dash * r_dash**3
                + N_φ_dash * φ
                + N_vvφ_dash * v_dash**2 * φ
                + N_vφφ_dash * v_dash * φ**2
                + N_rrφ_dash * r_dash**2 * φ
                + N_rφφ_dash * r_dash * φ**2
            )
        )
        
        
        C_XW_index = min(range(len(U_values)), key=lambda i: abs(U - U_values[i]))
        χ = χ0 + (ψ * 180 / np.pi)
        si_C_XW = si_C_XW_index[C_XW_index]
        
        C_XW = si_C_XW(χ)
        C_YW = si_C_YW(χ)
        C_NW = si_C_NW(χ)
        
        X_W = ρ * g * (h_a**2) * L_pp * C_XW
        Y_W = ρ * g * (h_a**2) * L_pp * C_YW
        N_W = ρ * g * (h_a**2) * (L_pp**2) * C_NW
        
        
        K_p = -2 / np.pi * a * np.sqrt(m * g * GM * (I_xx + J_xx))
        K_pp = -0.75 * b * (180 / np.pi) * (I_xx + J_xx)
        
        X = X_H + X_R + X_P + X_W
        Y = Y_H + Y_R + Y_W
        N = N_H + N_R + N_W
        K = -Y_H * z_H - Y_R * z_R - m * g * GM * φ + K_p * p + K_pp * p * np.abs(p) + z_W * Y_W
        
        A_ = (m + m_y) - (m_y * α_z + m * z_G)**2 / (I_xx + J_xx + m * z_G**2)
        B_ = x_G * m - (m_y * α_z + m * z_G) * m * z_G * x_G / (I_xx + J_xx + m * z_G**2)
        C_ = Y - (m + m_x) * u * r + (m_y * α_z + m * z_G) * (K + m * z_G * u * r) / (I_xx + J_xx + m * z_G**2)
        D_ = m * x_G * (1 - z_G * (m_y * α_z + m * z_G) / (I_xx + J_xx + m * z_G**2))
        E_ = (I_zz + J_zz + m * x_G**2) - m * z_G**2 * x_G / (I_xx + J_xx + m * z_G**2)
        F_ = N + m * x_G * (z_G * (K + m * z_G * u * r) / (I_xx + J_xx + m * z_G**2) - u * r)
        
        d_u = (X + (m + m_y) * v * r + m * x_G * (r**2) - m * z_G * r * p) / (m + m_x)
        d_v = (C_ * E_ - B_ * F_) / (A_ * E_ - B_ * D_)
        d_r = (C_ * D_ - A_ * F_) / (B_ * D_ - A_ * E_)
        d_p = (K + (m_y * α_z + m * z_G) * v_m + m * z_G * (x_G * d_r + u * r)) / (I_xx + J_xx + m * z_G**2)
        
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
        
        
        
