import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import csv
import pandas as pd
# Add envelope of the transmission curve
from scipy.signal import hilbert
import time 
import os

from scipy.ndimage import uniform_filter1d
#from ipywidgets import interact


from scipy.signal import savgol_filter


# Constants + Parameters
c = 3e8                      
lambda0 = 1063.999e-9
nu_detune = -0          
nu0 = c / lambda0 + nu_detune          
k0 = 2 * np.pi / lambda0     
dnu_arr = np.linspace(-0.001e9, 0.001e9, 1000) # alter upper and lower bounds according to dimensions
nu = nu0 + dnu_arr
k = 2 * np.pi * nu / c 

def BW_tunability(
    L1,
    L2,
    R1,
    R2,
    R3,
    lamb=1064e-9,
    tunability=0.55,
    heating_capacity=10,
    n=1.45,
    dn_dT=1.0e-5,
):
    c = 3e8
    prefactor = c / (8 * np.pi * L2)

    # Effective thermal expansion coefficient inferred from your tunability input
    alpha_eff = tunability * 1e-6

    # Thermal expansion of the etalon thickness
    dL1_dT = alpha_eff * L1

    # Etalon round-trip phase
    phi = 4 * np.pi * n * L1 / lamb

    # Effective transmissivity
    num = (1 - R1) * (1 - R2)
    denom = 1 - 2 * np.sqrt(R1 * R2) * np.cos(phi) + R1 * R2
    Teff = num / denom

    # Baseline cavity pole
    baseline_bw = prefactor * (Teff + (1 - R3))

    # dTeff/dphi
    dTeff_dphi = -num * (2 * np.sqrt(R1 * R2) * np.sin(phi)) / (denom ** 2)

    # Full thermal phase derivative including both expansion and thermo-optic effect
    dphi_dT = (4 * np.pi / lamb) * (n * dL1_dT + L1 * dn_dT)

    # Thermal tunability of cavity pole
    bw_expansion = prefactor * dTeff_dphi * dphi_dT

    # Keep this too if you still want length sensitivity separately
    dgamma_dL1 = prefactor * dTeff_dphi * (4 * np.pi * n / lamb)

    return baseline_bw, bw_expansion, dL1_dT, dgamma_dL1

def T_3_mirror(k, L1, L2, R1, R2, R3):
    r1, r2, r3 = np.sqrt(R1), np.sqrt(R2), np.sqrt(R3)
    t1, t2, t3 = np.sqrt(1 - R1), np.sqrt(1 - R2), np.sqrt(1 - R3)

    e_total = np.exp(2j * k * (L1 + L2))
    e_L2 = np.exp(2j * k * L2)
    e_L1 = np.exp(2j * k * L1)

    num = -t1 * t2 * t3 * np.exp(1j * k * (L1 + L2))
    denom = (e_total - r1 * r2 * e_L2 - r2 * r2 * e_L1 + r1 * r3 * (r2**2 + t2**2))

    return np.abs(num / denom)**2
    
# 2 miror transmission function
def T_effective_2_mirror(k, L2, R1, R2, R3):
    """Simplified 2-mirror transmission model using effective reflectivity."""
    # Compute Teff
    t1, t2 = np.sqrt(1 - R1), np.sqrt(1 - R2)
    r1, r2 = np.sqrt(R1), np.sqrt(R2)
    Teff = (t1 * t2)**2 / np.abs(1 - r1 * r2)**2
    Reff = 1 - Teff

    # Transmission function
    num = np.sqrt(Teff * (1 - R3))
    denom = 1 - np.sqrt(Reff * R3) * np.exp(2j * k * L2)
    return np.abs(num / denom)**2

def plot_bw_expansion(L1_best, L2_best, R1_best, R2_best, R3_best, Teff_target_ppm, output_dir):
    baseline_bw, bw_expansion, dL1_dT, dgamma_dL1 = BW_tunability(L1_best, L2_best, R1_best, R2_best, R3_best)
    print(f"L1 : {L1_best} m, Pole: {baseline_bw} Hz, Bandwidth Tunability: {bw_expansion} Hz/°C")
    transmissions = []
    plt.figure(figsize=(15, 10))
    L1_arr = [L1_best+ dL1_dT * T for T in np.arange(0, 10)]
    for i,L1 in enumerate(L1_arr):
        T3_test = T_3_mirror(k, L1, L2_best, R1_best, R2_best, R3_best)
        transmissions.append(T3_test)
        # if i == 0:
        #     color = 'blue'
        # else:
        #     color = 'grey'
        

        FSR_L1_Hz = c / (2 * L1)
        FSR_L1_MHz = FSR_L1_Hz * 1e-6
        plt.semilogy(dnu_arr * 1e-6,T3_test,label=(f"$L_1$={L1:.9f} m, $L_2$={L2_best} m, BW={baseline_bw + dgamma_dL1*dL1_dT*i:.1f} Hz"))
        # , FSR(L1) ≈ {FSR_L1_MHz:.1f} MHz"))
        # ,color=color)
        
        # envelope = savgol_filter(T3_test, window_length=100, polyorder=3)
        # plt.semilogy(dnu_arr*1e-6, envelope, label='Envelope', linewidth=2)
    plt.semilogy(dnu_arr * 1e-6, T_effective_2_mirror(k, L2_best, R1_best, R2_best, R3_best),label="Two mirror version")
        #plt.semilogy(dnu_arr * 1e-6,T3_smoothed, color='green')

    plt.xlabel("Frequency detuning $\\Delta\\nu$ [MHz]")
    plt.ylabel("Transmitted Power [arb. units]")
    plt.title(
        f"Transmission vs Frequency Detuning via Temperature Tuning of L1, "
        f"\nR1={R1_best:.3f}, R2={R2_best:.3f}, R3={R3_best:.3f},"
        f"\nPole: {baseline_bw:.2f} Hz, BW Tunability: {bw_expansion:.2f} Hz/°C"
    )
    plt.grid(True)
    plt.legend(loc='upper right')

   

    FSR_L2_Hz = c / (2 * L2_best)
    FSR_L2_MHz = FSR_L2_Hz * 1e-6
    

    plt.text(
    0.01, 0.02,  # x and y in axis fraction
    f"FSR(L2) ≈ {FSR_L2_MHz:.3f} MHz",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='left',
    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.8)
    )

    plt.ylim(1e-10, 1e-3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{Teff_target_ppm:.1f}ppm_L1{L1_best:0.3f}_L2{L2_best:0.1f}_R1{R1_best:0.2f}_R2{R2_best:0.2f}.png", dpi=300)
    plt.show()


def R_from_Teff_general(Teff_target, R2_fixed):
    def Teff(R1):
        num = (1 - R1) * (1 - R2_fixed)
        denom = (1 - np.sqrt(R1 * R2_fixed))**2
        return num / denom

    def objective(R1):
        return Teff(R1) - Teff_target

    R1_solution = brentq(objective, 0.8, 0.99999)
    return R1_solution

def optimize_L1(L1_min, L1_max, L2, R1, R2, R3, target_bw_range, heating_capacity=10, steps=10000):
    L1_vals = np.linspace(L1_min, L1_max, steps)
    results = []
    for L1 in L1_vals:
        bw, bw_tunability, _, _ = BW_tunability(L1, L2, R1, R2, R3)
        #print(f"bw: {bw} Hz")
        if target_bw_range[0] <= bw_tunability * heating_capacity <= target_bw_range[1] and 70 < bw < 110:
            results.append((L1, bw, bw_tunability))

    if results:
        df = pd.DataFrame(results, columns=["L1 (m)", "Pole (Hz)", "Bandwidth Tunability (Hz/°C)"])
        df_sorted = df.sort_values("Bandwidth Tunability (Hz/°C)", ascending=False)
        print(df_sorted[:10])
        return df_sorted.iloc[0]["L1 (m)"]
    else:
        print("No viable L1 found within the target bandwidth range.")
        return None
    
def R1_from_Teff(T_eff_target, R2):
    """ Given Teff and R2, solve for R1 that satisfies the effective transmission equation. """
    def func(R1):
        return Teff_formula(R1, R2) - T_eff_target
    try:
        R1_sol = brentq(func, 1e-6, R2 - 1e-5)  # Ensure R1 < R2
        return R1_sol
    except ValueError:
        return None

def generate_R1_R2_pairs(Teff_target_ppm, R2_range=np.linspace(0.7, 0.999999, 500)):
    Teff_target = Teff_target_ppm * 1e-6
    results = []
    for R2 in R2_range:
        R1 = R1_from_Teff(Teff_target, R2)
        if R1 is not None:
            results.append((R1, R2, Teff_formula(R1, R2)))
    return results

def Teff_formula(R1, R2):
    """ Effective power transmission of compound mirror formed by R1 and R2. """
    t1 = np.sqrt(1 - R1)
    t2 = np.sqrt(1 - R2)
    r1 = np.sqrt(R1)
    r2 = np.sqrt(R2)
    return (t1 * t2)**2 / abs(1 + r1 * r2)**2

def main():

    
    Ti, fdetune = 0.0013296, -50

    Teff_target_ppm = 1329.6  # Example

    print(f"----------------- Generating R1/R2 Optimal Params for T_target = {Teff_target_ppm} -----------------")
    pairs = generate_R1_R2_pairs(Teff_target_ppm)
    bestR1, bestR2 = 0,0
    if pairs:
        print(f"Found {len(pairs)} matching R1, R2 pairs for Teff = {Teff_target_ppm} ppm:\n")
        for i, (R1, R2, Teff) in enumerate(pairs[:10]):
            print(f"{i+1}: R1 = {R1:.9f}, T1 = {(1-R1)*1e6:.1f} ppm | R2 = {R2:.9f}, T2 = {(1-R2)*1e6:.1f} ppm | Teff = {Teff*1e6:.1f} ppm")
            if i == 5:
                bestR1 = R1
                bestR2 = R2
    else:
        print("No valid R1, R2 pairs found.")

    R2_fixed = bestR2
    R1_solution = bestR1

    L2 = 297.5
    R3 = 1 - 1e-6

    print(f"Loaded Ti = {Ti*1e6:.2f} ppm, fdetune = {fdetune:.2f} Hz")
    print(f"→ Required R1: {R1_solution:.9f}, R2: {R2_fixed:.9f}")

    # Create data folder for each run
    timestamp = time.strftime("%Y%m%d")

    target_bw_range = (10, 50)
    print(f"----------------- Generating L1 Optimal Params for T_target = {Teff_target_ppm} -----------------")
    L1_solution = optimize_L1(L1_min=0.019, L1_max=0.020, L2=L2, R1=R1_solution, R2=R2_fixed, R3=R3, target_bw_range=target_bw_range,steps=8000)
    if L1_solution:
        output_dir = f"plots_aditi/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        plot_bw_expansion(L1_best=L1_solution, 
                          L2_best=L2, 
                          R1_best=R1_solution, 
                          R2_best=R2_fixed, 
                          R3_best=R3, 
                          Teff_target_ppm=Teff_target_ppm, 
                          output_dir=output_dir)
    else:
        print("No L1 solution")


if __name__ == "__main__":
    main()
