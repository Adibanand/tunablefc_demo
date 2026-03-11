
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Bandwidth function from effective mirror model
def BW_TM(L1, L2, R1, R2, R3=0.999, wavelength=1064e-9):
    c = 3e8  # m/s
    k = 2 * np.pi / wavelength
    r1, r2 = np.sqrt(R1), np.sqrt(R2)
    T1 = 1 - R1
    T2 = 1 - R2
    T3 = 1 - R3
    Teff = (T1 * T2) / (1 - 2 * r1 * r2 * np.cos(2 * L1 * k) + (r1 * r2)**2)
    BW = (c / (8 * np.pi * (L2+L1))) * (Teff + T3)
    return BW


def T_three_mirror(k, L1, L2, r1, r2, r3):
    t1, t2, t3 = 1-r1,1-r2,1-r3
    e_total = np.exp(2j * k * (L1 + L2))
    e_L2 = np.exp(2j * k * L2)
    e_L1 = np.exp(2j * k * L1)

    num = -t1 * t2 * t3 * np.exp(1j * k * (L1 + L2))
    denom = (e_total - r1 * r2 * e_L2 - r2 * r3 * e_L1 + r1 * r3 * (r2**2 + t2**2))
    return np.abs(num / denom) ** 2

def T_fp(k, L, R1, R2, T1, T2):
    delta = 2 * k * L
    return (T1 * T2) / (1 + R1 * R2 - 2 * np.sqrt(R1 * R2) * np.cos(delta))

# Plotting
def plot_transmission_spectrum(R1, R2, L1, L2, save_dir, lambda0=1064e-9, R3=0.999):
    
    # Optical constants
    c = 3e8  # m/s
    nu0 = c / lambda0  # Hz

    # Detuning and wavenumber
    dnu = np.linspace(-150e6, 150e6, 1000)  # Hz
    nu = nu0 + dnu
    k = 2 * np.pi * nu / c

    t1, t2, t3 = np.sqrt(1 - R1), np.sqrt(1 - R2), np.sqrt(1 - R3)
    r1, r2, r3 = np.sqrt(R1), np.sqrt(R2), np.sqrt(R3)

    # Cavity lengths
    L_fp = L1+L2  # for comparison

    T3 = [T_three_mirror(ki, L1, L2, r1, r2, r3) for ki in k]
    TFP = [T_fp(ki, L_fp, R1, R2, 1 - R1, 1 - R2) for ki in k]

    plt.figure(figsize=(10, 6))
    plt.semilogy(dnu * 1e-6, T3, label=f'Three-mirror: $L_1={L1:.3f}$ mm, $L_2={L2:.3f}$ m')
    plt.semilogy(dnu * 1e-6, TFP, '--', label=f'FP cavity: $L={L_fp}$ m')

    plt.xlabel("Frequency detuning $\\Delta\\nu$ [MHz]")
    plt.ylabel("Transmitted Power [arb. units]")
    plt.title("Transmission vs Frequency Detuning")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Filename with parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detuning_L1_{L1:.3f}_L2_{L2:.3f}_R1_{R1:.3f}_R2_{R2:.3f}_{timestamp}.png"
    save_dir = "detuning_plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path)

    print(f"Saved detuning plot to: {save_path}")

# Objective function to maximize bandwidth (minimize negative bandwidth)
def objective(trial):
    R1 = trial.suggest_float("R1", 0.5, 0.9995)  # Reflectivity range extended
    R2 = trial.suggest_float("R2", 0.5, 0.9995)
    L1 = trial.suggest_float("L1", 5e-3, 8e-3)    # 6 mm nominal
    L2 = trial.suggest_float("L2", 1.0, 5.0)     # Lab-sized cavity

    BW = BW_TM(L1, L2, R1, R2)
    target_bw = 100.0  # Hz
    return (BW - target_bw) ** 2

# Main optimization loop
def main(n_trials=500, n_best=10, save_dir="results_bandwidth"):
    os.makedirs(save_dir, exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_trials = sorted(study.trials, key=lambda t: t.value)[:n_best]
    records = []
    for t in best_trials:
        params = t.params
        BW = BW_TM(**params)
        records.append({**params, "Bandwidth_Hz": BW, "Loss": t.value})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(save_dir, "best_bandwidth_parameters.csv"), index=False)
    print(f"\nSaved top {n_best} results to: {save_dir}/best_bandwidth_parameters.csv")

    # Plot bandwidth vs trial
    bandwidths = [-t.value for t in study.trials]
    plt.figure()
    plt.plot(range(len(bandwidths)), bandwidths, ".", alpha=0.6)
    plt.xlabel("Trial Index")
    plt.ylabel("Bandwidth (Hz)")
    plt.title("Bandwidth vs Trial")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bandwidth_vs_trial.png"))
    plt.close()

    # Parameter vs bandwidth plots
    for param in ["R1", "R2", "L1", "L2"]:
        values = [t.params[param] for t in study.trials]
        plt.figure()
        plt.scatter(values, bandwidths, alpha=0.6, s=10)
        plt.xlabel(param)
        plt.ylabel("Bandwidth (Hz)")
        plt.title(f"{param} vs Bandwidth")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{param}_vs_bandwidth.png"))
        plt.close()
    
    for idx, row in df.iterrows():
        plot_transmission_spectrum(row["R1"], row["R2"], row["L1"], row["L2"], save_dir)

    print(f"Saved transmission spectra with timestamps and parameters.")

if __name__ == "__main__":
    main()
