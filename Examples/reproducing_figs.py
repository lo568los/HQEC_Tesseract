import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import freeze_support

# --- HQEC Imports ---
from LEGO_HQEC.OperatorPush.PushingToolbox import batch_push
from LEGO_HQEC.OperatorPush.Presets.HaPPY_code import setup_zero_rate_happy
from LEGO_HQEC.QuDec.ErasureDecoder import calculate_recovery_rates_for_p_range
from LEGO_HQEC.QuDec.InputProcessor import extract_stabilizers_from_result_dict, extract_logicals_from_result_dict
from LEGO_HQEC.QuDec.TN_decoder import tn_quantum_error_correction_decoder_multiprocess

# --- Configuration ---
# Radii to simulate (Paper likely uses R=0, 1, 2, 3)
# Note: R=3 takes significantly longer. For a quick test, use [0, 1, 2].
RADII = [0, 1, 2] 

# Monte Carlo Trials (Paper likely uses N=1000 or N=10000)
# Reduced here for demonstration speed. Increase for smooth curves.
N_TRIALS_ERASURE = 200
N_TRIALS_PAULI = 200

def generate_data_fig6a(radii):
    """
    Reproduces Fig 6(a): Erasure Thresholds
    Plots Recovery Rate vs Erasure Probability (p) for different radii.
    """
    print("\n--- Generating Data for Fig 6(a): Erasure Threshold ---")
    data_storage = {}

    for R in radii:
        print(f"[*] Simulating Radius R={R}...")
        
        # 1. Setup Code & Push Operators
        tensor_list = setup_zero_rate_happy(R)
        results_dict = batch_push(tensor_list)
        
        stabilizers = extract_stabilizers_from_result_dict(results_dict)
        logical_zs, logical_xs = extract_logicals_from_result_dict(results_dict)
        
        # Combine logicals (recoverability checks if *any* logical is damaged)
        # For HaPPY code, we typically check recoverability of the encoded qubit.
        logical_operators = [logical_zs[0]] + [logical_xs[0]]

        # 2. Run Erasure Decoder
        # Range: 0.0 to 0.55 covers the typical threshold transition
        results = calculate_recovery_rates_for_p_range(
            n=N_TRIALS_ERASURE,
            p_start=0.01,
            p_end=0.60,
            p_step=0.04,
            stabilizers=stabilizers,
            logical_operators=logical_operators
        )
        data_storage[R] = results
        
    return data_storage

def generate_data_fig6b(radii):
    """
    Reproduces Fig 6(b): Pauli Thresholds (TN Decoder)
    Plots Logical Success Rate vs Depolarizing Probability (p) for different radii.
    """
    print("\n--- Generating Data for Fig 6(b): Pauli Threshold (Depolarizing) ---")
    data_storage = {}
    
    # Standard Depolarizing Channel: rx = ry = rz = p/3
    # We pass the ratios to the decoder, p is the total error probability
    rx_ratio, rz_ratio = 1/3, 1/3
    ry_ratio = 1 - rx_ratio - rz_ratio

    # Probability range for Pauli noise (usually lower than erasure)
    p_values = np.linspace(0.0, 0.5, 8)

    for R in radii:
        print(f"[*] Simulating Radius R={R}...")
        tensor_list = setup_zero_rate_happy(R)
        
        success_rates = []
        for p in p_values:
            print(f"    Testing p={p:.2f}...")
            rate = tn_quantum_error_correction_decoder_multiprocess(
                tensor_list=tensor_list,
                p=p,
                rx=rx_ratio,
                ry=ry_ratio,
                rz=rz_ratio,
                N=N_TRIALS_PAULI,
                n_process=4  # Adjust based on your CPU cores
            )
            success_rates.append((p, rate))
        
        data_storage[R] = success_rates
        
    return data_storage

def plot_results(erasure_data, pauli_data):
    """
    Uses Matplotlib to plot the data in a style similar to the paper.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 6(a): Erasure ---
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers = ['o', 's', '^', 'D']
    
    for i, R in enumerate(erasure_data):
        data = np.array(erasure_data[R])
        x, y = data[:, 0], data[:, 1]
        ax1.plot(x, y, label=f'R={R}', marker=markers[i%len(markers)], color=colors[i%len(colors)], alpha=0.8)
    
    ax1.set_title("Fig 6(a): Erasure Threshold")
    ax1.set_xlabel("Erasure Probability $p$")
    ax1.set_ylabel("Recovery Rate")
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)

    # --- Plot 6(b): Pauli (TN) ---
    for i, R in enumerate(pauli_data):
        data = np.array(pauli_data[R])
        x, y = data[:, 0], data[:, 1]
        ax2.plot(x, y, label=f'R={R}', marker=markers[i%len(markers)], color=colors[i%len(colors)], alpha=0.8)

    #Also plot the tesseract decoder results for comparison (if available)
        filename = f"hqec_tesseract_logical_error_rates_lin_R={R}.txt"
        tess_err = np.loadtxt(filename)
        p_values = np.linspace(0, 0.5, len(tess_err))
        ax2.plot(p_values, 1-tess_err, label=f'Tesseract R={R}', linestyle='--', color=colors[i%len(colors)], alpha=0.8)

    ax2.set_title("Fig 6(b): Pauli Threshold (Depolarizing)")
    ax2.set_xlabel("Physical Error Rate $p$")
    ax2.set_ylabel("Logical Success Rate")
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig('hqec_fig6_reproduction.png')
    print("\n[*] Plot saved to 'hqec_fig6_reproduction.png'")
    plt.show()

if __name__ == "__main__":
    freeze_support() # Required for Windows multiprocessing
    
    # 1. Generate Erasure Data
    erasure_results = generate_data_fig6a(RADII)
    
    # 2. Generate Pauli Data
    pauli_results = generate_data_fig6b(RADII)
    
    # 3. Plot
    plot_results(erasure_results, pauli_results)