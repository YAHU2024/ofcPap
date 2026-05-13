"""Phase B4: Performance Evaluation — BER curves, Q-factor comparison, complexity analysis.

Loads the trained MLP-NLC model and dataset, then produces publication-quality
comparative figures and analysis tables for the paper's Results section.

Produces:
  - ber_vs_power.png         BER vs Launch Power (log scale, EDC vs MLP)
  - qfactor_comparison.png   Q-factor: EDC vs MLP with gain annotations
  - qgain_vs_power.png       Q-factor gain vs Launch Power
  - evm_cdf_comparison.png   EVM CDF before/after compensation (0 dBm)
  - complexity_table.csv     Computational complexity summary
  - b4_summary.csv           Full per-power metrics
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from scipy.special import erfc
import time
import csv

# ── Paths ────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUT_DIR / "mlp_nlc_model.pt"
DATASET_PATH = OUT_DIR / "fiber_dataset.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config (must match train_mlp_nlc.py) ─────────────────────────────────────
MEMORY_TAPS = 2
HIDDEN_DIMS = [64, 32, 16]
BATCH_SIZE = 512
RANDOM_SEED = 42
MOD_ORDER = 16
BITS_PER_SYMBOL = 4
FEC_THRESHOLD = 1.5e-2  # typical HD-FEC threshold (BER)

# DBP complexity reference
N_SPANS = 10
SPS = 4
N_SYMBOLS = 16384
N_SAMPLES = N_SYMBOLS * SPS  # 65536 — full signal length
# Each complex FFT = 5*N*log2(N) real ops, then each complex multiply = 6 real ops
# For DBP: one step = 2×FFT + 2×phase multiplication (nonlinear + dispersion)
DBP_FFT_REAL_OPS = 5 * N_SAMPLES * np.log2(N_SAMPLES)  # real ops per FFT
DBP_PHASE_MULT_REAL_OPS = 6 * N_SAMPLES  # per complex phase multiply
DBP_OPS_PER_STEP = 2 * DBP_FFT_REAL_OPS + 2 * DBP_PHASE_MULT_REAL_OPS  # per step
# Per symbol (amortized: each DBP step processes full signal)
DBP_OPS_PER_SYMBOL_STANDARD = DBP_OPS_PER_STEP * N_SPANS * 500 / N_SYMBOLS  # 500 steps/span
DBP_OPS_PER_SYMBOL_REDUCED = DBP_OPS_PER_STEP * N_SPANS * 10 / N_SYMBOLS   # 10 steps/span

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


# ── Model definition (mirrors train_mlp_nlc.py) ──────────────────────────────
class MLPCompensator(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS
        layers = []
        prev = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev, h_dim))
            layers.append(nn.ReLU())
            prev = h_dim
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Gray-mapped 16-QAM decision ──────────────────────────────────────────────
def gray_code_4bit():
    return np.array([0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10])

def symbol_to_bits(symbols):
    """Hard-decision 16-QAM demapper with Gray decoding.
    Returns (n_symbols, 4) bit array.
    """
    n = len(symbols)
    symbols_norm = symbols * np.sqrt(10)  # undo constellation scaling
    bits = np.zeros((n, 4), dtype=int)

    # Decision thresholds at -2, 0, +2
    i_dec = np.digitize(symbols_norm.real, [-2, 0, 2])  # 0..3
    q_dec = np.digitize(symbols_norm.imag, [-2, 0, 2])

    gray = gray_code_4bit()
    gray_to_bits = {}
    for idx, g in enumerate(gray):
        gray_to_bits[g] = [(idx >> 3) & 1, (idx >> 2) & 1, (idx >> 1) & 1, idx & 1]

    # Map I/Q decision → Gray index → bits
    for k in range(n):
        i_val = 2 * i_dec[k] - 3
        q_val = 2 * q_dec[k] - 3
        # find Gray index for this (i_val, q_val) pair
        i_gray = (i_val + 3) // 2  # -3→0, -1→1, +1→2, +3→3
        q_gray = (q_val + 3) // 2
        gray_idx = i_gray * 4 + q_gray
        # reverse Gray mapping
        bits[k] = gray_to_bits.get(gray[gray_idx], [0, 0, 0, 0])

    return bits


def count_bit_errors(rx_symbols, tx_symbols):
    """Count bit errors via hard-decision Gray-mapped 16-QAM demapping."""
    rx_bits = symbol_to_bits(rx_symbols)
    tx_bits = symbol_to_bits(tx_symbols)
    return np.sum(rx_bits != tx_bits), rx_bits.size


def safe_log10_ber(ber):
    """Return log10(BER) with floor at -80 for zero/inf values."""
    return np.where(ber > 0, np.log10(np.maximum(ber, 1e-80)), -80)


# ── Data loading ─────────────────────────────────────────────────────────────
def build_features(symbols, memory=MEMORY_TAPS):
    n = len(symbols)
    features = []
    for k in range(memory, n - memory):
        window = symbols[k - memory : k + memory + 1]
        feat = np.column_stack([window.real, window.imag]).ravel()
        features.append(feat)
    return np.array(features, dtype=np.float32)


def build_target(symbols, memory=MEMORY_TAPS):
    n = len(symbols)
    targets = []
    for k in range(memory, n - memory):
        s = symbols[k]
        targets.append([s.real, s.imag])
    return np.array(targets, dtype=np.float32)


# ── BER estimation methods ───────────────────────────────────────────────────
def estimate_ber_theoretical(evm):
    """Theoretical BER for 16-QAM: BER ≈ 3/4 * erfc(sqrt(SNR_per_bit)).
    SNR_per_bit = SNR_per_symbol / 4 = (1/EVM²) / 4
    For Gray-coded 16-QAM: BER ≈ 3/4 * erfc(sqrt(SNR_per_symbol / 10))
    """
    snr_per_symbol = 1.0 / (evm ** 2)
    with np.errstate(over="ignore", under="ignore"):
        ber = 0.75 * erfc(np.sqrt(snr_per_symbol / 10.0))
    return np.clip(ber, 1e-80, 0.5)


def estimate_ber_counting(rx_symbols, tx_symbols):
    """BER via direct bit-error counting with hard decision."""
    n_errors, n_bits = count_bit_errors(rx_symbols, tx_symbols)
    ber = n_errors / n_bits if n_bits > 0 else 1.0
    return max(ber, 1e-80)


# ── Complexity analysis ──────────────────────────────────────────────────────
def analyze_complexity(model, input_dim, n_test=10000):
    """Compute computational complexity metrics for MLP-NLC vs DBP.

    Returns dict with params, MACs, inference time, and DBP comparison.
    """
    # MLP parameter count
    n_params = model.num_params

    # MLP MACs per symbol
    macs = 0
    prev = input_dim
    for h in HIDDEN_DIMS:
        macs += prev * h  # linear
        macs += h          # ReLU (~1 MAC per unit)
        prev = h
    macs += prev * 2       # output layer
    # Total MACs for the layer operations

    # DBP reference complexity
    dbp_standard = DBP_OPS_PER_SYMBOL_STANDARD
    dbp_reduced = DBP_OPS_PER_SYMBOL_REDUCED

    # Inference time benchmark
    model.eval()
    model = model.to(DEVICE)
    x_test = torch.randn(n_test, input_dim, device=DEVICE)

    # Warmup
    for _ in range(10):
        _ = model(x_test)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_runs = 100
    for _ in range(n_runs):
        _ = model(x_test)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    time_per_symbol = elapsed / (n_runs * n_test) * 1e9  # ns

    # Total real multiplications per symbol (more precise MAC count)
    real_mults = 0
    prev = input_dim
    for h in HIDDEN_DIMS:
        real_mults += prev * h  # weight matrix mults
        prev = h
    real_mults += prev * 2  # output layer

    return {
        "n_params": n_params,
        "real_mults_per_symbol": real_mults,
        "dbp_standard_ops_per_symbol": int(dbp_standard),
        "dbp_reduced_ops_per_symbol": int(dbp_reduced),
        "inference_ns_per_symbol": time_per_symbol,
        "inference_ms_per_batch": elapsed / n_runs * 1e3,
        "batch_size": n_test,
    }


# ── Plot: BER vs Launch Power ────────────────────────────────────────────────
def plot_ber_vs_power(launch_powers, results, filepath):
    """BER vs Launch Power, log scale. EDC vs MLP with FEC threshold."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ber_edc = np.array([results[p]["ber_counting_edc"] for p in launch_powers])
    ber_mlp = np.array([results[p]["ber_counting_mlp"] for p in launch_powers])

    ax.semilogy(launch_powers, ber_edc, "s-", color="tab:blue", markersize=8,
                linewidth=1.8, label="EDC (Linear Only)")
    ax.semilogy(launch_powers, ber_mlp, "o-", color="tab:orange", markersize=8,
                linewidth=1.8, label="EDC + MLP-NLC (Proposed)")

    # FEC threshold
    ax.axhline(FEC_THRESHOLD, color="tab:red", linestyle="--", linewidth=1.2,
               alpha=0.7)
    ax.annotate(f"HD-FEC Threshold ({FEC_THRESHOLD:.0e})",
                xy=(-3.5, FEC_THRESHOLD * 1.5), fontsize=9, color="tab:red",
                alpha=0.8)

    # Annotate BER reduction at each power
    for p, be, bm in zip(launch_powers, ber_edc, ber_mlp):
        if bm < be and be > 1e-20:
            reduction = be / bm
            ax.annotate(
                f"{reduction:.0f}×",
                (p, bm),
                textcoords="offset points",
                xytext=(0, -15),
                fontsize=8,
                ha="center",
                color="tab:red",
            )

    ax.set_xlabel("Launch Power [dBm]")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("BER vs Launch Power: EDC vs MLP Nonlinearity Compensation")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=1e-15)

    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] BER curve saved -> {filepath.name}")


# ── Plot: Q-factor Comparison ────────────────────────────────────────────────
def plot_qfactor_comparison(launch_powers, results, filepath):
    """Side-by-side: Q-factor vs Power + Q-gain bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    q_edc = np.array([results[p]["q_edc_db"] for p in launch_powers])
    q_mlp = np.array([results[p]["q_mlp_db"] for p in launch_powers])
    q_gain = np.array([results[p]["q_improvement_db"] for p in launch_powers])

    # Q-factor curves
    ax1.plot(launch_powers, q_edc, "s-", color="tab:blue", markersize=8,
             linewidth=1.8, label="EDC (Linear)")
    ax1.plot(launch_powers, q_mlp, "o-", color="tab:orange", markersize=8,
             linewidth=1.8, label="EDC + MLP-NLC")
    for i, p in enumerate(launch_powers):
        ax1.annotate(
            f"+{q_gain[i]:.1f}",
            (p, q_mlp[i]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=9, ha="center", color="tab:red", fontweight="bold",
        )
    ax1.set_xlabel("Launch Power [dBm]")
    ax1.set_ylabel("Q-factor [dB]")
    ax1.set_title("Q-factor vs Launch Power")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-gain bar chart
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(launch_powers)))
    bars = ax2.bar(launch_powers, q_gain, width=1.2, color=colors,
                   edgecolor="black", linewidth=0.5)
    for bar, gain in zip(bars, q_gain):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"+{gain:.1f} dB", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Launch Power [dBm]")
    ax2.set_ylabel("Q-factor Improvement [dB]")
    ax2.set_title("Q-factor Gain from MLP-NLC")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("MLP Nonlinearity Compensator: Q-factor Performance", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Q-factor comparison saved -> {filepath.name}")


# ── Plot: Q-gain vs Power (standalone) ──────────────────────────────────────
def plot_qgain_vs_power(launch_powers, results, filepath):
    """Q-factor gain as a function of launch power — shows nonlinearity dominance."""
    fig, ax = plt.subplots(figsize=(7, 5))

    q_gain = np.array([results[p]["q_improvement_db"] for p in launch_powers])
    evm_edc = np.array([results[p]["evm_edc"] for p in launch_powers])

    # Dual y-axis: Q-gain (left) and EDC EVM (right, showing nonlinearity)
    ax2_color = "tab:red"
    ax.bar(launch_powers, q_gain, width=1.2, color="tab:orange",
           edgecolor="black", linewidth=0.5, zorder=3)
    for p, gain in zip(launch_powers, q_gain):
        ax.text(p, gain + 0.5, f"+{gain:.1f}", ha="center", fontsize=10,
                fontweight="bold")

    ax.set_xlabel("Launch Power [dBm]")
    ax.set_ylabel("Q-factor Gain [dB]", color="tab:orange")
    ax.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_ylim(0, max(q_gain) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    ax2 = ax.twinx()
    ax2.plot(launch_powers, evm_edc, "s-", color=ax2_color, markersize=8,
             linewidth=1.8, label="EDC EVM (nonlinearity indicator)")
    ax2.set_ylabel("EDC EVM (pre-compensation)", color=ax2_color)
    ax2.tick_params(axis="y", labelcolor=ax2_color)
    ax2.legend(loc="lower right")

    ax.set_title("Q-factor Gain vs Launch Power\n(Gain increases with nonlinearity)", fontsize=13)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Q-gain curve saved -> {filepath.name}")


# ── Plot: EVM CDF ────────────────────────────────────────────────────────────
def plot_evm_cdf(results, filepath, power_dbm=0):
    """CDF of per-symbol EVM before and after MLP compensation."""
    r = results[power_dbm]
    rx_edc = r["rx_edc"]
    rx_mlp = r["rx_mlp"]
    tx = r["tx"]

    evm_per_sym_edc = np.abs(rx_edc - tx)
    evm_per_sym_mlp = np.abs(rx_mlp - tx)

    fig, ax = plt.subplots(figsize=(8, 5))

    # CDF
    for data, label, color, ls in [
        (evm_per_sym_edc, "EDC (Linear)", "tab:blue", "--"),
        (evm_per_sym_mlp, "EDC + MLP-NLC", "tab:orange", "-"),
    ]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.semilogy(sorted_data, 1 - cdf, color=color, linestyle=ls,
                    linewidth=1.8, label=f"{label} (mean={np.mean(data):.4f})")

    ax.set_xlabel("Per-Symbol EVM")
    ax.set_ylabel("Complementary CDF  P(EVM > x)")
    ax.set_title(f"EVM Distribution: EDC vs MLP-NLC ({power_dbm:+d} dBm Launch)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] EVM CDF saved -> {filepath.name}")


# ── Save complexity analysis ─────────────────────────────────────────────────
def save_complexity_csv(complexity, results, launch_powers, filepath):
    """Save complexity analysis and per-power detailed metrics."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["model_parameters", complexity["n_params"]])
        writer.writerow(["real_mults_per_symbol", complexity["real_mults_per_symbol"]])
        writer.writerow(["dbp_standard_ops_per_symbol", complexity["dbp_standard_ops_per_symbol"]])
        writer.writerow(["dbp_reduced_ops_per_symbol", complexity["dbp_reduced_ops_per_symbol"]])
        writer.writerow(["inference_ns_per_symbol", f"{complexity['inference_ns_per_symbol']:.2f}"])
        writer.writerow(["inference_ms_per_batch_10k", f"{complexity['inference_ms_per_batch']:.3f}"])
        writer.writerow([])
        writer.writerow(["Complexity comparison"])
        writer.writerow(["MLP real mults/symbol", complexity["real_mults_per_symbol"]])
        writer.writerow(["Standard DBP (500 steps/span)", complexity["dbp_standard_ops_per_symbol"]])
        writer.writerow(["Reduced DBP (10 steps/span)", complexity["dbp_reduced_ops_per_symbol"]])
        writer.writerow(["Complexity ratio MLP/Standard-DBP", f"{complexity['real_mults_per_symbol'] / complexity['dbp_standard_ops_per_symbol']:.6f}"])
        writer.writerow(["Complexity ratio MLP/Reduced-DBP", f"{complexity['real_mults_per_symbol'] / complexity['dbp_reduced_ops_per_symbol']:.4f}"])
    print(f"  [OK] Complexity CSV saved -> {filepath.name}")


# ── Save full per-power summary ──────────────────────────────────────────────
def save_b4_summary(launch_powers, results, filepath):
    """Save comprehensive per-power metrics including both BER methods."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "power_dbm", "evm_edc", "evm_mlp", "evm_reduction_pct",
            "q_edc_db", "q_mlp_db", "q_gain_db",
            "ber_theoretical_edc", "ber_theoretical_mlp",
            "ber_counting_edc", "ber_counting_mlp",
            "phase_error_edc_rad", "phase_error_mlp_rad",
        ])
        for p in launch_powers:
            r = results[p]
            phase_edc = np.mean(np.abs(np.angle(r["rx_edc"] * np.conj(r["tx"]))))
            phase_mlp = np.mean(np.abs(np.angle(r["rx_mlp"] * np.conj(r["tx"]))))
            evm_reduction = (1 - r["evm_mlp"] / r["evm_edc"]) * 100

            writer.writerow([
                p,
                f"{r['evm_edc']:.6f}",
                f"{r['evm_mlp']:.6f}",
                f"{evm_reduction:.2f}",
                f"{r['q_edc_db']:.2f}",
                f"{r['q_mlp_db']:.2f}",
                f"{r['q_improvement_db']:.2f}",
                f"{r['ber_theoretical_edc']:.6e}",
                f"{r['ber_theoretical_mlp']:.6e}",
                f"{r['ber_counting_edc']:.6e}",
                f"{r['ber_counting_mlp']:.6e}",
                f"{phase_edc:.6f}",
                f"{phase_mlp:.6f}",
            ])
    print(f"  [OK] B4 summary CSV saved -> {filepath.name}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" Phase B4: Performance Evaluation")
    print(f" Device: {DEVICE}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────
    print("\n[1/4] Loading trained model and dataset...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    launch_powers = checkpoint["config"]["launch_powers"]
    input_dim = checkpoint["config"]["input_dim"]

    # Load raw dataset for test symbols
    data = np.load(DATASET_PATH)

    # Build features and targets from raw data (same as training)
    memory = MEMORY_TAPS
    all_x, all_y = {}, {}
    for p in launch_powers:
        key = f"{p:+d}dBm"
        all_x[p] = data[f"X_{key}"]
        all_y[p] = data[f"Y_{key}"]

    # Recreate test split (same seed as training)
    datasets = {}
    TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
    for p in launch_powers:
        x_feat = build_features(all_x[p], memory)
        y_feat = build_target(all_y[p], memory)
        n = len(x_feat)
        idx = np.random.RandomState(RANDOM_SEED).permutation(n)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        datasets[p] = {
            "test": (x_feat[idx[n_train + n_val:]],
                     y_feat[idx[n_train + n_val:]]),
        }

    # ── Evaluate each power level ─────────────────────────────────────────
    print("\n[2/4] Running comprehensive evaluation...")
    results = {}

    for p in launch_powers:
        # Load and run model
        model = MLPCompensator(input_dim=input_dim, hidden_dims=HIDDEN_DIMS)
        model.load_state_dict(checkpoint["models"][p])
        model = model.to(DEVICE)
        model.eval()

        x_test, y_test = datasets[p]["test"]
        x_t = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred = model(x_t).cpu().numpy()

        # Reconstruct complex symbols
        center_pos = memory * 2
        rx_edc = x_test[:, center_pos] + 1j * x_test[:, center_pos + 1]
        rx_mlp = pred[:, 0] + 1j * pred[:, 1]
        tx = y_test[:, 0] + 1j * y_test[:, 1]

        # EVM
        evm_edc = np.sqrt(np.mean(np.abs(rx_edc - tx) ** 2))
        evm_mlp = np.sqrt(np.mean(np.abs(rx_mlp - tx) ** 2))

        # Q-factor
        q_edc = 20 * np.log10(1.0 / evm_edc) if evm_edc > 0 else 100
        q_mlp = 20 * np.log10(1.0 / evm_mlp) if evm_mlp > 0 else 100

        # BER: theoretical
        ber_theo_edc = estimate_ber_theoretical(evm_edc)
        ber_theo_mlp = estimate_ber_theoretical(evm_mlp)

        # BER: direct symbol counting
        ber_count_edc = estimate_ber_counting(rx_edc, tx)
        ber_count_mlp = estimate_ber_counting(rx_mlp, tx)

        results[p] = {
            "evm_edc": evm_edc,
            "evm_mlp": evm_mlp,
            "q_edc_db": q_edc,
            "q_mlp_db": q_mlp,
            "q_improvement_db": q_mlp - q_edc,
            "ber_theoretical_edc": ber_theo_edc,
            "ber_theoretical_mlp": ber_theo_mlp,
            "ber_counting_edc": ber_count_edc,
            "ber_counting_mlp": ber_count_mlp,
            "rx_edc": rx_edc,
            "rx_mlp": rx_mlp,
            "tx": tx,
        }

        print(f"  {p:+d} dBm: EVM {evm_edc:.4f}→{evm_mlp:.4f}  "
              f"Q {q_edc:.1f}→{q_mlp:.1f} dB (+{q_mlp - q_edc:.1f})  "
              f"BER(count) {ber_count_edc:.2e}→{ber_count_mlp:.2e}")

    # ── Complexity analysis ───────────────────────────────────────────────
    print("\n[3/4] Computing complexity metrics...")
    model_ref = MLPCompensator(input_dim=input_dim, hidden_dims=HIDDEN_DIMS)
    complexity = analyze_complexity(model_ref, input_dim)

    print(f"  MLP params:                  {complexity['n_params']:,}")
    print(f"  MLP real mults/symbol:       {complexity['real_mults_per_symbol']:,}")
    print(f"  Standard DBP (500 st/span):  {complexity['dbp_standard_ops_per_symbol']:,} ops/symbol")
    print(f"  Reduced DBP (10 st/span):    {complexity['dbp_reduced_ops_per_symbol']:,} ops/symbol")
    print(f"  Inference:                   {complexity['inference_ns_per_symbol']:.1f} ns/symbol")
    print(f"  Complexity vs Standard DBP:  {complexity['real_mults_per_symbol'] / complexity['dbp_standard_ops_per_symbol']:.6f}")
    print(f"  Complexity vs Reduced DBP:   {complexity['real_mults_per_symbol'] / complexity['dbp_reduced_ops_per_symbol']:.4f}")

    # ── Generate figures and tables ───────────────────────────────────────
    print("\n[4/4] Generating B4 figures and tables...")

    plot_ber_vs_power(launch_powers, results, OUT_DIR / "ber_vs_power.png")
    plot_qfactor_comparison(launch_powers, results, OUT_DIR / "qfactor_comparison.png")
    plot_qgain_vs_power(launch_powers, results, OUT_DIR / "qgain_vs_power.png")
    plot_evm_cdf(results, OUT_DIR / "evm_cdf_comparison.png", power_dbm=0)

    save_complexity_csv(complexity, results, launch_powers,
                        OUT_DIR / "complexity_table.csv")
    save_b4_summary(launch_powers, results, OUT_DIR / "b4_summary.csv")

    # ── Print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" B4 Performance Evaluation — Complete")
    print("=" * 80)
    print(f"\n{'Power':^8s} {'EVM EDC':^10s} {'EVM MLP':^10s} "
          f"{'Q EDC':^8s} {'Q MLP':^8s} {'Q Gain':^8s} "
          f"{'BER EDC':^12s} {'BER MLP':^12s}")
    print("-" * 80)
    for p in launch_powers:
        r = results[p]
        print(f" {p:+4d} dBm  {r['evm_edc']:^10.4f} {r['evm_mlp']:^10.4f} "
              f"{r['q_edc_db']:^8.2f} {r['q_mlp_db']:^8.2f} "
              f"{r['q_improvement_db']:^+8.2f} "
              f"{r['ber_counting_edc']:^12.2e} {r['ber_counting_mlp']:^12.2e}")
    print("-" * 80)
    print(f"\nComplexity: {complexity['n_params']:,} params  |  "
          f"{complexity['real_mults_per_symbol']:,} real mults/symbol  |  "
          f"{complexity['inference_ns_per_symbol']:.1f} ns/symbol")
    print(f"vs Standard DBP: {complexity['dbp_standard_ops_per_symbol']:,} ops/symbol "
          f"({complexity['real_mults_per_symbol'] / complexity['dbp_standard_ops_per_symbol']:.4f}×)")
    print(f"vs Reduced  DBP: {complexity['dbp_reduced_ops_per_symbol']:,} ops/symbol "
          f"({complexity['real_mults_per_symbol'] / complexity['dbp_reduced_ops_per_symbol']:.4f}×)")

    print("\nGenerated files:")
    for f in ["ber_vs_power.png", "qfactor_comparison.png", "qgain_vs_power.png",
              "evm_cdf_comparison.png", "complexity_table.csv", "b4_summary.csv"]:
        print(f"  -> {f}")


if __name__ == "__main__":
    main()
