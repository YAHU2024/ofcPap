"""SSFM simulator for single-channel 16-QAM fiber transmission.

Generates nonlinearly impaired signals for training a lightweight MLP
nonlinearity compensator, following the project Claude.md parameters.
"""

import numpy as np
from scipy import signal as sig
from scipy.constants import h, c
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Physical Constants & Simulation Parameters ───────────────────────────
CENTER_WAVELENGTH = 1550e-9           # m
CENTER_FREQ = c / CENTER_WAVELENGTH   # Hz

# 16-QAM
MOD_ORDER = 16
BITS_PER_SYMBOL = 4
SYMBOL_RATE = 32e9                    # 32 GBaud
SPS = 4                               # samples per symbol
ROLL_OFF = 0.1
RRC_SPAN = 20                         # filter span in symbols

# Fiber (80 km × 10 spans = 800 km)
SPAN_LENGTH = 80e3                    # m
N_SPANS = 10
ALPHA_DB = 0.2                        # dB/km
ALPHA_LIN = ALPHA_DB * np.log(10) / 10 / 1e3  # 1/m linear
D = 17e-6                             # s/m² (17 ps/nm/km)
BETA2 = -CENTER_WAVELENGTH**2 / (2 * np.pi * c) * D  # s²/m
GAMMA = 1.3e-3                        # /W/m (1.3 /W/km)
SSFM_STEPS_PER_SPAN = 500

# EDFA
NF_DB = 5.0
NF_LIN = 10 ** (NF_DB / 10)
N_SP = NF_LIN / 2                     # spontaneous emission factor

# Simulation
N_SYMBOLS = 16384                     # 2^14 symbols
LAUNCH_POWERS_DBM = [-4, -2, 0, 2, 4]  # input power sweep

OUT_DIR = Path(__file__).resolve().parent


# ── 16-QAM Constellation ──────────────────────────────────────────────────
def gray_code_4bit():
    """Return 4-bit Gray code mapping indices 0..15."""
    return np.array([0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10])


def generate_16qam_symbols(n_symbols):
    """Generate random 16-QAM symbols with Gray mapping, normalized to
    unit average power (E[|s|²] = 1).

    Constellation points: {±1 ± 1j, ±1 ± 3j, ±3 ± 1j, ±3 ± 3j} / sqrt(10)
    """
    bits = np.random.randint(0, 2, (n_symbols, BITS_PER_SYMBOL))
    # convert bits to integer indices 0..15
    idx = bits[:, 0] * 8 + bits[:, 1] * 4 + bits[:, 2] * 2 + bits[:, 3]
    # Gray decode
    gray = gray_code_4bit()
    gray_idx = gray[idx]
    # map to I/Q: each gray_idx maps to (I, Q) in {-3,-1,+1,+3}
    i_val = 2 * (gray_idx // 4) - 3       # -3, -1, +1, +3
    q_val = 2 * (gray_idx % 4) - 3        # -3, -1, +1, +3
    symbols = (i_val + 1j * q_val) / np.sqrt(10)  # normalize
    return symbols


# ── Pulse Shaping ─────────────────────────────────────────────────────────
def rrc_filter():
    """Design root-raised-cosine filter."""
    t = np.arange(-RRC_SPAN * SPS, RRC_SPAN * SPS + 1) / SPS
    h_rrc = np.zeros_like(t)

    # t = 0 (limit)
    idx0 = np.abs(t) < 1e-12
    h_rrc[idx0] = 1 - ROLL_OFF + 4 * ROLL_OFF / np.pi

    # t = +/- 1/(4*beta) (singularity)
    t_sing = 1.0 / (4 * ROLL_OFF)
    idx_sing = np.abs(np.abs(t) - t_sing) < 1e-12
    h_rrc[idx_sing] = ROLL_OFF / np.sqrt(2) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * ROLL_OFF))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * ROLL_OFF))
    )

    # normal case
    idx_normal = (~idx0) & (~idx_sing)
    t_n = t[idx_normal]
    pi_t = np.pi * t_n
    num = np.sin(pi_t * (1 - ROLL_OFF)) + 4 * ROLL_OFF * t_n * np.cos(pi_t * (1 + ROLL_OFF))
    den = pi_t * (1 - (4 * ROLL_OFF * t_n) ** 2)
    h_rrc[idx_normal] = num / den

    h_rrc /= np.sqrt(np.sum(h_rrc ** 2))
    return h_rrc


def pulse_shaping(symbols, h_rrc):
    """Upsample symbols and apply RRC pulse shaping (same-mode convolution).

    With 'same' mode and an odd-length filter, symbol k's RRC peak lands at
    output sample k * SPS (no net timing shift).
    """
    upsampled = np.zeros(len(symbols) * SPS, dtype=complex)
    upsampled[::SPS] = symbols
    shaped = np.convolve(upsampled, h_rrc, mode="same")
    return shaped


def matched_filter(rx_signal, h_rrc):
    """Apply matched filter (RRC) and downsample.

    Both TX and RX use same-mode convolution, so the combined RC peak for
    symbol k lands at output sample k * SPS.  Downsample starting at 0.
    """
    filtered = np.convolve(rx_signal, h_rrc, mode="same")
    downsampled = filtered[0::SPS]
    return downsampled


# ── SSFM Propagation ──────────────────────────────────────────────────────
def ssfm_propagation(signal, dz, dt):
    """Symmetric split-step Fourier propagation over one span.

    Uses the symmetric scheme: ½ N → D → ½ N for each step.
    """
    L = SPAN_LENGTH
    n_steps = int(L / dz)
    dz_actual = L / n_steps
    n_samples = len(signal)
    omega = 2 * np.pi * np.fft.fftfreq(n_samples, d=dt)

    # dispersion operator for FULL step dz
    d_step = np.exp(-1j * BETA2 / 2 * omega ** 2 * dz_actual)

    A = signal.copy()
    for _ in range(n_steps):
        # ½ nonlinear
        A *= np.exp(1j * GAMMA * np.abs(A) ** 2 * dz_actual / 2)
        # full dispersion
        A = np.fft.ifft(np.fft.fft(A) * d_step)
        # ½ nonlinear
        A *= np.exp(1j * GAMMA * np.abs(A) ** 2 * dz_actual / 2)
        # field attenuation (alpha is power coeff, field decays at alpha/2)
        A *= np.exp(-ALPHA_LIN * dz_actual / 2)

    return A


def add_ase_noise(signal, dt):
    """Post-span EDFA: amplify to compensate span loss and add ASE noise.

    Returns amplified signal with ASE noise.
    """
    g_lin = np.exp(ALPHA_LIN * SPAN_LENGTH)
    # amplify signal (field)
    signal = signal * np.sqrt(g_lin)
    # ASE noise power in the signal bandwidth (single polarization)
    noise_power = N_SP * h * CENTER_FREQ * (g_lin - 1) / dt
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


# ── EDC (Electronic Dispersion Compensation) ──────────────────────────────
def edc_equalization(rx_signal, dt):
    """Frequency-domain dispersion compensation over the full link."""
    n_samples = len(rx_signal)
    omega = 2 * np.pi * np.fft.fftfreq(n_samples, d=dt)
    L_total = SPAN_LENGTH * N_SPANS
    h_inv = np.exp(1j * BETA2 / 2 * omega ** 2 * L_total)
    return np.fft.ifft(np.fft.fft(rx_signal) * h_inv)


# ── Plotting ──────────────────────────────────────────────────────────────
def plot_constellation(rx_symbols, tx_symbols, power_dbm, filepath):
    """Side-by-side constellation: EDC-equalized vs. original transmitted."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(rx_symbols.real, rx_symbols.imag, s=1, alpha=0.5, color="tab:blue")
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel("In-Phase", fontsize=10)
    ax1.set_ylabel("Quadrature", fontsize=10)
    ax1.set_title(f"EDC Output ({power_dbm} dBm)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    ax2.scatter(tx_symbols.real, tx_symbols.imag, s=1, alpha=0.5, color="tab:green")
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel("In-Phase", fontsize=10)
    ax2.set_ylabel("Quadrature", fontsize=10)
    ax2.set_title("Original 16-QAM Transmitted", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  [OK] Constellation saved -> {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" SSFM 16-QAM Fiber Transmission Simulator")
    print(f" {N_SPANS} x {SPAN_LENGTH / 1e3:.0f} km  |  {SYMBOL_RATE / 1e9:.0f} GBaud  |  "
          f"alpha={ALPHA_DB} dB/km  D={D * 1e6:.0f} ps/nm/km  gamma={GAMMA * 1e3:.1f} /W/km")
    print(f" beta2 = {BETA2:.3e} s^2/m  |  dz = {SPAN_LENGTH / SSFM_STEPS_PER_SPAN:.1f} m")
    print("=" * 60)

    h_rrc = rrc_filter()
    dt = 1.0 / (SYMBOL_RATE * SPS)
    dz_step = SPAN_LENGTH / SSFM_STEPS_PER_SPAN

    # Generate transmit symbols once (same data for all power levels)
    tx_symbols_all = generate_16qam_symbols(N_SYMBOLS)
    print(f"\nGenerated {N_SYMBOLS} 16-QAM symbols")

    all_inputs = {}
    all_labels = {}
    # store symbols for later plotting at 0 dBm
    rx_0dbm = None
    tx_0dbm = None

    for p_dbm in LAUNCH_POWERS_DBM:
        print(f"\n--- Power sweep: {p_dbm:+.0f} dBm ---")
        p_lin = 10 ** (p_dbm / 10) * 1e-3  # dBm → W

        # Transmit signal with power scaling
        tx_signal = pulse_shaping(tx_symbols_all, h_rrc)
        tx_signal *= np.sqrt(p_lin)  # scale to launch power

        # SSFM propagation over N_SPANS
        rx_signal = tx_signal.copy()
        for span in range(N_SPANS):
            rx_signal = ssfm_propagation(rx_signal, dz_step, dt)
            rx_signal = add_ase_noise(rx_signal, dt)
            if (span + 1) % 2 == 0:
                print(f"  Span {span + 1:2d}/{N_SPANS} done")

        # EDC (dispersion compensation only)
        edc_signal = edc_equalization(rx_signal, dt)

        # Matched filter + downsampling
        rx_symbols = matched_filter(edc_signal, h_rrc)

        # Trim to same length (full conv may yield slightly more symbols)
        n_valid = min(len(rx_symbols), len(tx_symbols_all))
        rx_symbols = rx_symbols[:n_valid]
        tx_symbols = tx_symbols_all[:n_valid]

        # Power normalization to unit average
        rx_symbols /= np.sqrt(np.mean(np.abs(rx_symbols) ** 2))

        # Store
        all_inputs[f"{p_dbm:+d}dBm"] = rx_symbols
        all_labels[f"{p_dbm:+d}dBm"] = tx_symbols

        if p_dbm == 0:
            rx_0dbm = rx_symbols.copy()
            tx_0dbm = tx_symbols.copy()

        # Quick stats
        evm = np.sqrt(np.mean(np.abs(rx_symbols - tx_symbols) ** 2))
        print(f"  EVM = {evm:.4f}")

    # ── Save dataset ───────────────────────────────────────────────────
    np.savez_compressed(OUT_DIR / "fiber_dataset.npz",
                        **{f"X_{k}": v for k, v in all_inputs.items()},
                        **{f"Y_{k}": v for k, v in all_labels.items()},
                        launch_powers=LAUNCH_POWERS_DBM)
    print(f"\nDataset saved -> fiber_dataset.npz")

    # ── Verification plot at 0 dBm ─────────────────────────────────────
    plot_constellation(rx_0dbm, tx_0dbm, 0, OUT_DIR / "initial_test.png")

    # ── Multi-power constellation overview (optional) ──────────────────
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for idx, p_dbm in enumerate(LAUNCH_POWERS_DBM):
        rx = all_inputs[f"{p_dbm:+d}dBm"]
        tx = all_labels[f"{p_dbm:+d}dBm"]
        ax_rx = axes[0, idx]
        ax_tx = axes[1, idx]

        ax_rx.scatter(rx.real, rx.imag, s=0.5, alpha=0.5, color="tab:blue")
        ax_rx.set_xlim(-2, 2); ax_rx.set_ylim(-2, 2)
        ax_rx.set_title(f"EDC {p_dbm:+d} dBm", fontsize=10)
        ax_rx.set_aspect("equal")
        ax_rx.grid(True, alpha=0.2)

        ax_tx.scatter(tx.real, tx.imag, s=0.5, alpha=0.5, color="tab:green")
        ax_tx.set_xlim(-2, 2); ax_tx.set_ylim(-2, 2)
        ax_tx.set_title(f"TX {p_dbm:+d} dBm", fontsize=10)
        ax_tx.set_aspect("equal")
        ax_tx.grid(True, alpha=0.2)

    fig.suptitle("16-QAM Fiber Transmission: EDC Output vs. Original (per Launch Power)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "power_sweep_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Overview saved -> power_sweep_overview.png")

    print("\n" + "=" * 60)
    print(" Simulation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
