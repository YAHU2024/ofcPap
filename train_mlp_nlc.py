"""Lightweight MLP nonlinearity compensator for 16-QAM fiber transmission.

Implements a complexity-aware MLP (MT-NN simplified architecture) that
compensates Kerr nonlinearity from EDC-equalized symbols.

Reference: MT-NN simplified (Optics Express, 2025) — complexity-aware
lightweight MLP for fiber nonlinearity compensation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import erfc
import time

# ── Config ─────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent
DATASET_PATH = OUT_DIR / "fiber_dataset.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP architecture
MEMORY_TAPS = 2            # adjacent symbols each side (0 = single-symbol)
HIDDEN_DIMS = [64, 32, 16] # hidden layer sizes
ACTIVATION = "relu"

# Training
BATCH_SIZE = 512
MAX_EPOCHS = 500
LR_INIT = 1e-3
PATIENCE = 50              # early stopping
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Physical constants for BER estimation (16-QAM)
MOD_ORDER = 16


# ── Data Loading ────────────────────────────────────────────────────────────
def load_dataset():
    """Load fiber_dataset.npz and return dicts of X (EDC output) and Y (TX symbols)."""
    data = np.load(DATASET_PATH)
    launch_powers = data["launch_powers"]  # [-4, -2, 0, 2, 4]

    all_x, all_y = {}, {}
    for p in launch_powers:
        key = f"{p:+d}dBm"
        all_x[p] = data[f"X_{key}"]
        all_y[p] = data[f"Y_{key}"]

    return all_x, all_y, list(launch_powers)


def build_features(symbols, memory=MEMORY_TAPS):
    """Convert complex symbol sequence to real-valued feature matrix.

    For memory=M, each row contains 2*(2M+1) features:
      [Re(s_{k-M}), Im(s_{k-M}), ..., Re(s_{k+M}), Im(s_{k+M})]

    Boundary symbols are trimmed (output length = input_length - 2*memory).
    """
    n = len(symbols)
    features = []
    for k in range(memory, n - memory):
        window = symbols[k - memory : k + memory + 1]
        feat = np.column_stack([window.real, window.imag]).ravel()
        features.append(feat)
    return np.array(features, dtype=np.float32)


def build_target(symbols, memory=MEMORY_TAPS):
    """Extract center symbol I/Q as target (2 features per row)."""
    n = len(symbols)
    targets = []
    for k in range(memory, n - memory):
        s = symbols[k]
        targets.append([s.real, s.imag])
    return np.array(targets, dtype=np.float32)


def prepare_data(all_x, all_y, launch_powers, memory=MEMORY_TAPS):
    """Build train/val/test splits for each power level."""
    datasets = {}
    for p in launch_powers:
        x_feat = build_features(all_x[p], memory)
        y_feat = build_target(all_y[p], memory)

        n = len(x_feat)
        idx = np.random.RandomState(RANDOM_SEED).permutation(n)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        x_train = x_feat[idx[:n_train]]
        y_train = y_feat[idx[:n_train]]
        x_val = x_feat[idx[n_train:n_train + n_val]]
        y_val = y_feat[idx[n_train:n_train + n_val]]
        x_test = x_feat[idx[n_train + n_val:]]
        y_test = y_feat[idx[n_train + n_val:]]

        datasets[p] = {
            "train": (x_train, y_train),
            "val": (x_val, y_val),
            "test": (x_test, y_test),
        }
    return datasets


# ── MLP Model ───────────────────────────────────────────────────────────────
class MLPCompensator(nn.Module):
    """Lightweight MLP for per-symbol nonlinearity compensation.

    Input: 2*(2M+1) real-valued features (I/Q of symbol window)
    Output: 2 real values (compensated I/Q)
    """

    def __init__(self, input_dim=2, hidden_dims=None, activation="relu"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        layers = []
        prev = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev, h_dim))
            layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
            prev = h_dim
        layers.append(nn.Linear(prev, 2))  # output: I, Q

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Training ────────────────────────────────────────────────────────────────
def train_one_model(model, train_loader, val_loader, lr=LR_INIT):
    """Train MLP compensator with early stopping.

    Returns trained model, training history dict.
    """
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return model, history


# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_model(model, x_test, y_test):
    """Compute EVM, Q-factor improvement over EDC baseline."""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)
        pred = model(x_t).cpu().numpy()

    # Reconstruct complex symbols
    # Center symbol in x_test: columns [2*memory, 2*memory+1] in the window
    center_pos = MEMORY_TAPS * 2
    rx_edc = x_test[:, center_pos] + 1j * x_test[:, center_pos + 1]
    rx_mlp = pred[:, 0] + 1j * pred[:, 1]
    tx = y_test[:, 0] + 1j * y_test[:, 1]

    # EVM
    evm_edc = np.sqrt(np.mean(np.abs(rx_edc - tx) ** 2))
    evm_mlp = np.sqrt(np.mean(np.abs(rx_mlp - tx) ** 2))

    # Q-factor [dB] = 20*log10(1/EVM)
    q_edc_db = 20 * np.log10(1.0 / evm_edc) if evm_edc > 0 else 100
    q_mlp_db = 20 * np.log10(1.0 / evm_mlp) if evm_mlp > 0 else 100
    q_improvement = q_mlp_db - q_edc_db

    # BER estimation for 16-QAM
    # SNR = 1/EVM^2, BER_16QAM ≈ (3/4)*erfc(sqrt(SNR/10))
    snr_edc = 1.0 / (evm_edc ** 2)
    snr_mlp = 1.0 / (evm_mlp ** 2)
    ber_edc = 0.75 * erfc(np.sqrt(snr_edc / 10.0))
    ber_mlp = 0.75 * erfc(np.sqrt(snr_mlp / 10.0))

    return {
        "evm_edc": evm_edc,
        "evm_mlp": evm_mlp,
        "q_edc_db": q_edc_db,
        "q_mlp_db": q_mlp_db,
        "q_improvement_db": q_improvement,
        "ber_edc": ber_edc,
        "ber_mlp": ber_mlp,
        "rx_edc": rx_edc,
        "rx_mlp": rx_mlp,
        "tx": tx,
    }


def compute_phase_error(rx, tx):
    """Mean absolute phase error in radians between rx and tx symbols."""
    phase_diff = np.angle(rx * np.conj(tx))
    return np.mean(np.abs(phase_diff))


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_training_history(history, power_dbm, filepath):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.semilogy(epochs, history["train_loss"], label="Train", linewidth=1)
    ax1.semilogy(epochs, history["val_loss"], label="Val", linewidth=1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title(f"Training History ({power_dbm:+d} dBm)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["lr"], color="tab:red", linewidth=1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("LR Schedule")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_constellation_comparison(results, power_dbm, filepath):
    """3-panel: EDC output vs MLP output vs Original TX."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    r = results[power_dbm]

    for ax, symbols, title, color in [
        (ax1, r["rx_edc"], "EDC Output", "tab:blue"),
        (ax2, r["rx_mlp"], "MLP Compensated", "tab:orange"),
        (ax3, r["tx"], "Original TX", "tab:green"),
    ]:
        ax.scatter(symbols.real, symbols.imag, s=0.5, alpha=0.6, color=color)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("In-Phase")
        ax.set_ylabel("Quadrature")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle(
        f"16-QAM Constellation Comparison ({power_dbm:+d} dBm)\n"
        f"EVM: EDC={r['evm_edc']:.4f} -> MLP={r['evm_mlp']:.4f}  "
        f"Q-improvement: {r['q_improvement_db']:.2f} dB",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_power_sweep(results, launch_powers, filepath):
    """Q-factor and EVM vs launch power, EDC vs MLP comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    q_edc = [results[p]["q_edc_db"] for p in launch_powers]
    q_mlp = [results[p]["q_mlp_db"] for p in launch_powers]
    q_imp = [results[p]["q_improvement_db"] for p in launch_powers]
    evm_edc = [results[p]["evm_edc"] for p in launch_powers]
    evm_mlp = [results[p]["evm_mlp"] for p in launch_powers]

    ax1.plot(launch_powers, q_edc, "s-", label="EDC", color="tab:blue", markersize=6)
    ax1.plot(launch_powers, q_mlp, "o-", label="MLP", color="tab:orange", markersize=6)
    ax1.set_xlabel("Launch Power [dBm]")
    ax1.set_ylabel("Q-factor [dB]")
    ax1.set_title("Q-factor vs Launch Power")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(launch_powers, evm_edc, "s-", label="EDC", color="tab:blue", markersize=6)
    ax2.plot(launch_powers, evm_mlp, "o-", label="MLP", color="tab:orange", markersize=6)
    ax2.set_xlabel("Launch Power [dBm]")
    ax2.set_ylabel("EVM")
    ax2.set_title("EVM vs Launch Power")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Annotate Q improvement on first subplot
    for i, p in enumerate(launch_powers):
        ax1.annotate(
            f"+{q_imp[i]:.2f} dB",
            (p, q_mlp[i]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=8,
            ha="center",
            color="tab:red",
        )

    fig.suptitle("MLP Nonlinearity Compensation Performance", fontsize=13)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_phase_error(results, launch_powers, filepath):
    """Phase error comparison: EDC vs MLP across power levels."""
    fig, ax = plt.subplots(figsize=(7, 5))

    phase_edc = [
        compute_phase_error(results[p]["rx_edc"], results[p]["tx"])
        for p in launch_powers
    ]
    phase_mlp = [
        compute_phase_error(results[p]["rx_mlp"], results[p]["tx"])
        for p in launch_powers
    ]

    ax.plot(launch_powers, phase_edc, "s-", label="EDC", color="tab:blue", markersize=6)
    ax.plot(launch_powers, phase_mlp, "o-", label="MLP", color="tab:orange", markersize=6)
    ax.set_xlabel("Launch Power [dBm]")
    ax.set_ylabel("Mean Absolute Phase Error [rad]")
    ax.set_title("Phase Error: EDC vs MLP")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" MLP Nonlinearity Compensator — Training (Phase B3)")
    print(f" Device: {DEVICE}")
    print(f" Memory taps: {MEMORY_TAPS}  |  Hidden dims: {HIDDEN_DIMS}")
    print(f" Input dim: {2 * (2 * MEMORY_TAPS + 1)}  |  Output dim: 2")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading dataset...")
    all_x, all_y, launch_powers = load_dataset()
    print(f"  Launch powers: {launch_powers} dBm")
    for p in launch_powers:
        print(f"  {p:+d} dBm: {len(all_x[p])} symbols")

    # Prepare features
    print("\n[2/5] Building feature matrices...")
    datasets = prepare_data(all_x, all_y, launch_powers, MEMORY_TAPS)
    input_dim = 2 * (2 * MEMORY_TAPS + 1)

    # Train per power level
    print("\n[3/5] Training MLP compensators...")
    models = {}
    results = {}
    histories = {}

    for p in launch_powers:
        print(f"\n  --- Training for {p:+d} dBm ---")
        x_train, y_train = datasets[p]["train"]
        x_val, y_val = datasets[p]["val"]
        x_test, y_test = datasets[p]["test"]

        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = MLPCompensator(input_dim=input_dim, hidden_dims=HIDDEN_DIMS)
        print(f"    Model params: {model.num_params:,}")

        t0 = time.time()
        model, history = train_one_model(model, train_loader, val_loader)
        elapsed = time.time() - t0
        print(f"    Training time: {elapsed:.1f}s")

        # Evaluate
        eval_result = evaluate_model(model, x_test, y_test)
        print(
            f"    EVM:  EDC={eval_result['evm_edc']:.4f}  ->  "
            f"MLP={eval_result['evm_mlp']:.4f}  "
            f"(reduction: {(1 - eval_result['evm_mlp'] / eval_result['evm_edc']) * 100:.1f}%)"
        )
        print(
            f"    Q-factor:  EDC={eval_result['q_edc_db']:.2f} dB  ->  "
            f"MLP={eval_result['q_mlp_db']:.2f} dB  "
            f"(+{eval_result['q_improvement_db']:.2f} dB)"
        )
        print(
            f"    BER estimate:  EDC={eval_result['ber_edc']:.2e}  ->  "
            f"MLP={eval_result['ber_mlp']:.2e}"
        )

        models[p] = model
        results[p] = eval_result
        histories[p] = history

    # Summary table
    print("\n" + "-" * 70)
    print(f" {'Power':^8s} {'EVM EDC':^10s} {'EVM MLP':^10s} "
          f"{'Q EDC':^8s} {'Q MLP':^8s} {'Q Gain':^8s} {'BER EDC':^10s} {'BER MLP':^10s}")
    print("-" * 70)
    for p in launch_powers:
        r = results[p]
        print(f" {p:+4d} dBm  {r['evm_edc']:^10.4f} {r['evm_mlp']:^10.4f} "
              f"{r['q_edc_db']:^8.2f} {r['q_mlp_db']:^8.2f} "
              f"{r['q_improvement_db']:^+8.2f} "
              f"{r['ber_edc']:^10.2e} {r['ber_mlp']:^10.2e}")
    print("-" * 70)

    # Save model
    print("\n[4/5] Saving model and results...")
    torch.save(
        {
            "models": {p: m.state_dict() for p, m in models.items()},
            "config": {
                "memory_taps": MEMORY_TAPS,
                "hidden_dims": HIDDEN_DIMS,
                "input_dim": input_dim,
                "launch_powers": launch_powers,
            },
            "results": {p: {k: v for k, v in r.items()
                            if k not in ("rx_edc", "rx_mlp", "tx")}
                         for p, r in results.items()},
        },
        OUT_DIR / "mlp_nlc_model.pt",
    )
    print(f"  Model saved -> mlp_nlc_model.pt")

    # Save CSV
    csv_path = OUT_DIR / "mlp_results.csv"
    with open(csv_path, "w") as f:
        f.write("power_dbm,evm_edc,evm_mlp,q_edc_db,q_mlp_db,q_improvement_db,ber_edc,ber_mlp\n")
        for p in launch_powers:
            r = results[p]
            f.write(f"{p},{r['evm_edc']},{r['evm_mlp']},{r['q_edc_db']},"
                    f"{r['q_mlp_db']},{r['q_improvement_db']},{r['ber_edc']},{r['ber_mlp']}\n")
    print(f"  Results saved -> mlp_results.csv")

    # Plot
    print("\n[5/5] Generating plots...")
    for p in launch_powers:
        plot_training_history(histories[p], p,
                              OUT_DIR / f"mlp_training_{p:+d}dBm.png")
        plot_constellation_comparison(results, p,
                                      OUT_DIR / f"mlp_constellation_{p:+d}dBm.png")
    plot_power_sweep(results, launch_powers, OUT_DIR / "mlp_power_sweep.png")
    plot_phase_error(results, launch_powers, OUT_DIR / "mlp_phase_error.png")
    print("  All plots generated.")

    print("\n" + "=" * 60)
    print(" Phase B3 complete — MLP compensator trained and evaluated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
