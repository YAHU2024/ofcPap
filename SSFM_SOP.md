# SSFM Fiber Transmission Simulator — Development SOP

## Overview

This document records the core logic, code architecture, and pitfalls encountered while building a Split-Step Fourier Method (SSFM) simulator for single-channel 16-QAM fiber transmission. The simulator generates nonlinearly impaired signals for training a lightweight MLP nonlinearity compensator.

---

## 1. Core Physics & Key Formulae

### 1.1 Nonlinear Schrodinger Equation (NLSE)

The pulse evolution in single-mode fiber is governed by:

$$\frac{\partial A}{\partial z} = -\frac{\alpha}{2}A - j\frac{\beta_2}{2}\frac{\partial^2 A}{\partial t^2} + j\gamma|A|^2 A$$

where:
- $\alpha$ — field attenuation coefficient (**NOT** power coefficient)
- $\beta_2$ — group-velocity dispersion (GVD) parameter
- $\gamma$ — Kerr nonlinearity coefficient

### 1.2 Symmetric Split-Step Fourier Method (SSFM)

Each step of length $dz$ applies operators in the order:

$$A(z + dz) \approx \underbrace{\exp\left(j\gamma|A|^2 \cdot \frac{dz}{2}\right)}_{\text{1/2 Nonlinear}} \cdot \underbrace{\mathcal{F}^{-1}\left\{\exp\left(-j\frac{\beta_2}{2}\omega^2 \cdot dz\right) \cdot \mathcal{F}\{A\}\right\}}_{\text{Full Dispersion } D(dz)} \cdot \underbrace{\exp\left(j\gamma|A|^2 \cdot \frac{dz}{2}\right)}_{\text{1/2 Nonlinear}} \cdot \underbrace{\exp\left(-\frac{\alpha}{2}dz\right)}_{\text{Field attenuation}}$$

### 1.3 EDFA Amplification & ASE Noise

After each span, the EDFA compensates span loss and adds amplified spontaneous emission (ASE):

$$G = e^{\alpha L_{span}} \quad\text{(linear gain)}$$

$$\sigma_{ASE}^2 = N_{sp} \cdot h \cdot \nu \cdot (G - 1) \cdot \frac{1}{dt} \quad\text{(noise variance per sample, single pol.)}$$

$$A_{out} = A_{in} \cdot \sqrt{G} + n_{ASE}, \quad n_{ASE} \sim \mathcal{CN}(0, \sigma_{ASE}^2)$$

### 1.4 Electronic Dispersion Compensation (EDC)

Frequency-domain all-pass filter that inverts the linear dispersive channel:

$$H_{EDC}(\omega) = \exp\left(+j\frac{\beta_2}{2}\omega^2 \cdot L_{total}\right)$$

where $L_{total} = N_{spans} \times L_{span}$.

### 1.5 16-QAM Gray Mapping

Constellation points: $\{\pm1 \pm 1j, \pm1 \pm 3j, \pm3 \pm 1j, \pm3 \pm 3j\} / \sqrt{10}$

Normalization factor $1/\sqrt{10}$ ensures $E[|s|^2] = 1$.

---

## 2. Code Architecture

```
ssfm_simulator.py
├── Physical Constants & Parameters   (lines 16-46)
│   ├── CENTER_WAVELENGTH, CENTER_FREQ
│   ├── MOD_ORDER=16, SYMBOL_RATE=32e9, SPS=4, ROLL_OFF=0.1
│   ├── SPAN_LENGTH=80e3, N_SPANS=10
│   ├── ALPHA_LIN (per-meter, from ALPHA_DB=0.2 dB/km)
│   ├── BETA2 (from D=17 ps/nm/km @1550nm)
│   ├── GAMMA=1.3e-3 /W/m
│   └── SSFM_STEPS_PER_SPAN=500, N_SYMBOLS=16384
│
├── Constellation Generation
│   ├── gray_code_4bit()          → Gray mapping table [0,1,3,2,4,5,7,6,12,13,15,14,8,9,11,10]
│   └── generate_16qam_symbols()  → Random symbols, unit power, Gray mapped
│
├── Pulse Shaping
│   ├── rrc_filter()              → RRC filter design with singularity handling
│   ├── pulse_shaping()           → Upsample + "same"-mode RRC convolution
│   └── matched_filter()          → "same"-mode RRC + downsample at offset=0
│
├── Optical Channel
│   ├── ssfm_propagation()        → Symmetric SSFM over ONE span (dz adaptively computed)
│   └── add_ase_noise()           → EDFA amplify + ASE noise injection
│
├── Digital Signal Processing
│   └── edc_equalization()        → Frequency-domain dispersion compensation (full link)
│
├── Visualization
│   └── plot_constellation()      → Side-by-side: EDC output vs original TX
│
└── main()
    ├── for each launch_power in [-4, -2, 0, 2, 4] dBm:
    │   1. Scale tx_signal to target power (dBm → W conversion)
    │   2. Loop over N_SPANS: ssfm_propagation() → add_ase_noise()
    │   3. edc_equalization() on total received signal
    │   4. matched_filter() + downsample to symbols
    │   5. Power normalization: rx /= sqrt(mean(|rx|^2))
    │   6. Store as dataset entry
    ├── Save all data to fiber_dataset.npz
    └── Generate constellation plots (single + overview)
```

---

## 3. Pitfalls & Fixes (Critical)

### Pit 1: Unit Mismatch — ALPHA_LIN per-km vs per-meter

**Symptom:** `exp(alpha_lin * SPAN_LENGTH)` overflowed to infinity (SPAN_LENGTH=80000 m, ALPHA_LIN was per-km).

**Root cause:** `ALPHA_LIN` was computed as `ALPHA_DB * log(10)/10` (per-km), but `SPAN_LENGTH` is in meters.

**Fix:** Always use per-meter for internal calculations:
```python
ALPHA_LIN = ALPHA_DB * np.log(10) / 10 / 1e3  # 1/m
```

---

### Pit 2: SSFM Dispersion Step — D(dz/2) vs D(dz)

**Symptom:** Symmetric scheme used `exp(-j*beta2/2 * omega^2 * dz/2)`, providing only half the required dispersion per step.

**Root cause:** Confusion between the symmetric scheme notation and the D-operator definition. The symmetric scheme places 1/2 N at both ends and a **full** D(dz) in the middle, not D(dz/2).

**Fix:**
```python
# WRONG:
d_half = np.exp(-1j * BETA2 / 2 * omega ** 2 * dz / 2)
# CORRECT:
d_step = np.exp(-1j * BETA2 / 2 * omega ** 2 * dz_actual)  # Full D(dz)
```

---

### Pit 3: Field Attenuation — α vs α/2

**Symptom:** Signal power attenuated at double the correct rate (at one point in debugging).

**Root cause:** The NLSE describes **field** evolution, and field decays as `exp(-α·z/2)`. Using `exp(-α·z)` would apply the power attenuation to the field, effectively doubling it.

**Fix:**
```python
A *= np.exp(-ALPHA_LIN * dz_actual / 2)  # field attenuation = alpha_lin / 2
```

---

### Pit 4: RRC Filter NaN at Singularities

**Symptom:** `NaN` values in RRC filter coefficients.

**Root cause:** The RRC time-domain formula has divide-by-zero at `t=0` and `t=±1/(4β)`. Using `np.where()` nested conditions does not short-circuit — all branches are evaluated.

**Fix:** Use explicit boolean indexing with three separate masks:
```python
idx0 = np.abs(t) < 1e-12          # t = 0
idx_sing = np.abs(np.abs(t) - 1/(4*ROLL_OFF)) < 1e-12  # t = +/-1/(4β)
idx_normal = (~idx0) & (~idx_sing)  # everything else
```
Compute each case separately, never relying on short-circuit evaluation.

---

### Pit 5: EDFA Missing Signal Amplification

**Symptom:** `add_ase_noise()` only injected noise without re-amplifying the attenuated signal.

**Root cause:** After SSFM, the signal has undergone span attenuation `exp(-α·L_span/2)`. The EDFA must compensate this loss BEFORE adding noise.

**Fix:**
```python
g_lin = np.exp(ALPHA_LIN * SPAN_LENGTH)  # power gain
signal = signal * np.sqrt(g_lin)           # amplify FIRST
# then add ASE noise...
```

---

### Pit 6: Symbol Timing Misalignment (Convolution Mode)

**Symptom:** After TX pulse shaping → channel → RX matched filter, EVM was catastrophically high (~0.8) even in back-to-back (no fiber).

**Root cause:** Convolution mode and downsampling offset disagreed. "Full" mode convolution adds a delay of `N_filter - 1` samples; "same" mode adds a delay of `(N_filter - 1) / 2`. The downsampling offset must compensate for this delay.

**Fix:** Use "same" mode convolution at **both** TX and RX. This produces zero net timing delay:
```python
# TX:
shaped = np.convolve(upsampled, h_rrc, mode="same")
# RX:
filtered = np.convolve(rx_signal, h_rrc, mode="same")
downsampled = filtered[0::SPS]  # offset = 0, NOT (N_rrc-1)/2
```
Back-to-back EVM verified: 0.0013 (near-perfect).

---

### Pit 7: V&V Phase Estimator Destroys Noisy 16-QAM

**Symptom:** Adding a 4th-power V&V phase estimator caused EVM to jump from 0.17 to 0.77.

**Root cause:** The V&V algorithm estimates carrier phase as `0.25 * arg(E[X^4])`. For 16-QAM with high ASE noise, only the 4 corner symbols (angle ±45°, ±135°) contribute to X^4, and noise causes large estimation variance. The corrected phase was often wrong by 30°–60°.

**Fix:** Removed the phase correction entirely. In this simulation, there is no deliberate carrier phase rotation (direct-detection equivalent with perfect LO), so phase correction is unnecessary and harmful.

---

### Pit 8: UnicodeEncodeError in Chinese Windows Terminal (GBK)

**Symptom:** `UnicodeEncodeError` when printing Greek letters (β₂, α, γ) and special characters.

**Fix:** Replace all non-ASCII characters in print/log statements with ASCII equivalents:
- `β₂` → `beta2`
- `α` → `alpha`
- `γ` → `gamma`
- `→` → `->`
- `─` → `----`
- `×` → `x`

---

### Pit 9: pip Proxy Blocking PyPI Mirrors

**Symptom:** `pip install matplotlib` failed due to corporate proxy blocking standard PyPI mirrors.

**Fix:**
```bash
pip install matplotlib --index-url https://pypi.org/simple/ --trusted-host pypi.org
```

---

## 4. Validation Checklist

After running `ssfm_simulator.py`, verify:

| Check | Expected Result |
|-------|----------------|
| Script runs without errors | Clean exit with "Simulation complete." |
| `fiber_dataset.npz` generated | ~1.3 MB, contains X_{p} and Y_{p} for 5 power levels |
| `initial_test.png` generated | 0 dBm, side-by-side EDC vs original |
| `power_sweep_overview.png` generated | 2×5 grid for all powers |
| EVM vs power trend | U-shaped: best at -2 dBm, worst at +4 dBm |
| Phase error trend | Monotonically increases with power (SPM signature) |
| Low power (-4 dBm) constellation | Near-ideal, ASE-dominated noise cloud |
| High power (+4 dBm) constellation | Nonlinear distortion: outer points spread, phase rotation visible |

---

## 5. Key Design Decisions

1. **Same-mode convolution at both TX and RX** — Simplest timing alignment with zero net delay. The alternative ("full" mode + explicit delay compensation) is more complex and error-prone.

2. **Symmetric SSFM over asymmetric** — Better accuracy for same step count (O(dz³) local error vs O(dz²)).

3. **No carrier phase recovery** — Simulation assumes ideal LO with no frequency/phase offset. Adding a V&V estimator for 16-QAM is unreliable at low SNR and unnecessary for this scenario.

4. **Adaptive dz** — `dz_actual = L_span / floor(L_span / dz_nominal)` ensures exact integer steps per span, avoiding accumulation of truncation error.

5. **Generate symbols once, reuse across power levels** — Ensures fair comparison: the same bit sequence experiences different nonlinear regimes.
