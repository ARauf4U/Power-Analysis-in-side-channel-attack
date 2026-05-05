"""
cpa_attack.py
=============
Full Correlation Power Analysis demonstration.
Generates 7 publication-quality result figures saved to ../results/

Fixes vs original:
  - TIME_PTS / LEAK_TIME updated so all 16 byte leak points fit in trace.
  - fig_trace_count_effect: uses correct multi-byte trace generation.
  - fig_full_key_recovery: attacks all 16 bytes correctly (16/16 now).
  - fig_countermeasures: uses correct time_points.
  - All calls to simulate_power_traces now use consistent parameters.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from aes_sim import (
    SBOX, hamming_weight, subbytes_output,
    simulate_power_traces, correlation_power_analysis,
    leak_time_for_byte, BYTE_LEAK_SPACING,
)

# ── Palette ────────────────────────────────────────────────────────────────
DARK_BG  = "#0D1117"
PANEL_BG = "#161B22"
ACCENT1  = "#58A6FF"
ACCENT2  = "#F78166"
ACCENT3  = "#3FB950"
TEXT_CLR = "#E6EDF3"
MUTED    = "#8B949E"

plt.rcParams.update({
    "figure.facecolor" : DARK_BG,
    "axes.facecolor"   : PANEL_BG,
    "axes.edgecolor"   : MUTED,
    "axes.labelcolor"  : TEXT_CLR,
    "xtick.color"      : MUTED,
    "ytick.color"      : MUTED,
    "text.color"       : TEXT_CLR,
    "grid.color"       : "#21262D",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "font.family"      : "DejaVu Sans",
})

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Simulation parameters ──────────────────────────────────────────────────
BASE_LEAK  = 5                                      # byte-0 leaks at t=5
#   byte b leaks at t = BASE_LEAK + b * BYTE_LEAK_SPACING
#   byte 15 leaks at t = 5 + 15*3 = 50  → need TIME_PTS > 50
TIME_PTS   = 70    # safe margin (> 50)
BYTE_IDX   = 0     # which byte to focus on for single-byte demos
FOCUS_LEAK = leak_time_for_byte(BYTE_IDX, BASE_LEAK)  # = 5

N_TRACES   = 1000
RNG        = np.random.default_rng(2024)
TRUE_KEY   = RNG.integers(0, 256, 16, dtype=np.uint8)
PLAINTEXTS = RNG.integers(0, 256, (N_TRACES, 16), dtype=np.uint8)

print(f"True key byte {BYTE_IDX}: 0x{TRUE_KEY[BYTE_IDX]:02X}  ({TRUE_KEY[BYTE_IDX]})")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Power Traces Overview
# ══════════════════════════════════════════════════════════════════════════════

def fig_power_traces():
    traces_clean = simulate_power_traces(
        PLAINTEXTS, TRUE_KEY, noise_std=0.3, time_points=TIME_PTS, leak_time=BASE_LEAK)
    traces_noisy = simulate_power_traces(
        PLAINTEXTS, TRUE_KEY, noise_std=1.2, time_points=TIME_PTS, leak_time=BASE_LEAK)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)
    fig.suptitle("Simulated AES Power Traces", fontsize=18, fontweight='bold',
                 color=TEXT_CLR, y=0.98)

    for ax, traces, title in zip(
        axes,
        [traces_clean, traces_noisy],
        ["Clean Traces  (σ = 0.3)", "Noisy Traces  (σ = 1.2)  — real-world scenario"],
    ):
        for i in range(20):
            ax.plot(traces[i], color=ACCENT1, alpha=0.12, linewidth=0.7)
        ax.plot(traces.mean(axis=0), color=ACCENT3, linewidth=2.2,
                label="Mean trace", zorder=5)
        ax.axvline(FOCUS_LEAK, color=ACCENT2, linewidth=2, linestyle='--', alpha=0.9)
        ax.text(FOCUS_LEAK + 1, ax.get_ylim()[1] * 0.8,
                f"← Byte-0\n   leak (t={FOCUS_LEAK})",
                color=ACCENT2, fontsize=9, va='top')
        ax.set_title(title, color=TEXT_CLR, fontsize=12, pad=8)
        ax.set_xlabel("Time Sample", fontsize=10)
        ax.set_ylabel("Power (a.u.)", fontsize=10)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.3)
        ax.grid(True)

    plt.tight_layout()
    out = os.path.join(RESULTS, "01_power_traces.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")
    return traces_noisy


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Hamming Weight Distribution
# ══════════════════════════════════════════════════════════════════════════════

def fig_hamming_weight():
    k = int(TRUE_KEY[BYTE_IDX])
    hw_vals = [hamming_weight(subbytes_output(p, k)) for p in range(256)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG)
    fig.suptitle("Hamming Weight Leakage Model", fontsize=16,
                 fontweight='bold', color=TEXT_CLR)

    ax = axes[0]
    counts = np.bincount(hw_vals, minlength=9)
    bars = ax.bar(range(9), counts, color=ACCENT1, alpha=0.85,
                  edgecolor=DARK_BG, linewidth=0.8)
    ax.bar_label(bars, fmt='%d', color=TEXT_CLR, fontsize=9, padding=2)
    ax.set_title("HW Distribution of SubBytes Output", color=TEXT_CLR, fontsize=11)
    ax.set_xlabel("Hamming Weight (# of 1-bits)", fontsize=10)
    ax.set_ylabel("Count (over 256 plaintexts)", fontsize=10)
    ax.set_xticks(range(9))
    ax.grid(True, axis='y')

    ax2 = axes[1]
    sbox_array = np.array(SBOX, dtype=np.uint8).reshape(16, 16)
    hw_sbox = np.vectorize(hamming_weight)(sbox_array.astype(int))
    im = ax2.imshow(hw_sbox, cmap='plasma', aspect='auto', vmin=0, vmax=8)
    ax2.set_title("S-Box Hamming Weight Map", color=TEXT_CLR, fontsize=11)
    ax2.set_xlabel("Low nibble of input byte", fontsize=9)
    ax2.set_ylabel("High nibble of input byte", fontsize=9)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    cbar.set_label("HW of S-Box output", color=TEXT_CLR)

    plt.tight_layout()
    out = os.path.join(RESULTS, "02_hamming_weight.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – CPA Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_cpa_heatmap(traces_noisy):
    corr_matrix, best_guess = correlation_power_analysis(
        PLAINTEXTS, traces_noisy, BYTE_IDX)
    recovered = int(np.argmax(best_guess))
    print(f"  CPA recovered key byte {BYTE_IDX}: 0x{recovered:02X}  "
          f"({'CORRECT' if recovered == TRUE_KEY[BYTE_IDX] else 'WRONG'})")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=DARK_BG,
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"Correlation Power Analysis — Key Byte {BYTE_IDX}",
                 fontsize=16, fontweight='bold', color=TEXT_CLR)

    ax = axes[0]
    im = ax.imshow(np.abs(corr_matrix), aspect='auto', cmap='inferno',
                   origin='lower', vmin=0, vmax=0.6)
    ax.axhline(TRUE_KEY[BYTE_IDX], color=ACCENT3,
               linewidth=1.5, linestyle='--', alpha=0.9)
    ax.text(TIME_PTS + 0.5, TRUE_KEY[BYTE_IDX],
            f" k=0x{TRUE_KEY[BYTE_IDX]:02X}", color=ACCENT3, fontsize=8, va='center')
    ax.set_title("|Pearson Correlation|  ×  (Key Hypothesis, Time)",
                 color=TEXT_CLR, fontsize=11)
    ax.set_xlabel("Time Sample", fontsize=10)
    ax.set_ylabel("Key Hypothesis (0–255)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("|Correlation|", color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)

    ax2 = axes[1]
    colors = [ACCENT2] * 256
    colors[TRUE_KEY[BYTE_IDX]] = ACCENT3
    if recovered != TRUE_KEY[BYTE_IDX]:
        colors[recovered] = ACCENT1
    ax2.barh(range(256), best_guess, color=colors, height=1.0)
    ax2.axhline(TRUE_KEY[BYTE_IDX], color=ACCENT3, linewidth=1.5, linestyle='--')
    ax2.set_title("Max |Correlation|\nper Hypothesis", color=TEXT_CLR, fontsize=10)
    ax2.set_xlabel("|Correlation|", fontsize=9)
    ax2.set_ylabel("Key Hypothesis", fontsize=9)
    ax2.set_ylim(-1, 256)

    legend_handles = [
        plt.Line2D([0],[0], color=ACCENT3, lw=3,
                   label=f"True key 0x{TRUE_KEY[BYTE_IDX]:02X}"),
        plt.Line2D([0],[0], color=ACCENT1, lw=3,
                   label=f"Recovered 0x{recovered:02X}"),
        plt.Line2D([0],[0], color=ACCENT2, lw=3, label="Other hypotheses"),
    ]
    ax2.legend(handles=legend_handles, fontsize=8, loc='lower right', framealpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS, "03_cpa_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")
    return corr_matrix, best_guess, recovered


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Key Recovery: Best Correlation over Time
# ══════════════════════════════════════════════════════════════════════════════

def fig_key_recovery(corr_matrix, best_guess):
    true_k = int(TRUE_KEY[BYTE_IDX])
    t = np.arange(TIME_PTS)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor=DARK_BG)
    fig.suptitle("Key Recovery Process — CPA Step by Step",
                 fontsize=16, fontweight='bold', color=TEXT_CLR)

    ax = axes[0]
    wrong = [k for k in range(256) if k != true_k]
    rng2 = np.random.default_rng(7)
    sample_wrong = rng2.choice(wrong, 30, replace=False)
    for k_w in sample_wrong:
        ax.plot(t, corr_matrix[k_w], color=ACCENT2, alpha=0.15, linewidth=0.7)
    ax.plot(t, corr_matrix[true_k], color=ACCENT3, linewidth=2.5,
            label=f"True key  k=0x{true_k:02X}", zorder=10)
    ax.axvline(FOCUS_LEAK, color=ACCENT1, linewidth=1.5, linestyle=':', alpha=0.8)
    ax.text(FOCUS_LEAK + 0.5, 0.4, f"Leak\nt={FOCUS_LEAK}",
            color=ACCENT1, fontsize=9)
    ax.set_title("Correlation Traces  (True key highlighted in green)",
                 color=TEXT_CLR)
    ax.set_xlabel("Time Sample", fontsize=10)
    ax.set_ylabel("Pearson Correlation", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True)

    ax2 = axes[1]
    sorted_idx = np.argsort(best_guess)[::-1][:20]
    bar_colors = [ACCENT3 if idx == true_k else
                  ACCENT1 if i == 0 else ACCENT2
                  for i, idx in enumerate(sorted_idx)]
    x_labels = [f"0x{idx:02X}" for idx in sorted_idx]
    ax2.bar(range(20), best_guess[sorted_idx], color=bar_colors,
            edgecolor=DARK_BG, linewidth=0.6)
    ax2.set_xticks(range(20))
    ax2.set_xticklabels(x_labels, fontsize=8, rotation=45)
    ax2.set_title("Top-20 Key Hypotheses by Maximum Correlation", color=TEXT_CLR)
    ax2.set_xlabel("Key Hypothesis (hex)", fontsize=10)
    ax2.set_ylabel("Max |Correlation|", fontsize=10)
    ax2.grid(True, axis='y')

    legend_handles = [
        plt.Rectangle((0,0),1,1, color=ACCENT3,
                       label=f"True key 0x{true_k:02X}"),
        plt.Rectangle((0,0),1,1, color=ACCENT1, label="Best guess"),
        plt.Rectangle((0,0),1,1, color=ACCENT2, label="Other candidates"),
    ]
    ax2.legend(handles=legend_handles, fontsize=9, framealpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS, "04_key_recovery.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Effect of Trace Count on Attack Success
# ══════════════════════════════════════════════════════════════════════════════

def fig_trace_count_effect():
    trace_counts = [10, 25, 50, 100, 200, 400, 600, 800, 1000]
    RUNS = 15

    traces_all = simulate_power_traces(
        PLAINTEXTS, TRUE_KEY, noise_std=1.2,
        time_points=TIME_PTS, leak_time=BASE_LEAK)

    success_counts = []
    for n in trace_counts:
        successes = 0
        for _ in range(RUNS):
            idx = RNG.choice(N_TRACES, n, replace=False)
            _, bg = correlation_power_analysis(
                PLAINTEXTS[idx], traces_all[idx], BYTE_IDX)
            if np.argmax(bg) == TRUE_KEY[BYTE_IDX]:
                successes += 1
        pct = successes / RUNS * 100
        success_counts.append(pct)
        print(f"  N={n:5d}  success={successes}/{RUNS}  ({pct:.0f}%)")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.plot(trace_counts, success_counts, 'o-', color=ACCENT1, linewidth=2.5,
            markersize=8, markerfacecolor=ACCENT3)
    ax.fill_between(trace_counts, success_counts, alpha=0.12, color=ACCENT1)
    ax.axhline(100, color=ACCENT3, linewidth=1, linestyle='--',
               alpha=0.5, label="100% success")
    ax.set_title(
        "Attack Success Rate vs Number of Power Traces\n"
        "(σ = 1.2,  15 Monte Carlo runs per point)",
        color=TEXT_CLR, fontsize=13)
    ax.set_xlabel("Number of Traces (N)", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(-5, 110)
    ax.grid(True)
    ax.legend(fontsize=10, framealpha=0.3)

    for x, y in zip(trace_counts, success_counts):
        ax.annotate(f"{y:.0f}%", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8, color=MUTED)

    plt.tight_layout()
    out = os.path.join(RESULTS, "05_trace_count_effect.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Full 16-Byte Key Recovery  [THE FIXED VERSION]
# ══════════════════════════════════════════════════════════════════════════════

def fig_full_key_recovery():
    # Use a fresh set of plaintexts and a dedicated trace set
    rng6 = np.random.default_rng(999)
    pts6 = rng6.integers(0, 256, (500, 16), dtype=np.uint8)
    key6 = rng6.integers(0, 256, 16, dtype=np.uint8)

    # Generate traces with ALL 16 byte leaks embedded
    traces6 = simulate_power_traces(
        pts6, key6, noise_std=0.9,
        time_points=TIME_PTS, leak_time=BASE_LEAK)

    recovered_key = np.zeros(16, dtype=int)
    for b in range(16):
        _, bg = correlation_power_analysis(pts6, traces6, byte_idx=b)
        recovered_key[b] = int(np.argmax(bg))

    match = recovered_key == key6.astype(int)
    print(f"  Full key recovery: {match.sum()}/16 bytes correct")

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    x = np.arange(16)
    width = 0.35
    ax.bar(x - width/2, key6, width, label='True Key',
           color=ACCENT3, alpha=0.85)

    bar_colors = [ACCENT1 if m else ACCENT2 for m in match]
    ax.bar(x + width/2, recovered_key, width, label='Recovered Key',
           color=bar_colors, alpha=0.85)

    ax.set_title(
        f"Full 16-Byte AES Key: True vs Recovered  "
        f"({match.sum()}/16 correct)",
        color=TEXT_CLR, fontsize=14, fontweight='bold')
    ax.set_xlabel("Key Byte Index", fontsize=11)
    ax.set_ylabel("Byte Value (0–255)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{i}" for i in range(16)])
    ax.legend(fontsize=10, framealpha=0.3)
    ax.grid(True, axis='y')

    # Hex labels
    for i in range(16):
        ax.text(i - width/2, key6[i] + 4, f"0x{key6[i]:02X}",
                ha='center', fontsize=7, color=ACCENT3, rotation=45)
        clr = ACCENT1 if match[i] else ACCENT2
        ax.text(i + width/2, recovered_key[i] + 4, f"0x{recovered_key[i]:02X}",
                ha='center', fontsize=7, color=clr, rotation=45)

    plt.tight_layout()
    out = os.path.join(RESULTS, "06_full_key_recovery.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")
    return recovered_key, match


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Countermeasures
# ══════════════════════════════════════════════════════════════════════════════

def fig_countermeasures():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle("Countermeasures: Effect on Power Trace Leakage",
                 fontsize=15, fontweight='bold', color=TEXT_CLR)

    rng7 = np.random.default_rng(5)

    # Panel 1: no countermeasure
    ax = axes[0]
    tr = simulate_power_traces(
        PLAINTEXTS[:50], TRUE_KEY, noise_std=0.3,
        time_points=TIME_PTS, leak_time=BASE_LEAK)
    for i in range(20):
        ax.plot(tr[i], color=ACCENT1, alpha=0.2, linewidth=0.6)
    ax.plot(tr.mean(axis=0), color=ACCENT3, linewidth=2, label="Mean")
    ax.axvline(FOCUS_LEAK, color=ACCENT2, linewidth=1.5,
               linestyle='--', alpha=0.8, label="Leak point")
    ax.set_title("No Countermeasure\n(σ=0.3)", color=TEXT_CLR, fontsize=10)
    ax.set_xlabel("Time Sample", fontsize=9)
    ax.set_ylabel("Power (a.u.)", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True)

    # Panel 2: added noise (hiding)
    ax = axes[1]
    tr2 = simulate_power_traces(
        PLAINTEXTS[:50], TRUE_KEY, noise_std=2.0,
        time_points=TIME_PTS, leak_time=BASE_LEAK)
    for i in range(20):
        ax.plot(tr2[i], color=ACCENT1, alpha=0.2, linewidth=0.6)
    ax.plot(tr2.mean(axis=0), color=ACCENT3, linewidth=2, label="Mean")
    ax.axvline(FOCUS_LEAK, color=ACCENT2, linewidth=1.5,
               linestyle='--', alpha=0.8, label="Leak point")
    ax.set_title("Added Noise\n(σ=2.0  — hiding)", color=TEXT_CLR, fontsize=10)
    ax.set_xlabel("Time Sample", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True)

    # Panel 3: random delay jitter
    ax = axes[2]
    jitter_pts = TIME_PTS + 20
    jitter_traces = rng7.normal(0, 0.3, size=(50, jitter_pts))
    for i in range(50):
        j = rng7.integers(0, 20)
        lt = FOCUS_LEAK + j
        if lt < jitter_pts:
            hw = hamming_weight(
                subbytes_output(int(PLAINTEXTS[i, 0]), int(TRUE_KEY[0])))
            jitter_traces[i, lt] += hw
    for i in range(20):
        ax.plot(jitter_traces[i], color=ACCENT1, alpha=0.2, linewidth=0.6)
    ax.plot(jitter_traces.mean(axis=0), color=ACCENT3, linewidth=2, label="Mean")
    ax.set_title("Random Clock Jitter\n(temporal hiding)", color=TEXT_CLR, fontsize=10)
    ax.set_xlabel("Time Sample", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True)

    plt.tight_layout()
    out = os.path.join(RESULTS, "07_countermeasures.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n=== Power Analysis Side-Channel Attack Simulation ===\n")
    print("Generating figures...\n")

    traces_noisy = fig_power_traces()
    fig_hamming_weight()
    corr_matrix, best_guess, recovered = fig_cpa_heatmap(traces_noisy)
    fig_key_recovery(corr_matrix, best_guess)
    fig_trace_count_effect()
    recovered_key, match = fig_full_key_recovery()
    fig_countermeasures()

    print("\n=== Summary ===")
    print(f"True key      : {' '.join(f'0x{b:02X}' for b in TRUE_KEY)}")
    print(f"Recovered key : {' '.join(f'0x{b:02X}' for b in recovered_key)}")
    print(f"Bytes correct : {match.sum()}/16")
    print(f"\nAll figures saved to: {os.path.abspath(RESULTS)}")
