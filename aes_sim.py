"""
aes_sim.py
==========
Pure-Python AES-128 simulation used by the Power Analysis project.

Key fixes vs original:
  - simulate_power_traces() now injects leakage for ALL 16 key bytes,
    each at its own dedicated time sample (base_leak_time + byte_idx * 3).
  - Type hints use typing.Tuple for Python 3.8+ compatibility.
  - rng_seed works correctly across all callers.
  - Validates that time_points is large enough.

Reference: FIPS 197
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

# ---------------------------------------------------------------------------
# AES S-Box (FIPS 197, Figure 7)
# ---------------------------------------------------------------------------
SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]

BYTE_LEAK_SPACING = 3   # time samples between consecutive byte leak points


def hamming_weight(x: int) -> int:
    """Count the number of set bits (population count)."""
    return bin(x).count('1')


def hamming_distance(x: int, y: int) -> int:
    """Number of bit positions that differ between x and y."""
    return hamming_weight(x ^ y)


def subbytes_output(plaintext_byte: int, key_byte: int) -> int:
    """Return AES SubBytes( plaintext_byte XOR key_byte ) — the CPA target."""
    return SBOX[plaintext_byte ^ key_byte]


def leak_time_for_byte(byte_idx: int, base_leak_time: int = 5) -> int:
    """Time-sample index at which byte <byte_idx> leaks power."""
    return base_leak_time + byte_idx * BYTE_LEAK_SPACING


# ---------------------------------------------------------------------------
# Power trace simulation — all 16 bytes, each at its own time slot
# ---------------------------------------------------------------------------

def simulate_power_traces(
    plaintexts: np.ndarray,
    true_key: np.ndarray,
    noise_std: float = 0.5,
    time_points: int = 100,
    leak_time: int = 5,
    rng_seed: int = 42,
) -> np.ndarray:
    """
    Generate N synthetic power traces.

    Each trace has Gaussian noise everywhere.
    At time sample leak_time_for_byte(b), the trace also receives:
        HW( SubBytes( plaintext[b] XOR key[b] ) )
    for every byte b in 0..15.

    Parameters
    ----------
    plaintexts  : (N, 16) uint8 array of random plaintexts
    true_key    : (16,)   uint8 array — the secret key
    noise_std   : Gaussian noise standard deviation
    time_points : number of time samples per trace
    leak_time   : base leak time for byte 0
    rng_seed    : reproducibility seed

    Returns
    -------
    traces : (N, time_points) float64 array
    """
    # Validate that all 16 leak points fit within the trace length
    max_leak = leak_time_for_byte(15, base_leak_time=leak_time)
    if max_leak >= time_points:
        raise ValueError(
            f"time_points ({time_points}) is too small. "
            f"Byte 15 leaks at t={max_leak}. "
            f"Set time_points >= {max_leak + 1}."
        )

    rng = np.random.default_rng(rng_seed)
    N = plaintexts.shape[0]
    traces = rng.normal(0.0, noise_std, size=(N, time_points))

    # Inject leakage for all 16 key bytes
    for b in range(16):
        lt = leak_time_for_byte(b, base_leak_time=leak_time)
        for i in range(N):
            hw = hamming_weight(
                subbytes_output(int(plaintexts[i, b]), int(true_key[b]))
            )
            traces[i, lt] += hw

    return traces


# ---------------------------------------------------------------------------
# Correlation Power Analysis
# ---------------------------------------------------------------------------

def correlation_power_analysis(
    plaintexts: np.ndarray,
    traces: np.ndarray,
    byte_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CPA: correlate a Hamming Weight power model against measured traces.

    For each of 256 key hypotheses k_hyp:
      hw_model[i] = HW( SubBytes( plaintext[i, byte_idx] XOR k_hyp ) )
      correlations[k_hyp, t] = Pearson( hw_model, traces[:, t] )

    Returns
    -------
    correlations : (256, T) — full correlation matrix
    best_guess   : (256,)   — max |correlation| over time per hypothesis
    """
    N, T = traces.shape
    correlations = np.zeros((256, T), dtype=np.float64)

    # Pre-compute per time-point mean and std (vectorised)
    tr_mean = traces.mean(axis=0)           # (T,)
    tr_std  = traces.std(axis=0)            # (T,)
    tr_std[tr_std < 1e-12] = 1e-12         # guard against zero-std columns
    traces_centered = traces - tr_mean      # (N, T)

    for k_hyp in range(256):
        hw_model = np.array([
            hamming_weight(subbytes_output(int(plaintexts[i, byte_idx]), k_hyp))
            for i in range(N)
        ], dtype=np.float64)

        hw_mean = hw_model.mean()
        hw_std  = hw_model.std()
        if hw_std < 1e-12:
            continue    # all predictions identical — no discriminative power

        hw_centered = hw_model - hw_mean                                # (N,)
        cov = (traces_centered * hw_centered[:, np.newaxis]).mean(axis=0)  # (T,)
        correlations[k_hyp] = cov / (hw_std * tr_std)

    best_guess = np.max(np.abs(correlations), axis=1)   # (256,)
    return correlations, best_guess


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true_key   = rng.integers(0, 256, 16, dtype=np.uint8)
    plaintexts = rng.integers(0, 256, (500, 16), dtype=np.uint8)

    print("=== aes_sim.py self-test ===\n")
    print(f"True key : {' '.join(f'0x{b:02X}' for b in true_key)}\n")

    # Need time_points > leak_time_for_byte(15, base=5) = 5 + 15*3 = 50
    traces = simulate_power_traces(
        plaintexts, true_key, noise_std=0.8, time_points=60, leak_time=5
    )

    recovered = []
    for b in range(16):
        _, bg = correlation_power_analysis(plaintexts, traces, byte_idx=b)
        recovered.append(int(np.argmax(bg)))
    recovered = np.array(recovered)
    match = recovered == true_key.astype(int)

    print(f"True key  : {' '.join(f'0x{b:02X}' for b in true_key)}")
    print(f"Recovered : {' '.join(f'0x{b:02X}' for b in recovered)}")
    print(f"Match     : {' '.join('OK' if m else 'XX' for m in match)}")
    print(f"\nResult: {match.sum()}/16 bytes correct")
