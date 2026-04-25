import numpy as np
import sys
import os

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(str(_HERE), '..', 'mejora del modelo'))
sys.path.insert(0, os.path.join(str(_HERE), '..', 'GA beam'))

from config.config import REBAR_CATALOG
from config import config_ga

def _make_base_chrom(chrom_L, diam_A: int = 1, diam_B: int = 2) -> np.ndarray:
    z = np.zeros(chrom_L, dtype=np.int8)
    z[0] = diam_A
    z[1] = diam_B
    return z

def _activate_corners(z: np.ndarray, block_offset: int, n_slots: int,
                      diam_choice: int = 0):
    z[block_offset + 0] = diam_choice
    z[block_offset + 1] = 1
    last_slot_offset = block_offset + (n_slots - 1) * 2
    z[last_slot_offset]     = diam_choice
    z[last_slot_offset + 1] = 1

def _archetype_A(chrom_L, n_slots) -> np.ndarray:
    dA = np.random.randint(0, 4)
    dB = np.random.randint(dA, 4)
    z = _make_base_chrom(chrom_L, dA, dB)
    _activate_corners(z, 2, n_slots, diam_choice=0)
    return z

def _archetype_B(chrom_L, n_slots, block_size, _ZONE_ORDER, mu_zones, d_cm) -> np.ndarray:
    dA = np.random.randint(0, 4)
    dB = np.random.randint(dA, 4)
    z = _make_base_chrom(chrom_L, dA, dB)
    _activate_corners(z, 2, n_slots, diam_choice=0)

    as_corr = 2 * REBAR_CATALOG[dA]['area_cm2']

    for zone_ord, zone_id in enumerate(_ZONE_ORDER):
        mu = mu_zones[zone_id]
        if mu <= 0.01:
            continue
        mn_approx = as_corr * 4200 * d_cm / 100_000
        if mn_approx >= mu:
            continue
        n_extra = min(2, n_slots)
        bast_offset = 2 + block_size + zone_ord * block_size
        mid = n_slots // 2
        for j in range(n_extra):
            slot = mid + j
            if slot < n_slots:
                base = bast_offset + slot * 2
                z[base]     = 1
                z[base + 1] = 1
    return z

def _archetype_C(chrom_L) -> np.ndarray:
    dA = np.random.randint(0, 4)
    dB = np.random.randint(dA, 4)
    z = _make_base_chrom(chrom_L, dA, dB)
    for offset in range(2, chrom_L, 2):
        z[offset]     = np.random.randint(0, 2)
        z[offset + 1] = 1 if np.random.random() < 0.25 else 0
    return z

def _warm_start(chrom_L, beam, n_slots) -> np.ndarray | None:
    try:
        from beam_reconciler import find_optimal_design
        from zone_extractor import extract_zones
        zones = extract_zones(beam)
        design = find_optimal_design(beam, zones)
        corr = design['corrido']
        dA = corr['diam_a_idx']
        dB = corr['diam_b_idx'] if corr['diam_b_idx'] is not None else dA

        z = _make_base_chrom(chrom_L, dA, dB)
        n_a = corr['n_a']
        n_b = corr['n_b']
        offset = 2
        slots_used = 0
        for _ in range(min(n_a, n_slots)):
            base = offset + slots_used * 2
            z[base]     = 0
            z[base + 1] = 1
            slots_used += 1
        for _ in range(min(n_b, n_slots - slots_used)):
            base = offset + slots_used * 2
            z[base]     = 1
            z[base + 1] = 1
            slots_used += 1
        return z
    except Exception:
        return None

def _mutate(z: np.ndarray, block_size, P_MUT_DIAM, P_MUT_RESET, P_MUT_ONI, P_MUT_CHOICE) -> np.ndarray:
    z = z.copy()

    if np.random.random() < P_MUT_DIAM:
        which = np.random.randint(0, 2)
        delta = np.random.choice([-1, 1])
        z[which] = int(np.clip(z[which] + delta, 0, 3))
        if z[0] > z[1]:
            z[0], z[1] = z[1], z[0]

    offset = 2
    N_ZONES = len(config_ga.ZONE_IDS)
    for bi in range(1 + N_ZONES):
        if bi > 0 and np.random.random() < P_MUT_RESET:
            z[offset: offset + block_size] = 0
            offset += block_size
            continue

        for gi in range(block_size // 2):
            gene_off = offset + gi * 2
            if np.random.random() < P_MUT_ONI:
                z[gene_off + 1] = 1 - z[gene_off + 1]
            if z[gene_off + 1] == 1 and np.random.random() < P_MUT_CHOICE:
                z[gene_off] = 1 - z[gene_off]
        offset += block_size

    return z

def _crossover(p1: np.ndarray, p2: np.ndarray, block_size, P_CROSS) -> tuple:
    c1 = p1.copy()
    c2 = p2.copy()

    if np.random.random() > P_CROSS:
        return c1, c2

    N_ZONES = len(config_ga.ZONE_IDS)
    n_blocks = 1 + N_ZONES
    for bi in range(n_blocks):
        if np.random.random() < 0.5:
            off = 2 + bi * block_size
            end = off + block_size
            c1[off:end], c2[off:end] = c2[off:end].copy(), c1[off:end].copy()
            if np.random.random() < 0.3:
                pt = np.random.randint(1, block_size)
                c1[off: off + pt], c2[off: off + pt] = (
                    c2[off: off + pt].copy(), c1[off: off + pt].copy()
                )

    return c1, c2