"""Motor de cálculo unificado ACI-02+ (Ottazzi Cap. 17).

Calcula φMn para secciones de concreto reforzado con barras individuales
(sin depender de civdevAPI ni de otros módulos del proyecto).

Para M+: fibra comprimida = parte superior → ds_i = y_from_top
Para M-: fibra comprimida = parte inferior → ds_i = h_cm - y_from_top

Bisección 100 iteraciones para equilibrio axial Pn = 0.
"""

import math
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '..', 'mejora del modelo'))

from config import FY, ES, ECU

# ---------------------------------------------------------------------------
# Parámetros materiales
# ---------------------------------------------------------------------------
_FY  = float(FY)    # kg/cm2
_ES  = float(ES)    # kg/cm2
_ECU = float(ECU)   # deformación unitaria última del concreto


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def compute_beta1(fc: float) -> float:
    """Factor β₁ según E.060 / ACI 318-02."""
    if fc <= 280:
        return 0.85
    elif fc <= 560:
        return round(1.05 - 0.714 * (fc / 1000.0), 4)
    return 0.65


def compute_rho_min(fc: float, fy: float = _FY) -> float:
    """ρ_min = 0.7√fc/fy  (E.060 Art.10.5, secciones rectangulares)."""
    return 0.7 * math.sqrt(fc) / fy


def compute_phi_unified(epsilon_t: float, epsilon_y: float) -> float:
    """Factor φ de reducción de resistencia ACI 318-02 Sección 9.3.

    Zona controlada por tracción  (ε_t ≥ 0.005) : φ = 0.90
    Zona de transición            (ε_y ≤ ε_t < 0.005): interpolación lineal
    Zona controlada por compresión (ε_t < ε_y)  : φ = 0.70
    """
    if epsilon_t >= 0.005:
        return 0.90
    elif epsilon_t >= epsilon_y:
        # interpolación lineal entre 0.70 y 0.90
        return 0.70 + 0.20 * (epsilon_t - epsilon_y) / (0.005 - epsilon_y)
    else:
        return 0.70


def compute_section_mn(
    bars: list,
    b_cm: float,
    h_cm: float,
    fc: float,
    fy: float = _FY,
    Es: float = _ES,
    ecu: float = _ECU,
) -> dict:
    """Calcula φMn para M+ y M- de una sección con barras individuales.

    Parameters
    ----------
    bars : list of (y_from_top, area_cm2, diam_name_str)
        Posición de cada barra medida desde el borde superior (cm).
    b_cm : float
        Ancho de la viga (cm).
    h_cm : float
        Altura total de la viga (cm).
    fc : float
        Resistencia del concreto (kg/cm²).
    fy, Es, ecu : float
        Propiedades del acero y deformación última del concreto.

    Returns
    -------
    dict con claves 'positive' y 'negative', cada una con:
        phi_Mn (tonf·m), Mn (tonf·m), epsilon_t, phi, c (cm), a (cm).
    """
    epsilon_y = fy / Es
    beta1 = compute_beta1(fc)

    def _solve_direction(compression_top: bool) -> dict:
        """Resuelve equilibrio para una dirección de flexión.

        compression_top=True  → M+ (fibra comprimida arriba, ds_i = y_from_top)
        compression_top=False → M- (fibra comprimida abajo,  ds_i = h - y_from_top)
        """
        if not bars:
            return {'phi_Mn': 0.0, 'Mn': 0.0, 'epsilon_t': 0.0,
                    'phi': 0.70, 'c': 0.0, 'a': 0.0}

        # Distancias desde la fibra comprimida
        if compression_top:
            ds = [y for y, _, _ in bars]
            areas = [a for _, a, _ in bars]
        else:
            ds = [h_cm - y for y, _, _ in bars]
            areas = [a for _, a, _ in bars]

        # Distancia al acero de tracción más alejado
        d_max = max(ds)
        if d_max <= 0:
            return {'phi_Mn': 0.0, 'Mn': 0.0, 'epsilon_t': 0.0,
                    'phi': 0.70, 'c': 0.0, 'a': 0.0}

        def _equilibrium(c: float) -> float:
            """Suma de fuerzas internas (positivo = tensión > compresión)."""
            a = beta1 * c
            # Fuerza de compresión del concreto
            Cc = 0.85 * fc * b_cm * min(a, h_cm)
            # Fuerzas en el acero
            Fs_total = 0.0
            for d_i, A_i in zip(ds, areas):
                if d_i <= 0:
                    eps_i = -ecu  # barra en la fibra comprimida extrema
                else:
                    eps_i = ecu * (d_i - c) / c if c > 1e-9 else ecu
                fs_i = max(-fy, min(fy, eps_i * Es))
                Fs_total += A_i * fs_i
            return Fs_total - Cc

        # Bisección para encontrar c tal que equilibrio = 0.
        # equilibrium(c) = Fs - Cc es DECRECIENTE en c
        # (más c → más Cc, menos Fs) → f(c_lo)>0, f(c_hi)<0
        # Por eso: f>0 en c_mid → cero está a la DERECHA → c_lo = c_mid
        #          f<0 en c_mid → cero está a la IZQUIERDA → c_hi = c_mid
        c_lo, c_hi = 1e-6, h_cm * 2.0
        for _ in range(100):
            c_mid = 0.5 * (c_lo + c_hi)
            if _equilibrium(c_mid) > 0:
                c_lo = c_mid   # cero a la derecha
            else:
                c_hi = c_mid   # cero a la izquierda
        c = 0.5 * (c_lo + c_hi)
        a = beta1 * c

        # ε_t del acero de tracción más alejado
        if c > 1e-9:
            eps_t = ecu * (d_max - c) / c
        else:
            eps_t = ecu * 1000  # prácticamente sin compresión

        phi = compute_phi_unified(eps_t, epsilon_y)

        # Momento nominal Mn respecto a la fibra de compresión
        a_eff = min(a, h_cm)
        Mn = 0.0
        for d_i, A_i in zip(ds, areas):
            if d_i <= 0:
                eps_i = -ecu
            else:
                eps_i = ecu * (d_i - c) / c if c > 1e-9 else ecu
            fs_i = max(-fy, min(fy, eps_i * Es))
            Mn += A_i * fs_i * (d_i - a_eff / 2.0)

        Mn_tf_m = abs(Mn) / 100_000.0   # kg·cm → tonf·m
        phi_Mn  = phi * Mn_tf_m

        return {
            'phi_Mn':    round(phi_Mn,  4),
            'Mn':        round(Mn_tf_m, 4),
            'epsilon_t': round(eps_t,   6),
            'phi':       round(phi,     4),
            'c':         round(c,       4),
            'a':         round(a_eff,   4),
        }

    return {
        'positive': _solve_direction(compression_top=True),
        'negative': _solve_direction(compression_top=False),
    }


# ---------------------------------------------------------------------------
# Test básico (ejecutar como script)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Viga conocida: b=30cm, h=60cm, fc=210, 3ø5/8" (As=6.00cm²)
    # Barras en la cara de tension para M+ → y_from_top ≈ 54 cm (d = 54 cm)
    # a = 6*4200/(0.85*210*30) = 4.706 cm → c = 5.536 cm
    # phi_Mn ≈ 0.9 * 6*4200*(54 - 4.706/2)/100000 ≈ 11.71 tonf·m
    bars_test = [
        (54.0, 2.00, '5/8'),
        (54.0, 2.00, '5/8'),
        (54.0, 2.00, '5/8'),
    ]
    res = compute_section_mn(bars_test, b_cm=30, h_cm=60, fc=210)
    print("=== Test seccion 3x5/8\" ===")
    print(f"  M+: phi_Mn={res['positive']['phi_Mn']:.3f} tonf-m, "
          f"eps_t={res['positive']['epsilon_t']:.4f}, phi={res['positive']['phi']:.3f}")
    print(f"  M-: phi_Mn={res['negative']['phi_Mn']:.3f} tonf-m, "
          f"eps_t={res['negative']['epsilon_t']:.4f}, phi={res['negative']['phi']:.3f}")

    # Con β1=0.85, As=6cm², b=30, d=54: a=4.706, c=5.536
    import math
    b1 = compute_beta1(210)
    As = 6.0
    a_ = As * 4200 / (0.85 * 210 * 30)
    c_ = a_ / b1
    phi_mn_ref = 0.9 * As * 4200 * (54 - a_/2) / 100_000
    print(f"\nRef analitica: a={a_:.3f} c={c_:.3f} phi_Mn={phi_mn_ref:.3f} tonf-m")
    print(f"beta1={b1}, rho_min={compute_rho_min(210):.5f}")
