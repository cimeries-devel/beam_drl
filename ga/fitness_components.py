
def compute_fitness(eval_result: dict, b_cm: float, d_cm: float,
                    as_corrido: float, fc: float,
                    rho_min_func, ET_MIN_DUCTILITY, N_CAPAS_MAX,
                    LAMBDA_M, LAMBDA_N, LAMBDA_G, PENALTY_FIXED,
                    LAMBDA_EXC, LAMBDA_CORR_EXC, LAMBDA_CONSTR_BAST) -> float:
    """Calcula el fitness total."""
    W = eval_result['total_weight_kg']

    # P_capacidad: déficit en cada zona
    P_cap = 0.0
    for v in eval_result['zone_results'].values():
        mu = max(v['Mu'], 0.01)
        deficit = max(0.0, (mu - v['phi_Mn']) / mu * 100.0)
        P_cap += deficit

    # P_normativa
    rho_min = rho_min_func(fc)
    rho = eval_result['rho_corrido']
    eps_t_pos = eval_result['eps_t_corrido_pos']
    eps_t_neg = eval_result['eps_t_corrido_neg']
    fy = 4200.0
    Es = 2_000_000.0
    eps_y = fy / Es

    P_norm = 0.0
    if rho < rho_min:
        P_norm += (rho_min - rho) / max(rho_min, 1e-9) * 100.0

    if eps_t_pos < ET_MIN_DUCTILITY:
        P_norm += (ET_MIN_DUCTILITY - eps_t_pos) / ET_MIN_DUCTILITY * 100.0
    if eps_t_neg < ET_MIN_DUCTILITY:
        P_norm += (ET_MIN_DUCTILITY - eps_t_neg) / ET_MIN_DUCTILITY * 100.0

    if eps_t_pos < eps_y:
        P_norm += (eps_y - eps_t_pos) / eps_y * 100.0
    if eps_t_neg < eps_y:
        P_norm += (eps_y - eps_t_neg) / eps_y * 100.0

    # P_constructiva
    P_constr = 0.0
    dec = eval_result.get('decoded', {})

    corr_bars = eval_result.get('corr_bars', [])
    if not corr_bars:
        P_constr += 10.0

    if dec:
        corrido_mat = dec.get('corrido')
        if corrido_mat is not None:
            n_slots = corrido_mat.shape[1]
            for k in range(N_CAPAS_MAX):
                oni_k = corrido_mat[k, :, 1]
                n_on = int(oni_k.sum())
                if n_on == 0:
                    continue
                if oni_k[0] == 0:
                    P_constr += 1.0
                if oni_k[n_slots - 1] == 0:
                    P_constr += 1.0
                if n_on == 1:
                    P_constr += 1.0

    # P_exceso_bast: exceso de capacidad en zonas con bastones activos
    P_exc = 0.0
    for v in eval_result['zone_results'].values():
        if v.get('n_bast', 0) > 0 and v['Mu'] > 0.01:
            exceso = max(0.0, (v['phi_Mn'] - v['Mu']) / v['Mu'] * 100.0)
            P_exc += exceso

    # P_exceso_corrido: penaliza corrido sobredimensionado
    P_exc_corr = 0.0
    phi_corr_pos = eval_result['phi_mn_corrido']['positive']
    phi_corr_neg = eval_result['phi_mn_corrido']['negative']
    zone_extents = eval_result.get('zone_extents', {})
    xs_lst   = eval_result.get('xs', [])
    ms_lst   = eval_result.get('ms', [])
    ms_max_l = eval_result.get('ms_max')
    ms_min_l = eval_result.get('ms_min')
    L_val    = eval_result.get('L', 0.0)
    
    if xs_lst and ms_lst and L_val > 0.0 and zone_extents:
        L3 = L_val / 3.0
        tercios_x = [(0.0, L3), (L3, 2.0 * L3), (2.0 * L3, L_val)]
        tercio_names = ['LEFT', 'MID', 'RIGHT']
        for face, phi_corr in (('BOT', phi_corr_pos), ('TOP', phi_corr_neg)):
            if ms_max_l is not None and ms_min_l is not None:
                ms_face = ms_max_l if face == 'BOT' else ms_min_l
            else:
                ms_face = ms_lst
            for idx, (x_lo, x_hi) in enumerate(tercios_x):
                zone_id = f'{tercio_names[idx]}_{face}'
                if zone_extents.get(zone_id, {}).get('exists', False):
                    continue
                vals = [
                    max(float(m), 0.0) if face == 'BOT' else max(-float(m), 0.0)
                    for x, m in zip(xs_lst, ms_face)
                    if x_lo <= x <= x_hi
                ]
                mu_t = max(vals) if vals else 0.0
                if mu_t > 0.01 and phi_corr > mu_t:
                    P_exc_corr += (phi_corr - mu_t) / mu_t * 100.0

    # P_constr_bast: penalización de constructabilidad de bastones
    P_constr_bast = 0.0
    for v in eval_result['zone_results'].values():
        slots_per_layer = v.get('bast_slots_per_layer', [0, 0, 0])
        if slots_per_layer[1] > 2:
            P_constr_bast += (slots_per_layer[1] - 2)
        P_constr_bast += slots_per_layer[2] * 2

        n_bast = v.get('n_bast', 0)
        if n_bast > 2:
            P_constr_bast += (n_bast - 2)

    violations = {
        'P_capacidad':       round(P_cap, 4),
        'P_normativa':       round(P_norm, 4),
        'P_constructiva':    round(P_constr, 4),
        'P_exceso_bast':     round(P_exc, 4),
        'P_exceso_corrido':  round(P_exc_corr, 4),
        'P_constr_bast':     round(P_constr_bast, 4),
    }
    eval_result['violations'] = violations

    fitness = (W
               + LAMBDA_M           * P_cap
               + LAMBDA_N           * P_norm
               + LAMBDA_G           * P_constr * PENALTY_FIXED
               + LAMBDA_EXC         * P_exc
               + LAMBDA_CORR_EXC    * P_exc_corr
               + LAMBDA_CONSTR_BAST * P_constr_bast)
    return fitness