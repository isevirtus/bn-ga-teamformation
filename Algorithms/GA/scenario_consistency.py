import os
import glob
import json
from collections import Counter
from math import sqrt

# ==========================
# Config
# ==========================

BASE_DIR = os.path.dirname(__file__)
REPORT_DIR = os.path.join(BASE_DIR, "Reports")

LABELS = ["VL", "L", "M", "H", "VH"]
VALS = [0.1, 0.3, 0.5, 0.7, 0.9]  # centros p/ expected value
K = len(LABELS)

MIN_SAMPLES_SCENARIO = 10  # mínimo por cenário para entrar nas métricas globais


# ==========================
# 1) Cenários do especialista
# ==========================

SCENARIOS = {
    1: {
        "name": "S1_VL_VH",
        "AT_bin": "VL",
        "AC_bin": "VH",
        "expert": [
            0.2741935484,
            0.3225806452,
            0.2741935484,
            0.08064516129,
            0.04838709677,
        ],
    },
    2: {
        "name": "S2_VH_VL",
        "AT_bin": "VH",
        "AC_bin": "VL",
        "expert": [
            0.1724137931,
            0.2586206897,
            0.3448275862,
            0.1724137931,
            0.05172413793,
        ],
    },
    3: {
        "name": "S3_VL_VL",
        "AT_bin": "VL",
        "AC_bin": "VL",
        "expert": [
            0.3333333333,
            0.3333333333,
            0.2833333333,
            0.05,
            0.0,
        ],
    },
    4: {
        "name": "S4_VH_VH",
        "AT_bin": "VH",
        "AC_bin": "VH",
        "expert": [
            0.0,
            0.05454545455,
            0.2727272727,
            0.3090909091,
            0.3636363636,
        ],
    },
    5: {
        "name": "S5_VL_M",
        "AT_bin": "VL",
        "AC_bin": "M",
        "expert": [
            0.2,
            0.3,
            0.34,
            0.1,
            0.06,
        ],
    },
    6: {
        "name": "S6_M_VL",
        "AT_bin": "M",
        "AC_bin": "VL",
        "expert": [
            0.3571428571,
            0.3571428571,
            0.1785714286,
            0.1071428571,
            0.0,
        ],
    },
    7: {
        "name": "S7_VH_M",
        "AT_bin": "VH",
        "AC_bin": "M",
        "expert": [
            0.07462686567,
            0.1492537313,
            0.223880597,
            0.2537313433,
            0.2985074627,
        ],
    },
    8: {
        "name": "S8_M_VH",
        "AT_bin": "M",
        "AC_bin": "VH",
        "expert": [
            0.07246376812,
            0.2173913043,
            0.2173913043,
            0.2463768116,
            0.2463768116,
        ],
    },
}


# ==========================
# 2) Utils
# ==========================

def load_all_logs():
    rows = []
    pattern = os.path.join(REPORT_DIR, "GA_run_*.jsonl")
    files = sorted(glob.glob(pattern))
    print(f"[LOAD] Encontrados {len(files)} arquivos JSONL.")
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    print(f"[LOAD] Total de linhas combinadas: {len(rows)}")
    return rows


def make_bin_fn(values):
    """Cria função to_bin(x) usando quantis empíricos 20/40/60/80%."""
    vs = sorted(float(v) for v in values if v is not None)
    n = len(vs)
    if n == 0:
        return lambda x: "M"

    def q(p):
        idx = int(round(p * (n - 1)))
        return vs[idx]

    e1 = q(0.20)
    e2 = q(0.40)
    e3 = q(0.60)
    e4 = q(0.80)

    def to_bin(x):
        x = float(x)
        if x <= e1:
            return "VL"
        elif x <= e2:
            return "L"
        elif x <= e3:
            return "M"
        elif x <= e4:
            return "H"
        else:
            return "VH"

    print(f"[BINS] edges = {e1:.4f}, {e2:.4f}, {e3:.4f}, {e4:.4f}")
    return to_bin


def mean_distribution(dists):
    n = len(dists)
    if n == 0:
        return None
    acc = [0.0] * K
    for d in dists:
        for i in range(K):
            acc[i] += float(d[i])
    return [v / n for v in acc]


def expected_score(dist):
    return sum(p * v for p, v in zip(dist, VALS))


def brier(p, e):
    return sum((pc - ec) ** 2 for pc, ec in zip(p, e)) / K


def argmax_idx(xs):
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def spearman_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None

    def ranks(arr):
        # average ranks for ties
        sorted_idx = sorted(range(n), key=lambda i: arr[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[sorted_idx[j+1]] == arr[sorted_idx[i]]:
                j += 1
            rank = (i + j + 2) / 2.0  # 1-based média
            for k in range(i, j+1):
                r[sorted_idx[k]] = rank
            i = j + 1
        return r

    rx = ranks(xs)
    ry = ranks(ys)

    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    deny = sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


# ==========================
# 3) Monotonicidade
# ==========================

LABEL_IDX = {lab: i for i, lab in enumerate(LABELS)}

def dominates(sc_better, sc_worse):
    """Retorna True se cenário 'better' é >= em AT e AC."""
    ab = LABEL_IDX[sc_better["AT_bin"]]
    ac = LABEL_IDX[sc_better["AC_bin"]]
    wb = LABEL_IDX[sc_worse["AT_bin"]]
    wc = LABEL_IDX[sc_worse["AC_bin"]]
    return (ab >= wb) and (ac >= wc)


# ==========================
# 4) Main
# ==========================

def main():
    log = load_all_logs()
    if not log:
        print("[ERRO] Nenhum dado carregado.")
        return

    # coleta AT/AC contínuos válidos
    at_vals = [r.get("AT_cont") for r in log if r.get("AT_cont") is not None]
    ac_vals = [r.get("AC_cont") for r in log if r.get("AC_cont") is not None]

    if not at_vals or not ac_vals:
        print("[ERRO] Logs sem AT_cont/AC_cont.")
        return

    print(f"[RANGE] AT_cont: min={min(at_vals):.4f}, max={max(at_vals):.4f}")
    print(f"[RANGE] AC_cont: min={min(ac_vals):.4f}, max={max(ac_vals):.4f}")

    # funções de binagem empíricas
    to_bin_AT = make_bin_fn(at_vals)
    to_bin_AC = make_bin_fn(ac_vals)

    # debug: distribuição de bins globais
    at_bins = Counter(to_bin_AT(v) for v in at_vals)
    ac_bins = Counter(to_bin_AC(v) for v in ac_vals)
    print("[AT bins globais]", dict(at_bins))
    print("[AC bins globais]", dict(ac_bins))
    print()

    # agrupar equipes por cenário especialista (usando bins empíricos)
    grupos = {j: [] for j in SCENARIOS}

    for r in log:
        at = r.get("AT_cont")
        ac = r.get("AC_cont")
        dist = [
            r.get("dist_VL"),
            r.get("dist_L"),
            r.get("dist_M"),
            r.get("dist_H"),
            r.get("dist_VH"),
        ]
        if at is None or ac is None or any(v is None for v in dist):
            continue

        at_bin = to_bin_AT(float(at))
        ac_bin = to_bin_AC(float(ac))

        for j, sc in SCENARIOS.items():
            if at_bin == sc["AT_bin"] and ac_bin == sc["AC_bin"]:
                grupos[j].append(dist)
                break

    # métricas por cenário
    briers = []
    modes_ok = 0
    e_exp_list = []
    e_bn_list = []
    considered = []

    print("========== CENÁRIOS ==========")
    for j, sc in SCENARIOS.items():
        name = sc["name"]
        tgt = sc["expert"]
        dists = grupos[j]
        n = len(dists)
        print(f"\n[SCENÁRIO {j} - {name}] AT={sc['AT_bin']} AC={sc['AC_bin']}  n={n}")

        if n < MIN_SAMPLES_SCENARIO:
            print("  - poucas amostras, fora das métricas globais.")
            continue

        p_mean = mean_distribution(dists)
        bj = brier(p_mean, tgt)
        mode_exp = LABELS[argmax_idx(tgt)]
        mode_bn = LABELS[argmax_idx(p_mean)]
        ok = (mode_exp == mode_bn)
        if ok:
            modes_ok += 1

        e_exp = expected_score(tgt)
        e_bn = expected_score(p_mean)

        briers.append(bj)
        e_exp_list.append(e_exp)
        e_bn_list.append(e_bn)
        considered.append(j)

        print(f"  Expert: {['%.3f' % x for x in tgt]}")
        print(f"  BN mean: {['%.3f' % x for x in p_mean]}")
        print(f"  Brier_j = {bj:.4f}")
        print(f"  Mode expert = {mode_exp}, Mode BN = {mode_bn}  -> {'OK' if ok else 'DIFF'}")
        print(f"  E_expert(j) = {e_exp:.3f},  E_BN(j) = {e_bn:.3f}")

    # visão global
    print("\n========== VISÃO GLOBAL ==========")
    m = len(considered)
    if m == 0:
        print("[ALERTA] Nenhum cenário com amostras suficientes.")
        return

    mean_brier = sum(briers) / m
    mode_acc = modes_ok / m

    rho = spearman_corr(e_exp_list, e_bn_list)

    # monotonicidade entre cenários dominantes
    mono_total = 0
    mono_ok = 0
    considered_scenarios = [SCENARIOS[j] for j in considered]
    e_bn_map = {j: e_bn_list[i] for i, j in enumerate(considered)}

    for i in range(m):
        ji = considered[i]
        sci = SCENARIOS[ji]
        for k in range(m):
            jk = considered[k]
            if ji == jk:
                continue
            scj = SCENARIOS[jk]
            if dominates(sci, scj):
                mono_total += 1
                if e_bn_map[ji] + 1e-9 >= e_bn_map[jk]:
                    mono_ok += 1

    mono_rate = mono_ok / mono_total if mono_total > 0 else None

    print(f"Média Brier (cenários válidos): {mean_brier:.4f}")
    print(f"Mode agreement: {modes_ok}/{m} = {mode_acc:.3f}")
    if rho is not None:
        print(f"Spearman(E_expert, E_BN): {rho:.3f}")
    else:
        print("Spearman: insuficiente.")
    if mono_rate is not None:
        print(f"Monotonicidade entre cenários dominantes: {mono_ok}/{mono_total} = {mono_rate:.3f}")
    else:
        print("Monotonicidade: não aplicável.")


if __name__ == "__main__":
    main()
