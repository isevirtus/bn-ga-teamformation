# rt_ga_bn_only_engine.py
import sys, time, csv, statistics, random
from pathlib import Path

THIS = Path(__file__).resolve()
root = next((p for p in THIS.parents if (p / "STFP").exists()), None)
if root is None:
    raise RuntimeError("Não achei a raiz contendo 'STFP'.")
sys.path.insert(0, str(root))

# Importa o GA (engine.py ou GA4STF.py)
try:
    import STFP.Algorithms.GA.engine as engine
    run_ga = engine.run_ga_com_config
except Exception:
    import STFP.Algorithms.GA.GA4STF as engine
    run_ga = engine.run_ga_com_config

import STFP.Pipeline.evaluate_teams as et


def make_project(team_size=4):
    # Projeto-alvo — Web SaaS (Vue + Node.js + JS)
    return {
        "dominio": {"must": ["Web"], "should": ["Cloud"], "could": ["IoT"]},
        "ecossistema": {"must": ["Vue", "Node.js"], "should": ["MongoDB"], "could": ["AWS Lambda"]},
        "linguagens": {"must": ["JavaScript"], "should": ["TypeScript"], "could": ["Python"]},
        "tamanhoEquipe": team_size,
    }


def wrap_evaluator_bn():
    """
    Cria uma função avaliar_equipe(...) que conta chamadas
    e chama evaluate_teams.avaliar_equipe em mode="bn".
    Retorna (fn, counter_dict)
    """
    counter = {"calls": 0}

    def _fn(team_ids, projeto_alvo, log=False):
        counter["calls"] += 1
        return et.avaliar_equipe(team_ids, projeto_alvo, log=log, mode="bn")

    return _fn, counter


def percentile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def mean_ci95(xs):
    import math
    n = len(xs)
    if n < 2:
        return (None, None)
    m = statistics.mean(xs)
    s = statistics.stdev(xs)
    se = s / math.sqrt(n)
    t = 2.045 if n == 30 else 1.96
    return (m - t * se, m + t * se)


def iqr(xs):
    if not xs:
        return None
    q1 = percentile(xs, 0.25)
    q3 = percentile(xs, 0.75)
    return q3 - q1


def cv_pct(mean, std):
    return (100.0 * std / mean) if mean else None


def main():
    OUT_RUNS = THIS.parent / "ga_bn_only_runs.csv"
    OUT_SUM  = THIS.parent / "ga_bn_only_summary.csv"

    TEAM_SIZE = 4
    POP = 100
    GENS = 100          # teto (hard stop). Early stop fica no engine.py
    RUNS = 10
    BASE_SEED = 123

    PROJ = make_project(team_size=TEAM_SIZE)

    # Warm-up BN (constrói CPDs antes de medir)
    warm_team = random.sample(getattr(engine, "CANDIDATOS_IDS"), TEAM_SIZE)
    _ = et.avaliar_equipe(warm_team, PROJ, log=False, mode="bn")
    print("[WARMUP] BN ok")

    rows = []
    mode = "bn"

    for i in range(RUNS):
        seed = BASE_SEED + i

        # limpa caches/artefatos entre execuções
        if hasattr(engine, "FITNESS_CACHE"):
            engine.FITNESS_CACHE.clear()
        if hasattr(engine, "RUN_SUMMARY"):
            engine.RUN_SUMMARY.clear()

        # injeta evaluator BN com contador
        fn, counter = wrap_evaluator_bn()
        engine.avaliar_equipe = fn

        t0 = time.perf_counter()
        out = run_ga(
            PROJ, TEAM_SIZE,
            pop_size=POP,
            geracoes=GENS,
            seed=seed,
            verbose=False,
            report=True,
            log_eval=False
        )
        dur = time.perf_counter() - t0

        calls = counter["calls"]
        per_call_ms = (dur * 1000.0 / calls) if calls else None

        rows.append({
            "mode": mode,
            "run": i + 1,
            "seed": seed,
            "team_size": TEAM_SIZE,
            "pop": POP,
            "gens_max": GENS,
            "gens_executed": out.get("gens_executed", ""),
            "stable_gens_param": out.get("stable_gens_param", ""),
            "stable_gens_observed": out.get("stable_gens_observed", ""),
            "stop_reason": out.get("stop_reason", ""),
            "duration_sec": f"{dur:.6f}",
            "fitness_calls": calls,
            "ms_per_fitness": f"{per_call_ms:.6f}" if per_call_ms is not None else "",
            "best_fitness": f"{out.get('best_fitness', 0.0):.6f}",
            "best_team": str(out.get("best_team")),
        })

        print(f"[BN][run {i+1:02d}] dur={dur:.3f}s calls={calls} ms/fit={per_call_ms:.3f}")

    # salva runs
    with open(OUT_RUNS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)

    # summary (somente bn)
    durs = [float(r["duration_sec"]) for r in rows]
    mpf  = [float(r["ms_per_fitness"]) for r in rows if r["ms_per_fitness"]]
    calls_mode = [int(r["fitness_calls"]) for r in rows]

    d_mean = statistics.mean(durs)
    d_med  = statistics.median(durs)
    d_std  = statistics.stdev(durs) if len(durs) > 1 else 0.0

    d_p05 = percentile(durs, 0.05)
    d_p95 = percentile(durs, 0.95)
    d_p99 = percentile(durs, 0.99)
    d_iqr = iqr(durs)
    d_ci_lo, d_ci_hi = mean_ci95(durs)

    c_mean = statistics.mean(calls_mode)
    c_med  = statistics.median(calls_mode)
    c_std  = statistics.stdev(calls_mode) if len(calls_mode) > 1 else 0.0
    c_p95  = percentile(calls_mode, 0.95)

    summary = [
        ("N_runs", "bn", len(durs)),
        ("dur_mean_sec", "bn", d_mean),
        ("dur_median_sec", "bn", d_med),
        ("dur_std_sec", "bn", d_std),
        ("dur_cv_pct", "bn", cv_pct(d_mean, d_std)),
        ("dur_p05_sec", "bn", d_p05),
        ("dur_p95_sec", "bn", d_p95),
        ("dur_p99_sec", "bn", d_p99),
        ("dur_iqr_sec", "bn", d_iqr),
        ("dur_ci95_low_sec", "bn", d_ci_lo),
        ("dur_ci95_high_sec", "bn", d_ci_hi),
        ("dur_min_sec", "bn", min(durs)),
        ("dur_max_sec", "bn", max(durs)),
        ("calls_mean", "bn", c_mean),
        ("calls_median", "bn", c_med),
        ("calls_std", "bn", c_std),
        ("calls_p95", "bn", c_p95),
    ]

    if mpf:
        m_mean = statistics.mean(mpf)
        m_med  = statistics.median(mpf)
        m_std  = statistics.stdev(mpf) if len(mpf) > 1 else 0.0
        m_ci_lo, m_ci_hi = mean_ci95(mpf)
        summary += [
            ("ms_per_fitness_mean", "bn", m_mean),
            ("ms_per_fitness_median", "bn", m_med),
            ("ms_per_fitness_std", "bn", m_std),
            ("ms_per_fitness_ci95_low", "bn", m_ci_lo),
            ("ms_per_fitness_ci95_high", "bn", m_ci_hi),
            ("ms_per_fitness_p95", "bn", percentile(mpf, 0.95)),
        ]

    with open(OUT_SUM, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["metric", "group", "value"])
        for k, g, v in summary:
            w.writerow([k, g, v])

    print(f"\n[OK] Runs   : {OUT_RUNS}")
    print(f"[OK] Summary: {OUT_SUM}")


if __name__ == "__main__":
    main()
