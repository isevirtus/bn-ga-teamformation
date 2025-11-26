import os
import csv
import json

from STFP.Algorithms.GA.random_search_baseline import run_random_search

def _csv_list(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def build_projeto_alvo(dom_m, dom_s, dom_c,
                       eco_s, eco_c,
                       ling_s, ling_c):
    return {
        "dominio": {
            "must":   _csv_list(dom_m),
            "should": _csv_list(dom_s),
            "could":  _csv_list(dom_c),
        },
        "ecossistema": {
            "should": _csv_list(eco_s),
            "could":  _csv_list(eco_c),
        },
        "linguagens": {
            "should": _csv_list(ling_s),
            "could":  _csv_list(ling_c),
        },
    }

def main():
    # mesmo cenário do GA
    team_size  = 4
    pop_size   = 6
    generations = 10

    dom_m  = "Web,Mobile,Cloud"
    dom_s  = ""
    dom_c  = ""
    eco_s  = 'React,Angular,Vue,Node.js,"AWS Lambda"'
    eco_c  = ""
    ling_s = "JavaScript,TypeScript,Python,Java"
    ling_c = ""

    projeto_alvo = build_projeto_alvo(dom_m, dom_s, dom_c,
                                      eco_s, eco_c,
                                      ling_s, ling_c)

    # orçamento igual ao GA
    budget = pop_size * (generations + 1)  # 6 * 11 = 66

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    rows = []
    for seed in seeds:
        print(f"\n[Random] Rodando seed={seed}")
        res = run_random_search(
            PROJETO_ALVO_EXTERNO=projeto_alvo,
            team_size=team_size,
            num_evals=budget,
            seed=seed,
            run_name=f"Random_run_seed{seed}",
        )
        rows.append({
            "method": "Random",
            "seed": seed,
            "team_size": team_size,
            "budget": budget,
            "best_team": json.dumps(res["best_team"]),
            "best_fitness": res["best_fitness"],
            "duration_sec": res["duration_sec"],
            "unique_evals": res.get("unique_evals", ""),
        })

    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(base_dir, "random_10seeds_results.csv")

    fieldnames = [
        "method",
        "seed",
        "team_size",
        "budget",
        "best_team",
        "best_fitness",
        "duration_sec",
        "unique_evals",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[OK] Resultados Random Search salvos em: {out_path}")

if __name__ == "__main__":
    main()
