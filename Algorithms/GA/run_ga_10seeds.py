import os
import csv
import json

from STFP.Algorithms.GA.engine import run_ga_com_config

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
    # mesmo cenário do seu GA
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

    seeds = [0]

    rows = []
    for seed in seeds:
        print(f"\n[GA] Rodando seed={seed}")
        res = run_ga_com_config(
            PROJETO_ALVO_EXTERNO=projeto_alvo,
            team_size=team_size,
            pop_size=pop_size,
            geracoes=generations,
            seed=seed,
        )
        rows.append({
            "method": "GA",
            "seed": seed,
            "team_size": team_size,
            "pop_size": pop_size,
            "generations": generations,
            "best_team": json.dumps(res["best_team"]),
            "best_fitness": res["best_fitness"],
            "duration_sec": res["duration_sec"],
        })

    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(base_dir, "ga_10seeds_results.csv")

    fieldnames = [
        "method",
        "seed",
        "team_size",
        "pop_size",
        "generations",
        "best_team",
        "best_fitness",
        "duration_sec",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[OK] Resultados GA salvos em: {out_path}")

if __name__ == "__main__":
    main()
