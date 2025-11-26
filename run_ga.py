# STFP/run_ga.py
import argparse
from pathlib import Path
import json

from STFP.Algorithms.GA.engine import run_ga_com_config


def _csv_list(s: str):
    if not s: return []
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Executa GA para formação de equipes (novo fluxo).")
    parser.add_argument("--team-size", type=int, required=True)
    parser.add_argument("--dom-m", type=str, default="")
    parser.add_argument("--dom-s", type=str, default="")
    parser.add_argument("--dom-c", type=str, default="")
    parser.add_argument("--eco-s", type=str, default="")
    parser.add_argument("--eco-c", type=str, default="")
    parser.add_argument("--ling-s", type=str, default="")
    parser.add_argument("--ling-c", type=str, default="")
    
    args = parser.parse_args()

    projeto_alvo = {
        "dominio": {
            "must":   _csv_list(args.dom_m),
            "should": _csv_list(args.dom_s),
            "could":  _csv_list(args.dom_c),
        },
        "ecossistema": {
            "must":   [],
            "should": _csv_list(args.eco_s),
            "could":  _csv_list(args.eco_c),
        },
        "linguagens": {
            "must":   [],
            "should": _csv_list(args.ling_s),
            "could":  _csv_list(args.ling_c),
        },
    }

    result = run_ga_com_config(
        PROJETO_ALVO_EXTERNO=projeto_alvo,
        team_size=args.team_size,
        pop_size=10,
        geracoes=10,
        seed=123,
    )

    print("\n=== RESULTADO ===")
    print("Best team:", result["best_team"])
    print("Best fitness (AĒ):", f'{result["best_fitness"]:.4f}')
    print("Duration (s):", f'{result["duration_sec"]:.2f}')

if __name__ == "__main__":
    main()
