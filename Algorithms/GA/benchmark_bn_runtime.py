import os
import sys
import json
import random
import time
from statistics import mean

# ---------------------------------------------------------
# 1) Resolver caminho do projeto para importar STFP.Pipeline
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from STFP.Pipeline.evaluate_teams import avaliar_equipe

# ---------------------------------------------------------
# 2) Helpers para carregar devs e grafo (copiados do GA)
# ---------------------------------------------------------

def _csv_list(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _as_int(x):
    import re
    if x is None:
        return None
    if isinstance(x, int):
        return x
    m = re.search(r"(\d+)$", str(x))
    return int(m.group(1)) if m else None

def _carregar_devs(db_path):
    with open(db_path, encoding="utf-8") as f:
        raw = json.load(f)
    devs = raw if isinstance(raw, list) else raw.get("developers", [])
    norm = []
    for d in devs:
        d2 = dict(d)
        d2["ecossistema"] = ([d2["ecossistema"]] if isinstance(d2.get("ecossistema"), str)
                             else (d2.get("ecossistema") or []))
        d2["linguagens"]  = ([d2["linguagens"]] if isinstance(d2.get("linguagens"), str)
                             else (d2.get("linguagens") or []))
        projs_in = d2.get("projects", []) or d2.get("projectHistory", [])
        d2["projects"] = [
            {
                "id": p.get("id") or p.get("projectId"),
                "osf": p.get("osf") if "osf" in p else p.get("NPS"),
                "slf": p.get("slf") if "slf" in p else p.get("SLF"),
            }
            for p in projs_in
        ]
        norm.append(d2)
    return norm

def _load_grafo(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    edges = data.get("edges") or data.get("links") or (data if isinstance(data, list) else [])
    adj, ids = {}, set()
    for e in edges:
        u = _as_int(e.get("source_user_id")) or _as_int(e.get("source"))
        v = _as_int(e.get("target_user_id")) or _as_int(e.get("target"))
        if u is None or v is None:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        ids.update([u, v])
    return adj, ids

# ---------------------------------------------------------
# 3) Configuração do benchmark
# ---------------------------------------------------------

DB_PATH  = os.path.join(BASE_DIR, "..", "..", "Data", "Dev_DB.json")
GRAFO_PATH = os.path.join(BASE_DIR, "..", "..", "Data", "Graph_DB.json")

ADJ, VALID_IDS = _load_grafo(GRAFO_PATH)
DEVS = [d for d in _carregar_devs(DB_PATH) if int(d["id"]) in VALID_IDS]
CANDIDATOS_IDS = [int(d["id"]) for d in DEVS]

TEAM_SIZE = 4
NUM_RUNS = 1  # número de chamadas que você quer medir

if len(CANDIDATOS_IDS) < TEAM_SIZE:
    raise RuntimeError("Candidatos insuficientes para formar uma equipe no benchmark.")

PROJETO_ALVO = {
    "dominio": {
        "must":   _csv_list("Web,Mobile,Cloud"),
        "should": _csv_list(""),
        "could":  _csv_list(""),
    },
    "ecossistema": {
        "should": _csv_list('React,Angular,Vue,Node.js,"AWS Lambda"'),
        "could":  _csv_list(""),
    },
    "linguagens": {
        "should": _csv_list("JavaScript,TypeScript,Python,Java"),
        "could":  _csv_list(""),
    },
    "tamanhoEquipe": TEAM_SIZE,
}

def main():
    random.seed(123)

    print(f"[BENCH] Candidatos no grafo: {len(CANDIDATOS_IDS)}")
    print(f"[BENCH] Tamanho da equipe: {TEAM_SIZE}")
    print(f"[BENCH] Num. de execuções (avaliar_equipe): {NUM_RUNS}")

    # warm-up (opcional, tira custo de primeira chamada)
    for _ in range(3):
        team = random.sample(CANDIDATOS_IDS, TEAM_SIZE)
        _ = avaliar_equipe(team, PROJETO_ALVO, log=False)

    durations = []

    for i in range(NUM_RUNS):
        team = random.sample(CANDIDATOS_IDS, TEAM_SIZE)

        t0 = time.perf_counter()
        _ = avaliar_equipe(team, PROJETO_ALVO, log=False)
        t1 = time.perf_counter()

        dt = t1 - t0
        durations.append(dt)

        if (i + 1) % 10 == 0:
            print(f"[BENCH] {i+1} chamadas realizadas... última levou {dt:.4f} s")

    durations_sorted = sorted(durations)
    avg = mean(durations)
    med = durations_sorted[len(durations_sorted)//2]
    p10 = durations_sorted[int(0.1 * len(durations_sorted))]
    p90 = durations_sorted[int(0.9 * len(durations_sorted))]

    print("\n[RESULTADOS BN]")
    print(f"  Chamadas medidas: {NUM_RUNS}")
    print(f"  Média:     {avg:.4f} s por chamada")
    print(f"  Mediana:   {med:.4f} s por chamada")
    print(f"  P10:       {p10:.4f} s")
    print(f"  P90:       {p90:.4f} s")

if __name__ == "__main__":
    main()
