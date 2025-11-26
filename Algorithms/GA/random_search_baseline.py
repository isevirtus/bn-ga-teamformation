import sys, os, re, json, time, random, argparse
from datetime import datetime
from typing import Tuple

# ------------------------------------------------------------
#  Fitness cache and team key
# ------------------------------------------------------------

FITNESS_CACHE = {}  # chave: tuple(sorted(team_ids)) -> dict(res)
UNIQUE_EVALS = 0    # número de chamadas reais à avaliar_equipe (sem cache)

def _key(team) -> Tuple[int, ...]:
    """Chave canônica para identificar a equipe, independente de ordem."""
    return tuple(sorted(int(x) for x in team))

# -------------------- resolver caminho do projeto --------------------
# BASE_DIR = pasta deste arquivo ( ...\STFP\Algorithms\ga )
BASE_DIR = os.path.dirname(__file__)
# PROJECT_ROOT = pasta ACIMA de STFP ( ...\Works )
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# permitir importar evaluate_teams.py (via pacote STFP)
from STFP.Pipeline.evaluate_teams import avaliar_equipe
# permitir importar evaluate_teams.py (mesmo esquema do GA)
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from STFP.Pipeline.evaluate_teams import avaliar_equipe

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, '..', '..', 'Data', 'Dev_DB.json')
GRAFO_GRAPH_PATH = os.path.join(BASE_DIR, '..', '..', 'Data', 'Graph_DB.json')

# ------------------------------------------------------------
# REPORT / LOG
# ------------------------------------------------------------

REPORT_DIR = os.path.join(BASE_DIR, "Reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def _timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

_TEE_ORIG_STDOUT = None
_TEE_FILE_HANDLE = None

def enable_run_report(run_name="Random_run"):
    """Redireciona stdout para arquivos (latest + stamped)."""
    global _TEE_ORIG_STDOUT, _TEE_FILE_HANDLE
    latest_path  = os.path.join(REPORT_DIR, f"{run_name}_latest.txt")
    stamped_path = os.path.join(REPORT_DIR, f"{run_name}_{_timestamp()}.txt")
    fh_latest  = open(latest_path,  "w", encoding="utf-8")
    fh_stamped = open(stamped_path, "w", encoding="utf-8")
    _TEE_ORIG_STDOUT = sys.stdout
    sys.stdout = _TeeIO(sys.stdout, fh_latest, fh_stamped)
    _TEE_FILE_HANDLE = (fh_latest, fh_stamped)
    print(f"[REPORT] Logging habilitado. latest='{latest_path}', stamped='{stamped_path}'")
    return latest_path, stamped_path

def disable_run_report():
    """Restaura stdout original e fecha arquivos de log."""
    global _TEE_ORIG_STDOUT, _TEE_FILE_HANDLE
    if _TEE_ORIG_STDOUT is not None:
        sys.stdout.flush()
        sys.stdout = _TEE_ORIG_STDOUT
        _TEE_ORIG_STDOUT = None
    if _TEE_FILE_HANDLE is not None:
        for fh in _TEE_FILE_HANDLE:
            try: fh.close()
            except:
                pass
        _TEE_FILE_HANDLE = None

# artefatos estruturados
RUN_SUMMARY = []

def _append_run_row(gen, team, res_dict):
    """Adiciona linha em RUN_SUMMARY, mesmo formato do GA."""
    row = {
        "generation": gen,
        "team": list(team),
        "fitness": res_dict.get("media_AE"),

        # distribuição AE
        "dist_VL": None, "dist_L": None, "dist_M": None, "dist_H": None, "dist_VH": None,

        # info técnica 
        "dom_score": res_dict.get("scores", {}).get("dominio", {}).get("score")
                      if isinstance(res_dict.get("scores"), dict) else \
                      (res_dict.get("dominio", {}).get("score")
                       if isinstance(res_dict.get("dominio"), dict) else None),
        "dom_label": res_dict.get("scores", {}).get("dominio", {}).get("rotulo")
                      if isinstance(res_dict.get("scores"), dict) else \
                      (res_dict.get("dominio", {}).get("rotulo")
                       if isinstance(res_dict.get("dominio"), dict) else None),
        "fullP": res_dict.get("dominio", {}).get("fullP")
                 if isinstance(res_dict.get("dominio"), dict) else None,

        # colaboração agregada original (se você já usa)
        "pc_percent": res_dict.get("pc_percent"),

        # AT e AC contínuos
        "AT_cont": res_dict.get("AT_cont"),
        "AC_cont": res_dict.get("AC_cont"),
    }

    dist = res_dict.get("distribuicao")
    if isinstance(dist, (list, tuple)) and len(dist) == 5:
        row["dist_VL"], row["dist_L"], row["dist_M"], row["dist_H"], row["dist_VH"] = dist

    RUN_SUMMARY.append(row)


def _save_jsonl_and_csv(run_name="Random_run"):
    """Salva JSONL (todas as linhas) + CSV com melhor linha por 'generation'."""
    jsonl_latest = os.path.join(REPORT_DIR, f"{run_name}_latest.jsonl")
    jsonl_stamp  = os.path.join(REPORT_DIR, f"{run_name}_{_timestamp()}.jsonl")

    with open(jsonl_latest, "w", encoding="utf-8") as f_latest, \
         open(jsonl_stamp,  "w", encoding="utf-8") as f_stamp:
        for r in RUN_SUMMARY:
            line = json.dumps(r, ensure_ascii=False)
            f_latest.write(line + "\n")
            f_stamp.write(line + "\n")

    # best_by_gen para CSV
    best_by_gen = {}
    for r in RUN_SUMMARY:
        g = r["generation"]
        cur = best_by_gen.get(g)
        if (cur is None) or (r.get("fitness", -1) > cur.get("fitness", -1)):
            best_by_gen[g] = r

    header = [
        "generation",
        "fitness",
        "team",
        "dom_score",
        "dom_label",
        "fullP",
        "dist_VL",
        "dist_L",
        "dist_M",
        "dist_H",
        "dist_VH",
        "pc_percent",
        "AT_cont",
        "AC_cont",
    ]

    csv_top_path = os.path.join(REPORT_DIR, f"{run_name}_top_by_generation_latest.csv")
    with open(csv_top_path, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for g in sorted(best_by_gen):
            row = {k: best_by_gen[g].get(k) for k in header}
            w.writerow(row)

    print(f"[REPORT] JSONL latest: {jsonl_latest}")
    print(f"[REPORT] JSONL stamped: {jsonl_stamp}")
    print(f"[REPORT] CSV (top por geração) salvo em: {csv_top_path}")
    return jsonl_latest, csv_top_path


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def fmt_dur(segundos: float) -> str:
    m, s = divmod(segundos, 60)
    return f"{int(m)} min {s:05.2f} s"

def _csv_list(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

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
        d2["projects"] = [{"id": p.get("id") or p.get("projectId"),
                           "osf": p.get("osf") if "osf" in p else p.get("NPS"),
                           "slf": p.get("slf") if "slf" in p else p.get("SLF")} for p in projs_in]
        norm.append(d2)
    return norm

def _as_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    m = re.search(r"(\d+)$", str(x))
    return int(m.group(1)) if m else None

def _load_grafo(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    edges = data.get("edges") or data.get("links") or (data if isinstance(data, list) else [])
    mapa, adj, ids = {}, {}, set()
    for e in edges:
        u = _as_int(e.get("source_user_id")) or _as_int(e.get("source"))
        v = _as_int(e.get("target_user_id")) or _as_int(e.get("target"))
        if u is None or v is None:
            continue
        w = float(e.get("weight", 0.0))
        a, b = (u, v) if u < v else (v, u)
        mapa[(a, b)] = max(mapa.get((a, b), 0.0), w)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        ids.update([u, v])
    return mapa, adj, ids

# instâncias globais
MAPA_PESOS_GRAFO, ADJ, VALID_IDS = _load_grafo(GRAFO_GRAPH_PATH)
DEVS = [d for d in _carregar_devs(DB_PATH) if int(d["id"]) in VALID_IDS]
CANDIDATOS_IDS = [int(d["id"]) for d in DEVS]

# projeto alvo será configurado no run_random_search / main
PROJETO_ALVO = {}


def _avaliar_time_com_cache(team, projeto_alvo, log=False):
    """Wrapper para avaliar_equipe com cache e contagem de chamadas únicas."""
    global UNIQUE_EVALS
    k = _key(team)
    if k in FITNESS_CACHE:
        res = FITNESS_CACHE[k]
        if log:
            # se quiser forçar log, podemos re-chamar só para logging textual,
            # mas mantemos contagem de UNIQUE_EVALS inalterada.
            try:
                avaliar_equipe(team, projeto_alvo, log=True)
            except Exception:
                pass
    else:
        res = avaliar_equipe(team, projeto_alvo, log=log)
        FITNESS_CACHE[k] = res
        UNIQUE_EVALS += 1
    return res


# ------------------------------------------------------------
# Núcleo do Random Search
# ------------------------------------------------------------

def run_random_search(PROJETO_ALVO_EXTERNO, team_size, num_evals, seed=123, run_name="Random_run"):
    """
    Random Search baseline compatível com o GA.

    Args:
        PROJETO_ALVO_EXTERNO (dict): mesmo formato usado pelo GA.
        team_size (int): tamanho da equipe.
        num_evals (int): orçamento de avaliações.
        seed (int): semente aleatória.
        run_name (str): prefixo para arquivos de log/relatório.

    Returns:
        dict: {"best_team": [...], "best_fitness": float, "duration_sec": float,
               "unique_evals": int}
    """
    global PROJETO_ALVO, RUN_SUMMARY, FITNESS_CACHE, UNIQUE_EVALS

    # reset de estruturas globais para este run
    PROJETO_ALVO = dict(PROJETO_ALVO_EXTERNO)
    PROJETO_ALVO["tamanhoEquipe"] = team_size
    RUN_SUMMARY = []
    FITNESS_CACHE = {}
    UNIQUE_EVALS = 0

    if len(CANDIDATOS_IDS) < team_size:
        raise RuntimeError("Candidatos insuficientes presentes no grafo para formar uma equipe.")

    random.seed(seed)

    enable_run_report(run_name=run_name)
    print(f"[RandomSearch] team_size={team_size}, num_evals={num_evals}, seed={seed}")
    print(f"[RandomSearch] Total de candidatos no grafo: {len(CANDIDATOS_IDS)}")

    inicio = time.perf_counter()

    best_team = None
    best_fitness = float("-inf")
    seen_keys = set()

    eval_idx = 0
    while eval_idx < num_evals:
        # sorteia equipe única
        team = random.sample(CANDIDATOS_IDS, team_size)
        k = _key(team)
        if k in seen_keys:
            continue
        seen_keys.add(k)

        eval_idx += 1
        print(f"\n[RANDOM TEAM #{eval_idx:04d}] ids={team}")

        res = _avaliar_time_com_cache(team, PROJETO_ALVO, log=True)
        fit = res.get("media_AE", float("-inf"))

        _append_run_row(eval_idx, team, res)

        if fit > best_fitness:
            best_fitness = fit
            best_team = list(team)
            print(f"[RandomSearch] Novo melhor: fit={best_fitness:.4f}, team={best_team}")

    dur = time.perf_counter() - inicio

    print(f"\n[RandomSearch] concluído em {fmt_dur(dur)}  (total {dur:.2f} s)")
    print(f"[RandomSearch] Melhor fitness: {best_fitness:.4f}")
    print(f"[RandomSearch] Melhor equipe: {best_team}")
    print(f"[RandomSearch] Avaliações únicas de BN: {UNIQUE_EVALS}")

    _save_jsonl_and_csv(run_name=run_name)
    disable_run_report()

    return {
        "best_team": best_team,
        "best_fitness": best_fitness,
        "duration_sec": dur,
        "unique_evals": UNIQUE_EVALS,
    }


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def _build_projeto_alvo_from_args(args):
    return {
        "dominio": {
            "must":   _csv_list(args.dom_m),
            "should": _csv_list(args.dom_s),
            "could":  _csv_list(args.dom_c),
        },
        "ecossistema": {
            "should": _csv_list(args.eco_s),
            "could":  _csv_list(args.eco_c),
        },
        "linguagens": {
            "should": _csv_list(args.ling_s),
            "could":  _csv_list(args.ling_c),
        },
        # tamanhoEquipe será setado em run_random_search
    }


def main():
    parser = argparse.ArgumentParser(description="Random Search baseline para formação de equipes.")
    parser.add_argument("--team-size", type=int, default=4, help="Tamanho da equipe")
    parser.add_argument("--dom-m",   type=str, default="Cloud", help="Domínio MUST, CSV")
    parser.add_argument("--dom-s",   type=str, default="Web",   help="Domínio SHOULD, CSV")
    parser.add_argument("--dom-c",   type=str, default="AI",    help="Domínio COULD, CSV")
    parser.add_argument("--eco-s",   type=str, default="", help="Ecossistema SHOULD, CSV")
    parser.add_argument("--eco-c",   type=str, default="", help="Ecossistema COULD, CSV")
    parser.add_argument("--ling-s",  type=str, default="", help="Linguagens SHOULD, CSV")
    parser.add_argument("--ling-c",  type=str, default="", help="Linguagens COULD, CSV")

    parser.add_argument("--budget", type=int, default=None,
                        help="Orçamento de avaliações. Se None, usa pop_size * (generations + 1).")
    parser.add_argument("--pop-size", type=int, default=6,
                        help="Tamanho da população do GA para calcular orçamento padrão.")
    parser.add_argument("--generations", type=int, default=10,
                        help="Número de gerações do GA para calcular orçamento padrão.")
    parser.add_argument("--seed", type=int, default=123, help="Semente aleatória.")
    parser.add_argument("--run-name", type=str, default="Random_run",
                        help="Prefixo para arquivos de log/relatório.")

    args = parser.parse_args()

    projeto_alvo = _build_projeto_alvo_from_args(args)
    team_size = args.team_size

    if args.budget is not None and args.budget > 0:
        num_evals = args.budget
    else:
        num_evals = args.pop_size * (args.generations + 1)

    print(f"[CLI] Random Search baseline")
    print(f"[CLI] team_size={team_size}, num_evals={num_evals}, seed={args.seed}")
    print(f"[CLI] Projeto alvo: {json.dumps(projeto_alvo, ensure_ascii=False)}")

    result = run_random_search(
        PROJETO_ALVO_EXTERNO=projeto_alvo,
        team_size=team_size,
        num_evals=num_evals,
        seed=args.seed,
        run_name=args.run_name,
    )

    print("\n[CLI] Resultado final (Random Search):")
    print(f"  best_team   = {result['best_team']}")
    print(f"  best_fitness= {result['best_fitness']:.4f}")
    print(f"  duration_sec= {result['duration_sec']:.2f}")
    print(f"  unique_evals= {result['unique_evals']}")

if __name__ == "__main__":
    main()
