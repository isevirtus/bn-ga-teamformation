#eval_team_per_project_bn_full.py
# -*- coding: utf-8 -*-

"""
Avalia as equipes do CSV (6 projetos, 5 equipes por projeto) usando a BN via Pipeline/evaluate_teams.py
e gera outro CSV com AE_mean, ranking por projeto, AT_cont/AC_cont, cobertura técnica M/S/C e tempos.

Entrada:
- experimento_ranking/team_per_project.csv  (delimitado por ';')

Saída:
- experimento_ranking/team_per_project_bn_scored_full.csv  (delimitado por ';')
"""

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path

# ------------------------------------------------------------
# Importa o pipeline do seu projeto
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # .../STFP
sys.path.insert(0, str(BASE_DIR))

from Pipeline import evaluate_teams as et  # usa DB_NORM, get_bn, pesos, grafo, etc.


# ------------------------------------------------------------
# Normalização / split robusto de skills
# ------------------------------------------------------------
_SPLIT_RE = re.compile(r"[;,|/]+")  # separadores comuns dentro de strings

def norm_token(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def explode_tokens(value) -> list[str]:
    """
    Aceita string ou lista. Se string tiver "A; B, C" quebra em tokens.
    Mantém tokens multi-palavra ("aws lambda").
    """
    if value is None:
        return []
    out = []
    if isinstance(value, list):
        items = value
    else:
        items = [str(value)]

    for it in items:
        it = str(it).strip()
        if not it:
            continue
        parts = _SPLIT_RE.split(it)
        for p in parts:
            t = norm_token(p)
            if t:
                out.append(t)
    return out


# ------------------------------------------------------------
# Parsing do projeto alvo (DOM/ECO/LING com M/S/C)
# ------------------------------------------------------------
_RE_DIM_LINE = re.compile(r"^(DOM|ECO|LING)\s*(.*)$", re.IGNORECASE)
_RE_MSC = re.compile(r"M\[(.*?)\]\s*S\[(.*?)\]\s*C\[(.*?)\]", re.IGNORECASE)

def parse_projeto_alvo(req_text: str) -> dict:
    req_text = (req_text or "").strip()
    lines = [l.strip() for l in req_text.splitlines() if l.strip()]

    out = {
        "dominio": {"must": [], "should": [], "could": []},
        "ecossistema": {"must": [], "should": [], "could": []},
        "linguagens": {"must": [], "should": [], "could": []},
    }
    map_dim = {"DOM": "dominio", "ECO": "ecossistema", "LING": "linguagens"}

    for line in lines:
        m = _RE_DIM_LINE.match(line)
        if not m:
            continue
        dim = m.group(1).upper()
        rest = m.group(2)

        m2 = _RE_MSC.search(rest)
        if not m2:
            continue

        must_raw, should_raw, could_raw = m2.group(1), m2.group(2), m2.group(3)

        def split_list(x):
            x = (x or "").strip()
            if not x:
                return []
            return [norm_token(t) for t in x.split(",") if norm_token(t)]

        key = map_dim.get(dim)
        if key:
            out[key]["must"] = split_list(must_raw)
            out[key]["should"] = split_list(should_raw)
            out[key]["could"] = split_list(could_raw)

    return out


# ------------------------------------------------------------
# Parsing das equipes do CSV
# ------------------------------------------------------------
def dev_to_int(dev_str: str) -> int:
    m = re.search(r"(\d+)$", str(dev_str).strip())
    if not m:
        raise ValueError(f"Não consegui extrair ID do dev: {dev_str}")
    return int(m.group(1))

def team_str(devs: list[str]) -> str:
    return "[" + ";".join(devs) + "]"

def load_teams_from_csv(path_csv: Path):
    """
    Interpretação do seu CSV:
    - blocos por projeto (P1..P6)
    - 5 equipes por projeto
    - há uma linha no bloco com o texto DOM/ECO/LING (projeto alvo)
    """
    raw = path_csv.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(raw), delimiter=";", quotechar='"')
    rows = list(reader)
    if not rows:
        raise ValueError("CSV vazio.")

    header = [h.lstrip("\ufeff") for h in rows[0]]
    idx = {name: i for i, name in enumerate(header)}

    required_cols = ["projeto_id", "projeto e requisitos", "dev1", "dev2", "dev3", "dev4"]
    for c in required_cols:
        if c not in idx:
            raise ValueError(f"Coluna obrigatória não encontrada: {c}. Header={header}")

    data = rows[1:]

    teams = []
    projeto_atual = None
    projeto_alvo_atual = None
    team_counter = 0

    for r in data:
        if not r or all((x.strip() == "" for x in r)):
            continue

        pid = r[idx["projeto_id"]].strip()
        req_text = r[idx["projeto e requisitos"]].strip()

        devs = [
            r[idx["dev1"]].strip(),
            r[idx["dev2"]].strip(),
            r[idx["dev3"]].strip(),
            r[idx["dev4"]].strip(),
        ]
        has_team = all(devs)

        if pid:
            projeto_atual = pid
            projeto_alvo_atual = None
            team_counter = 0

        if projeto_atual is None:
            continue

        # Identifica a linha que contém DOM/ECO/LING no formato M/S/C
        if ("DOM" in req_text.upper()) and ("M[" in req_text) and ("ECO" in req_text.upper()):
            projeto_alvo_atual = parse_projeto_alvo(req_text)

        if has_team:
            t_id = f"{projeto_atual}-TEAM{team_counter}"
            teams.append({
                "projeto_id": projeto_atual,
                "team_id": t_id,
                "devs_str": devs,
                "team_str": team_str(devs),
                "team_ids_int": [dev_to_int(d) for d in devs],
                "projeto_alvo": projeto_alvo_atual,  # pode ser None até achar a linha de requisitos
            })
            team_counter += 1

    # Garantias: 5 equipes por projeto + projeto alvo existe
    by_proj = {}
    for t in teams:
        by_proj.setdefault(t["projeto_id"], []).append(t)

    for p, lst in by_proj.items():
        by_proj[p] = lst[:5]
        if len(by_proj[p]) != 5:
            raise ValueError(f"Projeto {p} ficou com {len(by_proj[p])} equipes (esperado 5).")

        pa = None
        for x in by_proj[p]:
            if x["projeto_alvo"] is not None:
                pa = x["projeto_alvo"]
                break
        if pa is None:
            raise ValueError(f"Não encontrei requisitos DOM/ECO/LING para o projeto {p} no CSV.")
        for x in by_proj[p]:
            x["projeto_alvo"] = pa

    # Ordena P1..Pn
    ordered = []
    for p in sorted(by_proj.keys(), key=lambda s: int(re.search(r"\d+", s).group(0))):
        ordered.extend(by_proj[p])

    return ordered


# ------------------------------------------------------------
# Cobertura técnica M/S/C por dimensão
# ------------------------------------------------------------
def build_dev_map():
    devs = et.DB_NORM.get("developers", [])
    mp = {}
    for d in devs:
        try:
            mp[int(d.get("id"))] = d
        except Exception:
            continue
    return mp

DEV_MAP = build_dev_map()

def team_skill_sets(team_ids: list[int]) -> dict:
    dom_set, eco_set, ling_set = set(), set(), set()

    for uid in team_ids:
        d = DEV_MAP.get(int(uid))
        if not d:
            continue

        # dominio já é lista (às vezes string)
        dom_set.update(explode_tokens(d.get("dominio", [])))

        # ecossistema_list existe no DB_NORM
        eco_set.update(explode_tokens(d.get("ecossistema_list", d.get("ecossistema", []))))

        # linguagens_list existe no DB_NORM
        ling_set.update(explode_tokens(d.get("linguagens_list", d.get("linguagens", []))))

    return {"dominio": dom_set, "ecossistema": eco_set, "linguagens": ling_set}

def coverage_block(req_list: list[str], skill_set: set[str]):
    req_norm = [norm_token(x) for x in (req_list or []) if norm_token(x)]
    req_unique = sorted(set(req_norm))
    covered = [x for x in req_unique if x in skill_set]
    return {
        "total": len(req_unique),
        "covered": len(covered),
        "covered_list": ",".join(covered) if covered else "",
    }

def compute_coverage(projeto_alvo: dict, team_ids: list[int]) -> dict:
    skills = team_skill_sets(team_ids)
    out = {}

    for dim in ("dominio", "ecossistema", "linguagens"):
        for level, tag in (("must", "m"), ("should", "s"), ("could", "c")):
            blk = coverage_block(projeto_alvo.get(dim, {}).get(level, []), skills[dim])
            out[f"{dim}_{tag}_total"] = blk["total"]
            out[f"{dim}_{tag}_covered"] = blk["covered"]
            out[f"{dim}_{tag}_covered_list"] = blk["covered_list"]

    return out


# ------------------------------------------------------------
# BN: tenta limpar evidência (se sua lib suportar)
# ------------------------------------------------------------
def try_clear_evidence(bn):
    for m in ("clearAllEvidence", "clearEvidence", "resetEvidence"):
        if hasattr(bn, m):
            try:
                getattr(bn, m)()
            except Exception:
                pass
            return


# ------------------------------------------------------------
# Avaliação de 1 equipe (com tempos + AT_cont/AC_cont)
# ------------------------------------------------------------
def avaliar_equipe_bn_full(team_ids: list[int], projeto_alvo: dict):
    t0_team = time.perf_counter()

    # 1) Dimension scoring (técnico)
    t0_dim = time.perf_counter()
    res_dims = et.avaliar_todos_as_dimensions(
        team_ids=team_ids,
        db=et.DB_NORM,
        projeto_alvo=projeto_alvo,
        pesos=et.PESOS,
        nota_sem_must=0.0,
    )
    t_dim = time.perf_counter() - t0_dim

    dom_rotulo = res_dims["dominio"]["rotulo"]
    eco_rotulo = res_dims["ecossistema"]["rotulo"]
    ling_rotulo = res_dims["linguagens"]["rotulo"]

    dom_score = res_dims["dominio"]["score"]
    eco_score = res_dims["ecossistema"]["score"]
    ling_score = res_dims["linguagens"]["score"]

    # 2) PCs do grafo
    t0_pc_graph = time.perf_counter()
    pcs = et.pcs_da_equipe_por_grafo(team_ids, et.MAPA_PESOS_GRAFO)
    t_pc_graph = time.perf_counter() - t0_pc_graph

    # 3) PC -> score
    t0_pc_transform = time.perf_counter()
    props = et.classificar_pc_por_faixa_e_score(pcs)
    t_pc_transform = time.perf_counter() - t0_pc_transform

    props_pc_score = props["props"]
    pc_score = props["pc_score"]
    pc_label = props["pc_label"]

    print("props:",props)
    print("pc_score:",pc_score)
    print("pc_label:",pc_label)

    # 4) BN inferência
    bn = et.get_bn()
    try_clear_evidence(bn)

    bn.setEvidence("Dom", dom_rotulo)
    if eco_rotulo is not None:
        bn.setEvidence("Eco", eco_rotulo)
    if ling_rotulo is not None:
        bn.setEvidence("Ling", ling_rotulo)
    bn.setEvidence("AC", pc_label)
    

    
    
    dist_ac = bn.calculateTPN("AC")
    AC_cont = float(sum(p * v for p, v in zip(dist_ac, et.CENTROIDES)))
    # Salvar distribuição (VL..VH) como string para CSV/dashboard
    dist_ac_str = ",".join([f"{x:.6f}" for x in dist_ac])
    
    print("dist_ac", dist_ac)

    print("AC_cont", AC_cont)

    t0_inf_at = time.perf_counter()
    dist_at = bn.calculateTPN("AT")
    t_infer_at = time.perf_counter() - t0_inf_at
    AT_cont = float(sum(p * v for p, v in zip(dist_at, et.CENTROIDES)))
    dist_at_str = ",".join([f"{x:.6f}" for x in dist_at])

    t0_inf_ae = time.perf_counter()
    dist_ae = bn.calculateTPN("AE")
    t_infer_ae = time.perf_counter() - t0_inf_ae
    ae_mean = float(sum(p * v for p, v in zip(dist_ae, et.CENTROIDES)))

    t_team_total = time.perf_counter() - t0_team

    return {
        "ae_bn_mean": ae_mean,
        "AT_cont": AT_cont,
        "AC_cont": AC_cont,
        "dist_AT": dist_at_str,
        "dist_AC": dist_ac_str,
        "dist_AE": ",".join([f"{x:.6f}" for x in dist_ae]),
        "pc_score": pc_score,
        "pc_label": pc_label,
        "dominio_tier": dom_rotulo,
        "ecossistema_tier": eco_rotulo,
        "linguagens_tier": ling_rotulo,
        "dominio_score": dom_score,
        "ecossistema_score": eco_score,
        "linguagens_score": ling_score,
        "t_dim_s": t_dim,
        "t_pc_graph_s": t_pc_graph,
        "t_pc_transform_s": t_pc_transform,
        "t_infer_at_s": t_infer_at,
        "t_infer_ae_s": t_infer_ae,
        "t_infer_total_s": (t_infer_at + t_infer_ae),
        "t_team_total_s": t_team_total,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV de entrada (team_per_project.csv)")
    ap.add_argument("--out", dest="out", required=True, help="CSV de saída")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    teams = load_teams_from_csv(inp)

    # Tempo para construir BN/CPT (1ª vez): get_bn() chama criar_rede_fitness() :contentReference[oaicite:2]{index=2}
    t0_cpt = time.perf_counter()
    _ = et.get_bn()
    t_cpt_build = time.perf_counter() - t0_cpt

    # Agrupa por projeto
    by_project = {}
    for t in teams:
        by_project.setdefault(t["projeto_id"], []).append(t)

    results = []

    for p in sorted(by_project.keys(), key=lambda s: int(re.search(r"\d+", s).group(0))):
        t0_proj = time.perf_counter()

        proj_rows = []
        for team in by_project[p]:
            metrics = avaliar_equipe_bn_full(team["team_ids_int"], team["projeto_alvo"])
            cov = compute_coverage(team["projeto_alvo"], team["team_ids_int"])

            row = {
                "projeto_id": p,
                "team_id": team["team_id"],
                "team_str": team["team_str"],

                "ae_bn_mean": metrics["ae_bn_mean"],
                "rank_bn": None,  # preenche depois

                "AT_cont": metrics["AT_cont"],
                "AC_cont": metrics["AC_cont"],
                "dist_AT": metrics["dist_AT"],
                "dist_AC": metrics["dist_AC"],
                "dist_AE": metrics["dist_AE"],

                "pc_score": metrics["pc_score"],
                "pc_label": metrics["pc_label"],

                
                "dominio_tier": metrics["dominio_tier"],
                "ecossistema_tier": metrics["ecossistema_tier"],
                "linguagens_tier": metrics["linguagens_tier"],
                "dominio_score": metrics["dominio_score"],
                "ecossistema_score": metrics["ecossistema_score"],
                "linguagens_score": metrics["linguagens_score"],

                "t_cpt_build_s": t_cpt_build,
                "t_dim_s": metrics["t_dim_s"],
                "t_pc_graph_s": metrics["t_pc_graph_s"],
                "t_pc_transform_s": metrics["t_pc_transform_s"],
                "t_infer_at_s": metrics["t_infer_at_s"],
                "t_infer_ae_s": metrics["t_infer_ae_s"],
                "t_infer_total_s": metrics["t_infer_total_s"],
                "t_team_total_s": metrics["t_team_total_s"],

                "t_project_total_s": None,
                "t_project_total_plus_cpt_s": None,
            }

            row.update(cov)
            proj_rows.append(row)

        t_project_total = time.perf_counter() - t0_proj

        # rank por projeto: 1 = maior AE
        proj_rows_sorted = sorted(proj_rows, key=lambda r: r["ae_bn_mean"], reverse=True)
        for rk, r in enumerate(proj_rows_sorted, start=1):
            r["rank_bn"] = rk

        

        # repete o tempo do projeto em todas as linhas do projeto
        for r in proj_rows:
            r["t_project_total_s"] = t_project_total
            r["t_project_total_plus_cpt_s"] = t_project_total + t_cpt_build

        results.extend(proj_rows)

    # Ordem de colunas
    cov_cols = []
    for dim in ("dominio", "ecossistema", "linguagens"):
        for tag in ("m", "s", "c"):
            cov_cols.extend([
                f"{dim}_{tag}_total",
                f"{dim}_{tag}_covered",
                f"{dim}_{tag}_covered_list",
            ])

    fieldnames = [
        "projeto_id", "team_id", "team_str",
        "ae_bn_mean", "rank_bn",
        "AT_cont", "AC_cont", "dist_AT", "dist_AC", "dist_AE", "pc_score", "pc_label",
        "dominio_tier", "ecossistema_tier", "linguagens_tier",
        "dominio_score", "ecossistema_score", "linguagens_score",
        "t_cpt_build_s",
        "t_dim_s", "t_pc_graph_s", "t_pc_transform_s",
        "t_infer_at_s", "t_infer_ae_s", "t_infer_total_s",
        "t_team_total_s",
        "t_project_total_s", "t_project_total_plus_cpt_s",
        *cov_cols,
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        wr.writeheader()
        for r in results:
            rr = dict(r)

            # Formatação numérica
            rr["ae_bn_mean"] = f"{float(rr['ae_bn_mean']):.6f}"
            rr["AT_cont"] = f"{float(rr['AT_cont']):.6f}"
            rr["AC_cont"] = f"{float(rr['AC_cont']):.6f}"
            rr["pc_score"] = f"{float(rr['pc_score']):.6f}"
            for k in ("dominio_score", "ecossistema_score", "linguagens_score"):
                if rr.get(k) is not None and rr.get(k) != "":
                    rr[k] = f"{float(rr[k]):.6f}"

            for k in fieldnames:
                if k.endswith("_s") and rr.get(k) is not None:
                    rr[k] = f"{float(rr[k]):.6f}"

            wr.writerow(rr)

    print(f"[OK] Saída gerada em: {out}")
    print(f"[INFO] t_cpt_build_s = {t_cpt_build:.6f}s (BN/CPT build na 1ª vez)")


if __name__ == "__main__":
    main()
