#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import pandas as pd
from collections import Counter
import csv

# ----------------------------
# IO helpers
# ----------------------------

def sniff_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","

def read_csv_auto(path: str) -> pd.DataFrame:
    sep = sniff_sep(path)
    return pd.read_csv(path, sep=sep, engine="python", quoting=csv.QUOTE_MINIMAL)

def normalize_project_col(df: pd.DataFrame) -> pd.DataFrame:
    if "project_id" not in df.columns and "projeto_id" in df.columns:
        df = df.rename(columns={"projeto_id": "project_id"})
    return df

def normalize_team_col(df: pd.DataFrame) -> pd.DataFrame:
    # mantido por compatibilidade (mesmo não usando team_id)
    if "team_id" not in df.columns and "equipe_id" in df.columns:
        df = df.rename(columns={"equipe_id": "team_id"})
    return df

def ensure_team_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante team_str no formato:
      [Dev169;Dev27;Dev81;Dev363]
    Se já existir, não mexe.
    Se não existir, tenta construir via dev1..dev4.
    """
    if "team_str" in df.columns:
        return df

    dev_cols = ["dev1", "dev2", "dev3", "dev4"]
    if all(c in df.columns for c in dev_cols):
        def build_row(r):
            if any(pd.isna(r[c]) for c in dev_cols):
                return None
            return f"[{r['dev1']};{r['dev2']};{r['dev3']};{r['dev4']}]"
        df["team_str"] = df.apply(build_row, axis=1)

    return df

# ----------------------------
# Metrics (mantidas do original)
# ----------------------------

def reciprocal_rank(g: pd.DataFrame, key_col: str, pred_col: str, exp_col: str) -> float:
    """
    RR por projeto:
    - pega o(s) Top-1 do expert (suporta empates),
    - encontra a melhor posição desses itens no ranking do modelo,
    - retorna 1/posição.
    """
    exp_min = g[exp_col].min()
    best_expert_teams = set(g.loc[g[exp_col] == exp_min, key_col])

    if not best_expert_teams:
        return 0.0

    g_sorted = g.sort_values([pred_col, key_col], ascending=[True, True])

    best_pos = None
    for i, (_, row) in enumerate(g_sorted.iterrows(), 1):
        if row[key_col] in best_expert_teams:
            best_pos = i
            break

    return 1.0 / best_pos if best_pos is not None else 0.0

def kendall_tau_b(x, y) -> float:
    """
    Kendall tau-b (suporta empates). O(n^2) suficiente para n=5.
    """
    n = len(x)
    if n < 2:
        return float("nan")

    n0 = n * (n - 1) / 2

    cx = Counter(x)
    cy = Counter(y)

    n1 = sum(v * (v - 1) / 2 for v in cx.values())  # ties em x
    n2 = sum(v * (v - 1) / 2 for v in cy.values())  # ties em y

    C = D = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or y[i] == y[j]:
                continue
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                C += 1
            elif s < 0:
                D += 1

    denom = math.sqrt((n0 - n1) * (n0 - n2))
    return float("nan") if denom == 0 else (C - D) / denom

def spearman_rho(x_ranks, y_ranks) -> float:
    import numpy as np
    x = np.array(x_ranks, dtype=float)
    y = np.array(y_ranks, dtype=float)
    if len(x) < 2:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def dcg_at_k(relevances, k: int) -> float:
    k = min(k, len(relevances))
    dcg = 0.0
    for i in range(k):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg

def ndcg_at_k(pred_order_ids, expert_rank_by_id, k: int = 5) -> float:
    rel_pred = []
    for tid in pred_order_ids:
        r = expert_rank_by_id.get(tid, None)
        if r is None or r <= 0:
            return float("nan")
        rel_pred.append(1.0 / float(r))

    dcg = dcg_at_k(rel_pred, k)

    rel_all = [1.0 / float(r) for r in expert_rank_by_id.values() if r is not None and r > 0]
    rel_all.sort(reverse=True)
    idcg = dcg_at_k(rel_all, k)

    return float("nan") if idcg == 0 else dcg / idcg

def top1_match_with_ties(g: pd.DataFrame, key_col: str, pred_col: str, exp_col: str) -> int:
    pred_best = set(g.loc[g[pred_col] == g[pred_col].min(), key_col].tolist())
    exp_best  = set(g.loc[g[exp_col]  == g[exp_col].min(),  key_col].tolist())
    return 1 if len(pred_best & exp_best) > 0 else 0

def mean_abs_rank_diff(pred_rank_by_id, expert_rank_by_id) -> float:
    diffs = []
    for tid, pr in pred_rank_by_id.items():
        er = expert_rank_by_id.get(tid, None)
        if er is None or er <= 0:
            return float("nan")
        diffs.append(abs(float(pr) - float(er)))
    return float("nan") if not diffs else sum(diffs) / len(diffs)

# ----------------------------
# Novas métricas pedidas (sem inventar outras)
# ----------------------------

def exact_match_count(g: pd.DataFrame, pred_col: str, exp_col: str) -> int:
    return int((g[pred_col] == g[exp_col]).sum())

def within_1_count(g: pd.DataFrame, pred_col: str, exp_col: str) -> int:
    return int((g[pred_col] - g[exp_col]).abs().le(1).sum())

def top1_within_1(g: pd.DataFrame, key_col: str, pred_col: str, exp_col: str) -> int:
    """
    "errou até 1 posição" no Top-1:
    - pega o(s) Top-1 do modelo (suporta empates)
    - verifica se algum deles está no Top-(min_rank_expert + 1)
      (para rank começando em 1, equivale ao Top-2 do expert)
    """
    exp_min = float(g[exp_col].min())
    threshold = exp_min + 1.0

    pred_best_ids = set(g.loc[g[pred_col] == g[pred_col].min(), key_col].tolist())
    if not pred_best_ids:
        return 0

    # melhor rank_expert entre os Top-1 do modelo
    best_er = float(g.loc[g[key_col].isin(pred_best_ids), exp_col].min())
    return 1 if best_er <= threshold else 0

def top1_miss_by_1(g: pd.DataFrame, key_col: str, pred_col: str, exp_col: str) -> int:
    """
    1 se NÃO acertou Top-1, mas ficou a até 1 posição (Top-2 do expert).
    """
    m = top1_match_with_ties(g, key_col, pred_col, exp_col)
    w = top1_within_1(g, key_col, pred_col, exp_col)
    return 1 if (m == 0 and w == 1) else 0

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV único com project_id/projeto_id, team_str, rank_expert e rank_bn")
    ap.add_argument("--expert-col", default="rank_expert")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="metrics_rank.csv")
    args = ap.parse_args()

    df = read_csv_auto(args.inp)

    df = normalize_project_col(df)
    df = normalize_team_col(df)
    df = ensure_team_str(df)

    # completa project_id se vier agrupado com vazios
    if "project_id" in df.columns:
        df["project_id"] = df["project_id"].ffill()

    # remove linhas vazias
    if "team_str" in df.columns:
        df = df[df["team_str"].notna()].copy()

    # valida colunas mínimas
    need = ["project_id", "team_str", "rank_bn", args.expert_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns {miss}. Found: {list(df.columns)}")

    merged = df[["project_id", "team_str", "rank_bn", args.expert_col]].copy()

    merged["rank_bn"] = pd.to_numeric(merged["rank_bn"], errors="coerce")
    merged[args.expert_col] = pd.to_numeric(merged[args.expert_col], errors="coerce")

    rows = []
    projects = list(merged.groupby("project_id").groups.keys())

    for pid, g in merged.groupby("project_id"):
        g = g.dropna(subset=["rank_bn", args.expert_col]).copy()

        if len(g) < 2:
            rows.append({
                "project_id": pid, "n_items": len(g),
                "kendall_tau_b": float("nan"), "spearman_rho": float("nan"),
                "ndcg_at_k": float("nan"), "top1_match": float("nan"),
                "mrr": float("nan"), "mad_rank": float("nan"),
                "match_exato_count": float("nan"), "within_1_count": float("nan"),
                "top1_within_1": float("nan"), "top1_miss_by_1": float("nan"),
                "note": "menos de 2 itens"
            })
            continue

        if (g[args.expert_col] <= 0).any():
            rows.append({
                "project_id": pid, "n_items": len(g),
                "kendall_tau_b": float("nan"), "spearman_rho": float("nan"),
                "ndcg_at_k": float("nan"), "top1_match": float("nan"),
                "mrr": float("nan"), "mad_rank": float("nan"),
                "match_exato_count": float("nan"), "within_1_count": float("nan"),
                "top1_within_1": float("nan"), "top1_miss_by_1": float("nan"),
                "note": "rank expert <=0 detectado"
            })
            continue

        # Ordem predita: rank_bn menor é melhor; desempate estável por team_str
        g_sorted_pred = g.sort_values(["rank_bn", "team_str"], ascending=True)
        pred_order = g_sorted_pred["team_str"].tolist()

        pred_rank_by_id = dict(zip(g["team_str"], g["rank_bn"]))
        expert_rank_by_id = dict(zip(g["team_str"], g[args.expert_col]))

        # listas alinhadas pela mesma ordem de keys (n=5, ok)
        keys = list(pred_rank_by_id.keys())
        x = [pred_rank_by_id[t] for t in keys]
        y = [expert_rank_by_id[t] for t in keys]

        mex = exact_match_count(g, "rank_bn", args.expert_col)
        w1c = within_1_count(g, "rank_bn", args.expert_col)

        rows.append({
            "project_id": pid,
            "n_items": len(g),
            "kendall_tau_b": kendall_tau_b(x, y),
            "spearman_rho": spearman_rho(x, y),
            "ndcg_at_k": ndcg_at_k(pred_order, expert_rank_by_id, k=args.k),
            "top1_match": top1_match_with_ties(g, "team_str", "rank_bn", args.expert_col),
            "mrr": reciprocal_rank(g, "team_str", "rank_bn", args.expert_col),
            "mad_rank": mean_abs_rank_diff(pred_rank_by_id, expert_rank_by_id),
            "match_exato_count": mex,
            "match_exato_rate": mex / float(len(g)) if len(g) else float("nan"),
            "within_1_count": w1c,
            "within_1_rate": w1c / float(len(g)) if len(g) else float("nan"),
            "top1_within_1": top1_within_1(g, "team_str", "rank_bn", args.expert_col),
            "top1_miss_by_1": top1_miss_by_1(g, "team_str", "rank_bn", args.expert_col),
            "note": ""
        })

    out = pd.DataFrame(rows)

    # Overall mean (macro-average) + contagens pedidas
    def nanmean(s): return float(pd.to_numeric(s, errors="coerce").mean(skipna=True))

    # contagens (não médias)
    top1_match_count = int(pd.to_numeric(out["top1_match"], errors="coerce").fillna(0).sum())
    top1_within_1_count = int(pd.to_numeric(out["top1_within_1"], errors="coerce").fillna(0).sum())
    top1_miss_by_1_count = int(pd.to_numeric(out["top1_miss_by_1"], errors="coerce").fillna(0).sum())
    match_exato_total = int(pd.to_numeric(out["match_exato_count"], errors="coerce").fillna(0).sum())
    within_1_total = int(pd.to_numeric(out["within_1_count"], errors="coerce").fillna(0).sum())
    n_projects = int(out["project_id"].nunique())

    summary = {
        "project_id": "OVERALL_MEAN",
        "n_items": int(out["n_items"].sum()),
        "kendall_tau_b": nanmean(out["kendall_tau_b"]),
        "spearman_rho": nanmean(out["spearman_rho"]),
        "ndcg_at_k": nanmean(out["ndcg_at_k"]),
        "top1_match": nanmean(out["top1_match"]),
        "mrr": nanmean(out["mrr"]),
        "mad_rank": nanmean(out["mad_rank"]),
        "match_exato_count": match_exato_total,
        "match_exato_rate": (match_exato_total / float(out["n_items"].sum())) if out["n_items"].sum() else float("nan"),
        "within_1_count": within_1_total,
        "within_1_rate": (within_1_total / float(out["n_items"].sum())) if out["n_items"].sum() else float("nan"),
        "top1_within_1": nanmean(out["top1_within_1"]),
        "top1_miss_by_1": nanmean(out["top1_miss_by_1"]),
        "top1_match_count": top1_match_count,
        "top1_within_1_count": top1_within_1_count,
        "top1_miss_by_1_count": top1_miss_by_1_count,
        "n_projects": n_projects,
        "note": "macro-mean across projects + counts"
    }

    out = pd.concat([out, pd.DataFrame([summary])], ignore_index=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] Wrote: {args.out}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
