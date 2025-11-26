#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Checking (RB do Felipe) - versão compatível com BNetwork.calculateTPN()
-------------------------------------------------------------------------------
Execute a partir da pasta-raiz do projeto (onde 'rede_bayesiana' é pacote):
  python -m rede_bayesiana.gradient_checking_rb_compat --funcao WMEAN --sigma 0.10
"""

import argparse
import numpy as np
import pandas as pd

# Importa a função de construção da RB dentro do pacote
from rede_bayesiana.rede_fitness_ag import criar_rede_fitness

ESTADOS = ['VL', 'L', 'M', 'H', 'VH']
CENTROS = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
EPS = 0.05

def soft_cpd_from_scalar(x: float, tau: float = 0.10) -> np.ndarray:
    x = float(np.clip(x, 0.0, 1.0))
    w = np.exp(-0.5 * ((x - CENTROS) / tau) ** 2)
    w = w / (w.sum() + 1e-12)
    return w.reshape(-1, 1)  # 5x1 para TabularCPD dos nós raiz

def expected_from_probs(probs) -> float:
    arr = np.array(probs, dtype=float).reshape(-1)
    arr = arr / (arr.sum() + 1e-12)
    return float((arr * CENTROS).sum())

def set_inputs_via_cpd(bn, x_dom: float, x_eco: float, x_ling: float, x_ac: float, tau_soft: float):
    """
    Define CPDs 'soft' nos nós raiz (Dom, Eco, Ling, AC).
    Como a BNetwork calcula marginais via calculateTPN(), não precisamos de updateBeliefs.
    """
    bn.setNodeCPD("Dom",  soft_cpd_from_scalar(x_dom,  tau_soft))
    bn.setNodeCPD("Eco",  soft_cpd_from_scalar(x_eco,  tau_soft))
    bn.setNodeCPD("Ling", soft_cpd_from_scalar(x_ling, tau_soft))
    bn.setNodeCPD("AC",   soft_cpd_from_scalar(x_ac,   tau_soft))

def get_marginal_AE(bn):
    """
    Usa a API própria do Felipe: bn.calculateTPN('AE') retorna lista de probs.
    """
    probs = bn.calculateTPN("AE")
    return np.array(probs).reshape(-1)

def gradient_check_at_fixed_ac(bn, at_base: float, ac_fixed: float, eps: float, tau_soft: float) -> dict:
    rows = []
    for delta in [-eps, 0.0, eps]:
        x_at = float(np.clip(at_base + delta, 0.0, 1.0))
        set_inputs_via_cpd(bn, x_at, x_at, x_at, ac_fixed, tau_soft)
        pAE = get_marginal_AE(bn)
        AE = expected_from_probs(pAE)
        rows.append({"delta": delta, "x_at": x_at, "ac": ac_fixed, "AE": AE})
    AE_minus, AE_base, AE_plus = [r["AE"] for r in rows]
    return {
        "kind": "AT|ACfixed",
        "at_base": at_base, "ac_fixed": ac_fixed,
        "AE(-eps)": AE_minus, "AE(base)": AE_base, "AE(+eps)": AE_plus,
        "monotonic_AT": (AE_minus <= AE_base <= AE_plus) or (AE_minus >= AE_base >= AE_plus),
        "delta_left": AE_base - AE_minus, "delta_right": AE_plus - AE_base
    }

def gradient_check_ac_fixed_at(bn, ac_base: float, at_fixed: float, eps: float, tau_soft: float) -> dict:
    rows = []
    for delta in [-eps, 0.0, eps]:
        x_ac = float(np.clip(ac_base + delta, 0.0, 1.0))
        set_inputs_via_cpd(bn, at_fixed, at_fixed, at_fixed, x_ac, tau_soft)
        pAE = get_marginal_AE(bn)
        AE = expected_from_probs(pAE)
        rows.append({"delta": delta, "x_ac": x_ac, "at": at_fixed, "AE": AE})
    AE_minus, AE_base, AE_plus = [r["AE"] for r in rows]
    return {
        "kind": "AC|ATfixed",
        "ac_base": ac_base, "at_fixed": at_fixed,
        "AE(-eps)": AE_minus, "AE(base)": AE_base, "AE(+eps)": AE_plus,
        "monotonic_AC": (AE_minus <= AE_base <= AE_plus) or (AE_minus >= AE_base >= AE_plus),
        "delta_left": AE_base - AE_minus, "delta_right": AE_plus - AE_base
    }

def extremes_check(bn, tau_soft: float) -> dict:
    set_inputs_via_cpd(bn, 0.0, 0.0, 0.0, 0.0, tau_soft)
    AE_min = expected_from_probs(get_marginal_AE(bn))
    set_inputs_via_cpd(bn, 1.0, 1.0, 1.0, 1.0, tau_soft)
    AE_max = expected_from_probs(get_marginal_AE(bn))
    return {"AE_min": AE_min, "AE_max": AE_max}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcao", default="WMEAN")
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--tau_soft", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Constrói a RB do Felipe
    bn = criar_rede_fitness(funcao_agregadora=args.funcao, variancia=args.sigma)

    bases = [0.20, 0.50, 0.80]
    rows = []

    # AT variando, AC fixo
    for ac_fixed in bases:
        for at_base in bases:
            r = gradient_check_at_fixed_ac(bn, at_base, ac_fixed, args.eps, args.tau_soft)
            rows.append(r)

    # AC variando, AT fixo
    for at_fixed in bases:
        for ac_base in bases:
            r = gradient_check_ac_fixed_at(bn, ac_base, at_fixed, args.eps, args.tau_soft)
            rows.append(r)

    # Extremos
    ext = extremes_check(bn, args.tau_soft)

    df = pd.DataFrame(rows)
    mono_at_ok = df[df["kind"]=="AT|ACfixed"]["monotonic_AT"].mean()
    mono_ac_ok = df[df["kind"]=="AC|ATfixed"]["monotonic_AC"].mean()
    deltas_at = df[df["kind"]=="AT|ACfixed"][["delta_left","delta_right"]].abs().mean()
    deltas_ac = df[df["kind"]=="AC|ATfixed"][["delta_left","delta_right"]].abs().mean()

    print("\n=== Gradient Checking (BNetwork.calculateTPN) ===")
    print(f"Função: {args.funcao} | σ={args.sigma:.3f} | ε={args.eps:.3f} | tau_soft={args.tau_soft:.3f}\n")
    print("-> Monotonicidade (AT|AC fixo): {:.0%}".format(mono_at_ok))
    print("-> Monotonicidade (AC|AT fixo): {:.0%}".format(mono_ac_ok))
    print("\n-> |ΔAE| médio (AT): left={:.4f}, right={:.4f}".format(deltas_at['delta_left'], deltas_at['delta_right']))
    print("-> |ΔAE| médio (AC): left={:.4f}, right={:.4f}".format(deltas_ac['delta_left'], deltas_ac['delta_right']))
    print("\n-> Extremos: AE_min={:.4f} | AE_max={:.4f}".format(ext["AE_min"], ext["AE_max"]))

    out = "gradient_check_results.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nResultados detalhados salvos em: {out}\n")
    print("Critérios sugeridos: monotonicidade ≥ 90%; |ΔAE| com ε=0.05 entre 0.02–0.05; AE_min<0.25; AE_max>0.75.")

if __name__ == "__main__":
    main()
