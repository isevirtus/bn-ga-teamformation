# pipeline/bench_bn_simple.py
import sys, os, time
from statistics import mean, median

# Deixa o sys.path apontando para .../STFP (raiz do pacote)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Algorithms.BN.team_fit_bn import criar_rede_fitness

INFER_COUNTS = {"AE": 0}
CENTROIDES = [0.1, 0.3, 0.5, 0.7, 0.9]

def set_evidence_all_VH(bn, use_pc_nodes=True):
    
    bn.setEvidence("Dom",  "VH")
    bn.setEvidence("Eco",  "VH")
    bn.setEvidence("Ling", "VH")
    # pc (VL..VH)
    bn.setEvidence("PC_VL", "VH")
    bn.setEvidence("PC_L","VH")
    bn.setEvidence("PC_M","VH")
    bn.setEvidence("PC_H","VH")
    bn.setEvidence("PC_VH","VH")
    
def set_evidence_all_VL(bn, use_pc_nodes=True):
    bn.setEvidence("Dom",  "VL")
    bn.setEvidence("Eco",  "VL")
    bn.setEvidence("Ling", "VL")
    # pc (VL..VH)
    bn.setEvidence("PC_VL", "VL")
    bn.setEvidence("PC_L","VL")
    bn.setEvidence("PC_M","VL")
    bn.setEvidence("PC_H","VL")
    bn.setEvidence("PC_VH","VL")

def infer_AE_once(bn):
    """Mede apenas o tempo da inferência de AE."""
    t0 = time.perf_counter()
    dist_AE = bn.calculateTPN("AE")  # distribuição [VL..VH]
    dt = time.perf_counter() - t0
    ae_mean = sum(p*v for p, v in zip(dist_AE, CENTROIDES))
    INFER_COUNTS["AE"] += 1     # <--- conta 1 inferência de AE
    return dt, dist_AE, ae_mean

def run_repeated(bn, reps=50):
    """Repete apenas a inferência para medir estabilidade de tempo."""
    times = []
    last = None
    for _ in range(reps):
        dt, dist, m = infer_AE_once(bn)
        times.append(dt)
        last = (dist, m)
    return times, last

def fmt_dist(d):
    return "[" + ", ".join(f"{x:.4f}" for x in d) + "]"

def main():
    print("=== BN Bench (evidências diretas, sem Dimension Scoring nem PC transformer) ===")

    # 1) Construção da rede (normalmente onde CPT/TPN é criada)
    t0 = time.perf_counter()
    bn = criar_rede_fitness(
    func_at="WMAX",  var_at=0.001, pesos_dom_eco_ling=[2, 2, 5],
    func_ac="WMEAN",  var_ac=0.5, pesos_pc=[1, 1, 1, 1, 5],
    func_ae="WMIN",  var_ae=0.005, pesos_at_ac=[3,1],
)


    

    t_build = time.perf_counter() - t0
    print(f"[BUILD] Rede criada em {t_build:.4f} s")

    # 2) VH
    print("\n--- CENÁRIO: tudo VH ---")
    set_evidence_all_VH(bn, use_pc_nodes=True)
    # Primeira inferência (detecta custo de 'warmup' se existir lazy compile)
    t1, dist1, m1 = infer_AE_once(bn)
    print(f"[INFER 1ª] AE em {t1:.4f} s | dist={fmt_dist(dist1)} | mean={m1:.4f}")

    times_vh, last_vh = run_repeated(bn, reps=50)
    print(f"[INFER x50] média={mean(times_vh):.4f} s | mediana={median(times_vh):.4f} s | piores={max(times_vh):.4f} s")
    print(f"[INFER x50] última dist={fmt_dist(last_vh[0])} | mean={last_vh[1]:.4f}")

    # 3) VL (mesma BN, só troca evidências)
    print("\n--- CENÁRIO: tudo VL ---")
    set_evidence_all_VL(bn, use_pc_nodes=True)
    t2, dist2, m2 = infer_AE_once(bn)
    print(f"[INFER 1ª] AE em {t2:.4f} s | dist={fmt_dist(dist2)} | mean={m2:.4f}")

    times_vl, last_vl = run_repeated(bn, reps=50)
    print(f"[INFER x50] média={mean(times_vl):.4f} s | mediana={median(times_vl):.4f} s | piores={max(times_vl):.4f} s")
    print(f"[INFER x50] última dist={fmt_dist(last_vl[0])} | mean={last_vl[1]:.4f}")

    # 4) Resumo
    print("\n=== RESUMO ===")
    print(f"Tempo de criação (build): {t_build:.4f} s")
    print(f"VH → 1ª infer: {t1:.4f} s | média 50x: {mean(times_vh):.4f} s")
    print(f"VL → 1ª infer: {t2:.4f} s | média 50x: {mean(times_vl):.4f} s")
    print("\nDica: se a 1ª inferência for claramente mais lenta que as demais, sua BN provavelmente\n"
          "gera/compila estruturas internamente na primeira consulta (lazy). O build já mede\n"
          "o tempo de criação (onde você pode optar por pré-gerar CPT/TPN).")
    print(f"\n[INFER-COUNT] AE total = {INFER_COUNTS['AE']} chamadas")
    # Com o script atual, serão 1 (primeira) + 50 (loop) para VH
    #      + 1 + 50 para VL = 102 chamadas de AE.
if __name__ == "__main__":
    main()
