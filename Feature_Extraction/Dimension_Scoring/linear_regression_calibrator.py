# calibrar_regressao_moscow_consolidado.py
import math
import numpy as np
from typing import Dict, List, Tuple, Iterable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------
# Utilitários
# ---------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def ranks_to_scores(ranks: Iterable[float]) -> np.ndarray:
    ranks = np.array(list(ranks), dtype=float)
    rmin, rmax = ranks.min(), ranks.max()
    if rmax == rmin:
        # Todos iguais → vire 1.0 para todo mundo (evita divisão por zero)
        return np.ones_like(ranks)
    return 1.0 - (ranks - rmin) / (rmax - rmin)

# ---------------------------
# Situação-2 (intra-prioridade)
# ---------------------------
# score_sit2 = clamp01(b + a_full*fullP + a_red2*red2P + a_bal*balP + a_cov*covP + a_help*helpF)

class Sit2Regressor:
    def __init__(self):
        self.model = LinearRegression(fit_intercept=True, positive=True)
        # pesos aprendidos
        self.b = 0.0
        self.a_full = 0.0
        self.a_red2 = 0.0
        self.a_bal  = 0.0
        self.a_cov  = 0.0
        self.a_help = 0.0

    @staticmethod
    def _features_for_priority(
        counts_by_tech: Dict[str, int],
        P: List[str],
        F: List[str] = None
    ) -> List[float]:
        """Extrai fullP, red2P, balP, covP, helpF a partir dos hits em P (e F opcional)."""
        F = F or []
        # hits por tech em P
        hitsP = [int(counts_by_tech.get(t, 0)) for t in P]
        totalP = len(P)

        if totalP == 0:
            # Sem techs em P → definimos tudo como 1.0 (neutro) para não punir nem favorecer
            covP = 1.0
            red2P = 1.0
            balP = 1.0
            fullP = 1.0
        else:
            covered = sum(1 for h in hitsP if h >= 1)
            red2    = sum(1 for h in hitsP if h >= 2)
            covP    = covered / totalP
            red2P   = red2 / totalP
            mx      = max(hitsP) if len(hitsP) > 0 else 0
            mn      = min(hitsP) if len(hitsP) > 0 else 0
            balP    = 1.0 if (mx == 0 or totalP == 1) else (mn / mx)
            fullP   = 1.0 if covP == 1.0 else 0.0

        # Fallback F
        if len(F) == 0:
            covF = 0.0
        else:
            hitsF = [int(counts_by_tech.get(t, 0)) for t in F]
            totalF = len(F)
            covF = 0.0 if totalF == 0 else (sum(1 for h in hitsF if h >= 1) / totalF)

        helpF = covF * (1.0 - fullP)

        # ordem: [fullP, red2P, balP, covP, helpF]
        return [float(fullP), float(red2P), float(balP), float(covP), float(helpF)]

    def fit(self, scenarios: List[Tuple[Dict[str, int], float]], P: List[str], F: List[str] = None):
        """
        scenarios: lista de (counts_by_tech, rank_integer)
        P: lista de tecnologias da prioridade alvo (ex.: SHOULD)
        F: lista de tecnologias fallback (ex.: COULD), opcional
        """
        F = F or []
        counts_list = [d for (d, _) in scenarios]
        ranks = [r for (_, r) in scenarios]
        Y = ranks_to_scores(ranks)

        X = np.array([self._features_for_priority(c, P, F) for c in counts_list], dtype=float)
        self.model.fit(X, Y)

        self.b, self.a_full, self.a_red2, self.a_bal, self.a_cov, self.a_help = \
            float(self.model.intercept_), *[float(x) for x in self.model.coef_]

        # Diagnóstico
        y_pred = self.model.predict(X)
        mse = mean_squared_error(Y, y_pred)

        return {
            "intercepto_b": self.b,
            "a_full": self.a_full,
            "a_red2": self.a_red2,
            "a_bal":  self.a_bal,
            "a_cov":  self.a_cov,
            "a_help": self.a_help,
            "mse_in_sample": float(mse),
        }

    def predict_score(self, counts_by_tech: Dict[str, int], P: List[str], F: List[str] = None) -> float:
        F = F or []
        f_full, f_red2, f_bal, f_cov, f_help = self._features_for_priority(counts_by_tech, P, F)
        y = self.b + self.a_full*f_full + self.a_red2*f_red2 + self.a_bal*f_bal + self.a_cov*f_cov + self.a_help*f_help
        return clamp01(y)

# ---------------------------
# Situação-1 (entre prioridades)
# ---------------------------
# y = clamp01(b + w_red2M*red2M + w_covM*covM + w_intMS*intMS + w_helpC*helpC)
# Penalização: se existe MUST no alvo e covM==0 → y *= nota_sem_must

class Sit1Regressor:
    def __init__(self, nota_sem_must: float = 0.0):
        self.model = LinearRegression(fit_intercept=True, positive=True)
        self.nota_sem_must = float(nota_sem_must)
        self.b = 0.0
        self.w_red2M = 0.0
        self.w_covM  = 0.0
        self.w_intMS = 0.0
        self.w_helpC = 0.0

    @staticmethod
    def _features_between_priorities(M: float, S: float, C: float, red2M: float) -> List[float]:
        # covM, covS, covC são frações ∈ [0,1] (ex.: #techs cobertas / |techs|)
        covM = float(M)
        covS = float(S)
        covC = float(C)
        red2M = float(red2M)
        intMS = covM * covS
        helpC = covM * covC * (1.0 - covS)
        # ordem: [covM, red2M, intMS, helpC]
        return [covM, red2M, intMS, helpC]

    def fit(self, scenarios_msc: List[Tuple[List[float], float]]):
        """
        scenarios_msc: lista de ([M,S,C], rank_integer)
        Interpretação de M,S,C aqui: contagens de pessoas por prioridade.
        Para treinar Sit-1, mapeamos para frações de cobertura (0/1 aqui, por simplicidade),
        e red2M como fração de redundância MUST (aqui aproximado por indicadores 0/1 com base no valor).
        """
        MSC = [msc for (msc, _) in scenarios_msc]
        ranks = [r for (_, r) in scenarios_msc]
        Y = ranks_to_scores(ranks)

        # Aproximação simples (a partir do seu dataset de exemplo):
        # - covX = 1 se X>=1 senão 0 (cobertura mínima por prioridade)
        # - red2M = 1 se M>=2 senão 0 (redundância mínima em MUST)
        feats = []
        for (M, S, C) in MSC:
            covM = 1.0 if M >= 1 else 0.0
            covS = 1.0 if S >= 1 else 0.0
            covC = 1.0 if C >= 1 else 0.0
            red2M = 1.0 if M >= 2 else 0.0
            feats.append(self._features_between_priorities(covM, covS, covC, red2M))

        X = np.array(feats, dtype=float)
        self.model.fit(X, Y)

        self.b, self.w_covM, self.w_red2M, self.w_intMS, self.w_helpC = \
            float(self.model.intercept_), *[float(x) for x in self.model.coef_]

        y_pred = self.model.predict(X)
        mse = mean_squared_error(Y, y_pred)

        return {
            "intercepto_b": self.b,
            "w_covM":  self.w_covM,
            "w_red2M": self.w_red2M,
            "w_intMS": self.w_intMS,
            "w_helpC": self.w_helpC,
            "mse_in_sample": float(mse),
        }

    def predict_score(self, covM: float, covS: float, covC: float, red2M: float, alvo_tem_MUST: bool) -> float:
        covM = float(covM); covS = float(covS); covC = float(covC); red2M = float(red2M)
        intMS = covM * covS
        helpC = covM * covC * (1.0 - covS)

        y = self.b + self.w_red2M*red2M + self.w_covM*covM + self.w_intMS*intMS + self.w_helpC*helpC
        y = clamp01(y)
        if alvo_tem_MUST and covM == 0.0:
            y *= self.nota_sem_must  # tipicamente 0.0
        return clamp01(y)

# ---------------------------
# Exemplo de uso com seus dados
# ---------------------------

if __name__ == "__main__":
    # ==========
    # A) Ecossistema (Situação-2: intra-prioridade) com seus cenários React/AWS/Twilio
    #    P = SHOULD = ["React", "AWS Lambda"]
    #    F = COULD  = ["Twilio API"] (fallback)
    # ==========
    cenarios_eco = [
        ({"React": 0, "AWS Lambda": 4, "Twilio API": 0}, 4),  # Team 1
        ({"React": 1, "AWS Lambda": 3, "Twilio API": 0}, 2),  # Team 3
        ({"React": 2, "AWS Lambda": 2, "Twilio API": 0}, 1),  # Team 5
        ({"React": 1, "AWS Lambda": 2, "Twilio API": 1}, 5),  # Team 7
        ({"React": 3, "AWS Lambda": 1, "Twilio API": 0}, 3),  # Team 8
    ]
    SHOULD = ["React", "AWS Lambda"]
    COULD  = ["Twilio API"]

    sit2 = Sit2Regressor()
    info_sit2 = sit2.fit(cenarios_eco, P=SHOULD, F=COULD)

    print("\n=== Pesos interpretáveis (Situação-2 / Ecossistema) ===")
    for k, v in info_sit2.items():
        if k != "mse_in_sample":
            print(f"{k:>18s} = {v:.6f}")
    print(f"{'mse_in_sample':>18s} = {info_sit2['mse_in_sample']:.6f}")

    # Predições e ordem prevista
    X_names = ["Team 1", "Team 3", "Team 5", "Team 7", "Team 8"]
    preds_eco = [sit2.predict_score(d, P=SHOULD, F=COULD) for (d, _) in cenarios_eco]
    ordem_eco = [x for _, x in sorted(zip([-p for p in preds_eco], X_names))]
    print("\n[Ecossistema/Sit-2] Ordem prevista (melhor → pior):")
    for nome, score in sorted(zip(X_names, preds_eco), key=lambda z: -z[1]):
        print(f"  {nome:6s}  score={score:.4f}")

    # ==========
    # B) MSC (Situação-1: entre prioridades) com seus cenários [M,S,C]
    # ==========
    cenarios_msc = [
        ([1, 2, 1], 5),  # Team 1
        ([2, 1, 1], 2),  # Team 2
        ([0, 2, 2], 8),  # Team 3
        ([3, 0, 1], 1),  # Team 5
        ([4, 0, 0], 4),  # Team 7
        ([2, 0, 0], 5),  # Team 8
    ]

    sit1 = Sit1Regressor(nota_sem_must=0.0)
    info_sit1 = sit1.fit(cenarios_msc)

    print("\n=== Pesos interpretáveis (Situação-1 / Entre Prioridades) ===")
    for k, v in info_sit1.items():
        if k != "mse_in_sample":
            print(f"{k:>15s} = {v:.6f}")
    print(f"{'mse_in_sample':>15s} = {info_sit1['mse_in_sample']:.6f}")

    # Predições e ordem prevista (mapeando contagens → frações 0/1 de cobertura e redundância mínima em MUST)
    nomes_msc = ["Team 1","Team 2","Team 3","Team 5","Team 7","Team 8"]
    preds_msc = []
    for (M, S, C), _rank in cenarios_msc:
        covM = 1.0 if M >= 1 else 0.0
        covS = 1.0 if S >= 1 else 0.0
        covC = 1.0 if C >= 1 else 0.0
        red2M = 1.0 if M >= 2 else 0.0
        score = sit1.predict_score(covM, covS, covC, red2M, alvo_tem_MUST=True)
        preds_msc.append(score)

    print("\n[MSC/Sit-1] Ordem prevista (melhor → pior):")
    for nome, score in sorted(zip(nomes_msc, preds_msc), key=lambda z: -z[1]):
        print(f"  {nome:6s}  score={score:.4f}")

    # ---------------
    # Observação:
    # Para aplicar na dimensão completa:
    # - Se só UMA prioridade estiver ativa → use Sit-2 daquele P (com F opcional).
    # - Se DUAS+ prioridades ativas → combine frações (cov/red2) extraídas da Sit-2 e aplique Sit-1.
    # ---------------
