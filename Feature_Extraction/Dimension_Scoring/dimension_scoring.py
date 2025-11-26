# teamformation/dimension_scoring.py
from typing import Dict, List, Optional, Tuple
from collections import Counter
import math
DEBUG_DIM = True
# deixar true p mostrar logs
def _log(msg: str):
    if DEBUG_DIM:
        print(msg)
# ----------------------------- utils -----------------------------

def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def _norm_list(xs):
    return [str(s).strip().lower() for s in _as_list(xs)]

def _get_dev_field_as_set(dev: Dict, campo: str):
    """
    campo: 'dominio' | 'ecossistema' | 'linguagens'
    Aceita lista ou string na base; devolve set normalizado.
    """
    raw = dev.get(campo)
    if isinstance(raw, (list, tuple, set)):
        return set(_norm_list(raw))
    if isinstance(raw, str):
        return set(_norm_list([raw]))
    return set()

def categorizar(score: float) -> str:
    if score >= 0.8: return "VH"
    if score >= 0.6: return "H"
    if score >= 0.4: return "M"
    if score >= 0.2: return "L"
    return "VL"

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

# ------------------------- Pesos obtidos na regressao  ----------------------------
pesos = {
    "dominio": {
        "sit2": {  # Situação 2 - Intra-prioridade
            "b": -0.333333,
            "a_full": 0.166667,
            "a_red2": 1.166667,
            "a_bal": 0.0,
            "a_cov": 0.0,
            "a_help": 0.0
        },
        "sit1": {  # Situação 1 - Entre prioridades
            "b": 0.0,
            "w_covM": 0.071429,
            "w_red2M": 0.428571,
            "w_intMS": 0.357143,
            "w_helpC": 0.5
        }
    },
    "ecossistema": {
        "sit2": {  # usa os mesmos pesos
            "b": -0.333333,
            "a_full": 0.166667,
            "a_red2": 1.166667,
            "a_bal": 0.0,
            "a_cov": 0.0,
            "a_help": 0.0
        },
        "sit1": {
            "b": 0.0,
            "w_covM": 0.071429,
            "w_red2M": 0.428571,
            "w_intMS": 0.357143,
            "w_helpC": 0.5
        }
    },
    "linguagens": {
        "sit2": {  # idem
            "b": -0.333333,
            "a_full": 0.166667,
            "a_red2": 1.166667,
            "a_bal": 0.0,
            "a_cov": 0.0,
            "a_help": 0.0
        },
        "sit1": {
            "b": 0.0,
            "w_covM": 0.071429,
            "w_red2M": 0.428571,
            "w_intMS": 0.357143,
            "w_helpC": 0.5
        }
    }
}



# ------------------------- Situação 2 ----------------------------
# intra-prioridade (um dimension principal P e opcional fallback F)

def extrair_sit2_features(
    team_ids: List[int],
    db: Dict,
    tecnologias_principal: List[str],
    campo: str,
    fallback: Optional[List[str]] = None,
) -> Dict:
    """
    Retorna features por prioridade (intra-dimension):
      - covP, red2P, balP, fullP
      - (se fallback): covF, helpF = covF * (1 - fullP)
      - hits detalhados por tecnologia (debug)
    """
    dev_by_id = {d["id"]: d for d in db.get("developers", [])}
    P = set(_norm_list(tecnologias_principal))
    F = set(_norm_list(fallback)) if fallback else set()
    nP = len(P)
    team_size = max(1, len(team_ids))

    # Contar hits por tecnologia do dimension principal
    hitsP = {t: 0 for t in P}
    for did in team_ids:
        dev = dev_by_id.get(did)
        if not dev: continue
        techs = _get_dev_field_as_set(dev, campo)
        for t in P:
            if t in techs:
                hitsP[t] += 1

    # Frações no principal
    if nP == 0:
        covP = 1.0   # neutro
        red2P = 0.0
        balP = 1.0
        fullP = 1.0
    else:
        covered = sum(1 for t, h in hitsP.items() if h >= 1)
        red2    = sum(1 for t, h in hitsP.items() if h >= 2)
        covP  = covered / nP
        red2P = red2 / nP
        if nP == 1:
            balP = 1.0
        else:
            hvals = [h for _, h in hitsP.items()]
            hmin, hmax = min(hvals), max(hvals)
            balP = (hmin / hmax) if hmax > 0 else 1.0
        fullP = 1.0 if covP == 1.0 else 0.0

    # Fallback (opcional)
    covF = 0.0
    helpF = 0.0
    hitsF = {}
    if F:
        hitsF = {t: 0 for t in F}
        for did in team_ids:
            dev = dev_by_id.get(did)
            if not dev: continue
            techs = _get_dev_field_as_set(dev, campo)
            for t in F:
                if t in techs:
                    hitsF[t] += 1
        nF = len(F) if len(F) > 0 else 1
        coveredF = sum(1 for t, h in hitsF.items() if h >= 1)
        covF = coveredF / (len(F) if len(F) > 0 else 1)
        helpF = covF * (1.0 - fullP)
    #log 
    _log(
        "[SIT2.FEATS]"
        f" P={sorted(list(P))} F={sorted(list(F))} | "
        f"covP={covP:.3f} red2P={red2P:.3f} balP={balP:.3f} fullP={fullP:.0f} "
        f"covF={covF:.3f} helpF={helpF:.3f}"
    )
    return {
        "covP": covP, "red2P": red2P, "balP": balP, "fullP": fullP,
        "covF": covF, "helpF": helpF,
        "hitsP": hitsP, "hitsF": hitsF,
        "P": sorted(list(P)), "F": sorted(list(F)),
        "team_size": team_size,
    }

def score_sit2(features: Dict, pesos: Dict) -> float:
    b      = float(pesos.get("b", 0.0))
    a_full = float(pesos.get("a_full", 0.0))
    a_red2 = float(pesos.get("a_red2", 0.0))
    a_bal  = float(pesos.get("a_bal",  0.0))
    a_cov  = float(pesos.get("a_cov",  0.0))
    a_help = float(pesos.get("a_help", 0.0))

    fullP = features.get("fullP", 0.0)
    red2P = features.get("red2P", 0.0)
    balP  = features.get("balP",  0.0)
    covP  = features.get("covP",  0.0)
    helpF = features.get("helpF", 0.0)

    y = b + a_full*fullP + a_red2*red2P + a_bal*balP + a_cov*covP + a_help*helpF
    score = clamp01(y)
    gargalo_score = 0.0 if fullP < 1.0 else score

    _log(
        "[SIT2.SCORE] "
        f"b={b:+.3f} + a_full({a_full:+.3f})*{fullP:.3f} "
        f"+ a_red2({a_red2:+.3f})*{red2P:.3f} "
        f"+ a_bal({a_bal:+.3f})*{balP:.3f} "
        f"+ a_cov({a_cov:+.3f})*{covP:.3f} "
        f"+ a_help({a_help:+.3f})*{helpF:.3f} "
        f"= y={y:.3f} -> clamp={score:.3f} "
        f"{'(GARGALO: fullP<1 ⇒ score=0.000)' if fullP < 1.0 else ''}"
    )
    return gargalo_score



# ------------------------- Situação 1 ----------------------------
# entre prioridades (combina M/S/C usando as features da Sit2)

def score_sit1_from_feats(
    featsM: Optional[Dict],
    featsS: Optional[Dict],
    featsC: Optional[Dict],
    pesos: Dict,
    must_exists: bool,
    nota_sem_must: float = 0.0,
) -> float:
    """
    Fórmula mínima da Situação 1 (calibre w_* com regressão):
      y = intercepto_b + w_red2M*red2M + w_covM*covM + w_intMS*(covM*covS) + w_helpC*(covM*covC*(1-covS))
    Penalização 'sem MUST' se existe MUST e covM==0: y *= nota_sem_must
    """
    intercepto_b        = float(pesos.get("b", 0.0))
    w_red2M  = float(pesos.get("w_red2M", 0.0))
    w_covM   = float(pesos.get("w_covM", 0.0))
    w_intMS  = float(pesos.get("w_intMS", 0.0))
    w_helpC  = float(pesos.get("w_helpC", 0.0))

    # DEBUG
    print(f"[SIT1 WEIGHTS] b={intercepto_b} w_red2M={w_red2M} w_covM={w_covM} w_intMS={w_intMS} w_helpC={w_helpC}")


    # extrai frações da Sit2
    covM  = featsM.get("covP", 1.0) if featsM is not None else 1.0  # neutro se MUST não existe
    red2M = featsM.get("red2P", 0.0) if featsM is not None else 0.0
    covS  = featsS.get("covP", 0.0) if featsS is not None else 0.0
    covC  = featsC.get("covP", 0.0) if featsC is not None else 0.0
    

    intMS = covM * covS
    helpC = covM * covC * (1.0 - covS)
    
    #log
    _log(
        "[SIT1.VARS] "
        f"covM={covM:.3f} red2M={red2M:.3f} covS={covS:.3f} covC={covC:.3f} "
        f"| intMS=covM*covS={intMS:.3f} helpC=covM*covC*(1-covS)={(covM*covC*(1.0-covS)):.3f}"
    )
        
    y = (intercepto_b
         + w_red2M * red2M
         + w_covM  * covM
         + w_intMS * intMS
         + w_helpC * helpC)

    if must_exists and covM == 0.0:
        _log(f"[SIT1.PENALTY] must_exists=True & covM=0 → y *= {nota_sem_must}")
        y *= float(nota_sem_must)

    score = clamp01(y)
    _log(f"[SIT1.SCORE] y={y:.3f} -> clamp={score:.3f}")
    return score


# --------------------- Avaliação por dimension ---------------------

def avaliar_dimension(
    team_ids: List[int],
    db: Dict,
    alvo_dimension: Dict,      # {'must': [...], 'should': [...], 'could': [...]}
    campo: str,              # 'dominio' | 'ecossistema' | 'linguagens'
    pesos_sit1: Dict,        # pesos calibrados para Situação 1 (entre prioridades)
    pesos_sit2: Dict,        # pesos calibrados para Situação 2 (intra prioridade)
    nota_sem_must: float = 0.0,
) -> Dict:
    """
    Retorna: {'score': float, 'rotulo': str, 'debug': {...}}
    - Se 0 prioridades ativas: retorna {'score': None, 'rotulo': None}
    - Se 1 prioridade ativa: score sai da Sit2
    - Se 2+ prioridades: usa Sit2 por prioridade e combina com Sit1
    """
    must   = _norm_list(alvo_dimension.get("must", []))
    should = _norm_list(alvo_dimension.get("should", []))
    could  = _norm_list(alvo_dimension.get("could", []))
    actives = []
    if must:   actives.append("must")
    if should: actives.append("should")
    if could:  actives.append("could")
    #log
    _log(f"\n[DIM] campo={campo} | must={must} should={should} could={could} | ativas={actives}")

    if len(actives) == 0:
        _log("[DIM] 0 prioridades ativas → NÃO calcula (BN usa prior)")
        return {"score": None, "rotulo": None, "debug": {"reason": "dimension vazio"}}


    # Sit2 por prioridade
    feats = {"must": None, "should": None, "could": None}
    if must:
        feats["must"] = extrair_sit2_features(team_ids, db, must, campo, fallback=None)
    if should:
        # fallback para SHOULD é opcionalmente o COULD
        feats["should"] = extrair_sit2_features(team_ids, db, should, campo, fallback=could if could else None)
    if could:
        feats["could"] = extrair_sit2_features(team_ids, db, could, campo, fallback=None)
    #log
    _log("[DIM] SIT2 executada para: " +
         ", ".join([p for p in ("must","should","could") if feats[p] is not None]) )

    # Se apenas 1 prioridade ativa → score = Sit2 dessa prioridade
    if len(actives) == 1:
        p = actives[0]
        _log(f"[DIM] 1 prioridade ativa ({p}) → usa SOMENTE SIT2")
        score = score_sit2(feats[p], pesos_sit2)
        rot = categorizar(score)
        _log(f"[DIM] RESULT (SOMENTE SIT2) → score={score:.3f} rotulo={rot}")
        return {"score": score, "rotulo": rot,
                "debug": {"sit": "sit2-only", "priority": p, "feats": feats[p]}}


    # 2+ prioridades → Sit1 (entre prioridades), usando features da Sit2
    _log("[DIM] 2+ prioridades ativas → SIT2 (todas) + SIT1 (combinação)")
    score = score_sit1_from_feats(
        featsM = feats["must"],
        featsS = feats["should"],
        featsC = feats["could"],
        pesos  = pesos_sit1,
        must_exists = bool(must),
        nota_sem_must = nota_sem_must,
    )
    rot = categorizar(score)
    _log(f"[DIM] RESULT (SIT2+SIT1) → score={score:.3f} rotulo={rot}")
    debug = {
        "sit": "sit2+sit1",
        "featsM": feats["must"],
        "featsS": feats["should"],
        "featsC": feats["could"],
        "nota_sem_must": nota_sem_must,
    }
    return {"score": score, "rotulo": rot, "debug": debug}


# --------------------- Avaliar os 3 dimensions ---------------------

def avaliar_todos_as_dimensions(
    team_ids: List[int],
    db: Dict,
    projeto_alvo: Dict,              # {'dominio': {...}, 'ecossistema': {...}, 'linguagens': {...}}
    pesos: Dict,                     # {'dominio': {'sit1':..., 'sit2':...}, 'ecossistema': {...}, 'linguagens': {...}}
    nota_sem_must: float = 0.0,
) -> Dict:
    resultados = {}
    for dimension, campo in (("dominio","dominio"), ("ecossistema","ecossistema"), ("linguagens","linguagens")):
        alvo = projeto_alvo.get(dimension, {}) or {}
        ps1  = (pesos.get(dimension, {}) or {}).get("sit1", {})  # pesos Sit1 do dimension
        ps2  = (pesos.get(dimension, {}) or {}).get("sit2", {})  # pesos Sit2 do dimension
        #log
        _log(f" ======== AVALIANDO DIMENSÃO: {campo.upper()} ========")

        resultados[dimension] = avaliar_dimension(
            team_ids=team_ids, db=db, alvo_dimension=alvo, campo=campo,
            pesos_sit1=ps1, pesos_sit2=ps2, nota_sem_must=nota_sem_must
        )
    return resultados
