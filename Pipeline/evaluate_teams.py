import sys, os
from pathlib import Path
import json, re
from itertools import combinations
import time

# Deixa o sys.path apontando para .../STFP
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = Path(__file__).resolve().parents[1]  # .../STFP

from Feature_Extraction.Dimension_Scoring.dimension_scoring import avaliar_todos_as_dimensions
from Algorithms.BN.team_fit_bn import criar_rede_fitness
from Feature_Extraction.PC_Transformer.pc_transformer import classificar_pc_por_faixa

_BN_SINGLETON = None
def get_bn():
    global _BN_SINGLETON
    if _BN_SINGLETON is None:
        _BN_SINGLETON = criar_rede_fitness()  # passe configs aqui se precisar
    return _BN_SINGLETON


CENTROIDES = [0.1, 0.3, 0.5, 0.7, 0.9]

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # .../STFP
DB_PATH = str(BASE_DIR / "Data" / "Dev_DB.json")
GRAFO_GRAPH_PATH = str(BASE_DIR / "Data" / "Graph_DB.json")
WEIGHTS_PATH = str(BASE_DIR / "Feature_Extraction" / "Dimension_Scoring" / "pesos_calibrados.json")


# ------------------------------------------------------------------
# Weights (calibrated)
# ------------------------------------------------------------------
with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
    PESOS = json.load(f)
print(f"[WEIGHTS LOADED FROM] {WEIGHTS_PATH}")

# ------------------------------------------------------------------
# Utils de normalização da base
# ------------------------------------------------------------------
def _normalizar_db_devs(db_raw):
    """
    Aceita:
      - lista de devs (novo formato)
      - ou {"developers": [...]} (formato antigo)
    Retorna: {"developers": [ ... ]}
    """
    if isinstance(db_raw, list):
        devs = db_raw
    elif isinstance(db_raw, dict) and "developers" in db_raw:
        devs = db_raw["developers"]
    else:
        devs = []

    norm_devs = []
    for d in devs:
        d2 = dict(d)

        # id -> int (se possível)
        if "id" in d2:
            try:
                d2["id"] = int(d2["id"])
            except Exception:
                pass

        # dominio -> lista
        dom = d2.get("dominio", [])
        if isinstance(dom, str):
            d2["dominio"] = [dom] if dom else []
        elif not isinstance(dom, list) or dom is None:
            d2["dominio"] = []

        # ecossistema -> manter lista e também 1 string (backcompat)
        eco = d2.get("ecossistema", [])
        if isinstance(eco, str):
            eco_list = [eco] if eco else []
        elif isinstance(eco, list):
            eco_list = eco
        else:
            eco_list = []
        d2["ecossistema_list"] = eco_list
        d2["ecossistema"] = eco_list[0] if eco_list else ""

        # linguagens -> manter lista e também 1 string (backcompat)
        lig = d2.get("linguagens", [])
        if isinstance(lig, str):
            lig_list = [lig] if lig else []
        elif isinstance(lig, list):
            lig_list = lig
        else:
            lig_list = []
        d2["linguagens_list"] = lig_list
        d2["linguagens"] = lig_list[0] if lig_list else ""

        # projects -> padronizar chaves
        projs_in = d2.get("projects", []) or d2.get("projectHistory", [])
        norm_projs = []
        for p in projs_in:
            pid = p.get("id") or p.get("projectId")
            osf = p.get("osf") if "osf" in p else p.get("NPS")
            slf = p.get("slf") if "slf" in p else p.get("SLF")
            norm_projs.append({"id": pid, "osf": osf, "slf": slf})
        d2["projects"] = norm_projs

        norm_devs.append(d2)

    return {"developers": norm_devs}

def _load_db_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # aceita lista na raiz ou dict {"developers": [...]}
    if isinstance(data, list):
        data = {"developers": data}
    return data
DB_NORM = _normalizar_db_devs(_load_db_json(DB_PATH))  # <-- carrega 1x
# ------------------------------------------------------------------
# Grafo de colaboração (PCs)
# ------------------------------------------------------------------
def _as_int(x):
    """Converte 558, '558' ou 'Dev558' -> 558."""
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x)
    m = re.search(r'(\d+)$', s)
    return int(m.group(1)) if m else None

def _carregar_mapa_pesos_grafo(caminho_json):
    """
    Retorna: mapa[(min(u,v), max(u,v))] = weight
    Aceita arquivos com:
      - raiz {"edges":[...]} OU {"links":[...]} OU lista direta
      - campos *_user_id OU 'source'/'target' como 'Dev###'
    """
    with open(caminho_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        edges = data.get("edges") or data.get("links")
    elif isinstance(data, list):
        edges = data
    else:
        edges = None

    if not edges:
        raise ValueError(f"Nenhuma aresta em {caminho_json}.")

    mapa = {}
    for e in edges:
        u = _as_int(e.get("source_user_id"))
        v = _as_int(e.get("target_user_id"))
        if u is None: u = _as_int(e.get("source"))
        if v is None: v = _as_int(e.get("target"))
        w = e.get("weight", 0.0)
        try:
            w = float(w)
        except Exception:
            continue
        if u is None or v is None:
            continue
        a, b = (u, v) if u < v else (v, u)
        k = (a, b)
        mapa[k] = max(mapa.get(k, 0.0), w)
    return mapa

MAPA_PESOS_GRAFO = _carregar_mapa_pesos_grafo(GRAFO_GRAPH_PATH)

def pcs_da_equipe_por_grafo(team_ids, mapa_pesos, default_zero=True):
    team = [int(x) for x in team_ids]
    pcs = []
    for a, b in combinations(sorted(team), 2):
        k = (a, b) if a < b else (b, a)
        w = mapa_pesos.get(k)
        if w is None:
            if default_zero:
                pcs.append(0.0)
        else:
            pcs.append(float(w))
    return pcs

# ------------------------------------------------------------------
# Avaliação da Equipe (usa dimension_scoring + PC_tranformer + BN)
# ------------------------------------------------------------------
def avaliar_equipe(team_ids, PROJETO_ALVO, log=True):
    """
    Avalia uma equipe:
      - Avalia Dom/Eco/Ling com dimension_scoring + pesos calibrados
      - Injeta rótulos na BN (Dom obrigatório; Eco/Ling se houver)
      - AC vem da distribuição de PCs via grafo
      - Retorna distribuição de AE e fitness (média ponderada)
    """
    db = DB_NORM

    t0_dim = time.perf_counter()
    res_dims = avaliar_todos_as_dimensions(
        team_ids=team_ids,
        db=db,
        projeto_alvo=PROJETO_ALVO,
        pesos=PESOS,
        nota_sem_must=0.0,
    )
    t_dim = time.perf_counter() - t0_dim

    dom_score, dom_rotulo = res_dims["dominio"]["score"], res_dims["dominio"]["rotulo"]
    eco_score, eco_rotulo = res_dims["ecossistema"]["score"], res_dims["ecossistema"]["rotulo"]
    ling_score, ling_rotulo = res_dims["linguagens"]["score"], res_dims["linguagens"]["rotulo"]

    
    t0_pcs = time.perf_counter()
    pcs = pcs_da_equipe_por_grafo(team_ids, MAPA_PESOS_GRAFO, default_zero=True)
    t_pcs = time.perf_counter() - t0_pcs

    
    #pcs = pcs_da_equipe_por_grafo(team_ids, MAPA_PESOS_GRAFO, default_zero=True)
    t0_pc = time.perf_counter()
    props = classificar_pc_por_faixa(pcs)
    t_pc = time.perf_counter() - t0_pc
    ac_dist = [props['PC_VL'], props['PC_L'], props['PC_M'], props['PC_H'], props['PC_VH']]

    #bn = criar_rede_fitness()
    bn = get_bn()
    
    bn.setEvidence("Dom", dom_rotulo)    
    bn.setEvidence("Eco", eco_rotulo)  
    bn.setEvidence("Ling", ling_rotulo)

    bn.setEvidence("AC", ac_dist)

    # --- AT contínuo a partir da BN ---
    dist_AT = bn.calculateTPN("AT")              # distribuição em [VL,L,M,H,VH]
    AT_cont = float(sum(p * v for p, v in zip(dist_AT, CENTROIDES)))

    # --- AC contínuo a partir da distribuição de PCs ---
    AC_cont = float(sum(p * v for p, v in zip(ac_dist, CENTROIDES)))

    # --- AE (como já fazia) ---
    dist_AE = bn.calculateTPN("AE")
    media_AE = float(sum(p * v for p, v in zip(dist_AE, CENTROIDES)))

    if log:
        print("\nEquipe:", team_ids)
        print("Domínio:", f"{dom_score:.4f} → {dom_rotulo}")
        if eco_score is not None:
            print("Ecossistema:", f"{eco_score:.4f} → {eco_rotulo}")
        if ling_score is not None:
            print("Linguagens:", f"{ling_score:.4f} → {ling_rotulo}")
        print("PCs pares:", pcs)
        print("PC % [VL,L,M,H,VH]:", [round(props[k],3) for k in ('PC_VL','PC_L','PC_M','PC_H','PC_VH')])
        print("AC_cont:", round(AC_cont, 4))
        print("AT_cont:", round(AT_cont, 4))
        print("Distribuição AE (VL..VH):", [round(x, 4) for x in dist_AE])
        print("Fitness(AE):", round(media_AE, 4))
        print(f"[FEATURE][DIM_SCORING] tempo={t_dim:.6f}s") #tudo do Dimension Scoring.
        print(f"[FEATURE][PC_TRANSFORMER] tempo={t_pc:.6f}s (n_pares={len(pcs)})") #tempo para transformar esses PCs em proporções VL/L/M/H/VH.
        
        print(f"[FEATURE][PC_GRAPH] tempo={t_pcs:.6f}s (n_pares={len(pcs)})")#tempo só para varrer o grafo e montar a lista de PCs.
       


    return {
        "scores": res_dims,
        "distribuicao": dist_AE,
        "media_AE": media_AE,
        "AT_cont": AT_cont,
        "AC_cont": AC_cont,
    }


# ------------------------------------------------------------------
# Main de teste simples
# ------------------------------------------------------------------
def main():
    # Exemplo de PROJETO_ALVO no formato esperado
    PROJETO_ALVO = {
        "dominio":     {"must": ["cloud"], "should": ["web"], "could": ["ai"]},
        "ecossistema": {"must": [], "should": ["react", "aws lambda"], "could": ["twilio api"]},
        "linguagens":  {"must": [], "should": [], "could": []}
    }

    # Uma equipe de exemplo (preencha)
    teams = [
        [14, 77, 97, 261],
    ]
    print("Avaliando a equipe tal [14, 77, 97, 261],")
    for team_ids in teams:
        avaliar_equipe(team_ids, PROJETO_ALVO, log=True)

if __name__ == "__main__":
    main()
