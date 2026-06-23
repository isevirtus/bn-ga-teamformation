#team_fit_bn.py
from pgmpy.factors.discrete import TabularCPD
from Algorithms.BN.bnetwork import BNetwork
from Algorithms.BN.utils import funcoes, carregar_repositorio, gerar_cpt

import numpy as np

#ver a qtd de vezes q a cpt é gerada
import time, hashlib
_CPT_BUILD_COUNT = {"AT": 0, "AE": 0}

def _log_cpt_build(label: str, cpt_array, dt: float):
    _CPT_BUILD_COUNT[label] += 1
    h = hashlib.md5(cpt_array.tobytes()).hexdigest()[:8]
    print(f"[BUILD][{label}] #{_CPT_BUILD_COUNT[label]} shape={cpt_array.shape} "
          f"tempo={dt:.4f}s hash={h}")

def criar_rede_fitness(


    
    func_at="WMEAN",
    sigma_at=0.05,
    pesos_dom_eco_ling=[3, 1, 5],

    func_ae="MIXMINMAX",
    sigma_ae=0.05,
    pesos_at_ac=[5, 1],

    cpd_dom=None, cpd_eco=None, cpd_ling=None, cpd_ac=None
):


    # Carregar o repositório de amostras 
    repo, _ = carregar_repositorio() 
    
    # Criar rede 
    bn = BNetwork()
    
    # Definir estados
    estados = ['VL', 'L', 'M', 'H', 'VH']

    # Criar nós
    bn.createNode("Dom", "Domínio", estados)
    bn.createNode("Eco", "Ecossistema", estados)
    bn.createNode("Ling", "Linguagens", estados)
    
    bn.createNode("AC", "Aptidão Colaborativa", estados)
    
    bn.createNode("AT", "Aptidão Técnica", estados)
    bn.createNode("AE", "Aptidão da Equipe", estados)

    # Adicionar ligações
    bn.addEdge("Dom", "AT")
    bn.addEdge("Eco", "AT")
    bn.addEdge("Ling", "AT")
    
    bn.addEdge("AT", "AE")
    bn.addEdge("AC", "AE")

    # Selecionar função agregadora
    funcao_at = funcoes[func_at]
    
    funcao_ae = funcoes[func_ae]


    # CPDs dos nós de entrada
    uniforme = np.array([[0.2], [0.2], [0.2], [0.2], [0.2]])
    bn.setNodeCPD("Dom",  cpd_dom  if cpd_dom  is not None else uniforme)
    bn.setNodeCPD("Eco",  cpd_eco  if cpd_eco  is not None else uniforme)
    bn.setNodeCPD("Ling", cpd_ling if cpd_ling is not None else uniforme)

    bn.setNodeCPD("AC", cpd_ac if cpd_ac is not None else uniforme)

    

    # CPDs calculados
    t0 = time.perf_counter()
    cpt_at = gerar_cpt(['Dom', 'Eco', 'Ling'], funcao_at, pesos_dom_eco_ling, sigma_at, repo)
    dt = time.perf_counter() - t0
    _log_cpt_build("AT", cpt_at, dt)
    bn.setNodeCPD("AT", cpt_at)

   

    t0 = time.perf_counter()
    cpt_ae = gerar_cpt(['AT', 'AC'], funcao_ae, pesos_at_ac, sigma_ae, repo)
    dt = time.perf_counter() - t0
    _log_cpt_build("AE", cpt_ae, dt)
    bn.setNodeCPD("AE", cpt_ae)
    # --- guarda metadados da config usada (para debug/prints externos) ---
    bn._stfp_cfg = {
        "func_at": func_at,
        "sigma_at": sigma_at,
        "pesos_dom_eco_ling": list(pesos_dom_eco_ling),

        "func_ae": func_ae,
        "sigma_ae": sigma_ae,
        "pesos_at_ac": list(pesos_at_ac),
    }
    return bn

