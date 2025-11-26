import numpy as np
from scipy.stats import truncnorm
import json

# -----------------------------
# Funções de Agregação
# -----------------------------
def media_ponderada(*args):
    if len(args) % 2 != 0:
        raise ValueError("Quantidade inválida de argumentos (pares peso-valor esperados).")
    soma_valores = 0.0
    soma_pesos = 0.0
    for i in range(0, len(args), 2):
        peso = args[i]
        valor = args[i + 1]
        soma_valores += peso * valor
        soma_pesos += peso
    if soma_pesos == 0:
        return None
    return soma_valores / soma_pesos

def minimo_ponderado(*args):
    if len(args) % 2 != 0:
        raise ValueError("Quantidade inválida de argumentos (pares peso-valor esperados).")
    n = len(args) // 2
    if n < 2:
        return None
    pesos = [args[2 * i] for i in range(n)]
    valores = [args[2 * i + 1] for i in range(n)]
    if sum(pesos) == 0:
        return None
    soma_valores = sum(valores)
    minimo_atual = float('inf')
    for i in range(n):
        denominador = pesos[i] + (n - 1)
        numerador = pesos[i] * valores[i] + (soma_valores - valores[i])
        e_i = numerador / denominador
        minimo_atual = np.minimum(minimo_atual, e_i)
    return minimo_atual
import numpy as np

def maximo_ponderado(*args):
    
    if len(args) % 2 != 0:
        return None

    n = len(args) // 2
    if n < 2:
        return None

    weights = []
    values = []
    for i in range(n):
        w_i = args[2 * i]
        x_i = args[2 * i + 1]
        if w_i < 0 or not np.all((0 <= x_i) & (x_i <= 1)): #adaptando essa linha para trabalhar com vetores de amostras, e não com valores escalares.
            return None
        weights.append(w_i)
        values.append(x_i)

    all_zero_denominators = True
    for i in range(n):
        denom = weights[i] + (n - 1)
        if denom != 0:
            all_zero_denominators = False
            break
    if all_zero_denominators:
        return None

    max_e = None
    sum_all = sum(values)
    for i in range(n):
        w_i = weights[i]
        x_i = values[i]
        denom = w_i + (n - 1)
        numerator = w_i * x_i + (sum_all - x_i)
        e_i = numerator / denom
        if max_e is None: #adaptando aqui para comparar arrays posição a posição.
            max_e = e_i
        else:
            max_e = np.maximum(max_e, e_i)
    return max_e
funcoes = {
    "WMEAN": media_ponderada,
    "WMIN": minimo_ponderado,
    "WMAX": maximo_ponderado
}

# -----------------------------
# Repositório e Geração de CPT
# -----------------------------
import os

def carregar_repositorio(path = os.path.join(os.path.dirname(__file__), 'repositorio.json')):
    with open(path, 'r', encoding='utf-8') as f:
        repo = json.load(f)
    
    sample_format = 'amostras' in next(iter(repo.values()))
    for estado in repo:
        if sample_format:
            repo[estado]['amostras'] = np.array(repo[estado]['amostras'])
    
    return repo, sample_format

def gerar_cpt(estados_pais, funcao, pesos, variancia, repositorio):
    num_estados = 5
    estados = ['VL', 'L', 'M', 'H', 'VH']
    cpt = []

    for combinacao in np.ndindex(*(num_estados for _ in estados_pais)):
        valores = []
        for idx, estado_idx in enumerate(combinacao):
            estado_nome = estados[estado_idx]
            valores.append(np.mean(repositorio[estado_nome]['amostras']))
        
        args = [item for par in zip(pesos, valores) for item in par]
        valor_agregado = funcao(*args)
        media = np.clip(valor_agregado, 0.001, 0.999)
        sigma = np.sqrt(variancia)

        lower_bound = (0 - media) / sigma
        upper_bound = (1 - media) / sigma
        dist = truncnorm(lower_bound, upper_bound, loc=media, scale=sigma)

        limites = np.linspace(0, 1, 6)
        probs = [dist.cdf(limites[i+1]) - dist.cdf(limites[i]) for i in range(5)]
        probs = np.array(probs)
        probs /= probs.sum()
        cpt.append(probs)

    return np.array(cpt).T
