# Função para transformar os valores de compatibilidade entre pares (PC)
# em um vetor com as proporções de cada categoria: PC_VL, PC_L, PC_M, PC_H, PC_VH

def classificar_pc_por_faixa(pc_valores): 
    """
    Recebe uma lista de valores de compatibilidade de pares (entre 0 e 1)
    Retorna as proporções de cada categoria (PC_VL, PC_L, PC_M, PC_H, PC_VH)
    """
    # Inicializa contadores
    bins = {
        'PC_VL': 0,
        'PC_L': 0,
        'PC_M': 0,
        'PC_H': 0,
        'PC_VH': 0
    }

    total = len(pc_valores)
    if total == 0:
        raise ValueError("A lista de valores de PC não pode estar vazia.")

    for valor in pc_valores:
        if 0.0 <= valor < 0.2:
            bins['PC_VL'] += 1
        elif 0.2 <= valor < 0.4:
            bins['PC_L'] += 1
        elif 0.4 <= valor < 0.6:
            bins['PC_M'] += 1
        elif 0.6 <= valor < 0.8:
            bins['PC_H'] += 1
        elif 0.8 <= valor <= 1.0:
            bins['PC_VH'] += 1
        else:
            raise ValueError(f"Valor fora da faixa esperada (0 a 1): {valor}")

    # Converte contagens em proporções
    proporcoes = {k: v / total for k, v in bins.items()}
    return proporcoes

# Exemplo de uso:
if __name__ == "__main__":
    pc_exemplo = [0.643, 0.643, 0.643, 0.643, 0.643, 0.643]  # Compatibilidades entre pares da equipe
    print(pc_exemplo)
    resultado = classificar_pc_por_faixa(pc_exemplo)
    print("Proporções por faixa:")
    for categoria, proporcao in resultado.items():
        print(f"{categoria}: {proporcao:.2f}")

 
    pc_exemplo = [0.593, 0.643, 0.643, 0.643, 0.643, 0.643]  # Compatibilidades entre pares da equipe
    print(pc_exemplo)
    resultado = classificar_pc_por_faixa(pc_exemplo)
    print("Proporções por faixa:")
    for categoria, proporcao in resultado.items():
        print(f"{categoria}: {proporcao:.2f}")