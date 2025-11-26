import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import sys
import os

# Adiciona o diretório pai ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Tenta importar de diferentes locais possíveis
try:
    from rede_bayesiana.rede_fitness_ag import criar_rede_fitness
    print("✓ Rede bayesiana importada de rede_bayesiana.rede_fitness_ag")
except ImportError:
    try:
        from feature_Extraction.Dimension_Scoring.dimension_scoring import criar_rede_fitness
        print("✓ Rede bayesiana importada de feature_Extraction")
    except ImportError:
        try:
            # Importa os componentes necessários diretamente
            from Algorithms.BN.bnetwork import BNetwork
            from Algorithms.BN.utils import funcoes, carregar_repositorio, gerar_cpt
            from pgmpy.factors.discrete import TabularCPD
            
            def criar_rede_fitness(funcao_agregadora="WMEAN", variancia=0.1):
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
                funcao = funcoes[funcao_agregadora]

                # CPDs dos nós de entrada
                uniforme = np.array([[0.2], [0.2], [0.2], [0.2], [0.2]])
                bn.setNodeCPD("Dom", uniforme)
                bn.setNodeCPD("Eco", uniforme)
                bn.setNodeCPD("Ling", uniforme)
                bn.setNodeCPD("AC", uniforme)

                # CPDs calculados
                cpt_at = gerar_cpt(['Dom', 'Eco', 'Ling'], funcao, [1, 1, 1], variancia, repo)
                bn.setNodeCPD("AT", cpt_at)

                cpt_ae = gerar_cpt(['AT', 'AC'], funcao, [2, 1], variancia, repo)
                bn.setNodeCPD("AE", cpt_ae)

                return bn
            print("✓ Rede bayesiana criada localmente")
        except ImportError as e:
            print(f"❌ Erro ao importar módulos necessários: {e}")
            print("📂 Estrutura de pastas esperada:")
            print("   STFP/")
            print("   ├── rede_bayesiana/")
            print("   │   └── rede_fitness_ag.py")
            print("   ├── Algorithms/")
            print("   │   └── BN/")
            print("   │       ├── bnetwork.py")
            print("   │       └── utils.py")
            print("   └── validation/")
            print("       └── sensibilidade_completa.py")
            sys.exit(1)

# Constantes
ESTADOS = ['VL', 'L', 'M', 'H', 'VH']
CENTROS = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)

def soft_cpd_from_scalar(x: float, tau: float = 0.10) -> np.ndarray:
    x = float(np.clip(x, 0.0, 1.0))
    w = np.exp(-0.5 * ((x - CENTROS) / tau) ** 2)
    w = w / (w.sum() + 1e-12)
    return w.reshape(-1, 1)

def expected_from_probs(probs) -> float:
    arr = np.array(probs, dtype=float).reshape(-1)
    arr = arr / (arr.sum() + 1e-12)
    return float((arr * CENTROS).sum())

def set_inputs_via_cpd(bn, x_dom: float, x_eco: float, x_ling: float, x_ac: float, tau_soft: float):
    bn.setNodeCPD("Dom",  soft_cpd_from_scalar(x_dom,  tau_soft))
    bn.setNodeCPD("Eco",  soft_cpd_from_scalar(x_eco,  tau_soft))
    bn.setNodeCPD("Ling", soft_cpd_from_scalar(x_ling, tau_soft))
    bn.setNodeCPD("AC",   soft_cpd_from_scalar(x_ac,   tau_soft))

def get_marginal_AE(bn):
    probs = bn.calculateTPN("AE")
    return np.array(probs).reshape(-1)

def analise_sensibilidade_completa(bn, pontos=6, tau_soft=0.1):
    """Análise de sensibilidade sistemática varrendo todo o espaço"""
    print(f"🎯 Iniciando análise com {pontos} pontos por dimensão...")
    print(f"📊 Total de combinações: {pontos ** 4}")
    
    # Gera todas as combinações
    valores = np.linspace(0.0, 1.0, pontos)
    combinacoes = list(product(valores, repeat=4))
    
    resultados = []
    
    for i, (dom, eco, ling, ac) in enumerate(combinacoes):
        if i % 100 == 0:
            print(f"⏳ Processando {i}/{len(combinacoes)}...")
            
        set_inputs_via_cpd(bn, dom, eco, ling, ac, tau_soft)
        pAE = get_marginal_AE(bn)
        AE = expected_from_probs(pAE)
        
        resultados.append({
            'Dom': dom, 'Eco': eco, 'Ling': ling, 'AC': ac,
            'AE': AE
        })
    
    df = pd.DataFrame(resultados)
    
    # Análises estatísticas
    sens_individual = {
        'Dom': df.groupby('Dom')['AE'].std().mean(),
        'Eco': df.groupby('Eco')['AE'].std().mean(), 
        'Ling': df.groupby('Ling')['AE'].std().mean(),
        'AC': df.groupby('AC')['AE'].std().mean()
    }
    
    correlacoes = df.corr()['AE'].drop('AE')
    
    return df, sens_individual, correlacoes

def plot_resultados(df, sens_ind, corr):
    """Gera gráficos da análise de sensibilidade"""
    
    plt.figure(figsize=(15, 5))
    
    # 1. Gráfico de correlações
    plt.subplot(1, 3, 1)
    bars = plt.bar(corr.index, corr.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Correlação com AE')
    plt.xticks(rotation=45)
    for bar, value in zip(bars, corr.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Gráfico de sensibilidade individual
    plt.subplot(1, 3, 2)
    bars = plt.bar(sens_ind.keys(), sens_ind.values(), color = ['#646464', '#989898', '#707070', '#5b5b5b'])

    plt.title('Sensibilidade Individual')
    plt.xticks(rotation=45)
    for bar, value in zip(bars, sens_ind.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Distribuição de AE
    plt.subplot(1, 3, 3)
    plt.hist(df['AE'], bins=20, alpha=0.7, color='grey', edgecolor='black')
    plt.title('Distribuição de AE')
    plt.xlabel('AE')
    plt.ylabel('Frequência')
    
    plt.tight_layout()
    plt.savefig('sensibilidade_resultados.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_compact_results(df, sens_ind, corr):
    import numpy as np
    import matplotlib.pyplot as plt

    vars_order = ["Dom", "Eco", "Ling", "AC"]

    # Normaliza para comparar visualmente (0-1)
    corr_abs = np.array([abs(corr[v]) for v in vars_order])
    sens_vals = np.array([sens_ind[v] for v in vars_order])

    corr_norm = corr_abs / (corr_abs.max() + 1e-12)
    sens_norm = sens_vals / (sens_vals.max() + 1e-12)

    y = np.arange(len(vars_order))
    width = 0.35

    # Figura 1: Global sensitivity / importance
    plt.figure(figsize=(6, 3))
    plt.barh(
        y - width/2,
        corr_norm,
        height=width,
        edgecolor="black",
        facecolor="0.7",
        label="|corr(AE)| (norm.)"
    )
    plt.barh(
        y + width/2,
        sens_norm,
        height=width,
        edgecolor="black",
        facecolor="0.4",
        label="Sensitivity (norm.)"
    )
    plt.xlabel("Normalized importance")
    plt.yticks(y, vars_order)
    plt.title("Global sensitivity indices for AE")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("sens_global_compact.png", dpi=300, bbox_inches="tight")

    # Figura 2: Distribuição de AE
    plt.figure(figsize=(6, 3))
    plt.hist(df["AE"], bins=20, edgecolor="black", facecolor="0.8")
    plt.xlabel("AE")
    plt.ylabel("Frequency")
    plt.title("Distribution of AE over sampled input space")
    plt.tight_layout()
    plt.savefig("ae_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pontos", type=int, default=4, help="Pontos por dimensão")
    parser.add_argument("--funcao", default="WMEAN", help="Função agregadora")
    parser.add_argument("--sigma", type=float, default=0.10, help="Variância")
    parser.add_argument("--tau_soft", type=float, default=0.10, help="Suavização CPDs")
    args = parser.parse_args()

    print("🔧 Construindo rede Bayesiana...")
    bn = criar_rede_fitness(funcao_agregadora=args.funcao, variancia=args.sigma)

    print("📈 Executando análise de sensibilidade...")
    df, sens_ind, corr = analise_sensibilidade_completa(
        bn, pontos=args.pontos, tau_soft=args.tau_soft
    )

    # Salva resultados
    output_file = f"sensibilidade_completa_p{args.pontos}.csv"
    df.to_csv(output_file, index=False)
    
    # Exibe resultados
    print("\n=== 📊 RESULTADOS DA ANÁLISE DE SENSIBILIDADE ===")
    print(f"Total de simulações: {len(df)}")
    print(f"AE médio: {df['AE'].mean():.4f}")
    print(f"AE mínimo: {df['AE'].min():.4f}")
    print(f"AE máximo: {df['AE'].max():.4f}")
    
    print("\n--- Sensibilidade Individual ---")
    for var, sens in sens_ind.items():
        print(f"  {var}: {sens:.4f}")
    
    print("\n--- Correlações com AE ---")
    for var, corr_val in corr.items():
        print(f"  {var}: {corr_val:.4f}")
    
    print(f"\n💾 Resultados salvos em: {output_file}")
    
    # Gera gráficos
    try:
        plot_resultados(df, sens_ind, corr)
        print("🖼️  Gráficos salvos como 'sensibilidade_resultados.png'")
    except Exception as e:
        print(f"⚠️  Erro ao gerar gráficos: {e}")


    # Gera gráficos compactos para o paper
    try:
        plot_compact_results(df, sens_ind, corr)
        print("🖼️  Figuras salvas: 'sens_global_compact.png' e 'ae_distribution.png'")
    except Exception as e:
        print(f"⚠️  Erro ao gerar gráficos: {e}")

if __name__ == "__main__":
    main()