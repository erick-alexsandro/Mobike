"""
Árvore de Decisão para Classificação de Risco de Ciclistas
"""

import math
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================================ #
# FUNÇÕES PARA CALCULAR ENTROPIA E GANHO DE INFORMAÇÃO
# ============================================================================ #

def calcular_entropia(y):
    n = len(y)
    if n == 0:
        return 0.0
    contagem = Counter(y)
    entropia_val = 0.0
    for classe, qtd in contagem.items():
        p = qtd / n
        if p > 0:
            entropia_val -= p * math.log2(p)
    return entropia_val

def calcular_entropia_ponderada(y_esq, y_dir):
    n_total = len(y_esq) + len(y_dir)
    if n_total == 0:
        return 0.0
    e_esq = calcular_entropia(y_esq)
    e_dir = calcular_entropia(y_dir)
    return (len(y_esq) / n_total) * e_esq + (len(y_dir) / n_total) * e_dir

def calcular_ganho_informacao(y_inicial, y_esq, y_dir):
    e_inicial = calcular_entropia(y_inicial)
    e_pon = calcular_entropia_ponderada(y_esq, y_dir)
    return e_inicial - e_pon

def testar_cortes_em_coluna(coluna_x, y, max_cortes=20):
    coluna_x = np.asarray(coluna_x)
    valores = np.unique(coluna_x[~np.isnan(coluna_x)])
    if len(valores) <= 1:
        return None, 0.0
    if len(valores) > max_cortes:
        percentis = np.percentile(valores, np.linspace(0, 100, max_cortes))
        candidatos = np.unique(percentis)
    else:
        candidatos = (valores[:-1] + valores[1:]) / 2.0
    y_array = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    melhor_ganho, melhor_limite = -1.0, None
    for threshold in candidatos:
        mascara_esq = coluna_x <= threshold
        mascara_dir = coluna_x > threshold
        if mascara_esq.sum() == 0 or mascara_dir.sum() == 0:
            continue
        y_esq = y_array[mascara_esq]
        y_dir = y_array[mascara_dir]
        ganho = calcular_ganho_informacao(y_array, y_esq, y_dir)
        if ganho > melhor_ganho:
            melhor_ganho, melhor_limite = ganho, float(threshold)
    return melhor_limite, melhor_ganho

# ============================================================================ #
# CLASSE NÓ E ÁRVORE DE DECISÃO
# ============================================================================ #

class NoArvore:
    def __init__(self, coluna=None, limite=None, esquerda=None, direita=None,
                 classe_predita=None, entropia=None, n_amostras=None):
        self.coluna = coluna
        self.limite = limite
        self.esquerda = esquerda
        self.direita = direita
        self.classe_predita = classe_predita
        self.entropia = entropia
        self.n_amostras = n_amostras
    def eh_folha(self):
        return self.classe_predita is not None

def construir_arvore_decisao(X, y, profundidade=0, prof_max=5, min_amostras_folha=2, verbose=False):
    n = len(y)
    y_array = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    e_atual = calcular_entropia(y_array)
    classe_mai = Counter(y_array).most_common(1)[0][0]
    prefixo = "  " * profundidade

    if len(set(y_array)) == 1:
        if verbose:
            print(f"{prefixo}[FOLHA PURA] Classe: {y_array[0]} | Entropia: 0.0 | n={n}")
        return NoArvore(classe_predita=y_array[0], entropia=0.0, n_amostras=n)

    if profundidade >= prof_max or n <= min_amostras_folha:
        if verbose:
            motivo = "profundidade máxima" if profundidade >= prof_max else "poucas amostras"
            print(f"{prefixo}[PARADA - {motivo}] Classe majoritária: {classe_mai} | Entropia: {e_atual:.4f} | n={n}")
        return NoArvore(classe_predita=classe_mai, entropia=e_atual, n_amostras=n)

    melhor_ganho, melhor_coluna, melhor_limite = -1.0, None, None
    for col in X.columns:
        x_col = X[col].to_numpy()
        limite, ganho = testar_cortes_em_coluna(x_col, y)
        if limite is not None and ganho > melhor_ganho:
            melhor_ganho, melhor_coluna, melhor_limite = ganho, col, limite

    if melhor_ganho <= 0 or melhor_coluna is None:
        if verbose:
            print(f"{prefixo}[SEM GANHO] Classe majoritária: {classe_mai} | Entropia: {e_atual:.4f} | n={n}")
        return NoArvore(classe_predita=classe_mai, entropia=e_atual, n_amostras=n)

    if verbose:
        print(f"{prefixo}[DIVISÃO] {melhor_coluna} <= {melhor_limite:.4f} | Ganho: {melhor_ganho:.6f} | Entropia: {e_atual:.6f}")

    mascara_esq = X[melhor_coluna] <= melhor_limite
    mascara_dir = ~mascara_esq
    X_esq, y_esq = X[mascara_esq].reset_index(drop=True), y[mascara_esq].reset_index(drop=True)
    X_dir, y_dir = X[mascara_dir].reset_index(drop=True), y[mascara_dir].reset_index(drop=True)

    no_esq = construir_arvore_decisao(X_esq, y_esq, profundidade+1, prof_max, min_amostras_folha, verbose)
    no_dir = construir_arvore_decisao(X_dir, y_dir, profundidade+1, prof_max, min_amostras_folha, verbose)

    return NoArvore(coluna=melhor_coluna, limite=melhor_limite, esquerda=no_esq, direita=no_dir,
                    entropia=e_atual, n_amostras=n)

# ============================================================================ #
# FUNÇÕES DE PREVISÃO, CARREGAMENTO, PREPARO E AVALIAÇÃO
# ============================================================================ #

def prever_uma_amostra(no, amostra):
    if no.eh_folha():
        return no.classe_predita
    if amostra[no.coluna] <= no.limite:
        return prever_uma_amostra(no.esquerda, amostra)
    else:
        return prever_uma_amostra(no.direita, amostra)

def prever(arvore, X):
    return np.array([prever_uma_amostra(arvore, linha) for _, linha in X.iterrows()])

def carregar_dados(caminho='data/raw/ciclovias.csv'):
    return pd.read_csv(caminho)

def preparar_features(df):
    features = [
        'weather_code','wind_speed_10m','precipitation',
        'sensacao_termica','chuva_acumulada_3h','rajada_maxima_3h'
    ]
    for c in features:
        if c not in df.columns:
            df[c] = 0.0
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    if 'rótulo' not in df.columns and 'rotulo' in df.columns:
        df['rótulo'] = df['rotulo']
    if 'rótulo' not in df.columns:
        raise ValueError("Coluna 'rótulo' não encontrada em ciclovias.csv")
    mapa = {'Baixo': 0, 'Médio': 1, 'Medio': 1, 'Alto': 2}
    y = df['rótulo'].map(mapa).fillna(0).astype(int)
    return X, y

def dividir_estratificado(X, y, proporcao_teste=0.2, seed=42):
    np.random.seed(seed)
    indices_por_classe = {}
    for idx, classe in enumerate(y):
        indices_por_classe.setdefault(classe, []).append(idx)

    train_idx, test_idx = [], []
    for classe, idxs in indices_por_classe.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        n_teste = max(1, int(len(idxs) * proporcao_teste))
        test_idx.extend(idxs[:n_teste].tolist())
        train_idx.extend(idxs[n_teste:].tolist())

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    return X_train, X_test, y_train, y_test

def avaliar_modelo(y_real, y_pred, modelo_nome="Árvore de Decisão"):
    acc = accuracy_score(y_real, y_pred)
    print(f"\n{'='*70}")
    print(f"AVALIAÇÃO: {modelo_nome}")
    print(f"{'='*70}")
    print(f"Acurácia: {acc:.4f} ({100*acc:.2f}%)")
    print(f"\nRelatório de classificação:")
    print(classification_report(y_real, y_pred, target_names=['Baixo','Médio','Alto'], zero_division=0))
    print(f"\nMatriz de confusão:")
    print(confusion_matrix(y_real, y_pred))
    return acc

# ============================================================================ #
# MAIN OPCIONAL: imprime dados de treino/teste e métricas da árvore
# (sem testes de ciclovias fictícias — isso fica no notebook)
# ============================================================================ #

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ÁRVORE DE DECISÃO - CLASSIFICAÇÃO DE RISCO PARA CICLISTAS")
    print("="*70)

    # Carregar e preparar
    df = carregar_dados('data/raw/ciclovias.csv')
    X, y = preparar_features(df)

    # Dividir estratificado
    X_train, X_test, y_train, y_test = dividir_estratificado(X, y, proporcao_teste=0.2, seed=42)

    # Mostrar tamanhos e distribuição
    print(f"\nTreino: {len(X_train)} amostras | Distribuição: {dict(Counter(y_train))}")
    print(f"Teste:  {len(X_test)} amostras | Distribuição: {dict(Counter(y_test))}")

    # Treinar árvore
    arvore = construir_arvore_decisao(X_train, y_train, prof_max=5, min_amostras_folha=2, verbose=True)

    # Avaliar
    y_pred = prever(arvore, X_test)
    acc = avaliar_modelo(y_test, y_pred, "Árvore de Decisão (Implementação do Algoritmo)")

    # Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    print(f"Acurácia da Árvore de Decisão: {100*acc:.2f}%")
    print(f"Dataset: {len(df)} ciclovias com dados meteorológicos sintéticos")
    print(f"Features: {', '.join(X.columns)}")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    

    # ============================
    # TESTE DE EFICÁCIA COM CICLOVIAS FICTÍCIAS
    # ============================
    novas_ciclovias = pd.DataFrame([
        {
            'weather_code': 0, 'wind_speed_10m': 8.0, 'precipitation': 0.0,
            'sensacao_termica': 24.0, 'chuva_acumulada_3h': 0.0, 'rajada_maxima_3h': 12.0
        },  # Clima ideal → risco baixo
        {
            'weather_code': 3, 'wind_speed_10m': 22.0, 'precipitation': 1.5,
            'sensacao_termica': 19.0, 'chuva_acumulada_3h': 6.0, 'rajada_maxima_3h': 30.0
        },  # Chuva leve → risco médio
        {
            'weather_code': 5, 'wind_speed_10m': 45.0, 'precipitation': 18.0,
            'sensacao_termica': 12.0, 'chuva_acumulada_3h': 28.0, 'rajada_maxima_3h': 70.0
        }   # Tempestade forte → risco alto
    ])

    mapa_inv = {0: "Baixo", 1: "Médio", 2: "Alto"}
    preds = prever(arvore, novas_ciclovias)

    # Mostrar resultados em tabela
    novas_ciclovias['Previsão'] = [mapa_inv[p] for p in preds]
    print("\nResultados das ciclovias fictícias:\n")
    print(novas_ciclovias)

    # ============================
    # RESUMO FINAL
    # ============================
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    print(f"Acurácia da Árvore de Decisão: {100*acc:.2f}%")
    print(f"Dataset: {len(df)} ciclovias com dados meteorológicos sintéticos")
    print(f"Features: {', '.join(X.columns)}")
    print("\nPrevisão para ciclovias fictícias:", [mapa_inv[p] for p in preds])
    print("\n" + "="*70 + "\n")






