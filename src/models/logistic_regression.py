# logistic_regression_score_threshold.py
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# -------------------------
# Carregamento e preparação
# -------------------------
def carregar_dados(caminho='data/raw/ciclovias.csv'):
    return pd.read_csv(caminho)

def preparar_features(df):
    features = [
        'weather_code','wind_speed_10m','precipitation',
        'sensacao_termica','chuva_acumulada_3h','rajada_maxima_3h'
    ]
    # garante colunas presentes
    for c in features:
        if c not in df.columns:
            df[c] = 0.0
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # normalizar nome da coluna de rótulo
    if 'rótulo' not in df.columns and 'rotulo' in df.columns:
        df['rótulo'] = df['rotulo']
    if 'rótulo' not in df.columns:
        raise ValueError("Coluna 'rótulo' não encontrada em ciclovias.csv")

    # Mapeamento para binário: Baixo -> Seguro (0); Médio/Alto -> Não seguro (1)
    mapa_bin = {'Baixo': 0, 'Médio': 1, 'Medio': 1, 'Alto': 1}
    y = df['rótulo'].map(mapa_bin).fillna(1).astype(int)
    return X, y

# -------------------------
# Divisão estratificada
# -------------------------
def dividir_estratificado(X, y, proporcao_teste=0.2, seed=42):
    return train_test_split(X, y, test_size=proporcao_teste, random_state=seed, stratify=y)

# -------------------------
# Treinamento do modelo
# -------------------------
def treinar_regressao_logistica(X_train, y_train, seed=42, max_iter=1000, class_weight='balanced'):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=seed, max_iter=max_iter, class_weight=class_weight))
    ])
    pipe.fit(X_train, y_train)
    return pipe

# -------------------------
# Predição com limiar ajustável
# -------------------------
def prever_com_threshold(modelo, X, threshold=0.5):
    """
    Retorna um DataFrame com colunas:
    - probabilidade: probabilidade prevista de classe 1 (Não seguro)
    - pred_label: rótulo binário com base no threshold (0 = Seguro, 1 = Não seguro)
    """
    probs = modelo.predict_proba(X)[:, 1]  # probabilidade de classe 1
    preds = (probs >= threshold).astype(int)
    return pd.DataFrame({'probabilidade': probs, 'pred_label': preds})

# -------------------------
# Avaliação
# -------------------------
def avaliar_modelo(y_real, probs, threshold=0.5, modelo_nome="Regressão Logística"):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_real, preds)
    auc = None
    try:
        auc = roc_auc_score(y_real, probs)
    except Exception:
        auc = float('nan')
    print(f"\n{'='*70}")
    print(f"AVALIAÇÃO: {modelo_nome}")
    print(f"{'='*70}")
    print(f"Acurácia (threshold={threshold}): {acc:.4f} ({100*acc:.2f}%)")
    if not np.isnan(auc):
        print(f"AUC ROC: {auc:.4f}")
    print(f"\nRelatório de classificação:")
    print(classification_report(y_real, preds, target_names=['Seguro','Não seguro'], zero_division=0))
    print(f"\nMatriz de confusão:")
    print(confusion_matrix(y_real, preds))
    return {'accuracy': acc, 'auc': auc}

# -------------------------
# MAIN
# -------------------------
if __name__ == '__main__':
    # Parâmetros configuráveis
    CAMINHO_CSV = 'data/raw/ciclovias.csv'   
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    MAX_ITER = 2000
    CLASS_WEIGHT = 'balanced'   
    THRESHOLD_PADRAO = 0.5      # limiar padrão (ajustável entre 0 e 1) o critério de decisão pode ser alterado aqui

    print("\n" + "="*70)
    print("REGRESSÃO LOGÍSTICA COM PONTUAÇÃO 0-1 E LIMIAR AJUSTÁVEL")
    print("="*70)

    # Carregar e preparar
    df = carregar_dados(CAMINHO_CSV)
    X, y = preparar_features(df)

    # Dividir
    X_train, X_test, y_train, y_test = dividir_estratificado(X, y, proporcao_teste=TEST_SIZE, seed=RANDOM_SEED)

    print(f"\nTreino: {len(X_train)} amostras | Distribuição: {dict(Counter(y_train))}")
    print(f"Teste:  {len(X_test)} amostras | Distribuição: {dict(Counter(y_test))}")

    # Treinar
    modelo = treinar_regressao_logistica(X_train, y_train, seed=RANDOM_SEED, max_iter=MAX_ITER, class_weight=CLASS_WEIGHT)

    # Prever probabilidades no conjunto de teste
    probs_test = modelo.predict_proba(X_test)[:, 1]

    # Avaliar com limiar padrão
    resultados = avaliar_modelo(y_test, probs_test, threshold=THRESHOLD_PADRAO, modelo_nome="Regressão Logística (Binária)")

    # Mostrar exemplos de probabilidades + rótulos no teste
    df_test_result = X_test.copy().reset_index(drop=True)
    df_test_result['probabilidade'] = probs_test
    df_test_result['pred_label'] = (df_test_result['probabilidade'] >= THRESHOLD_PADRAO).astype(int)
    df_test_result['true_label'] = y_test.reset_index(drop=True)
    print("\nAmostras de teste com probabilidades (primeiras 10 linhas):")
    print(df_test_result.head(10))

    # ============================
    # TESTE DE EFICÁCIA COM CICLOVIAS FICTÍCIAS
    # ============================
    novas_ciclovias = pd.DataFrame([
        {
            'weather_code': 0, 'wind_speed_10m': 8.0, 'precipitation': 0.0,
            'sensacao_termica': 24.0, 'chuva_acumulada_3h': 0.0, 'rajada_maxima_3h': 12.0
        },  # Clima ideal → esperado: Seguro
        {
            'weather_code': 3, 'wind_speed_10m': 22.0, 'precipitation': 1.5,
            'sensacao_termica': 19.0, 'chuva_acumulada_3h': 6.0, 'rajada_maxima_3h': 30.0
        },  # Chuva leve → esperado: Não seguro (possivelmente)
        {
            'weather_code': 5, 'wind_speed_10m': 45.0, 'precipitation': 18.0,
            'sensacao_termica': 12.0, 'chuva_acumulada_3h': 28.0, 'rajada_maxima_3h': 70.0
        }   # Tempestade forte → esperado: Não seguro
    ])

    # Garante mesma ordem de features
    novas_ciclovias = novas_ciclovias[X.columns]

    # Probabilidades e previsões com limiar configurável
    resultados_novas = prever_com_threshold(modelo, novas_ciclovias, threshold=THRESHOLD_PADRAO)
    mapa_inv = {0: "É seguro", 1: "Não seguro"}
    novas_ciclovias_exib = novas_ciclovias.copy()
    novas_ciclovias_exib['probabilidade'] = resultados_novas['probabilidade']
    novas_ciclovias_exib['previsao'] = resultados_novas['pred_label'].map(mapa_inv)

    print("\nResultados das ciclovias fictícias:\n")
    print(novas_ciclovias_exib)

    # Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    print(f"Acurácia da Regressão Logística (threshold={THRESHOLD_PADRAO}): {100*resultados['accuracy']:.2f}%")
    print(f"AUC ROC (teste): {resultados['auc']}")
    print(f"Dataset: {len(df)} ciclovias com dados meteorológicos")
    print(f"Features: {', '.join(X.columns)}")
    print("\nPrevisão para ciclovias fictícias:", list(novas_ciclovias_exib['previsao']))
    print("\n" + "="*70 + "\n")
