"""
MLP com TensorFlow/Keras para prever risco de ciclistas
Saída: valor contínuo entre 0 e 1 (interpretação como porcentagem de risco)
"""

import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================ #
# FUNÇÕES DE CARREGAMENTO E PREPARO
# ============================================================================ #

def carregar_dados(caminho='data/raw/ciclovias.csv'):
    return pd.read_csv(caminho)

def preparar_features(df):
    # Remover colunas desnecessárias (identificação, não são features)
    df = df.drop(columns=['NOME_LOGRADOURO', 'COORDENADAS'], errors='ignore')
    
    # Features meteorológicas utilizadas pelo modelo
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
    # Mapear para valores contínuos (0.0 a 1.0)
    mapa = {'Baixo': 0.0, 'Médio': 0.5, 'Medio': 0.5, 'Alto': 1.0}
    y = df['rótulo'].map(mapa).fillna(0).astype(float)
    return X, y

# ============================================================================ #
# FUNÇÃO DE AVALIAÇÃO
# ============================================================================ #

def avaliar_modelo(y_real, y_pred, modelo_nome="MLP TensorFlow"):
    mse = mean_squared_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    print(f"\n{'='*70}")
    print(f"AVALIAÇÃO: {modelo_nome}")
    print(f"{'='*70}")
    print(f"Erro quadrático médio (MSE): {mse:.4f}")
    print(f"Coeficiente de determinação (R²): {r2:.4f}")
    print("\nExemplos de previsões:")
    for i in range(5):
        print(f"Real: {y_real.iloc[i]:.2f} | Previsto: {y_pred[i]:.2f} ({y_pred[i]*100:.1f}% de risco)")
    return mse, r2

# ============================================================================ #
# MAIN: Treino e avaliação do MLP com TensorFlow
# ============================================================================ #

if __name__ == '__main__':
    print("\n" + "="*70)
    print("MLP TENSORFLOW - PREVISÃO DE RISCO PARA CICLISTAS")
    print("="*70)

    # Carregar e preparar
    df = carregar_dados('data/raw/ciclovias.csv')
    X, y = preparar_features(df)

    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar features (boa prática para redes neurais)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Construir modelo MLP
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # saída entre 0 e 1
    ])

    # Compilar modelo
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Treinar modelo
    history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                        validation_split=0.2, verbose=1)

    # Avaliar
    y_pred = model.predict(X_test).flatten()
    mse, r2 = avaliar_modelo(y_test, y_pred, "Rede Neural MLP (TensorFlow)")

    # ============================
    # TESTAR COM CICLOVIAS FICTÍCIAS
    # ============================
    print("\n" + "="*70)
    print("PREVISÕES EM CICLOVIAS FICTÍCIAS")
    print("="*70)
    
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

    # Normalizar com o mesmo scaler
    novas_ciclovias_norm = scaler.transform(novas_ciclovias)
    preds_ficticias = model.predict(novas_ciclovias_norm).flatten()
    
    # Garantir que as previsões fiquem entre 0 e 1
    preds_ficticias = np.clip(preds_ficticias, 0, 1)
    
    # Criar coluna com interpretação (arredonda para a categoria mais próxima)
    def categorizar_risco(valor):
        if valor < 0.25:
            return "Baixo"
        elif valor < 0.75:
            return "Médio"
        else:
            return "Alto"
    
    resultado_ficticias = novas_ciclovias.copy()
    resultado_ficticias['Risco (%)'] = (preds_ficticias * 100).round(1)
    resultado_ficticias['Categoria'] = [categorizar_risco(p) for p in preds_ficticias]
    
    print("\nResultados das ciclovias fictícias:\n")
    print(resultado_ficticias.to_string(index=False))

    # Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    print(f"MSE: {mse:.4f} | R²: {r2:.4f}")
    print(f"Dataset: {len(df)} ciclovias com dados meteorológicos sintéticos")
    print(f"Features utilizadas: {', '.join(X.columns)}")
    print("\nPrevisões para ciclovias fictícias:")
    for i, cat in enumerate(resultado_ficticias['Categoria']):
        print(f"  Ciclovia {i+1}: {cat} ({resultado_ficticias['Risco (%)'].iloc[i]}% de risco)")
    print("\n" + "="*70 + "\n")
