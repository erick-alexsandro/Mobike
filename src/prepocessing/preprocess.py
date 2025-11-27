import pandas as pd
import numpy as np
import os
import json

# --- CONFIGURAÇÕES ---
# Ajuste estes nomes de colunas conforme o seu CSV original
COL_TEMP = 'temperature_2m'         # Coluna de temperatura
COL_HUMID = 'relative_humidity_2m'  # Coluna de umidade
COL_WIND = 'wind_speed_10m'         # Coluna de vento
COL_RAIN = 'precipitation'          # Coluna de chuva
COL_DATE = 'time'                   # Coluna de data/hora

def load_data(filepath):
    """Carrega os dados brutos."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    # Verifica se é JSON ou CSV
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Extrai os dados horários
        df = pd.DataFrame(data['hourly'])
        # Converte a coluna 'time' para datetime
        df['time'] = pd.to_datetime(df['time'])
    else:
        # Tenta ler inferindo formato de data
        df = pd.read_csv(filepath, parse_dates=[COL_DATE])
    
    print(f"Dados carregados: {df.shape}")
    return df

def clean_data(df):
    """
    Tarefa 1: Limpeza dos dados
    - Tratar nulos
    - Padronizar unidades
    """
    df_clean = df.copy()

    # 1. Tratar Nulos (Exemplo: preencher com a média ou valor anterior)
    # Para dados climáticos/temporais, 'ffill' (forward fill) é comum para não quebrar a sequência
    cols_to_fill = [COL_TEMP, COL_HUMID, COL_WIND, COL_RAIN]
    for col in cols_to_fill:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].ffill().fillna(0)

    # 2. Padronizar Unidades (Exemplo hipotético)
    # Se a temperatura estiver em Kelvin, converter para Celsius
    # (Supondo que se a média for > 200, está em Kelvin)
    if df_clean[COL_TEMP].mean() > 200:
        df_clean[COL_TEMP] = df_clean[COL_TEMP] - 273.15
    
    # Arredondar valores float para 2 casas decimais para padronização
    df_clean = df_clean.round(2)
    
    print("Limpeza e padronização concluídas.")
    return df_clean

def calculate_heat_index(temp, humidity):
    """
    Calcula a Sensação Térmica (Heat Index) simplificada.
    Fórmula baseada em aproximações usuais para T > 20C.
    """
    # Fórmula simplificada do Heat Index (Steadman)
    # HI = 0.5 * {T + 61.0 + [(T-68.0)*1.2] + (RH*0.094)} (versão simplificada)
    # Ou uso direto da temperatura aparente australiana (AT) que é comum em datasets simples:
    # AT = Ta + 0.33×e − 0.70×ws − 4.00 (requer pressão de vapor)
    
    # Abaixo, uma fórmula genérica de "Feels Like" combinando T e Vento/Umidade
    # Se T < 10, usa Wind Chill. Se T > 20, usa Heat Index.
    
    # Vamos usar uma aproximação vetorizada simples para o exercício:
    # HI = T - 0.55 * (1 - 0.01 * RH) * (T - 14.5)
    return temp - 0.55 * (1 - 0.01 * humidity) * (temp - 14.5)

def feature_engineering(df):
    """
    Tarefa 2: Criar variáveis derivadas
    - Chuva acumulada
    - Rajada máxima
    - Sensação térmica
    """
    df_eng = df.copy()

    # 1. Sensação Térmica
    if COL_TEMP in df_eng.columns and COL_HUMID in df_eng.columns:
        df_eng['sensacao_termica'] = calculate_heat_index(
            df_eng[COL_TEMP], 
            df_eng[COL_HUMID]
        )
    
    # 2. Chuva Acumulada
    # Se os dados forem horários, cria uma janela móvel de 24h, por exemplo
    if COL_RAIN in df_eng.columns:
        # Soma móvel das últimas 3 horas (exemplo) ou acumulado diário
        df_eng['chuva_acumulada_3h'] = df_eng[COL_RAIN].rolling(window=3, min_periods=1).sum()
        
    # 3. Rajada Máxima
    # Se tivermos dados de vento instantâneo, a rajada pode ser o max numa janela
    if COL_WIND in df_eng.columns:
        # Máxima das últimas 3 horas
        df_eng['rajada_maxima_3h'] = df_eng[COL_WIND].rolling(window=3, min_periods=1).max()

    print("Engenharia de features concluída.")
    return df_eng

def save_data(df, output_path):
    """Tarefa 3: Salvar dataset final em data/processed/"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset salvo com sucesso em: {output_path}")

def main():
    # Caminhos (ajuste o nome do arquivo de entrada conforme necessário)
    base_path = os.getcwd() # Ou defina o caminho absoluto da pasta Mobike
    raw_path = os.path.join(base_path, 'data', 'raw', 'location_001_raw.json') 
    processed_path = os.path.join(base_path, 'data', 'processed', 'weather_processed.csv')

    try:
        # Pipeline de execução
        df = load_data(raw_path)
        df = clean_data(df)
        df = feature_engineering(df)
        save_data(df, processed_path)
        
    except Exception as e:
        print(f"Erro no processamento: {e}")

if __name__ == "__main__":
    main()