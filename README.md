# 🚴 Mobike - Sistema de Previsão de Risco para Ciclistas

Um sistema inteligente de previsão de risco para ciclistas que utiliza dados meteorológicos para classificar o nível de segurança em ciclovias. O projeto integra três modelos de machine learning (Árvore de Decisão, Regressão Logística e Rede Neural MLP) para prever riscos em diferentes condições climáticas.

## 📋 Visão Geral

O Mobike analisa condições meteorológicas em tempo real para:
- **Classificar risco** em ciclovias como Baixo, Médio ou Alto
- **Prever segurança** para ciclistas baseado em dados climáticos
- **Comparar modelos** de ML para melhor acurácia
- **Testar cenários** com dados sintéticos

## 🏗️ Arquitetura do Projeto

```
Mobike/
├── data/                          # Dados
│   ├── raw/                       # Dados brutos da API
│   │   ├── ciclovias.csv         # Dataset principal
│   │   ├── location_*.json        # Dados brutos por local
│   │   ├── metadata.json          # Metadados da coleta
│   │   └── collection_log.txt     # Log de execução
│   └── processed/
│       └── weather_processed.csv  # Dados processados
│
├── src/                           # Código-fonte
│   ├── models/                    # Modelos de ML
│   │   ├── decision_tree.py      # Árvore de Decisão
│   │   ├── logistic_regression.py # Regressão Logística
│   │   └── mlp.py                # Rede Neural MLP
│   └── prepocessing/              # Preparação de dados
│       ├── config_stations.json   # Configuração de locais
│       ├── fetch_weather_data.py  # Coleta de dados da API
│       └── preprocess.py          # Limpeza e engenharia de features
│
├── README.md                      # Este arquivo
├── DATA_COLLECTION.md             # Documentação de coleta de dados
└── requiriments.txt               # Dependências do projeto
```

## 🎯 Features Utilizadas

O sistema analisa **6 variáveis meteorológicas** para prever risco:

| Feature | Descrição | Unidade |
|---------|-----------|---------|
| `weather_code` | Código WMO de tipo de clima | - |
| `wind_speed_10m` | Velocidade do vento a 10m | km/h |
| `precipitation` | Precipitação acumulada | mm |
| `sensacao_termica` | Sensação térmica | °C |
| `chuva_acumulada_3h` | Chuva acumulada (últimas 3h) | mm |
| `rajada_maxima_3h` | Rajada máxima de vento (últimas 3h) | km/h |

## 📊 Modelos Implementados

### 1. **Árvore de Decisão** (`decision_tree.py`)
- Classificação com 3 classes: Baixo, Médio, Alto
- Implementação customizada com cálculo de entropia
- Profundidade máxima: 5 níveis
- Ideal para interpretabilidade

```bash
python src/models/decision_tree.py
```

### 2. **Regressão Logística** (`logistic_regression.py`)
- Classificação binária: Seguro (0) vs Não Seguro (1)
- Pipeline com normalização automática
- Threshold ajustável (padrão: 0.5)
- Melhor para probabilidades calibradas

```bash
python src/models/logistic_regression.py
```

### 3. **Rede Neural MLP** (`mlp.py`)
- Regressão contínua: saída entre 0 e 1
- Arquitetura: 64 → 32 → 1 neurônios
- Ativação Sigmoid na saída (garante 0-100%)
- 50 épocas de treinamento

```bash
python src/models/mlp.py
```



## 📈 Resultados Esperados

Cada modelo exibe:

- **Métricas de desempenho** (Acurácia, MSE, R², etc.)
- **Matriz de confusão**
- **Relatório de classificação**
- **Exemplos de previsões** em dados de teste
- **Testes com ciclovias fictícias** para validação

## 🗂️ Estrutura de Dados

### CSV Principal (`data/raw/ciclovias.csv`)

```
NOME_LOGRADOURO,COORDENADAS,weather_code,wind_speed_10m,precipitation,...,rótulo
Av. Afonso Pena,-19.932..., -43.929...,3,4.7,2.42,...,Médio
...
```

### Formato JSON (Dados Brutos)

```json
{
  "hourly": {
    "time": ["2025-12-03T00:00", "2025-12-03T01:00", ...],
    "temperature_2m": [22.5, 21.8, ...],
    "weather_code": [0, 0, ...],
    ...
  }
}
```



## 📚 Documentação Adicional

- **[DATA_COLLECTION.md](DATA_COLLECTION.md)** - Detalhes sobre coleta de dados da API Open-Meteo
- **[Decision Tree](src/models/decision_tree.py)** - Implementação de Árvore de Decisão customizada
- **[Logistic Regression](src/models/logistic_regression.py)** - Regressão Logística com threshold ajustável
- **[MLP](src/models/mlp.py)** - Rede Neural com TensorFlow

## 🔌 API Integrada

### Open-Meteo

- **URL**: https://open-meteo.com/
- **Tipo**: API pública, sem autenticação
- **Limite**: 10.000 chamadas/dia
- **Variáveis**: Temperatura, Umidade, Vento, Chuva, Código WMO

## 🧪 Testando com Dados Fictícios

Cada modelo inclui testes automáticos com 3 cenários:

1. **Clima Ideal** → Risco Baixo (vento 8 km/h, sem chuva)
2. **Chuva Leve** → Risco Médio (chuva 1.5mm, vento 22 km/h)
3. **Tempestade** → Risco Alto (chuva 18mm, vento 45 km/h)

## 📊 Comparação de Modelos

| Aspecto | Decision Tree | Logistic Reg. | MLP |
|---------|---------------|---------------|-----|
| **Tipo** | Classificação | Classificação | Regressão |
| **Classes** | 3 (Baixo/Médio/Alto) | 2 (Binário) | Contínuo (0-1) |
| **Interpretabilidade** | ⭐⭐⭐ Alta | ⭐⭐ Média | ⭐ Baixa |
| **Tempo Treino** | Rápido | Muito Rápido | Moderado |
| **Precisão** | Boa | Boa | Melhor |

