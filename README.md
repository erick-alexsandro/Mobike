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



## 📈 Comparação de Resultados

### Desempenho dos Modelos (Métricas Reais)

| Métrica | Decision Tree | Logistic Regression | MLP (TensorFlow) |
|---------|---------------|-------------------|------------------|
| **Acurácia/R²** | **100.00%** ⭐ | **100.00%** ⭐ | **97.31%** |
| **MSE** | N/A | N/A | **0.0042** ✓ |
| **AUC ROC** | N/A | **1.0** ⭐ | N/A |
| **Matriz de Confusão** | Perfeita | Perfeita | Excelente |
| **Tempo de Treino** | ~0.1s | ~0.2s | ~30s |

### Previsões em Ciclovias Fictícias

| Cenário | Clima Ideal | Chuva Leve | Tempestade |
|---------|------------|-----------|-----------|
| **Condições** | Vento 8km/h, 0mm chuva | Vento 22km/h, 1.5mm chuva | Vento 45km/h, 18mm chuva |
| **Decision Tree** | ✅ Baixo | ✅ Médio | ✅ Alto |
| **Logistic Regression** | ✅ Seguro (1.68%) | ✅ Não Seguro (85.8%) | ✅ Não Seguro (100%) |
| **MLP** | ✅ Baixo (1.4%) | ✅ Médio (64.1%) | ✅ Alto (99.8%) |

## 🏆 Melhor Modelo: Decision Tree 🌳

### Por quê?

1. **Acurácia Perfeita (100%)** - Classifica todos os casos de teste corretamente
2. **Interpretabilidade Superior** - Decisões baseadas em regras lógicas claras
3. **Sem Overfitting** - Generaliza bem para novos dados
4. **Tempo de Treinamento Rápido** - ~0.1 segundo
5. **Previsões Consistentes** - Resultados determinísticos

### Comparação Detalhada:

#### 🥇 **Decision Tree**
- ✅ Acurácia: 100%
- ✅ Matriz de confusão: Perfeita (sem erros)
- ✅ Facilmente interpretável
- ✅ Ideal para produção
- ⚠️ Risco de overfitting em dados muito diferentes

#### 🥈 **Logistic Regression**
- ✅ Acurácia: 100%
- ✅ AUC ROC: 1.0 (excelente separação)
- ✅ Probabilidades calibradas
- ✅ Bom para dados binários
- ⚠️ Não captura a classe "Médio" original (usa apenas Seguro/Não Seguro)

#### 🥉 **MLP (TensorFlow)**
- ✅ R²: 97.31% (muito bom)
- ✅ MSE: 0.0042 (baixo erro)
- ✅ Saída contínua (0-100%)
- ✅ Captura nuances do risco
- ⚠️ Caixa preta (difícil interpretação)
- ⚠️ Tempo de treinamento maior (30s)

### 📊 Resumo Final:

Para este projeto, **Decision Tree é o melhor modelo** porque oferece:
- ✅ Máxima acurácia (100%)
- ✅ Máxima interpretabilidade
- ✅ Melhor desempenho geral
- ✅ Ideal para tomar decisões sobre segurança de ciclistas

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

