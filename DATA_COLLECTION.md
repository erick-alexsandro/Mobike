# Documentação de Coleta de Dados - Mobike

## Visão Geral

Este documento descreve a infraestrutura configurada para coleta de dados meteorológicos via API Open-Meteo e a utilização do modelo sintético. 

## API Open-Meteo

### Fonte
- **URL**: https://open-meteo.com/
- **Documentação**: https://open-meteo.com/en/docs
- **Tipo**: API pública, sem autenticação necessária
- **Limite de taxa**: 10.000 chamadas/dia

### Dados Coletados

#### Variáveis Horárias Disponíveis
| Variável | Descrição | Unidade |
|----------|-----------|---------|
| `temperature_2m` | Temperatura do ar a 2m de altura | °C |
| `relative_humidity_2m` | Umidade relativa do ar a 2m | % |
| `weather_code` | Código de tipo de clima (WMO) | - |
| `wind_speed_10m` | Velocidade do vento a 10m | km/h |
| `precipitation` | Precipitação acumulada | mm |

#### Dados do modelo sintético
| Variável | Descrição | Unidade |
|----------|-----------|---------|
| `temperature_2m` | Temperatura do ar a 2m de altura | °C |
| `relative_humidity_2m` | Umidade relativa do ar a 2m | % |
| `weather_code` | Código de tipo de clima (WMO) | - |
| `wind_speed_10m` | Velocidade do vento a 10m | km/h |
| `precipitation` | Precipitação acumulada | mm |
| `sensacao_termica` | Sensação térmica | °C |
| `chuva_acumulada_3h` | Chuva acumulada nas últimas 3 horas | mm |
| `rajada_maxima_3h` | Velocidade máxima de rajada de vento nas últimas 3 horas | km/h |

## Local escolhido

As coordenadas são configuráveis em `src/prepocessing/config_stations.json`:

```json
{
  "id": "location_001",
  "name": "Centro",
  "latitude": -19.9167,
  "longitude": -43.9345
}
```
## Estrutura de Dados Brutos

Os dados brutos são salvos em `data/raw/` com a seguinte estrutura:

```
data/raw/
├── metadata.json              # Metadados de toda coleta
├── collection_log.txt         # Log de execução
├── station_001_raw.json       # Dados brutos do local
└── ...
```
### Metadados (`metadata.json`)

```json
{
  "api_source": "Open-Meteo Archive API",
  "api_url": "https://archive-api.open-meteo.com/v1/archive",
  "variables": [...],
  "locations": [...],
  "collection_timestamp": "2025-01-15T14:30:00.123456",
  "coordinate_system": "WGS84 (EPSG:4326)",
  "timezone": "America/Sao_Paulo"
}
```

## Como Usar

### 1. Configurar Local

Edite `src/prepocessing/config_stations.json` com as coordenadas do local desejado:

```json
{
  "id": "location_XXX",
  "name": "Nome do Local",
  "latitude": -19.9XXX,
  "longitude": -43.9XXX
}
```

### 2. Instalar Dependências

```powershell
pip install -r requiriments.txt
```

### 3. Executar Coleta

```powershell
python src/prepocessing/fetch_weather_data.py
```

Isso coletará dados de previsão do dia escolhido e os salvará em `data/raw/`.

## Qual o propósito de utilizar um banco de dados sintético?

Utilizamos um banco de dados sintético durante a fase de treinamento e desenvolvimento pelos seguintes motivos:

### 1. **Estabilidade para Treinamento do Modelo**
   - O banco sintético fornece dados **consistentes e previsíveis**, permitindo que o modelo seja treinado sem variações inesperadas
   - Evita que mudanças futuras nos dados reais da API afetem modelos já treinados

### 2. **Controle de Qualidade**
   - Podemos gerar dados com características específicas e conhecidas
   - Facilita testes e validação
   - Permite reproduzibilidade total dos experimentos

### 3. **Independência de Fontes Externas**
   - Não dependemos da disponibilidade, limite de taxa ou mudanças na API Open-Meteo durante desenvolvimento
   - Evita bloqueios relacionados a requisições externas

### 4. **Transição para Dados Reais**
   - Após a conclusão do treinamento e validação do modelo, migraremos para a **API Open-Meteo real**
   - Isso garante que o modelo esteja otimizado e pronto para trabalhar com dados reais do mundo

## Referências

- [Open-Meteo API Documentation](https://open-meteo.com/en/docs)