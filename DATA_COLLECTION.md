# Documentação de Coleta de Dados - Mobike

## Visão Geral

Este documento descreve a infraestrutura configurada para coleta de dados meteorológicos via API Open-Meteo. 

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

## Referências

- [Open-Meteo API Documentation](https://open-meteo.com/en/docs)