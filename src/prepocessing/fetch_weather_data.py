import requests
import json
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/raw/collection_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OpenMeteoFetcher:
    """Classe para coletar dados da API Open-Meteo."""
    
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "weather_code",
        "wind_speed_10m",
        "precipitation"
    ]
    
    def __init__(self, config_path="src/prepocessing/config_stations.json"):
        self.config_path = config_path
        self.locations = self._load_config()
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config['locations']
    
    def _validate_coordinates(self, latitude, longitude):
        if not (-90 <= latitude <= 90):
            logger.warning(f"Latitude inválida: {latitude}")
            return False
        if not (-180 <= longitude <= 180):
            logger.warning(f"Longitude inválida: {longitude}")
            return False
        return True
    
    def fetch_station_data(self, location):
        # Validar coordenadas
        if not self._validate_coordinates(location['latitude'], location['longitude']):
            logger.error(f"Coordenadas inválidas para {location['name']}")
            return None
        
        # Usa API Forecast para previsões futuras
        params = {
            "latitude": location['latitude'],
            "longitude": location['longitude'],
            "hourly": ",".join(self.VARIABLES),
            "timezone": "America/Sao_Paulo"
        }
        
        try:
            logger.info(f"Coletando dados de {location['name']}")
            response = requests.get(self.FORECAST_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Filtrar para apenas 24 horas (um dia)
            if 'hourly' in data and 'time' in data['hourly']:
                hourly_data = data['hourly']
                # Manter apenas as primeiras 24 horas
                for key in hourly_data:
                    if isinstance(hourly_data[key], list):
                        hourly_data[key] = hourly_data[key][:24]
            
            data['location_id'] = location['id']
            data['location_name'] = location['name']
            data['collection_date'] = datetime.now().isoformat() # Data da coleta
            data['forecast_hours'] = 24
            
            logger.info(f"Sucesso: {location['name']}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao coletar dados de {location['name']}: {str(e)}")
            return None
    
    def save_raw_data(self, location_data, location_id):
        if location_data is None:
            return
        
        filename = self.raw_data_dir / f"{location_id}_raw.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(location_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Dados salvos em {filename}")
        except Exception as e:
            logger.error(f"Erro ao salvar {filename}: {str(e)}")
    
    def save_metadata(self):
        metadata = {
            "api_source": "Open-Meteo Forecast API",
            "api_url": self.FORECAST_URL,
            "api_type": "Previsão",
            "variables": self.VARIABLES,
            "locations": self.locations,
            "collection_timestamp": datetime.now().isoformat(),
            "coordinate_system": "WGS84 (EPSG:4326)",
            "timezone": "America/Sao_Paulo",
        }
        
        metadata_file = self.raw_data_dir / "metadata.json"
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadados salvos em {metadata_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar metadados: {str(e)}")
    
    def collect_all(self):
        logger.info("Iniciando coleta de previsão meteorológica")
        
        if self.locations:
            location = self.locations[0]
            data = self.fetch_station_data(location)
            if data:
                self.save_raw_data(data, location['id'])
        
        self.save_metadata()
        logger.info("Coleta concluída")


if __name__ == "__main__":
    # Exemplo de uso
    fetcher = OpenMeteoFetcher()
    
    # Coleta dados de previsão meteorológica
    fetcher.collect_all()
