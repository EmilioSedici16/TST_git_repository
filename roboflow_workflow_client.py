#!/usr/bin/env python3
"""
HTTP клиент для работы с Roboflow Workflows
Работает без inference-sdk, используя прямые HTTP запросы
"""

import os
import requests
from typing import Optional, Dict, Any, Union
from pathlib import Path
import base64


class RoboflowWorkflowClient:
    """
    HTTP клиент для работы с Roboflow Workflows
    Альтернатива InferenceHTTPClient, работающая с Python 3.13+
    """
    
    def __init__(self, api_url: str = "https://serverless.roboflow.com", 
                 api_key: Optional[str] = None):
        """
        Инициализация клиента
        
        Args:
            api_url: URL API Roboflow (по умолчанию serverless)
            api_key: API ключ Roboflow. Если не указан, берется из переменной окружения
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key or os.getenv('ROBOFLOW_API_KEY')
        
        if not self.api_key:
            raise ValueError("API ключ Roboflow не найден. Установите ROBOFLOW_API_KEY или передайте api_key")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Кодирует изображение в base64
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Base64 строка изображения
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def run_workflow(self, workspace_name: str, workflow_id: str, 
                    images: Dict[str, Union[str, Path]], 
                    use_cache: bool = True) -> Dict[str, Any]:
        """
        Запускает workflow в Roboflow
        
        Args:
            workspace_name: Название workspace
            workflow_id: ID workflow
            images: Словарь с изображениями {имя_входа: путь_к_изображению}
            use_cache: Использовать кэш определения workflow (15 минут)
            
        Returns:
            Результат выполнения workflow
        """
        # Подготавливаем изображения
        encoded_images = {}
        for key, image_path in images.items():
            if isinstance(image_path, (str, Path)):
                encoded_images[key] = self._encode_image(image_path)
            else:
                # Если уже base64 строка
                encoded_images[key] = image_path
        
        # Формируем запрос
        payload = {
            "images": encoded_images,
            "use_cache": use_cache
        }
        
        # URL для workflow
        workflow_url = f"{self.api_url}/workflows/{workspace_name}/{workflow_id}"
        
        try:
            response = requests.post(
                workflow_url,
                json=payload,
                headers=self.headers,
                timeout=60  # 60 секунд таймаут
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка выполнения workflow: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Ответ сервера: {e.response.text}")
            raise
    
    def run_workflow_from_url(self, workspace_name: str, workflow_id: str,
                             image_urls: Dict[str, str],
                             use_cache: bool = True) -> Dict[str, Any]:
        """
        Запускает workflow с изображениями по URL
        
        Args:
            workspace_name: Название workspace
            workflow_id: ID workflow
            image_urls: Словарь с URL изображений {имя_входа: url}
            use_cache: Использовать кэш
            
        Returns:
            Результат выполнения workflow
        """
        # Загружаем изображения по URL и кодируем
        encoded_images = {}
        for key, url in image_urls.items():
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                image_data = base64.b64encode(response.content).decode('utf-8')
                encoded_images[key] = image_data
            except Exception as e:
                print(f"❌ Ошибка загрузки изображения {url}: {e}")
                raise
        
        payload = {
            "images": encoded_images,
            "use_cache": use_cache
        }
        
        workflow_url = f"{self.api_url}/workflows/{workspace_name}/{workflow_id}"
        
        try:
            response = requests.post(
                workflow_url,
                json=payload,
                headers=self.headers,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка выполнения workflow: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Ответ сервера: {e.response.text}")
            raise


# Пример использования
if __name__ == "__main__":
    # Инициализация клиента
    client = RoboflowWorkflowClient(
        api_url="https://serverless.roboflow.com",
        api_key="WJqw53R6TlCZVbqB3fdU"  # Замените на ваш ключ или используйте переменную окружения
    )
    
    # Пример запуска workflow
    try:
        result = client.run_workflow(
            workspace_name="tst-workspace",
            workflow_id="find-people-helmets-and-lifts",
            images={
                "image": "data/test_image.jpg"  # Путь к вашему изображению
            },
            use_cache=True
        )
        
        print("✅ Workflow выполнен успешно!")
        print(f"Результат: {result}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
