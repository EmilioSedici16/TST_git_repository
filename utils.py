"""
Утилиты для проекта компьютерного зрения с Roboflow
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
from typing import Optional, Tuple, List, Dict, Any
import requests
from PIL import Image
import io


class RoboflowManager:
    """Класс для управления интеграцией с Roboflow"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация RoboflowManager
        
        Args:
            api_key: API ключ Roboflow. Если не указан, берется из переменной окружения
        """
        self.api_key = api_key or os.getenv('ROBOFLOW_API_KEY')
        if not self.api_key:
            print("⚠️ Предупреждение: API ключ Roboflow не найден. Установите переменную ROBOFLOW_API_KEY")
        else:
            self.rf = Roboflow(api_key=self.api_key)
    
    def get_project(self, workspace: str, project: str):
        """Получить проект из Roboflow"""
        try:
            return self.rf.workspace(workspace).project(project)
        except Exception as e:
            print(f"❌ Ошибка получения проекта: {e}")
            return None
    
    def download_dataset(self, workspace: str, project: str, version: int, format: str = "yolov8"):
        """Скачать датасет из Roboflow"""
        try:
            project = self.get_project(workspace, project)
            if project:
                dataset = project.version(version).download(format)
                print(f"✅ Датасет скачан в: {dataset.location}")
                return dataset
        except Exception as e:
            print(f"❌ Ошибка скачивания датасета: {e}")
            return None
    
    def get_model(self, workspace: str, project: str, version: int):
        """Получить модель из Roboflow"""
        try:
            project = self.get_project(workspace, project)
            if project:
                return project.version(version).model
        except Exception as e:
            print(f"❌ Ошибка получения модели: {e}")
            return None


class ImageProcessor:
    """Класс для обработки изображений"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Загрузить изображение"""
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(io.BytesIO(response.content))
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                return cv2.imread(image_path)
        except Exception as e:
            print(f"❌ Ошибка загрузки изображения: {e}")
            return None
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Изменить размер изображения"""
        return cv2.resize(image, target_size)
    
    @staticmethod
    def draw_bboxes(image: np.ndarray, detections: List[Dict], class_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Нарисовать ограничивающие рамки на изображении
        
        Args:
            image: Исходное изображение
            detections: Список детекций в формате [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class': int}]
            class_names: Список названий классов
        """
        img_with_boxes = image.copy()
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            confidence = detection.get('confidence', 0.0)
            class_id = detection.get('class', 0)
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Цвет рамки
                color = (0, 255, 0)  # Зеленый
                
                # Рисуем рамку
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # Подпись
                label = f"Class {class_id}"
                if class_names and class_id < len(class_names):
                    label = class_names[class_id]
                label += f" {confidence:.2f}"
                
                # Рисуем фон для текста
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_with_boxes, (x1, y1 - 20), (x1 + w, y1), color, -1)
                
                # Рисуем текст
                cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return img_with_boxes
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """Сохранить изображение"""
        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения изображения: {e}")
            return False


class Visualizer:
    """Класс для визуализации результатов"""
    
    @staticmethod
    def plot_image_with_detections(image: np.ndarray, detections: List[Dict], 
                                 title: str = "Детекция объектов", figsize: Tuple[int, int] = (12, 8)):
        """Отобразить изображение с детекциями"""
        plt.figure(figsize=figsize)
        
        # Конвертируем BGR в RGB для matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title(title)
        plt.axis('off')
        
        # Рисуем рамки
        for detection in detections:
            bbox = detection.get('bbox', [])
            confidence = detection.get('confidence', 0.0)
            class_id = detection.get('class', 0)
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                rect = plt.Rectangle((x1, y1), width, height, 
                                   fill=False, color='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                # Добавляем подпись
                plt.text(x1, y1 - 10, f"Class {class_id}: {confidence:.2f}", 
                        color='red', fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_detection_stats(detections: List[Dict], class_names: Optional[List[str]] = None):
        """Построить статистику детекций"""
        if not detections:
            print("Нет детекций для отображения статистики")
            return
        
        # Подсчет классов
        class_counts = {}
        confidences = []
        
        for detection in detections:
            class_id = detection.get('class', 0)
            confidence = detection.get('confidence', 0.0)
            
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"Class {class_id}"
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(confidence)
        
        # Создаем графики
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График классов
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        ax1.bar(classes, counts)
        ax1.set_title('Количество детекций по классам')
        ax1.set_xlabel('Классы')
        ax1.set_ylabel('Количество')
        ax1.tick_params(axis='x', rotation=45)
        
        # Гистограмма уверенности
        ax2.hist(confidences, bins=20, alpha=0.7, color='blue')
        ax2.set_title('Распределение уверенности детекций')
        ax2.set_xlabel('Уверенность')
        ax2.set_ylabel('Частота')
        
        plt.tight_layout()
        plt.show()


def setup_roboflow_env():
    """Настройка окружения для работы с Roboflow"""
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        print("🔑 Для работы с Roboflow установите API ключ:")
        print("export ROBOFLOW_API_KEY='ваш_api_ключ'")
        print("\nИли создайте файл .env с содержимым:")
        print("ROBOFLOW_API_KEY=ваш_api_ключ")
        return False
    else:
        print("✅ API ключ Roboflow найден")
        return True


def create_sample_detection():
    """Создать пример детекции для тестирования"""
    return [
        {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.95,
            'class': 0
        },
        {
            'bbox': [300, 150, 400, 250],
            'confidence': 0.87,
            'class': 1
        }
    ]


if __name__ == "__main__":
    # Пример использования
    print("🔧 Утилиты компьютерного зрения с Roboflow")
    setup_roboflow_env()