#!/usr/bin/env python3
"""
Специализированный скрипт для детекции безопасности на рабочих площадках
Детектирует: людей, каски, подъемные платформы
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utils import ImageProcessor, RoboflowManager, convert_ultralytics_to_supervision, SUPERVISION_AVAILABLE
import json

# Импорт supervision если доступен
if SUPERVISION_AVAILABLE:
    try:
        import supervision as sv
    except ImportError:
        sv = None
else:
    sv = None

class SafetyDetector:
    """Класс для детекции объектов безопасности"""
    
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        Инициализация детектора безопасности
        
        Args:
            model_path: Путь к модели или Roboflow модель
            conf_threshold: Порог уверенности
        """
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        
        # Классы для детекции безопасности
        self.safety_classes = {
            'person': 0,      # Человек (стандартный COCO класс)
            'helmet': None,   # Каска (нужно обучить или найти кастомную модель)
            'lift_platform': None  # Подъемная платформа (кастомный класс)
        }
        
        # Статистика безопасности
        self.safety_stats = {
            'total_people': 0,
            'people_with_helmets': 0,
            'people_without_helmets': 0,
            'lift_platforms': 0,
            'safety_score': 0.0
        }
    
    def load_roboflow_model(self, workspace, project, version=1):
        """Загрузить кастомную модель из Roboflow"""
        try:
            rf_manager = RoboflowManager()
            if rf_manager.api_key:
                model = rf_manager.get_model(workspace, project, version)
                if model:
                    self.model = model
                    print(f"✅ Загружена модель Roboflow: {project}")
                    return True
            return False
        except Exception as e:
            print(f"❌ Ошибка загрузки Roboflow модели: {e}")
            return False
    
    def load_rf_detr_model(self, workspace, project, version=1):
        """
        Загрузить RF-DETR модель (трансформерная детекция)
        RF-DETR лучше работает с мелкими объектами (каски)
        """
        try:
            rf_manager = RoboflowManager()
            if rf_manager.api_key:
                model = rf_manager.get_rf_detr_model(workspace, project, version)
                if model:
                    self.model = model
                    print(f"✅ Загружена RF-DETR модель: {project}")
                    return True
            return False
        except Exception as e:
            print(f"❌ Ошибка загрузки RF-DETR модели: {e}")
            return False
    
    def detect_objects(self, image):
        """Детекция объектов на изображении"""
        results = self.model(image, conf=self.conf_threshold)
        
        detections = []
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confs, classes):
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': self.get_class_name(cls)
                    })
        
        return detections
    
    def get_class_name(self, class_id):
        """Получить название класса по ID"""
        if hasattr(self.model, 'names'):
            return self.model.names.get(class_id, f"Class_{class_id}")
        return f"Class_{class_id}"
    
    def analyze_safety(self, detections):
        """Анализ безопасности на основе детекций"""
        people = []
        helmets = []
        platforms = []
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            
            if 'person' in class_name:
                people.append(detection)
            elif 'helmet' in class_name or 'hard_hat' in class_name:
                helmets.append(detection)
            elif 'lift' in class_name or 'platform' in class_name or 'crane' in class_name:
                platforms.append(detection)
        
        # Обновляем статистику
        self.safety_stats['total_people'] = len(people)
        self.safety_stats['lift_platforms'] = len(platforms)
        
        # Простая эвристика для определения людей в касках
        # В реальном проекте нужна более сложная логика
        people_with_helmets = self.match_people_with_helmets(people, helmets)
        
        self.safety_stats['people_with_helmets'] = people_with_helmets
        self.safety_stats['people_without_helmets'] = len(people) - people_with_helmets
        
        # Расчет индекса безопасности
        if len(people) > 0:
            self.safety_stats['safety_score'] = people_with_helmets / len(people)
        else:
            self.safety_stats['safety_score'] = 1.0
        
        return self.safety_stats
    
    def match_people_with_helmets(self, people, helmets):
        """Сопоставление людей с касками (упрощенная версия)"""
        if not helmets:
            return 0
        
        matched = 0
        for person in people:
            person_bbox = person['bbox']
            person_center_x = (person_bbox[0] + person_bbox[2]) / 2
            person_top_y = person_bbox[1]
            
            # Ищем каски в верхней части области человека
            for helmet in helmets:
                helmet_bbox = helmet['bbox']
                helmet_center_x = (helmet_bbox[0] + helmet_bbox[2]) / 2
                helmet_center_y = (helmet_bbox[1] + helmet_bbox[3]) / 2
                
                # Простая проверка: каска рядом с головой человека
                x_distance = abs(person_center_x - helmet_center_x)
                y_distance = abs(person_top_y - helmet_center_y)
                
                if x_distance < 50 and y_distance < 100:  # Пиксели
                    matched += 1
                    break
        
        return matched
    
    def draw_safety_results(self, image, detections, stats, use_supervision=True):
        """
        Отрисовка результатов с акцентом на безопасность
        
        Args:
            image: Исходное изображение
            detections: Список детекций
            stats: Статистика безопасности
            use_supervision: Использовать supervision для визуализации (если доступен)
        """
        # Используем supervision если доступен
        if use_supervision and SUPERVISION_AVAILABLE and detections:
            return self._draw_safety_results_supervision(image, detections, stats)
        else:
            return self._draw_safety_results_opencv(image, detections, stats)
    
    def _draw_safety_results_supervision(self, image, detections, stats):
        """Визуализация с использованием supervision"""
        boxes = []
        confidences = []
        class_ids = []
        labels = []
        colors = []
        
        # Цветовая схема для разных типов объектов
        if sv is not None:
            color_map = {
                'person': sv.Color.RED,
                'helmet': sv.Color.GREEN,
                'lift': sv.Color.BLUE,
                'platform': sv.Color.BLUE,
                'crane': sv.Color.BLUE
            }
        else:
            color_map = {}
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) == 4:
                boxes.append(bbox)
                confidences.append(detection.get('confidence', 0.0))
                class_ids.append(detection.get('class', 0))
                
                class_name = detection.get('class_name', 'unknown')
                label = f"{class_name}: {detection.get('confidence', 0.0):.2f}"
                labels.append(label)
                
                # Определяем цвет
                if sv is not None:
                    color = sv.Color.WHITE
                    for key, sv_color in color_map.items():
                        if key in class_name.lower():
                            color = sv_color
                            break
                    colors.append(color)
                else:
                    colors.append((255, 255, 255))  # Белый по умолчанию
        
        if not boxes:
            result_image = image.copy()
        else:
            # Создаем Detections объект
            detections_sv = sv.Detections(
                xyxy=np.array(boxes),
                confidence=np.array(confidences),
                class_id=np.array(class_ids)
            )
            
            # Создаем аннотаторы
            if sv is not None:
                box_annotator = sv.BoxAnnotator()
                label_annotator = sv.LabelAnnotator()
            else:
                # Fallback если supervision недоступен
                return self._draw_safety_results_opencv(image, detections, stats)
            
            # Аннотируем
            result_image = box_annotator.annotate(
                scene=image.copy(),
                detections=detections_sv
            )
            
            result_image = label_annotator.annotate(
                scene=result_image,
                detections=detections_sv,
                labels=labels
            )
        
        # Добавляем панель статистики
        self.draw_safety_panel(result_image, stats)
        
        return result_image
    
    def _draw_safety_results_opencv(self, image, detections, stats):
        """Визуализация с использованием OpenCV (fallback)"""
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Цветовая схема для разных типов объектов
            if 'person' in class_name.lower():
                color = (0, 0, 255)  # Красный для людей
            elif 'helmet' in class_name.lower():
                color = (0, 255, 0)  # Зеленый для касок
            elif any(keyword in class_name.lower() for keyword in ['lift', 'platform', 'crane']):
                color = (255, 0, 0)  # Синий для техники
            else:
                color = (128, 128, 128)  # Серый для остального
            
            # Рисуем рамку
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Подпись
            label = f"{class_name}: {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Добавляем панель статистики
        self.draw_safety_panel(result_image, stats)
        
        return result_image
    
    def draw_safety_panel(self, image, stats):
        """Отрисовка панели со статистикой безопасности"""
        h, w = image.shape[:2]
        panel_height = 150
        panel_color = (50, 50, 50)
        
        # Создаем панель
        cv2.rectangle(image, (10, 10), (400, panel_height), panel_color, -1)
        cv2.rectangle(image, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        # Заголовок
        cv2.putText(image, "SAFETY ANALYSIS", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Статистика
        y_offset = 60
        line_height = 20
        
        texts = [
            f"People detected: {stats['total_people']}",
            f"With helmets: {stats['people_with_helmets']}",
            f"Without helmets: {stats['people_without_helmets']}",
            f"Lift platforms: {stats['lift_platforms']}",
            f"Safety score: {stats['safety_score']:.1%}"
        ]
        
        for i, text in enumerate(texts):
            color = (0, 255, 0) if 'safety score' in text.lower() and stats['safety_score'] > 0.8 else (255, 255, 255)
            if 'without helmets' in text.lower() and stats['people_without_helmets'] > 0:
                color = (0, 0, 255)  # Красный для людей без касок
            
            cv2.putText(image, text, (20, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Детекция безопасности на рабочих площадках')
    parser.add_argument('--source', type=str, required=True,
                       help='Путь к изображению или видео')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Модель для детекции')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Порог уверенности')
    parser.add_argument('--save', action='store_true',
                       help='Сохранить результаты')
    parser.add_argument('--output', type=str, default='runs/safety',
                       help='Папка для сохранения')
    parser.add_argument('--roboflow-workspace', type=str,
                       help='Workspace в Roboflow')
    parser.add_argument('--roboflow-project', type=str,
                       help='Проект в Roboflow')
    
    args = parser.parse_args()
    
    print("🦺 Система анализа безопасности на рабочих площадках")
    print("-" * 60)
    
    # Создаем детектор
    detector = SafetyDetector(args.model, args.conf)
    
    # Загружаем Roboflow модель если указана
    if args.roboflow_workspace and args.roboflow_project:
        detector.load_roboflow_model(args.roboflow_workspace, args.roboflow_project)
    
    # Загружаем изображение
    image = ImageProcessor.load_image(args.source)
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {args.source}")
        return
    
    print(f"📷 Анализ изображения: {args.source}")
    
    # Выполняем детекцию
    detections = detector.detect_objects(image)
    print(f"🔍 Найдено объектов: {len(detections)}")
    
    # Анализируем безопасность
    safety_stats = detector.analyze_safety(detections)
    
    # Выводим результаты
    print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА БЕЗОПАСНОСТИ:")
    print(f"👥 Обнаружено людей: {safety_stats['total_people']}")
    print(f"🦺 Людей в касках: {safety_stats['people_with_helmets']}")
    print(f"⚠️  Людей без касок: {safety_stats['people_without_helmets']}")
    print(f"🏗️  Подъемных платформ: {safety_stats['lift_platforms']}")
    print(f"📈 Индекс безопасности: {safety_stats['safety_score']:.1%}")
    
    if safety_stats['people_without_helmets'] > 0:
        print("\n🚨 ПРЕДУПРЕЖДЕНИЕ: Обнаружены люди без защитных касок!")
    else:
        print("\n✅ Все люди используют защитное снаряжение")
    
    # Отрисовываем результаты
    result_image = detector.draw_safety_results(image, detections, safety_stats)
    
    # Показываем результат
    cv2.imshow('Safety Detection Results', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Сохраняем результаты
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем изображение
        output_path = output_dir / f"safety_result_{Path(args.source).name}"
        cv2.imwrite(str(output_path), result_image)
        
        # Сохраняем JSON с результатами
        json_path = output_dir / f"safety_stats_{Path(args.source).stem}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'source': args.source,
                'detections': detections,
                'safety_stats': safety_stats
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результаты сохранены:")
        print(f"   Изображение: {output_path}")
        print(f"   Данные: {json_path}")

if __name__ == "__main__":
    main()