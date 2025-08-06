#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный pipeline компьютерного зрения: извлечение кадров → Roboflow → детекция
Автор: TST Project
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Optional, List

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TSTPipeline:
    """Класс для управления полным pipeline компьютерного зрения"""
    
    def __init__(self, config: dict = None):
        self.config = config or self.get_default_config()
        self.setup_directories()
    
    def get_default_config(self) -> dict:
        """Получить конфигурацию по умолчанию"""
        return {
            'videos_dir': 'Videos',
            'images_dir': 'Images', 
            'data_dir': 'data',
            'models_dir': 'models',
            'num_frames': 100,
            'roboflow_api_key': os.environ.get('ROBOFLOW_API_KEY'),
            'confidence_threshold': 0.5,
            'image_format': 'jpg',
            'batch_size': 32
        }
    
    def setup_directories(self):
        """Создать необходимые директории"""
        dirs = [
            self.config['videos_dir'],
            self.config['images_dir'],
            self.config['data_dir'],
            self.config['models_dir'],
            f"{self.config['data_dir']}/raw",
            f"{self.config['data_dir']}/processed",
            f"{self.config['data_dir']}/results"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Создана/проверена директория: {dir_path}")
    
    def check_dependencies(self) -> bool:
        """Проверить установленные зависимости"""
        required_packages = [
            'cv2', 'roboflow', 'ultralytics', 
            'streamlit', 'PIL', 'numpy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} установлен")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} не найден")
        
        if missing_packages:
            logger.error(f"Установите недостающие пакеты: {missing_packages}")
            logger.info("Запустите: pip install -r requirements.txt")
            return False
        
        return True
    
    def extract_frames(self, video_files: List[str] = None) -> bool:
        """
        Этап 1: Извлечение кадров из видео
        
        Args:
            video_files: Список видеофайлов. Если None, обрабатывает все видео
        
        Returns:
            bool: Успешность выполнения
        """
        logger.info("🎬 ЭТАП 1: Извлечение кадров из видео")
        
        if not video_files:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = [
                f for f in os.listdir(self.config['videos_dir'])
                if any(f.lower().endswith(ext) for ext in video_extensions)
            ]
        
        if not video_files:
            logger.warning(f"Видеофайлы не найдены в {self.config['videos_dir']}")
            return False
        
        try:
            # Импорт модуля извлечения кадров
            import cv2
            
            total_extracted = 0
            for video_file in video_files:
                video_path = os.path.join(self.config['videos_dir'], video_file)
                
                if not os.path.exists(video_path):
                    logger.warning(f"Файл не найден: {video_path}")
                    continue
                
                logger.info(f"📹 Обработка: {video_file}")
                extracted = self._extract_frames_from_video(video_path)
                total_extracted += extracted
                logger.info(f"✅ Извлечено {extracted} кадров из {video_file}")
            
            logger.info(f"🎉 Всего извлечено кадров: {total_extracted}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при извлечении кадров: {e}")
            return False
    
    def _extract_frames_from_video(self, video_path: str) -> int:
        """Извлечь кадры из одного видеофайла"""
        import cv2
        
        # Получить имя файла без расширения
        video_name = Path(video_path).stem
        
        # Открыть видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {video_path}")
            return 0
        
        # Получить информацию о видео
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Видео: {total_frames} кадров, {fps:.2f} FPS")
        
        # Вычислить интервал для извлечения
        if total_frames < self.config['num_frames']:
            step = 1
            frames_to_extract = total_frames
        else:
            step = total_frames // self.config['num_frames']
            frames_to_extract = self.config['num_frames']
        
        extracted_count = 0
        for i in range(0, total_frames, step):
            if extracted_count >= frames_to_extract:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                # Сохранить кадр
                output_filename = f"{video_name}_frame_{i:06d}.{self.config['image_format']}"
                output_path = os.path.join(self.config['images_dir'], output_filename)
                
                cv2.imwrite(output_path, frame)
                extracted_count += 1
        
        cap.release()
        return extracted_count
    
    def prepare_for_roboflow(self) -> bool:
        """
        Этап 2: Подготовка данных для Roboflow
        
        Returns:
            bool: Успешность выполнения
        """
        logger.info("📤 ЭТАП 2: Подготовка данных для Roboflow")
        
        try:
            # Подсчет извлеченных кадров
            image_files = [
                f for f in os.listdir(self.config['images_dir'])
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if not image_files:
                logger.warning("Извлеченные кадры не найдены. Запустите извлечение кадров.")
                return False
            
            logger.info(f"📊 Найдено {len(image_files)} кадров для загрузки в Roboflow")
            
            # Создание файла со списком изображений
            image_list_path = os.path.join(self.config['data_dir'], 'image_list.txt')
            with open(image_list_path, 'w') as f:
                for img_file in image_files:
                    f.write(f"{os.path.join(self.config['images_dir'], img_file)}\\n")
            
            logger.info(f"📄 Создан список изображений: {image_list_path}")
            
            # Инструкции для Roboflow
            instructions = f"""
            📋 ИНСТРУКЦИИ ДЛЯ ROBOFLOW:
            
            1. Перейдите на https://roboflow.com
            2. Создайте новый проект или откройте существующий
            3. Загрузите изображения из папки: {self.config['images_dir']}
            4. Выполните разметку данных (аннотирование)
            5. Настройте аугментацию данных
            6. Обучите модель
            7. Экспортируйте обученную модель
            
            📊 Статистика:
            - Кадров для загрузки: {len(image_files)}
            - Формат изображений: {self.config['image_format']}
            - Рекомендуемый размер датасета: 100+ изображений на класс
            """
            
            print(instructions)
            logger.info("📋 Инструкции для Roboflow выведены")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при подготовке для Roboflow: {e}")
            return False
    
    def upload_to_roboflow(self, project_name: str, batch_name: str = None) -> bool:
        """
        Автоматическая загрузка в Roboflow (если настроен API)
        
        Args:
            project_name: Название проекта в Roboflow
            batch_name: Название батча для загрузки
        
        Returns:
            bool: Успешность выполнения
        """
        logger.info("🚀 ЭТАП 2б: Автоматическая загрузка в Roboflow")
        
        if not self.config['roboflow_api_key']:
            logger.warning("API ключ Roboflow не настроен. Используйте ручную загрузку.")
            return self.prepare_for_roboflow()
        
        try:
            from roboflow import Roboflow
            
            # Подключение к Roboflow
            rf = Roboflow(api_key=self.config['roboflow_api_key'])
            project = rf.workspace().project(project_name)
            
            # Загрузка изображений
            batch_name = batch_name or f"extracted_frames_{len(os.listdir(self.config['images_dir']))}"
            
            logger.info(f"📤 Загрузка в проект: {project_name}, батч: {batch_name}")
            
            project.upload(
                self.config['images_dir'],
                batch_name=batch_name,
                num_workers=4
            )
            
            logger.info("✅ Изображения успешно загружены в Roboflow")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке в Roboflow: {e}")
            logger.info("💡 Попробуйте ручную загрузку через веб-интерфейс")
            return self.prepare_for_roboflow()
    
    def run_detection(self, model_path: str = None, source: str = None) -> bool:
        """
        Этап 3: Запуск детекции
        
        Args:
            model_path: Путь к модели (YOLO или Roboflow)
            source: Источник для детекции (изображение, видео, папка)
        
        Returns:
            bool: Успешность выполнения
        """
        logger.info("🔍 ЭТАП 3: Запуск детекции объектов")
        
        try:
            # Запуск детекции через существующий скрипт
            detect_script = "detect.py"
            if not os.path.exists(detect_script):
                logger.error(f"Скрипт детекции не найден: {detect_script}")
                return False
            
            # Построение команды
            cmd = [sys.executable, detect_script]
            
            if model_path:
                cmd.extend(["--weights", model_path])
            
            if source:
                cmd.extend(["--source", source])
            else:
                # Использовать папку с изображениями по умолчанию
                cmd.extend(["--source", self.config['images_dir']])
            
            # Добавить папку для результатов
            results_dir = os.path.join(self.config['data_dir'], 'results')
            cmd.extend(["--project", results_dir])
            
            logger.info(f"🚀 Запуск команды: {' '.join(cmd)}")
            
            # Выполнение детекции
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Детекция завершена успешно")
                logger.info(f"📂 Результаты сохранены в: {results_dir}")
                return True
            else:
                logger.error(f"❌ Ошибка при детекции: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка при запуске детекции: {e}")
            return False
    
    def run_web_interface(self):
        """Запустить веб-интерфейс Streamlit"""
        logger.info("🌐 Запуск веб-интерфейса")
        
        app_script = "app.py"
        if not os.path.exists(app_script):
            logger.error(f"Веб-приложение не найдено: {app_script}")
            return False
        
        try:
            cmd = [sys.executable, "-m", "streamlit", "run", app_script]
            logger.info("🚀 Запуск Streamlit приложения...")
            subprocess.run(cmd)
            
        except KeyboardInterrupt:
            logger.info("⏹️ Веб-интерфейс остановлен пользователем")
        except Exception as e:
            logger.error(f"❌ Ошибка при запуске веб-интерфейса: {e}")
    
    def run_full_pipeline(
        self, 
        extract_frames: bool = True,
        prepare_roboflow: bool = True,
        roboflow_project: str = None,
        run_detection: bool = False,
        detection_model: str = None
    ) -> bool:
        """
        Запустить полный pipeline
        
        Args:
            extract_frames: Выполнить извлечение кадров
            prepare_roboflow: Подготовить данные для Roboflow
            roboflow_project: Название проекта Roboflow для автозагрузки
            run_detection: Запустить детекцию
            detection_model: Модель для детекции
        
        Returns:
            bool: Успешность выполнения всех этапов
        """
        logger.info("🎯 ЗАПУСК ПОЛНОГО PIPELINE")
        logger.info("=" * 50)
        
        success = True
        
        # Проверка зависимостей
        if not self.check_dependencies():
            return False
        
        # Этап 1: Извлечение кадров
        if extract_frames:
            if not self.extract_frames():
                logger.error("❌ Этап 1 провален: извлечение кадров")
                success = False
        
        # Этап 2: Подготовка для Roboflow
        if prepare_roboflow and success:
            if roboflow_project:
                if not self.upload_to_roboflow(roboflow_project):
                    logger.error("❌ Этап 2 провален: загрузка в Roboflow")
                    success = False
            else:
                if not self.prepare_for_roboflow():
                    logger.error("❌ Этап 2 провален: подготовка для Roboflow")
                    success = False
        
        # Этап 3: Детекция (опционально)
        if run_detection and success:
            if not self.run_detection(detection_model):
                logger.error("❌ Этап 3 провален: детекция")
                success = False
        
        if success:
            logger.info("🎉 PIPELINE ЗАВЕРШЕН УСПЕШНО!")
            logger.info("📊 Сводка:")
            logger.info(f"   📁 Кадры: {self.config['images_dir']}")
            logger.info(f"   📊 Данные: {self.config['data_dir']}")
            if run_detection:
                logger.info(f"   🎯 Результаты: {self.config['data_dir']}/results")
        else:
            logger.error("❌ PIPELINE ЗАВЕРШЕН С ОШИБКАМИ")
        
        return success


def main():
    """Главная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description="TST Pipeline: Полный pipeline компьютерного зрения",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Полный pipeline (извлечение + подготовка для Roboflow)
  python ПОЛНЫЙ_PIPELINE.py --full

  # Только извлечение кадров
  python ПОЛНЫЙ_PIPELINE.py --extract-frames

  # Подготовка для Roboflow
  python ПОЛНЫЙ_PIPELINE.py --prepare-roboflow

  # Автозагрузка в Roboflow
  python ПОЛНЫЙ_PIPELINE.py --upload-roboflow --project "my-project"

  # Детекция с моделью
  python ПОЛНЫЙ_PIPELINE.py --detect --model "yolov8n.pt"

  # Веб-интерфейс
  python ПОЛНЫЙ_PIPELINE.py --web

  # Настройка количества кадров
  python ПОЛНЫЙ_PIPELINE.py --extract-frames --num-frames 200
        """
    )
    
    parser.add_argument('--full', action='store_true', 
                       help='Запустить полный pipeline')
    parser.add_argument('--extract-frames', action='store_true',
                       help='Извлечь кадры из видео')
    parser.add_argument('--prepare-roboflow', action='store_true',
                       help='Подготовить данные для Roboflow')
    parser.add_argument('--upload-roboflow', action='store_true',
                       help='Автоматически загрузить в Roboflow')
    parser.add_argument('--detect', action='store_true',
                       help='Запустить детекцию')
    parser.add_argument('--web', action='store_true',
                       help='Запустить веб-интерфейс')
    
    parser.add_argument('--project', type=str,
                       help='Название проекта Roboflow')
    parser.add_argument('--model', type=str,
                       help='Путь к модели для детекции')
    parser.add_argument('--source', type=str,
                       help='Источник для детекции (изображение/видео/папка)')
    parser.add_argument('--num-frames', type=int, default=100,
                       help='Количество кадров для извлечения (по умолчанию: 100)')
    parser.add_argument('--videos-dir', type=str, default='Videos',
                       help='Папка с видео (по умолчанию: Videos)')
    parser.add_argument('--images-dir', type=str, default='Images',
                       help='Папка для кадров (по умолчанию: Images)')
    
    args = parser.parse_args()
    
    # Создание конфигурации
    config = {
        'videos_dir': args.videos_dir,
        'images_dir': args.images_dir,
        'data_dir': 'data',
        'models_dir': 'models',
        'num_frames': args.num_frames,
        'roboflow_api_key': os.environ.get('ROBOFLOW_API_KEY'),
        'confidence_threshold': 0.5,
        'image_format': 'jpg',
        'batch_size': 32
    }
    
    # Создание pipeline
    pipeline = TSTPipeline(config)
    
    # Выполнение команд
    if args.full:
        success = pipeline.run_full_pipeline(
            extract_frames=True,
            prepare_roboflow=True,
            roboflow_project=args.project,
            run_detection=False
        )
        sys.exit(0 if success else 1)
    
    elif args.extract_frames:
        success = pipeline.extract_frames()
        sys.exit(0 if success else 1)
    
    elif args.prepare_roboflow:
        success = pipeline.prepare_for_roboflow()
        sys.exit(0 if success else 1)
    
    elif args.upload_roboflow:
        if not args.project:
            logger.error("Укажите название проекта с --project")
            sys.exit(1)
        success = pipeline.upload_to_roboflow(args.project)
        sys.exit(0 if success else 1)
    
    elif args.detect:
        success = pipeline.run_detection(args.model, args.source)
        sys.exit(0 if success else 1)
    
    elif args.web:
        pipeline.run_web_interface()
    
    else:
        parser.print_help()
        print("\\n💡 Для быстрого старта используйте: python ПОЛНЫЙ_PIPELINE.py --full")


if __name__ == "__main__":
    main()