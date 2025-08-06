#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Извлечение 30 кадров из каждого видео с равномерным распределением по времени
Автор: TST Project
Версия: 2.0 для UTair Cloud
"""

import cv2
import os
import sys
import time
from pathlib import Path
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_30_frames.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FrameExtractor30:
    """Класс для извлечения 30 кадров из видео с равномерным распределением"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.setup_directories()
        
    def get_default_config(self):
        """Получить конфигурацию по умолчанию"""
        return {
            'videos_dir': 'Videos',
            'output_base_dir': None,  # Будет определена автоматически
            'frames_per_video': 30,
            'image_format': 'jpg',
            'image_quality': 95,
            'video_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv'],
            'skip_existing': True,
            'create_video_subfolders': True
        }
    
    def setup_directories(self):
        """Настроить директории для сохранения"""
        # Определить папку для сохранения кадров
        if self.config['output_base_dir'] is None:
            # Попробовать найти путь к облаку
            cloud_paths = [
                # Если Videos - ссылка на облако, подняться на уровень выше
                Path("Videos").parent / "Extracted_Frames" if Path("Videos").is_symlink() else None,
                # Поиск смонтированного облачного диска
                Path("Z:/TST_project/Extracted_Frames"),
                Path("Y:/TST_project/Extracted_Frames"), 
                Path("X:/TST_project/Extracted_Frames"),
                # Локальная папка как fallback
                Path("Extracted_Frames")
            ]
            
            # Найти первый доступный путь
            for path in cloud_paths:
                if path is None:
                    continue
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    # Проверить, что можно записать
                    test_file = path / "test_write.tmp"
                    test_file.write_text("test")
                    test_file.unlink()
                    self.config['output_base_dir'] = path
                    logger.info(f"📁 Использую папку для кадров: {path}")
                    break
                except Exception as e:
                    logger.debug(f"Не удалось использовать {path}: {e}")
                    continue
            
            # Если ничего не найдено, использовать локальную папку
            if self.config['output_base_dir'] is None:
                self.config['output_base_dir'] = Path("Extracted_Frames")
                self.config['output_base_dir'].mkdir(exist_ok=True)
                logger.warning("⚠️ Сохранение в локальную папку. Рекомендуется настроить облако.")
        
        logger.info(f"📂 Кадры будут сохранены в: {self.config['output_base_dir']}")
    
    def get_video_files(self):
        """Получить список видеофайлов для обработки"""
        videos_dir = Path(self.config['videos_dir'])
        
        if not videos_dir.exists():
            logger.error(f"❌ Папка {videos_dir} не найдена")
            logger.info("💡 Запустите НАСТРОИТЬ_ОБЛАКО.bat для настройки доступа к видео")
            return []
        
        # Поиск видеофайлов
        video_files = []
        for ext in self.config['video_extensions']:
            video_files.extend(videos_dir.glob(f"*{ext}"))
            video_files.extend(videos_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            logger.warning(f"⚠️ Видеофайлы не найдены в {videos_dir}")
            logger.info("💡 Проверьте содержимое облака: https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos")
            return []
        
        logger.info(f"📹 Найдено видеофайлов: {len(video_files)}")
        return sorted(video_files)
    
    def extract_frames_from_video(self, video_path):
        """
        Извлечь 30 кадров из одного видео с равномерным распределением
        
        Args:
            video_path (Path): Путь к видеофайлу
            
        Returns:
            int: Количество извлеченных кадров
        """
        video_name = video_path.stem
        logger.info(f"🎬 Обработка: {video_path.name}")
        
        # Создать папку для видео (если включено)
        if self.config['create_video_subfolders']:
            output_dir = self.config['output_base_dir'] / video_name
        else:
            output_dir = self.config['output_base_dir']
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Проверить, нужно ли пропустить (если кадры уже извлечены)
        if self.config['skip_existing']:
            existing_frames = list(output_dir.glob(f"{video_name}_frame_*.{self.config['image_format']}"))
            if len(existing_frames) >= self.config['frames_per_video']:
                logger.info(f"⏭️ Пропуск {video_name} - кадры уже извлечены ({len(existing_frames)} шт.)")
                return len(existing_frames)
        
        # Открыть видео
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"❌ Не удалось открыть видео: {video_path}")
            return 0
        
        try:
            # Получить информацию о видео
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📊 Видео: {total_frames} кадров, {fps:.2f} FPS, {duration:.1f} сек")
            
            # Рассчитать позиции кадров для равномерного распределения
            if total_frames < self.config['frames_per_video']:
                logger.warning(f"⚠️ В видео {total_frames} кадров, меньше чем требуется {self.config['frames_per_video']}")
                frame_positions = list(range(total_frames))
            else:
                # Разделить видео на 30 равных сегментов
                segment_size = total_frames / self.config['frames_per_video']
                frame_positions = []
                
                for i in range(self.config['frames_per_video']):
                    # Взять первый кадр из каждого сегмента
                    frame_pos = int(i * segment_size)
                    frame_positions.append(frame_pos)
                
                logger.info(f"📐 Сегменты по {segment_size:.1f} кадров, позиции: {frame_positions[:5]}...{frame_positions[-5:]}")
            
            # Извлечение кадров
            extracted_count = 0
            for i, frame_pos in enumerate(frame_positions):
                # Переместиться к нужному кадру
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"⚠️ Не удалось прочитать кадр на позиции {frame_pos}")
                    continue
                
                # Сформировать имя файла
                frame_filename = f"{video_name}_frame_{i+1:02d}_pos_{frame_pos:06d}.{self.config['image_format']}"
                frame_path = output_dir / frame_filename
                
                # Сохранить кадр
                encode_params = []
                if self.config['image_format'].lower() in ['jpg', 'jpeg']:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config['image_quality']]
                elif self.config['image_format'].lower() == 'png':
                    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
                
                success = cv2.imwrite(str(frame_path), frame, encode_params)
                
                if success:
                    extracted_count += 1
                    if extracted_count <= 5 or extracted_count % 10 == 0:
                        logger.info(f"   ✅ Кадр {extracted_count:2d}/{len(frame_positions)}: {frame_filename}")
                else:
                    logger.error(f"❌ Ошибка сохранения кадра: {frame_path}")
            
            logger.info(f"🎉 Извлечено {extracted_count} кадров из {video_name}")
            return extracted_count
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {video_name}: {e}")
            return 0
            
        finally:
            cap.release()
    
    def extract_all_videos(self):
        """Извлечь кадры из всех видео"""
        logger.info("🚀 ЗАПУСК ИЗВЛЕЧЕНИЯ 30 КАДРОВ ИЗ КАЖДОГО ВИДЕО")
        logger.info("=" * 60)
        logger.info(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📁 Исходные видео: {self.config['videos_dir']}")
        logger.info(f"💾 Сохранение кадров: {self.config['output_base_dir']}")
        logger.info(f"🎯 Кадров на видео: {self.config['frames_per_video']}")
        logger.info("=" * 60)
        
        # Получить список видео
        video_files = self.get_video_files()
        if not video_files:
            return 0, 0
        
        # Обработать каждое видео
        total_extracted = 0
        processed_videos = 0
        failed_videos = []
        
        start_time = time.time()
        
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"\\n📹 ВИДЕО {i}/{len(video_files)}: {video_file.name}")
            logger.info("-" * 50)
            
            try:
                extracted = self.extract_frames_from_video(video_file)
                if extracted > 0:
                    total_extracted += extracted
                    processed_videos += 1
                else:
                    failed_videos.append(video_file.name)
                    
            except Exception as e:
                logger.error(f"❌ Критическая ошибка при обработке {video_file.name}: {e}")
                failed_videos.append(video_file.name)
        
        # Итоговая статистика
        elapsed_time = time.time() - start_time
        
        logger.info("\\n" + "=" * 60)
        logger.info("📊 ИТОГОВАЯ СТАТИСТИКА")
        logger.info("=" * 60)
        logger.info(f"✅ Обработано видео: {processed_videos}/{len(video_files)}")
        logger.info(f"🖼️ Извлечено кадров: {total_extracted}")
        logger.info(f"⏱️ Время выполнения: {elapsed_time:.1f} сек")
        
        if processed_videos > 0:
            avg_frames = total_extracted / processed_videos
            avg_time = elapsed_time / processed_videos
            logger.info(f"📈 Среднее кадров/видео: {avg_frames:.1f}")
            logger.info(f"⚡ Среднее время/видео: {avg_time:.1f} сек")
        
        if failed_videos:
            logger.warning(f"⚠️ Не удалось обработать: {len(failed_videos)} видео")
            for failed in failed_videos:
                logger.warning(f"   ❌ {failed}")
        
        logger.info(f"💾 Кадры сохранены в: {self.config['output_base_dir']}")
        
        if str(self.config['output_base_dir']).startswith(('Z:', 'Y:', 'X:')):
            logger.info("🌐 Кадры сохранены в облаке UTair")
            logger.info("🔗 Доступ: https://cloud.utair.ru/apps/files/?dir=/TST_project")
        
        return processed_videos, total_extracted
    
    def print_summary(self):
        """Вывести сводку о настройках"""
        print("🎯 НАСТРОЙКИ ИЗВЛЕЧЕНИЯ КАДРОВ:")
        print("-" * 40)
        print(f"📁 Папка с видео: {self.config['videos_dir']}")
        print(f"💾 Папка для кадров: {self.config['output_base_dir']}")
        print(f"🎬 Кадров на видео: {self.config['frames_per_video']}")
        print(f"🖼️ Формат изображений: {self.config['image_format']}")
        print(f"📊 Качество JPEG: {self.config['image_quality']}%")
        print(f"📂 Подпапки для видео: {'Да' if self.config['create_video_subfolders'] else 'Нет'}")
        print(f"⏭️ Пропускать готовые: {'Да' if self.config['skip_existing'] else 'Нет'}")
        print()


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Извлечение 30 кадров из каждого видео с равномерным распределением",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Стандартное извлечение 30 кадров
  python extract_30_frames.py

  # Изменить количество кадров
  python extract_30_frames.py --frames 50

  # Сохранить в определенную папку
  python extract_30_frames.py --output "D:/Frames"

  # Без создания подпапок для каждого видео
  python extract_30_frames.py --no-subfolders

  # Переизвлечь все кадры заново
  python extract_30_frames.py --force

  # Только показать настройки
  python extract_30_frames.py --dry-run
        """
    )
    
    parser.add_argument('--frames', type=int, default=30,
                       help='Количество кадров для извлечения (по умолчанию: 30)')
    parser.add_argument('--videos-dir', default='Videos',
                       help='Папка с видеофайлами (по умолчанию: Videos)')
    parser.add_argument('--output', 
                       help='Папка для сохранения кадров (авто-определение)')
    parser.add_argument('--format', default='jpg', choices=['jpg', 'png'],
                       help='Формат изображений (по умолчанию: jpg)')
    parser.add_argument('--quality', type=int, default=95,
                       help='Качество JPEG 0-100 (по умолчанию: 95)')
    parser.add_argument('--no-subfolders', action='store_true',
                       help='Не создавать подпапки для каждого видео')
    parser.add_argument('--force', action='store_true',
                       help='Переизвлечь кадры даже если они уже существуют')
    parser.add_argument('--dry-run', action='store_true',
                       help='Показать настройки без выполнения')
    
    args = parser.parse_args()
    
    # Создание конфигурации
    config = {
        'videos_dir': args.videos_dir,
        'output_base_dir': Path(args.output) if args.output else None,
        'frames_per_video': args.frames,
        'image_format': args.format,
        'image_quality': args.quality,
        'video_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv'],
        'skip_existing': not args.force,
        'create_video_subfolders': not args.no_subfolders
    }
    
    # Создание экстрактора
    extractor = FrameExtractor30(config)
    
    if args.dry_run:
        extractor.print_summary()
        video_files = extractor.get_video_files()
        print(f"📹 Будет обработано видео: {len(video_files)}")
        print(f"🖼️ Ожидается кадров: {len(video_files) * args.frames}")
        return
    
    # Показать настройки
    extractor.print_summary()
    
    # Запуск извлечения
    try:
        processed, extracted = extractor.extract_all_videos()
        
        if processed > 0:
            print("\\n🎉 ИЗВЛЕЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print(f"📊 Результат: {extracted} кадров из {processed} видео")
            
            # Предложить следующие шаги
            print("\\n💡 Следующие шаги:")
            print("   1. Проверьте кадры в облаке: https://cloud.utair.ru/apps/files/?dir=/TST_project")
            print("   2. Загрузите кадры в Roboflow для разметки")
            print("   3. Обучите модель детекции")
            print("   4. Используйте python detect.py для анализа")
        else:
            print("\\n❌ Не удалось обработать ни одного видео")
            print("💡 Проверьте:")
            print("   - Доступность папки Videos")
            print("   - Подключение к UTair Cloud")
            print("   - Наличие видеофайлов")
            
    except KeyboardInterrupt:
        print("\\n⏹️ Извлечение прервано пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()