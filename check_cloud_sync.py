#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки синхронизации с UTair Cloud
Проверяет доступность видеофайлов и их готовность к обработке
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess

class UTairCloudChecker:
    """Класс для проверки синхронизации с UTair Cloud"""
    
    def __init__(self):
        self.videos_dir = Path("Videos")
        self.cloud_url = "https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos"
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv']
    
    def print_header(self):
        """Вывести заголовок"""
        print("=" * 60)
        print("🌐 ПРОВЕРКА СИНХРОНИЗАЦИИ С UTAIR CLOUD")
        print("=" * 60)
        print(f"📅 Время проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Облако: {self.cloud_url}")
        print("=" * 60)
    
    def check_videos_directory(self):
        """Проверить состояние папки Videos"""
        print("📁 ПРОВЕРКА ПАПКИ VIDEOS:")
        print("-" * 40)
        
        if not self.videos_dir.exists():
            print("❌ Папка Videos не существует")
            print("💡 Запустите НАСТРОИТЬ_ОБЛАКО.bat для настройки")
            return False
        
        # Проверка, является ли папка символической ссылкой
        if self.videos_dir.is_symlink():
            target = self.videos_dir.resolve()
            print(f"🔗 Videos - символическая ссылка")
            print(f"   Цель: {target}")
            
            if target.exists():
                print("✅ Ссылка работает корректно")
                return True
            else:
                print("❌ Цель ссылки недоступна")
                print("💡 Проверьте подключение к облаку")
                return False
        else:
            print("📂 Videos - обычная папка (не ссылка)")
            print("💡 Рекомендуется настроить ссылку на облако")
            return True
    
    def scan_video_files(self):
        """Сканировать видеофайлы"""
        print("\\n📹 СКАНИРОВАНИЕ ВИДЕОФАЙЛОВ:")
        print("-" * 40)
        
        try:
            # Поиск видеофайлов
            video_files = []
            for ext in self.video_extensions:
                video_files.extend(self.videos_dir.glob(f"*{ext}"))
                video_files.extend(self.videos_dir.glob(f"*{ext.upper()}"))
            
            if not video_files:
                print("⚠️ Видеофайлы не найдены")
                print("💡 Возможные причины:")
                print("   • Видео еще не загружены в облако")
                print("   • Проблемы с синхронизацией")
                print("   • Неправильный путь к облаку")
                print(f"🌐 Проверьте: {self.cloud_url}")
                return []
            
            print(f"✅ Найдено видеофайлов: {len(video_files)}")
            print()
            
            total_size = 0
            for i, video_file in enumerate(video_files, 1):
                try:
                    size = video_file.stat().st_size
                    size_mb = size / (1024 * 1024)
                    total_size += size
                    
                    # Проверка доступности файла
                    if size > 0:
                        status = "✅"
                    else:
                        status = "⚠️ (0 байт)"
                    
                    print(f"   {i:2d}. {status} {video_file.name}")
                    print(f"       Размер: {size_mb:.1f} MB")
                    
                except Exception as e:
                    print(f"   {i:2d}. ❌ {video_file.name}")
                    print(f"       Ошибка: {e}")
            
            total_gb = total_size / (1024 * 1024 * 1024)
            avg_mb = (total_size / len(video_files)) / (1024 * 1024) if video_files else 0
            
            print()
            print(f"📊 СТАТИСТИКА:")
            print(f"   Всего файлов: {len(video_files)}")
            print(f"   Общий размер: {total_gb:.2f} GB")
            print(f"   Средний размер: {avg_mb:.1f} MB/файл")
            
            return video_files
            
        except Exception as e:
            print(f"❌ Ошибка при сканировании: {e}")
            return []
    
    def check_extracted_frames(self):
        """Проверить извлеченные кадры"""
        print("\\n🖼️ ПРОВЕРКА ИЗВЛЕЧЕННЫХ КАДРОВ:")
        print("-" * 40)
        
        images_dir = Path("Images")
        if not images_dir.exists():
            print("📂 Папка Images не существует")
            print("💡 Кадры еще не извлекались")
            return
        
        # Подсчет изображений
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        if image_files:
            total_size = sum(f.stat().st_size for f in image_files)
            size_mb = total_size / (1024 * 1024)
            
            print(f"✅ Найдено кадров: {len(image_files)}")
            print(f"📊 Общий размер: {size_mb:.1f} MB")
            
            # Группировка по видео
            video_groups = {}
            for img in image_files:
                # Извлечение имени видео из имени кадра
                parts = img.stem.split('_frame_')
                if len(parts) >= 2:
                    video_name = parts[0]
                    if video_name not in video_groups:
                        video_groups[video_name] = 0
                    video_groups[video_name] += 1
            
            if video_groups:
                print("\\n📹 Кадры по видео:")
                for video, count in video_groups.items():
                    print(f"   {video}: {count} кадров")
        else:
            print("⚠️ Извлеченные кадры не найдены")
            print("💡 Запустите: python extract_frames.py")
    
    def check_connectivity(self):
        """Проверить подключение к облаку"""
        print("\\n🌐 ПРОВЕРКА ПОДКЛЮЧЕНИЯ:")
        print("-" * 40)
        
        try:
            # Проверка доступности домена
            import socket
            host = "cloud.utair.ru"
            port = 443
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"✅ {host} доступен")
            else:
                print(f"❌ {host} недоступен")
                print("💡 Проверьте интернет-соединение")
                
        except Exception as e:
            print(f"⚠️ Не удалось проверить подключение: {e}")
    
    def suggest_actions(self, video_files):
        """Предложить действия пользователю"""
        print("\\n🎯 РЕКОМЕНДУЕМЫЕ ДЕЙСТВИЯ:")
        print("-" * 40)
        
        if not video_files:
            print("1. 🌐 Проверьте облако UTair:")
            print(f"   {self.cloud_url}")
            print("2. 📤 Загрузите видеофайлы в облако")
            print("3. 🔄 Проверьте синхронизацию")
            print("4. 🔧 Настройте подключение: НАСТРОИТЬ_ОБЛАКО.bat")
            return
        
        # Проверка, есть ли новые видео для обработки
        images_dir = Path("Images")
        processed_videos = set()
        
        if images_dir.exists():
            for img in images_dir.glob("*.jpg"):
                parts = img.stem.split('_frame_')
                if len(parts) >= 2:
                    processed_videos.add(parts[0])
        
        new_videos = []
        for video in video_files:
            video_name = video.stem
            if video_name not in processed_videos:
                new_videos.append(video)
        
        if new_videos:
            print(f"🎬 Найдено {len(new_videos)} новых видео для обработки:")
            for video in new_videos[:5]:  # Показать первые 5
                print(f"   • {video.name}")
            if len(new_videos) > 5:
                print(f"   ... и еще {len(new_videos) - 5}")
            print()
            print("💡 Рекомендуемые действия:")
            print("   1. python extract_frames.py")
            print("   2. python ПОЛНЫЙ_PIPELINE.py --prepare-roboflow")
        else:
            print("✅ Все видео уже обработаны")
            print("💡 Возможные действия:")
            print("   1. python detect.py --source Images/")
            print("   2. streamlit run app.py")
            print("   3. Загрузите новые видео в облако")
    
    def run_interactive_mode(self):
        """Запустить интерактивный режим"""
        print("\\n⚡ БЫСТРЫЕ ДЕЙСТВИЯ:")
        print("-" * 40)
        print("1. Извлечь кадры из всех видео")
        print("2. Подготовить данные для Roboflow") 
        print("3. Запустить детекцию")
        print("4. Открыть веб-интерфейс")
        print("5. Настроить облако")
        print("6. Открыть облако в браузере")
        print("0. Выход")
        print()
        
        try:
            choice = input("Выберите действие (0-6): ").strip()
            
            if choice == "1":
                print("🎬 Запуск извлечения кадров...")
                os.system("python extract_frames.py")
            elif choice == "2":
                print("📤 Подготовка для Roboflow...")
                os.system("python ПОЛНЫЙ_PIPELINE.py --prepare-roboflow")
            elif choice == "3":
                print("🔍 Запуск детекции...")
                os.system("python detect.py --source Images/")
            elif choice == "4":
                print("🌐 Запуск веб-интерфейса...")
                os.system("streamlit run app.py")
            elif choice == "5":
                print("⚙️ Настройка облака...")
                os.system("НАСТРОИТЬ_ОБЛАКО.bat")
            elif choice == "6":
                print("🌐 Открытие облака в браузере...")
                import webbrowser
                webbrowser.open(self.cloud_url)
            elif choice == "0":
                print("👋 До свидания!")
            else:
                print("❌ Неверный выбор")
                
        except KeyboardInterrupt:
            print("\\n👋 До свидания!")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    def run_full_check(self):
        """Запустить полную проверку"""
        self.print_header()
        
        # Проверка папки Videos
        videos_ok = self.check_videos_directory()
        
        # Сканирование видеофайлов
        video_files = self.scan_video_files() if videos_ok else []
        
        # Проверка извлеченных кадров
        self.check_extracted_frames()
        
        # Проверка подключения
        self.check_connectivity()
        
        # Рекомендации
        self.suggest_actions(video_files)
        
        return video_files


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Проверка синхронизации с UTair Cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Полная проверка
  python check_cloud_sync.py

  # Только проверка видео
  python check_cloud_sync.py --videos-only

  # Интерактивный режим
  python check_cloud_sync.py --interactive

  # Быстрая проверка
  python check_cloud_sync.py --quick
        """
    )
    
    parser.add_argument('--videos-only', action='store_true',
                       help='Проверить только видеофайлы')
    parser.add_argument('--interactive', action='store_true',
                       help='Интерактивный режим с быстрыми действиями')
    parser.add_argument('--quick', action='store_true',
                       help='Быстрая проверка без подробностей')
    
    args = parser.parse_args()
    
    checker = UTairCloudChecker()
    
    try:
        if args.interactive:
            video_files = checker.run_full_check()
            if video_files:
                checker.run_interactive_mode()
        elif args.videos_only:
            checker.print_header()
            checker.check_videos_directory()
            checker.scan_video_files()
        elif args.quick:
            videos_dir = Path("Videos")
            if videos_dir.exists():
                video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
                print(f"📊 Видео: {len(video_files)} файлов")
                if video_files:
                    total_size = sum(f.stat().st_size for f in video_files) / (1024**3)
                    print(f"📦 Размер: {total_size:.2f} GB")
            else:
                print("❌ Папка Videos не найдена")
        else:
            video_files = checker.run_full_check()
            
            # Опция для интерактивного режима
            if video_files:
                try:
                    answer = input("\\nЗапустить интерактивный режим? (y/n): ").strip().lower()
                    if answer in ['y', 'yes', 'да', 'д']:
                        checker.run_interactive_mode()
                except KeyboardInterrupt:
                    print("\\n👋 До свидания!")
                    
    except KeyboardInterrupt:
        print("\\n👋 Операция прервана пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()