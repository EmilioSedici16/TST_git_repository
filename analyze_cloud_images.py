#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Анализатор структуры изображений в UTair Cloud
Анализирует и отображает информацию о разделенных по папкам изображениях
"""

import os
from pathlib import Path
from datetime import datetime
import json

class CloudImagesAnalyzer:
    """Анализатор структуры изображений в облаке"""
    
    def __init__(self):
        self.cloud_url = "https://cloud.utair.ru/apps/files/files/3350055?dir=/TST_project/Images"
        self.folders = [
            "1 секция_ задняя правая_20250723_101500",
            "1 секция_ задняя правая_20250723_102500", 
            "1 секция_ задняя средняя_20250723_101500",
            "1 секция_ задняя средняя_20250723_102500",
            "Моечная средняя_2_20250723_101500",
            "Моечная средняя_2_20250723_102500",
            "Склад_ зона самовыдачи-1_20250527_105059",
            "Склад_ зона самовыдачи-2_20250527_105047"
        ]
        
    def analyze_structure(self):
        """Анализ структуры изображений"""
        print("🖼️ Анализ структуры изображений в UTair Cloud")
        print("=" * 60)
        print(f"🌐 Облако: {self.cloud_url}")
        print(f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Статистика по папкам
        total_folders = len(self.folders)
        total_images = total_folders * 30
        
        print("📊 Статистика:")
        print(f"   📁 Всего папок: {total_folders}")
        print(f"   🖼️ Всего изображений: {total_images}")
        print(f"   📸 Кадров в папке: 30")
        print()
        
        # Детальная информация по папкам
        print("🗂️ Структура папок:")
        for i, folder in enumerate(self.folders, 1):
            print(f"  {i:2d}. {folder}/")
            print(f"      📸 30 кадров (frame_001.jpg - frame_030.jpg)")
            
            # Анализ названия папки
            zone, description, date_str, time_str = self._parse_folder_name(folder)
            print(f"      🏷️ Зона: {zone}")
            print(f"      📍 Описание: {description}")
            print(f"      📅 Дата: {date_str}")
            print(f"      🕐 Время: {time_str}")
            print()
    
    def _parse_folder_name(self, folder_name):
        """Парсинг названия папки для извлечения информации"""
        try:
            # Разделяем по последним двум подчеркиваниям
            parts = folder_name.rsplit('_', 2)
            if len(parts) >= 3:
                base_name = parts[0]
                date_str = parts[1]
                time_str = parts[2]
                
                # Форматируем дату
                if len(date_str) == 8:  # YYYYMMDD
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                else:
                    formatted_date = date_str
                
                # Форматируем время
                if len(time_str) == 6:  # HHMMSS
                    formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                else:
                    formatted_time = time_str
                
                return base_name, "N/A", formatted_date, formatted_time
            else:
                return folder_name, "N/A", "N/A", "N/A"
        except:
            return folder_name, "N/A", "N/A", "N/A"
    
    def generate_summary_report(self):
        """Генерация сводного отчета"""
        print("📋 Сводный отчет")
        print("=" * 60)
        
        # Группировка по зонам
        zones = {}
        for folder in self.folders:
            zone = folder.split('_')[0]
            if zone not in zones:
                zones[zone] = []
            zones[zone].append(folder)
        
        print("🏭 Группировка по зонам:")
        for zone, folders in zones.items():
            print(f"   {zone}: {len(folders)} папок")
            for folder in folders:
                print(f"     - {folder}")
            print()
        
        # Рекомендации по анализу
        print("💡 Рекомендации по анализу:")
        print("   1. 🏭 Секции: анализ рабочих процессов и безопасности")
        print("   2. 🧽 Моечная: контроль качества уборки и ТБ")
        print("   3. 📦 Склад: мониторинг зоны самовыдачи")
        print()
        
        print("🔗 Ссылки:")
        print(f"   🌐 UTair Cloud: {self.cloud_url}")
        print("   📚 Документация: docs/РАБОТА_С_ИЗОБРАЖЕНИЯМИ.md")
    
    def export_to_json(self, filename="cloud_images_structure.json"):
        """Экспорт структуры в JSON файл"""
        structure_data = {
            "cloud_url": self.cloud_url,
            "analysis_date": datetime.now().isoformat(),
            "total_folders": len(self.folders),
            "total_images": len(self.folders) * 30,
            "folders": []
        }
        
        for folder in self.folders:
            zone, description, date_str, time_str = self._parse_folder_name(folder)
            folder_info = {
                "name": folder,
                "zone": zone,
                "description": description,
                "date": date_str,
                "time": time_str,
                "frame_count": 30,
                "frame_format": "frame_XXX.jpg"
            }
            structure_data["folders"].append(folder_info)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(structure_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Структура экспортирована в {filename}")
    
    def check_local_images(self):
        """Проверка локальных изображений"""
        print("🔍 Проверка локальных изображений")
        print("=" * 60)
        
        local_images = Path("Images")
        if not local_images.exists():
            print("❌ Локальная папка Images не найдена")
            return
        
        # Подсчет локальных папок
        local_folders = [d for d in local_images.iterdir() if d.is_dir()]
        print(f"📁 Локальных папок: {len(local_folders)}")
        
        if len(local_folders) > 0:
            print("📂 Найденные папки:")
            for folder in local_folders:
                image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                print(f"   {folder.name}/ ({len(image_files)} изображений)")
        
        # Сравнение с облаком
        cloud_folders_set = set(self.folders)
        local_folders_set = {f.name for f in local_folders}
        
        missing_locally = cloud_folders_set - local_folders_set
        if missing_locally:
            print(f"\n⚠️ Отсутствуют локально: {len(missing_locally)} папок")
            for folder in missing_locally:
                print(f"   - {folder}")
        
        extra_locally = local_folders_set - cloud_folders_set
        if extra_locally:
            print(f"\n➕ Дополнительные локально: {len(extra_locally)} папок")
            for folder in extra_locally:
                print(f"   - {folder}")

def main():
    """Основная функция"""
    analyzer = CloudImagesAnalyzer()
    
    print("🚀 Запуск анализатора изображений UTair Cloud")
    print()
    
    # Основной анализ
    analyzer.analyze_structure()
    print()
    
    # Сводный отчет
    analyzer.generate_summary_report()
    print()
    
    # Проверка локальных изображений
    analyzer.check_local_images()
    print()
    
    # Экспорт в JSON
    analyzer.export_to_json()
    
    print("\n✅ Анализ завершен!")
    print("📚 Подробная информация: docs/РАБОТА_С_ИЗОБРАЖЕНИЯМИ.md")

if __name__ == "__main__":
    main() 