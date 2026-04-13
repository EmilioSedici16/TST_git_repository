#!/usr/bin/env python3
"""
Скрипт для очистки проекта от временных и ненужных файлов
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_python_cache():
    """Удаление Python кэша"""
    print("🧹 Очистка Python кэша...")
    
    # Удаляем __pycache__ папки
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    print(f"   ✅ Удалена: {cache_path}")
                except Exception as e:
                    print(f"   ❌ Ошибка удаления {cache_path}: {e}")
    
    # Удаляем .pyc и .pyo файлы
    pyc_files = glob.glob('**/*.pyc', recursive=True)
    pyo_files = glob.glob('**/*.pyo', recursive=True)
    
    for file_path in pyc_files + pyo_files:
        try:
            os.remove(file_path)
            print(f"   ✅ Удален: {file_path}")
        except Exception as e:
            print(f"   ❌ Ошибка удаления {file_path}: {e}")

def cleanup_temp_files():
    """Удаление временных файлов"""
    print("\n🗑️ Очистка временных файлов...")
    
    # Паттерны временных файлов
    temp_patterns = [
        '*.tmp', '*.temp', '*.log', '*.bak', '*.swp',
        '*.swo', '*~', '.DS_Store', 'Thumbs.db'
    ]
    
    for pattern in temp_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            try:
                os.remove(file_path)
                print(f"   ✅ Удален: {file_path}")
            except Exception as e:
                print(f"   ❌ Ошибка удаления {file_path}: {e}")

def cleanup_empty_dirs():
    """Удаление пустых папок"""
    print("\n📁 Удаление пустых папок...")
    
    for root, dirs, files in os.walk('.', topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Папка пустая
                    os.rmdir(dir_path)
                    print(f"   ✅ Удалена пустая папка: {dir_path}")
            except Exception as e:
                pass  # Игнорируем ошибки

def cleanup_test_files():
    """Удаление тестовых файлов"""
    print("\n🧪 Очистка тестовых файлов...")
    
    test_files = [
        'test_terminal.bat',  # Созданный диагностикой
        'temp_test_file.txt',
        'debug_output.log'
    ]
    
    for file_name in test_files:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                print(f"   ✅ Удален тестовый файл: {file_name}")
            except Exception as e:
                print(f"   ❌ Ошибка удаления {file_name}: {e}")

def show_project_status():
    """Показать статус проекта"""
    print("\n📊 Статус проекта после очистки:")
    
    # Подсчет файлов по типам
    py_files = len(glob.glob('**/*.py', recursive=True))
    bat_files = len(glob.glob('**/*.bat', recursive=True))
    md_files = len(glob.glob('**/*.md', recursive=True))
    
    print(f"   📁 Python файлы: {py_files}")
    print(f"   📁 Bat файлы: {bat_files}")
    print(f"   📁 Markdown файлы: {md_files}")
    
    # Размер проекта
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
                file_count += 1
            except:
                pass
    
    print(f"   📊 Всего файлов: {file_count}")
    print(f"   📊 Общий размер: {total_size / (1024*1024):.1f} MB")

def main():
    """Основная функция"""
    print("🧹 ОЧИСТКА ПРОЕКТА TST")
    print("=" * 50)
    
    # Проверяем, что мы в правильной папке
    if not os.path.exists('README.md'):
        print("❌ Ошибка: README.md не найден. Убедитесь, что вы в корне проекта.")
        return
    
    print(f"📍 Рабочая директория: {os.getcwd()}")
    print()
    
    # Выполняем очистку
    cleanup_python_cache()
    cleanup_temp_files()
    cleanup_empty_dirs()
    cleanup_test_files()
    
    # Показываем результат
    show_project_status()
    
    print("\n" + "=" * 50)
    print("✅ Очистка проекта завершена!")
    print("🚀 Проект готов к работе!")
    
    input("\nНажмите Enter для выхода...")

if __name__ == "__main__":
    main()


