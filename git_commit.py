#!/usr/bin/env python3
"""
Python скрипт для коммита - обходим проблемы терминала Cursor
"""

import subprocess
import sys

def run_git_command(command, description):
    """Выполнить Git команду"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
        
        if result.stdout:
            print("📤 Вывод:")
            print(result.stdout)
            
        if result.stderr:
            print("⚠️ Сообщения:")
            print(result.stderr)
            
        if result.returncode == 0:
            print(f"✅ {description} - успешно!")
        else:
            print(f"❌ {description} - ошибка (код: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False

def main():
    """Основная функция коммита"""
    print("💾 Python скрипт для Git коммита")
    print("=" * 50)
    
    # Проверка статуса
    print("\n📋 Проверка статуса Git:")
    run_git_command("git status", "Проверка статуса")
    
    # Добавление файлов
    print("\n📤 Добавление файлов:")
    success = run_git_command("git add .", "Добавление всех изменений")
    
    if not success:
        print("❌ Не удалось добавить файлы!")
        return
    
    # Коммит
    print("\n💬 Создание коммита:")
    commit_message = '''Организация структуры проекта + полная настройка ПК№2

✅ Выполнено:
- Создана оптимальная структура папок (scripts/, docs/, models/, config/, results/)
- Перемещены файлы по категориям для лучшей организации
- Созданы удобные ярлыки (БЫСТРОЕ_МЕНЮ.bat, ЗАПУСК_STREAMLIT.bat)
- Удалены лишние и дублирующиеся файлы (~40 файлов очищено)
- Установлены все зависимости Python (YOLOv8, OpenCV, Streamlit, etc.)
- Скачаны 8 видеофайлов для анализа (~350MB)

🎯 Готово к работе:
- Python 3.13.5 настроен и работает
- Все библиотеки для компьютерного зрения установлены
- Видео готовы для извлечения кадров и анализа безопасности
- Обходной путь для проблем терминала Cursor через bat-файлы

📁 Структура:
- Основные Python модули в корне
- Скрипты и утилиты в scripts/
- Документация в docs/
- Модели в models/
- Конфигурация в config/
- Готовая папка results/ для выводов

🚀 Следующие шаги: анализ видео и детекция безопасности'''
    
    success = run_git_command(f'git commit -m "{commit_message}"', "Создание коммита")
    
    if not success:
        print("❌ Не удалось создать коммит!")
        return
    
    # Push
    print("\n🚀 Отправка в GitHub:")
    success = run_git_command("git push origin main", "Отправка в удаленный репозиторий")
    
    if success:
        print("\n🎉 Коммит успешно завершен!")
        print("✅ Все изменения сохранены в Git и отправлены в GitHub")
        print("🚀 Проект готов к анализу видео!")
    else:
        print("\n⚠️ Коммит создан локально, но возможны проблемы с push")

if __name__ == "__main__":
    main()
    input("\nНажмите Enter для выхода...")