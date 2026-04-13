#!/usr/bin/env python3
"""
Диагностика проблем с терминалом Cursor
"""

import os
import sys
import subprocess
import platform

def check_system_info():
    """Проверка системной информации"""
    print("🔍 Диагностика системы:")
    print(f"   ОС: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   Рабочая директория: {os.getcwd()}")
    print(f"   PATH: {os.environ.get('PATH', 'НЕ НАЙДЕН')[:100]}...")

def check_python_commands():
    """Проверка команд Python"""
    print("\n🐍 Проверка Python команд:")
    
    commands = ['python', 'python3', 'pip', 'pip3']
    for cmd in commands:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"   ✅ {cmd}: {result.stdout.strip()}")
            else:
                print(f"   ❌ {cmd}: ошибка {result.returncode}")
        except FileNotFoundError:
            print(f"   ❌ {cmd}: не найден")
        except Exception as e:
            print(f"   ⚠️ {cmd}: исключение {e}")

def check_git():
    """Проверка Git"""
    print("\n📚 Проверка Git:")
    try:
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"   ✅ Git: {result.stdout.strip()}")
        else:
            print(f"   ❌ Git: ошибка {result.returncode}")
    except FileNotFoundError:
        print("   ❌ Git: не найден")
    except Exception as e:
        print(f"   ⚠️ Git: исключение {e}")

def check_environment_variables():
    """Проверка переменных окружения"""
    print("\n🌍 Проверка переменных окружения:")
    
    important_vars = ['PATH', 'PYTHONPATH', 'ROBOFLOW_API_KEY', 'HOME', 'USERPROFILE']
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"   ✅ {var}: {value[:50]}...")
        else:
            print(f"   ❌ {var}: не установлена")

def test_subprocess():
    """Тест subprocess"""
    print("\n🧪 Тест subprocess:")
    
    try:
        # Тест простой команды
        result = subprocess.run(['echo', 'test'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"   ✅ echo test: {result.stdout.strip()}")
        else:
            print(f"   ❌ echo test: ошибка {result.returncode}")
    except Exception as e:
        print(f"   ❌ echo test: исключение {e}")
    
    try:
        # Тест команды Windows
        if platform.system() == 'Windows':
            result = subprocess.run(['dir'], 
                                  capture_output=True, text=True, timeout=5, shell=True)
            if result.returncode == 0:
                print(f"   ✅ dir: команда выполнена успешно")
            else:
                print(f"   ❌ dir: ошибка {result.returncode}")
    except Exception as e:
        print(f"   ❌ dir: исключение {e}")

def create_test_bat():
    """Создание тестового bat файла"""
    print("\n📝 Создание тестового bat файла...")
    
    test_bat_content = """@echo off
chcp 65001
echo Тест терминала Cursor
echo.
echo Проверка Python:
python --version
echo.
echo Проверка Git:
git --version
echo.
echo Проверка директории:
dir
echo.
echo Тест завершен
pause
"""
    
    try:
        with open('test_terminal.bat', 'w', encoding='utf-8') as f:
            f.write(test_bat_content)
        print("   ✅ Создан файл test_terminal.bat")
        print("   💡 Запустите его для проверки терминала")
    except Exception as e:
        print(f"   ❌ Ошибка создания bat файла: {e}")

def main():
    """Основная функция"""
    print("🚨 ДИАГНОСТИКА ПРОБЛЕМ ТЕРМИНАЛА CURSOR")
    print("=" * 50)
    
    check_system_info()
    check_python_commands()
    check_git()
    check_environment_variables()
    test_subprocess()
    create_test_bat()
    
    print("\n" + "=" * 50)
    print("📋 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ:")
    print("   Если команды Python и Git работают через subprocess,")
    print("   но не работают в терминале Cursor - это проблема Cursor")
    print("\n🔧 РЕШЕНИЯ:")
    print("   1. Перезапустить Cursor")
    print("   2. Проверить настройки терминала")
    print("   3. Использовать внешний терминал (cmd, PowerShell)")
    print("   4. Использовать bat файлы для обхода проблем")
    
    input("\nНажмите Enter для выхода...")

if __name__ == "__main__":
    main()

