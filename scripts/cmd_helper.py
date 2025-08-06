#!/usr/bin/env python3
"""
Помощник для выполнения команд через Python
Обходит проблемы с терминалом Cursor
"""

import subprocess
import sys
import os

def run_command(command):
    """Выполнить команду и показать результат"""
    try:
        print(f"🔧 Выполняю: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
        
        if result.stdout:
            print("📤 Вывод:")
            print(result.stdout)
            
        if result.stderr:
            print("❌ Ошибки:")
            print(result.stderr)
            
        print(f"↩️ Код возврата: {result.returncode}")
        
    except Exception as e:
        print(f"❌ Исключение: {e}")

def main():
    """Интерактивный помощник команд"""
    print("🐍 Помощник командной строки через Python")
    print("Введите команды (или 'exit' для выхода):")
    print("-" * 50)
    
    while True:
        try:
            command = input("💻 > ")
            
            if command.lower() in ['exit', 'quit', 'q']:
                print("👋 До свидания!")
                break
                
            if command.strip():
                run_command(command)
                print("-" * 30)
                
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except EOFError:
            print("\n👋 До свидания!")
            break

if __name__ == "__main__":
    main()