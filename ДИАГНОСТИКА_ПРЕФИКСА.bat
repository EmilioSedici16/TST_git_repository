@echo off
chcp 65001
echo 🔍 Диагностика проблемы с префиксом "с" в терминале Cursor
echo ================================================================
echo.

echo 📋 1. Проверка системной локали:
echo Текущая кодовая страница:
chcp
echo.

echo Региональные настройки:
powershell -Command "Get-Culture | Select-Object Name, DisplayName, KeyboardLayoutId"
echo.

echo 📋 2. Проверка переменных окружения:
echo LANG: %LANG%
echo LC_ALL: %LC_ALL%  
echo PYTHONIOENCODING: %PYTHONIOENCODING%
echo.

echo 📋 3. Проверка раскладки клавиатуры:
powershell -Command "Get-WinUserLanguageList | Select-Object LanguageTag, InputMethodTips"
echo.

echo 📋 4. Тест различных способов ввода:
echo.
echo 🧪 Тест 1 - Прямой вызов python.exe:
C:\Python313\python.exe --version 2>nul || echo "Python не найден по стандартному пути"
echo.

echo 🧪 Тест 2 - Через PowerShell:
powershell -Command "python --version"
echo.

echo 🧪 Тест 3 - Через cmd:
cmd /c "python --version"
echo.

echo 📋 5. Проверка истории команд PowerShell:
echo Последние команды:
powershell -Command "Get-History | Select-Object -Last 5"
echo.

echo 📋 6. Информация о терминале:
echo TERM: %TERM%
echo TERMINAL_EMULATOR: %TERMINAL_EMULATOR%
echo.

echo 💡 Возможные решения:
echo 1. Переключить раскладку на английскую перед работой в терминале
echo 2. Использовать Ctrl+Shift для очистки буфера ввода
echo 3. Перезапустить Cursor
echo 4. Использовать Windows Terminal вместо встроенного терминала Cursor
echo 5. Проверить настройки IME в Windows

pause