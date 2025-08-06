@echo off
echo 🚀 Открываем проект в VS Code...
echo.

echo 📁 Путь к проекту: %CD%
echo 💻 Запускаем VS Code...

"C:\Users\user\AppData\Local\Programs\Microsoft VS Code\Code.exe" "%CD%"

echo ✅ VS Code запущен!
echo 💡 Теперь можете использовать Terminal → New Terminal в VS Code
echo 🎯 Команды будут работать без префикса "с"!

pause