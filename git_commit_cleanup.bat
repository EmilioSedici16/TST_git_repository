@echo off
chcp 65001 >nul
echo 🚀 Фиксация результатов уборки рабочего стола...
echo.

echo 📝 Добавляем все изменения в staging area...
git add .

echo.
echo 📦 Создаем коммит с описанием изменений...
git commit -m "🧹 Уборка рабочего стола: удалены устаревшие файлы, обновлена документация

✅ Удалено 11 устаревших файлов:
- README_extract_frames.md, ФИНАЛЬНАЯ_ИНСТРУКЦИЯ.md, ЗАПУСК_СЕЙЧАС.md
- ИНСТРУКЦИЯ_ЗАПУСКА.md, simple_extract.py
- run_extract.bat, run_extract.ps1, run_simple.bat
- auto_install_and_run.bat, find_and_run_python.bat
- install_and_run.ps1, promt.xml

📚 Обновлена документация:
- README.md: актуализирована структура проекта
- docs/README_ROBOFLOW_PIPELINE.md: создан индекс документации
- docs/last_chat_export_03-01-2025.md: экспорт истории чата

🎯 Результат: чистая структура проекта, все функции сохранены в улучшенных версиях"

echo.
echo 🌐 Отправляем изменения в GitHub...
git push origin main

echo.
echo ✅ Коммит завершен! Все изменения зафиксированы в Git.
echo 📊 Проект готов к работе на втором ПК.
echo.
pause