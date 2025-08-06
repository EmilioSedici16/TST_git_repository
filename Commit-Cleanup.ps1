# Скрипт для фиксации результатов уборки рабочего стола
Write-Host "🚀 Фиксация результатов уборки рабочего стола..." -ForegroundColor Green
Write-Host ""

Write-Host "📝 Добавляем все изменения в staging area..." -ForegroundColor Yellow
git add .

Write-Host ""
Write-Host "📦 Создаем коммит с описанием изменений..." -ForegroundColor Yellow
$commitMessage = @"
🧹 Уборка рабочего стола: удалены устаревшие файлы, обновлена документация

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

🎯 Результат: чистая структура проекта, все функции сохранены в улучшенных версиях
"@

git commit -m $commitMessage

Write-Host ""
Write-Host "🌐 Отправляем изменения в GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host ""
Write-Host "✅ Коммит завершен! Все изменения зафиксированы в Git." -ForegroundColor Green
Write-Host "📊 Проект готов к работе на втором ПК." -ForegroundColor Green
Write-Host ""
Read-Host "Нажмите Enter для завершения"