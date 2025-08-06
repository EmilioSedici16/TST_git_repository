# Настройка Git для TST_Project
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    Настройка Git для TST_Project" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Проверка наличия Git
try {
    $gitVersion = git --version
    Write-Host "✅ Git найден: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git не найден!" -ForegroundColor Red
    Write-Host "📥 Скачайте Git с: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "🔄 После установки перезапустите этот скрипт" -ForegroundColor Yellow
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

# Проверка, инициализирован ли уже Git
if (Test-Path ".git") {
    Write-Host "ℹ️  Git репозиторий уже инициализирован" -ForegroundColor Yellow
} else {
    Write-Host "🚀 Инициализация Git репозитория..." -ForegroundColor Blue
    git init
    
    Write-Host "📁 Добавление файлов в репозиторий..." -ForegroundColor Blue
    git add .
    
    Write-Host "💾 Создание первого коммита..." -ForegroundColor Blue
    git commit -m "Первый коммит: настройка проекта TST_Project"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "       Следующие шаги:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Создайте репозиторий на GitHub:" -ForegroundColor White
Write-Host "   - Перейдите на https://github.com" -ForegroundColor Gray
Write-Host "   - Нажмите 'New repository'" -ForegroundColor Gray
Write-Host "   - Назовите: TST_Project" -ForegroundColor Gray
Write-Host "   - НЕ добавляйте README, .gitignore" -ForegroundColor Gray
Write-Host "   - Нажмите 'Create repository'" -ForegroundColor Gray
Write-Host ""

$username = Read-Host "2. Введите ваш GitHub username"

Write-Host ""
Write-Host "🔗 Подключение к GitHub..." -ForegroundColor Blue

try {
    git remote add origin "https://github.com/$username/TST_Project.git"
    git branch -M main
    
    Write-Host "📤 Отправка кода на GitHub..." -ForegroundColor Blue
    git push -u origin main
    
    Write-Host ""
    Write-Host "✅ Успешно! Репозиторий настроен" -ForegroundColor Green
    Write-Host "🌐 Ваш проект: https://github.com/$username/TST_Project" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Для ежедневной работы используйте:" -ForegroundColor Yellow
    Write-Host "   git pull origin main    (перед работой)" -ForegroundColor Gray
    Write-Host "   git add ." -ForegroundColor Gray
    Write-Host "   git commit -m ""описание""" -ForegroundColor Gray
    Write-Host "   git push origin main    (после работы)" -ForegroundColor Gray
    
} catch {
    Write-Host ""
    Write-Host "❌ Ошибка при настройке GitHub" -ForegroundColor Red
    Write-Host "🔧 Проверьте:" -ForegroundColor Yellow
    Write-Host "   - Правильность username" -ForegroundColor Gray
    Write-Host "   - Создан ли репозиторий на GitHub" -ForegroundColor Gray
    Write-Host "   - Настроена ли аутентификация Git" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Для настройки аутентификации Git:" -ForegroundColor Yellow
    Write-Host "git config --global user.name ""Ваше Имя""" -ForegroundColor Gray
    Write-Host "git config --global user.email ""your.email@example.com""" -ForegroundColor Gray
}

Write-Host ""
Read-Host "Нажмите Enter для завершения"