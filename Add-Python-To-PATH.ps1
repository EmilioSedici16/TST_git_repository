# PowerShell скрипт для добавления Python в PATH
# Автор: TST Project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    Добавление Python в PATH" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Функция для поиска Python
function Find-PythonInstallations {
    $pythonPaths = @()
    
    # Стандартные пути установки
    $searchPaths = @(
        "C:\Python*",
        "C:\Program Files\Python*", 
        "C:\Program Files (x86)\Python*",
        "$env:LOCALAPPDATA\Programs\Python\Python*",
        "$env:APPDATA\Local\Programs\Python\Python*",
        "$env:USERPROFILE\AppData\Local\Programs\Python\Python*",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps"
    )
    
    Write-Host "🔍 Поиск установленного Python..." -ForegroundColor Blue
    
    foreach ($searchPath in $searchPaths) {
        $found = Get-ChildItem -Path $searchPath -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "python" }
        foreach ($dir in $found) {
            $pythonExe = Join-Path $dir.FullName "python.exe"
            if (Test-Path $pythonExe) {
                $pythonPaths += @{
                    Path = $dir.FullName
                    Executable = $pythonExe
                    Version = & $pythonExe --version 2>$null
                }
                Write-Host "✅ Найден: $($dir.FullName)" -ForegroundColor Green
            }
        }
    }
    
    # Поиск через реестр
    try {
        $regPaths = Get-ChildItem "HKLM:\SOFTWARE\Python\PythonCore" -ErrorAction SilentlyContinue
        foreach ($regPath in $regPaths) {
            $installPath = Get-ItemProperty "$($regPath.PSPath)\InstallPath" -ErrorAction SilentlyContinue
            if ($installPath -and $installPath."(default)") {
                $pythonExe = Join-Path $installPath."(default)" "python.exe"
                if (Test-Path $pythonExe) {
                    $pythonPaths += @{
                        Path = $installPath."(default)"
                        Executable = $pythonExe
                        Version = & $pythonExe --version 2>$null
                    }
                    Write-Host "✅ Найден в реестре: $($installPath."(default)")" -ForegroundColor Green
                }
            }
        }
    } catch {
        Write-Host "⚠️ Не удалось проверить реестр" -ForegroundColor Yellow
    }
    
    return $pythonPaths
}

# Функция для проверки PATH
function Test-PythonInPath {
    param($pythonPath)
    
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [Environment]::GetEnvironmentVariable("PATH", "User")
    return $currentPath -like "*$pythonPath*"
}

# Функция для добавления в PATH
function Add-ToPath {
    param(
        [string]$PathToAdd,
        [string]$Scope = "User"  # User или Machine
    )
    
    try {
        $currentPath = [Environment]::GetEnvironmentVariable("PATH", $Scope)
        
        if ($currentPath -notlike "*$PathToAdd*") {
            $newPath = "$currentPath;$PathToAdd"
            [Environment]::SetEnvironmentVariable("PATH", $newPath, $Scope)
            return $true
        } else {
            Write-Host "⚠️ Путь уже есть в PATH: $PathToAdd" -ForegroundColor Yellow
            return $true
        }
    } catch {
        Write-Host "❌ Ошибка добавления в PATH: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Основная логика
try {
    # Поиск Python
    $pythonInstallations = Find-PythonInstallations
    
    if ($pythonInstallations.Count -eq 0) {
        Write-Host "❌ Python не найден в системе!" -ForegroundColor Red
        Write-Host ""
        Write-Host "💡 Решения:" -ForegroundColor Yellow
        Write-Host "   1. Установите Python с https://python.org/downloads" -ForegroundColor Gray
        Write-Host "   2. При установке отметьте 'Add to PATH'" -ForegroundColor Gray
        Write-Host "   3. Перезагрузите компьютер после установки" -ForegroundColor Gray
        Read-Host "Нажмите Enter для выхода"
        exit 1
    }
    
    Write-Host ""
    Write-Host "📍 Найденные установки Python:" -ForegroundColor Blue
    Write-Host "-" * 40
    
    for ($i = 0; $i -lt $pythonInstallations.Count; $i++) {
        $python = $pythonInstallations[$i]
        Write-Host "$($i + 1). $($python.Path)" -ForegroundColor White
        Write-Host "   Версия: $($python.Version)" -ForegroundColor Gray
        Write-Host "   В PATH: $(if (Test-PythonInPath $python.Path) { '✅ Да' } else { '❌ Нет' })" -ForegroundColor Gray
        Write-Host ""
    }
    
    # Выбор Python для добавления
    if ($pythonInstallations.Count -eq 1) {
        $selectedPython = $pythonInstallations[0]
        Write-Host "🎯 Использую единственную найденную установку" -ForegroundColor Green
    } else {
        do {
            $choice = Read-Host "Выберите номер Python для добавления в PATH (1-$($pythonInstallations.Count))"
            $choiceNum = [int]$choice - 1
        } while ($choiceNum -lt 0 -or $choiceNum -ge $pythonInstallations.Count)
        
        $selectedPython = $pythonInstallations[$choiceNum]
    }
    
    Write-Host ""
    Write-Host "🎯 Выбранный Python:" -ForegroundColor Green
    Write-Host "   Путь: $($selectedPython.Path)" -ForegroundColor Gray
    Write-Host "   Версия: $($selectedPython.Version)" -ForegroundColor Gray
    Write-Host ""
    
    # Проверка текущего состояния PATH
    $pythonInPath = Test-PythonInPath $selectedPython.Path
    $scriptsPath = Join-Path $selectedPython.Path "Scripts"
    $scriptsInPath = Test-PythonInPath $scriptsPath
    
    if ($pythonInPath -and $scriptsInPath) {
        Write-Host "✅ Python уже правильно настроен в PATH!" -ForegroundColor Green
        Write-Host "💡 Возможно, нужно перезапустить терминал или компьютер" -ForegroundColor Yellow
    } else {
        Write-Host "🔧 Добавление Python в PATH..." -ForegroundColor Blue
        Write-Host ""
        
        # Выбор области видимости
        Write-Host "Выберите область видимости:" -ForegroundColor Yellow
        Write-Host "1. Текущий пользователь (рекомендуется)" -ForegroundColor White
        Write-Host "2. Вся система (требует админ прав)" -ForegroundColor White
        
        do {
            $scopeChoice = Read-Host "Выберите (1 или 2)"
        } while ($scopeChoice -ne "1" -and $scopeChoice -ne "2")
        
        $scope = if ($scopeChoice -eq "2") { "Machine" } else { "User" }
        $scopeName = if ($scopeChoice -eq "2") { "системный" } else { "пользовательский" }
        
        Write-Host ""
        Write-Host "🔧 Добавление в $scopeName PATH..." -ForegroundColor Blue
        
        $success = $true
        
        # Добавить основной путь Python
        if (-not $pythonInPath) {
            Write-Host "   Добавляю: $($selectedPython.Path)" -ForegroundColor Gray
            $success = $success -and (Add-ToPath -PathToAdd $selectedPython.Path -Scope $scope)
        }
        
        # Добавить путь Scripts
        if (-not $scriptsInPath -and (Test-Path $scriptsPath)) {
            Write-Host "   Добавляю: $scriptsPath" -ForegroundColor Gray
            $success = $success -and (Add-ToPath -PathToAdd $scriptsPath -Scope $scope)
        }
        
        if ($success) {
            Write-Host ""
            Write-Host "✅ Python успешно добавлен в PATH!" -ForegroundColor Green
            Write-Host ""
            Write-Host "🔄 Для применения изменений:" -ForegroundColor Yellow
            Write-Host "   1. Перезапустите терминал/PowerShell" -ForegroundColor Gray
            Write-Host "   2. Перезапустите Cursor" -ForegroundColor Gray
            Write-Host "   3. При необходимости перезагрузите компьютер" -ForegroundColor Gray
        } else {
            Write-Host ""
            Write-Host "❌ Ошибки при добавлении в PATH" -ForegroundColor Red
            Write-Host "💡 Попробуйте запустить от имени администратора" -ForegroundColor Yellow
        }
    }
    
    # Тестирование
    Write-Host ""
    Write-Host "🧪 Тестирование доступности Python..." -ForegroundColor Blue
    Write-Host "-" * 40
    
    # Прямой вызов
    try {
        $version = & $selectedPython.Executable --version 2>&1
        Write-Host "✅ Прямой вызов работает: $version" -ForegroundColor Green
    } catch {
        Write-Host "❌ Прямой вызов не работает" -ForegroundColor Red
    }
    
    # Вызов через PATH (может не работать до перезапуска)
    try {
        $pathVersion = python --version 2>&1
        Write-Host "✅ Вызов через PATH: $pathVersion" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Вызов через PATH не работает (перезапустите терминал)" -ForegroundColor Yellow
    }
    
    # Проверка pip
    $pipPath = Join-Path $selectedPython.Path "Scripts\pip.exe"
    if (Test-Path $pipPath) {
        try {
            $pipVersion = & $pipPath --version 2>&1
            Write-Host "✅ pip доступен: $pipVersion" -ForegroundColor Green
        } catch {
            Write-Host "⚠️ pip найден, но есть проблемы с запуском" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ pip не найден" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "              РЕЗУЛЬТАТ" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    if ($success) {
        Write-Host "🎉 Настройка завершена успешно!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 Следующие шаги:" -ForegroundColor Yellow
        Write-Host "   1. Перезапустите терминал/Cursor" -ForegroundColor Gray
        Write-Host "   2. Запустите: УСТАНОВИТЬ_БИБЛИОТЕКИ.bat" -ForegroundColor Gray
        Write-Host "   3. Проверьте: test_python.py" -ForegroundColor Gray
        Write-Host "   4. Извлеките кадры: ИЗВЛЕЧЬ_30_КАДРОВ.bat" -ForegroundColor Gray
        Write-Host ""
        Write-Host "🔧 Добавленные пути:" -ForegroundColor Blue
        Write-Host "   • $($selectedPython.Path)" -ForegroundColor Gray
        Write-Host "   • $scriptsPath" -ForegroundColor Gray
    } else {
        Write-Host "⚠️ Настройка завершена с проблемами" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "💡 Альтернативные решения:" -ForegroundColor Yellow
        Write-Host "   1. Переустановите Python с галочкой 'Add to PATH'" -ForegroundColor Gray
        Write-Host "   2. Добавьте пути вручную через системные настройки" -ForegroundColor Gray
        Write-Host "   3. Используйте полные пути к python.exe" -ForegroundColor Gray
    }
    
} catch {
    Write-Host "❌ Критическая ошибка: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "📋 Полная информация об ошибке:" -ForegroundColor Yellow
    Write-Host $_.Exception -ForegroundColor Gray
}

Write-Host ""
Read-Host "Нажмите Enter для завершения"