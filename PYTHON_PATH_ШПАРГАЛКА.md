# ⚡ Python PATH - Быстрая шпаргалка

## 🚨 Проблема
```
'python' не является внутренней или внешней командой
```

## 🚀 Быстрое решение

### 1️⃣ Автоматически:
```
ДОБАВИТЬ_PYTHON_В_PATH.bat
```

### 2️⃣ PowerShell:
```
Add-Python-To-PATH.ps1
```

### 3️⃣ Вручную:
```
Win + X → Система → Дополнительные параметры → 
Переменные среды → PATH → Добавить:

C:\Users\ВАШ_ИМЯ\AppData\Local\Programs\Python\Python311
C:\Users\ВАШ_ИМЯ\AppData\Local\Programs\Python\Python311\Scripts
```

## 🔍 Найти Python

### Командная строка:
```cmd
where python
dir C:\ /s /b python.exe
```

### PowerShell:
```powershell
Get-ChildItem C:\ -name python.exe -Recurse -ErrorAction SilentlyContinue
```

### Типичные места:
```
C:\Python39\
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\
%LOCALAPPDATA%\Microsoft\WindowsApps\
```

## ✅ Проверка

### Команды для теста:
```cmd
python --version
pip --version
python -c "print('OK')"
```

### Через скрипт:
```
test_python.py
```

## 🔧 Если не работает

### Перезапуск:
1. Закрыть терминал
2. Открыть новый терминал  
3. Перезапустить Cursor
4. Перезагрузить ПК

### Переустановка:
1. Удалить Python
2. Скачать с python.org
3. **Отметить "Add to PATH"**
4. Установить

## 💡 Быстрые команды

```cmd
# Временно добавить
set PATH=%PATH%;C:\путь\к\python

# Постоянно (пользователь)
setx PATH "%PATH%;C:\путь\к\python"

# Постоянно (система, админ)
setx PATH "%PATH%;C:\путь\к\python" /M
```

## 📋 Следующие шаги

После настройки PATH:
1. ✅ `УСТАНОВИТЬ_БИБЛИОТЕКИ.bat`
2. ✅ `test_python.py`
3. ✅ `ИЗВЛЕЧЬ_30_КАДРОВ.bat`

---

**🎯 Главное:** Перезапустите терминал после изменения PATH!