# 🐍 Добавление Python в PATH

## 🎯 Проблема
Python установлен, но команды `python` и `pip` не работают в терминале. Это означает, что Python не добавлен в переменную PATH.

## 🚀 Быстрое решение

### ⚡ Автоматический способ:
```
Дважды кликните: ДОБАВИТЬ_PYTHON_В_PATH.bat
```

Скрипт:
- 🔍 Найдёт установленный Python
- 🔧 Автоматически добавит в PATH
- 🧪 Протестирует работоспособность

## 🔧 Ручные способы

### 📋 Способ 1: Через Windows UI
1. Нажмите `Win + X` → "Система"
2. "Дополнительные параметры системы"
3. "Переменные среды" 
4. В "Системные переменные" найдите `PATH`
5. Нажмите "Изменить" → "Создать"
6. Добавьте пути:
   ```
   C:\Users\ВАШ_ПОЛЬЗОВАТЕЛЬ\AppData\Local\Programs\Python\Python311
   C:\Users\ВАШ_ПОЛЬЗОВАТЕЛЬ\AppData\Local\Programs\Python\Python311\Scripts
   ```
7. OK → OK → Перезагрузка

### 💻 Способ 2: Через командную строку (от администратора)
```cmd
# Найти Python
where python
dir "C:\Users\%USERNAME%\AppData\Local\Programs\Python" /s /b python.exe

# Добавить в PATH (замените на ваш путь)
setx PATH "%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\Scripts" /M
```

### 🔄 Способ 3: Переустановка Python
1. Скачайте Python с https://python.org/downloads
2. **ВАЖНО:** При установке отметьте "Add to PATH"
3. Выберите "Customize installation"
4. Отметьте "Add Python to environment variables"
5. Завершите установку

## 🔍 Проверка установки

### ✅ Тест в командной строке:
```cmd
python --version
pip --version
python -c "print('Python работает!')"
```

### 🧪 Тест через скрипт:
```
Дважды кликните: test_python.py
```

## 📍 Типичные пути Python

### Windows 10/11:
```
# Стандартная установка
C:\Python39\
C:\Python310\
C:\Python311\

# Пользовательская установка
C:\Users\ВАШ_ИМЯ\AppData\Local\Programs\Python\Python311\

# Microsoft Store
C:\Users\ВАШ_ИМЯ\AppData\Local\Microsoft\WindowsApps\

# Anaconda
C:\Users\ВАШ_ИМЯ\anaconda3\
```

### Что добавлять в PATH:
```
# Основная папка Python
C:\путь\к\python\

# Папка со скриптами pip
C:\путь\к\python\Scripts\
```

## 🚨 Решение проблем

### ❌ "python не найден" после добавления
- Перезапустите терминал
- Перезапустите Cursor  
- Перезагрузите компьютер

### ❌ "Отказано в доступе" при автоматическом добавлении
- Запустите батник от имени администратора:
  - ПКМ на `ДОБАВИТЬ_PYTHON_В_PATH.bat`
  - "Запуск от имени администратора"

### ❌ Несколько версий Python
```cmd
# Посмотреть все установленные Python
where python /R C:\

# Выбрать нужную версию для PATH
```

### ❌ Microsoft Store Python не работает
- Установите стандартный Python с python.org
- Или включите Microsoft Store Python в PATH:
  ```
  %LOCALAPPDATA%\Microsoft\WindowsApps
  ```

## 💡 Полезные команды

### 🔍 Диагностика:
```cmd
# Показать текущий PATH
echo %PATH%

# Найти Python в системе  
where python
dir C:\ /s /b python.exe

# Проверить переменные среды
set | findstr PYTHON
```

### 🔧 Управление PATH:
```cmd
# Временно добавить в PATH (до закрытия окна)
set PATH=%PATH%;C:\путь\к\python

# Постоянно добавить в PATH пользователя
setx PATH "%PATH%;C:\путь\к\python"

# Постоянно добавить в системный PATH (админ)
setx PATH "%PATH%;C:\путь\к\python" /M
```

## 📋 Проверочный список

После настройки PATH:

- [ ] ✅ `python --version` работает
- [ ] ✅ `pip --version` работает
- [ ] ✅ `python -c "print('test')"` работает
- [ ] ✅ `test_python.py` показывает успех
- [ ] ✅ Можно установить библиотеки через pip

## 🎯 Следующие шаги

После успешной настройки PATH:

1. **Установите библиотеки:**
   ```
   УСТАНОВИТЬ_БИБЛИОТЕКИ.bat
   ```

2. **Проверьте систему:**
   ```
   test_python.py
   ```

3. **Настройте облако:**
   ```
   НАСТРОИТЬ_ОБЛАКО.bat
   ```

4. **Извлеките кадры:**
   ```
   ИЗВЛЕЧЬ_30_КАДРОВ.bat
   ```

## ⚡ Быстрое решение

**Если ничего не помогает:**

1. Переустановите Python:
   - Удалите текущий Python
   - Скачайте с https://python.org/downloads
   - **Обязательно отметьте "Add to PATH"**
   - Установите и перезагрузитесь

2. Используйте абсолютные пути:
   ```cmd
   C:\путь\к\python\python.exe --version
   C:\путь\к\python\Scripts\pip.exe install opencv-python
   ```

---

**🚀 Рекомендация:** Используйте `ДОБАВИТЬ_PYTHON_В_PATH.bat` - он автоматически найдёт Python и настроит PATH!