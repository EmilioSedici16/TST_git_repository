# ⚡ Быстрая настройка Git

## 1️⃣ Установка Git (если нет)
Скачать: https://git-scm.com/download/win

## 2️⃣ Настройка (один раз)
```bash
git config --global user.name "Ваше Имя"
git config --global user.email "your.email@example.com"
```

## 3️⃣ Инициализация проекта
```bash
git init
git add .
git commit -m "Первый коммит"
```

## 4️⃣ Создание репозитория на GitHub
1. Зайти на github.com
2. New repository → `TST_Project`
3. Create repository

## 5️⃣ Подключение к GitHub
```bash
git remote add origin https://github.com/ВАШ_USERNAME/TST_Project.git
git branch -M main
git push -u origin main
```

## 📱 Ежедневная работа
```bash
# Перед работой
git pull origin main

# После работы  
git add .
git commit -m "Описание изменений"
git push origin main
```

**👀 Полная инструкция:** см. файл `GIT_SETUP_INSTRUCTIONS.md`