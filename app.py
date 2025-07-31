#!/usr/bin/env python3
"""
Веб-приложение для детекции объектов с Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
from utils import ImageProcessor, RoboflowManager
import os

# Настройка страницы
st.set_page_config(
    page_title="Детекция объектов с Roboflow",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model(model_name="yolov8n.pt"):
    """Загрузка модели с кэшированием"""
    try:
        model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

def process_image(image, model, confidence=0.5):
    """Обработка изображения и детекция объектов"""
    if model is None:
        return None, []
    
    # Конвертируем PIL в OpenCV формат
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Выполняем детекцию
    results = model(image_cv, conf=confidence)
    
    # Обрабатываем результаты
    detections = []
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class': int(cls)
                })
    
    # Рисуем результаты
    if detections:
        class_names = model.names if hasattr(model, 'names') else None
        result_image_cv = ImageProcessor.draw_bboxes(image_cv, detections, class_names)
        # Конвертируем обратно в RGB для отображения
        result_image = cv2.cvtColor(result_image_cv, cv2.COLOR_BGR2RGB)
        result_image_pil = Image.fromarray(result_image)
        return result_image_pil, detections
    
    return image, detections

def main():
    """Основная функция приложения"""
    
    # Заголовок
    st.title("🔍 Детекция объектов с компьютерным зрением")
    st.markdown("### Проект с использованием YOLOv8 и Roboflow")
    
    # Боковая панель с настройками
    st.sidebar.header("⚙️ Настройки")
    
    # Выбор модели
    model_options = {
        "YOLOv8 Nano (быстрая)": "yolov8n.pt",
        "YOLOv8 Small": "yolov8s.pt", 
        "YOLOv8 Medium": "yolov8m.pt",
        "YOLOv8 Large": "yolov8l.pt"
    }
    
    selected_model = st.sidebar.selectbox(
        "Выберите модель:",
        options=list(model_options.keys()),
        index=0
    )
    
    # Настройка уверенности
    confidence = st.sidebar.slider(
        "Порог уверенности:",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Информация о Roboflow
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Roboflow Integration")
    
    api_key = st.sidebar.text_input(
        "API ключ Roboflow:",
        type="password",
        help="Введите ваш API ключ с roboflow.com"
    )
    
    if api_key:
        os.environ['ROBOFLOW_API_KEY'] = api_key
        st.sidebar.success("✅ API ключ установлен")
    
    # Основной интерфейс
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Загрузка изображения")
        
        # Варианты загрузки
        upload_option = st.radio(
            "Выберите способ загрузки:",
            ["Загрузить файл", "URL изображения", "Пример изображения"]
        )
        
        uploaded_image = None
        
        if upload_option == "Загрузить файл":
            uploaded_file = st.file_uploader(
                "Выберите изображение:",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Поддерживаемые форматы: JPG, PNG, BMP"
            )
            
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file)
        
        elif upload_option == "URL изображения":
            image_url = st.text_input(
                "Введите URL изображения:",
                placeholder="https://example.com/image.jpg"
            )
            
            if image_url:
                try:
                    import requests
                    response = requests.get(image_url)
                    uploaded_image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    st.error(f"Ошибка загрузки изображения: {e}")
        
        elif upload_option == "Пример изображения":
            sample_images = {
                "Люди на улице": "https://ultralytics.com/images/bus.jpg",
                "Животные": "https://ultralytics.com/images/dog.jpg", 
                "Городская сцена": "https://ultralytics.com/images/city.jpg"
            }
            
            selected_sample = st.selectbox(
                "Выберите пример:",
                options=list(sample_images.keys())
            )
            
            if st.button("Загрузить пример"):
                try:
                    import requests
                    response = requests.get(sample_images[selected_sample])
                    uploaded_image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    st.error(f"Ошибка загрузки примера: {e}")
        
        # Отображение исходного изображения
        if uploaded_image:
            st.subheader("Исходное изображение:")
            st.image(uploaded_image, use_column_width=True)
            
            # Информация об изображении
            st.markdown(f"**Размер:** {uploaded_image.size[0]} x {uploaded_image.size[1]} пикселей")
    
    with col2:
        st.header("🎯 Результаты детекции")
        
        if uploaded_image:
            # Загружаем модель
            with st.spinner("Загрузка модели..."):
                model = load_model(model_options[selected_model])
            
            if model:
                # Выполняем детекцию
                with st.spinner("Обработка изображения..."):
                    result_image, detections = process_image(uploaded_image, model, confidence)
                
                if result_image:
                    st.subheader("Результат детекции:")
                    st.image(result_image, use_column_width=True)
                    
                    # Статистика
                    st.subheader("📊 Статистика:")
                    st.metric("Найдено объектов", len(detections))
                    
                    if detections:
                        # Детальная информация о найденных объектах
                        st.subheader("🏷️ Найденные объекты:")
                        
                        class_names = model.names if hasattr(model, 'names') else {}
                        
                        for i, detection in enumerate(detections):
                            class_id = detection['class']
                            confidence_val = detection['confidence']
                            class_name = class_names.get(class_id, f"Class {class_id}")
                            
                            st.write(f"**{i+1}.** {class_name} - {confidence_val:.2%}")
                        
                        # График распределения классов
                        if len(detections) > 1:
                            st.subheader("📈 Распределение классов:")
                            
                            class_counts = {}
                            for detection in detections:
                                class_id = detection['class']
                                class_name = class_names.get(class_id, f"Class {class_id}")
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            
                            st.bar_chart(class_counts)
                    
                    # Кнопка скачивания
                    if result_image:
                        buf = io.BytesIO()
                        result_image.save(buf, format='PNG')
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="💾 Скачать результат",
                            data=byte_im,
                            file_name="detection_result.png",
                            mime="image/png"
                        )
        else:
            st.info("👆 Загрузите изображение для начала детекции")
    
    # Нижняя панель с информацией
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚀 Возможности")
        st.markdown("""
        - Детекция 80+ классов объектов
        - Настраиваемый порог уверенности
        - Поддержка различных форматов
        - Интеграция с Roboflow
        """)
    
    with col2:
        st.markdown("### 🛠️ Технологии")
        st.markdown("""
        - **YOLOv8** для детекции
        - **Streamlit** для веб-интерфейса
        - **OpenCV** для обработки изображений
        - **Roboflow** для управления данными
        """)
    
    with col3:
        st.markdown("### 📖 Инструкция")
        st.markdown("""
        1. Загрузите изображение
        2. Настройте параметры
        3. Дождитесь результатов
        4. Скачайте обработанное изображение
        """)

if __name__ == "__main__":
    main()