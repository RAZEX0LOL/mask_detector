"""
Mask Detection System - Real-time Face Mask Detection Application

Это приложение использует глубокое обучение для обнаружения масок на лицах
в режиме реального времени через веб-камеру.

Архитектура:
    - Face Detection: SSD (Single Shot Detector) на базе ResNet-10
    - Mask Classification: MobileNetV2 с Transfer Learning
    - UI: Tkinter GUI

Паттерны проектирования:
    - MVC (Model-View-Controller): разделение логики и представления
    - Singleton: единственный экземпляр VideoStream

Авторы: Хаттаев Расул, Замараева Ксения, Буров Владислав
Дата: 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== КОНСТАНТЫ ====================
# Пути к моделям (кроссплатформенные)
FACE_DETECTOR_PROTOTXT = Path("face_detector") / "deploy.prototxt"
FACE_DETECTOR_WEIGHTS = Path("face_detector") / "res10_300x300_ssd_iter_140000.caffemodel"
MASK_DETECTOR_MODEL = Path("mask_detector.keras")

# Параметры обнаружения лиц
FACE_CONFIDENCE_THRESHOLD = 0.5  # Минимальная уверенность для детекции лица
FACE_INPUT_SIZE = (224, 224)     # Размер входного изображения для SSD
FACE_MEAN_VALUES = (104.0, 177.0, 123.0)  # Средние значения для нормализации

# Параметры классификации масок
MASK_INPUT_SIZE = (224, 224)     # Размер входного изображения для MobileNetV2
MASK_BATCH_SIZE = 32             # Размер батча для предсказаний

# Параметры видео
VIDEO_WIDTH = 800                # Ширина обрабатываемого кадра
FRAME_UPDATE_DELAY = 10          # Задержка между обновлениями кадров (мс)
CAMERA_WARMUP_TIME = 2.0         # Время прогрева камеры (сек)

# Параметры визуализации
COLOR_MASK = (0, 255, 0)         # Зеленый цвет для "с маской"
COLOR_NO_MASK = (0, 0, 255)      # Красный цвет для "без маски"
FONT_SCALE = 0.45                # Размер шрифта для меток
FONT_THICKNESS = 2               # Толщина текста
BOX_THICKNESS = 2                # Толщина рамки


class MaskDetectionApp:
    """
    Главный класс приложения для обнаружения масок на лицах.

    Реализует паттерн MVC, где:
        - Model: нейронные сети (faceNet, maskNet)
        - View: GUI компоненты (Tkinter)
        - Controller: методы обработки событий

    Attributes:
        root (tk.Tk): Главное окно приложения
        vs (VideoStream): Поток видео с камеры
        faceNet (cv2.dnn.Net): Модель для обнаружения лиц
        maskNet (keras.Model): Модель для классификации масок
        is_running (bool): Флаг активности детекции
        current_frame (np.ndarray): Текущий кадр видео
    """
    def __init__(self, root):
        """
        Инициализация приложения для обнаружения масок.

        Args:
            root (tk.Tk): Корневое окно Tkinter

        Raises:
            Exception: При ошибке загрузки моделей
        """
        self.root = root
        self.root.title("Mask Detection System")
        self.root.geometry("1000x700")

        # Инициализация атрибутов
        self.vs = None                  # VideoStream объект
        self.current_frame = None       # Текущий кадр для обработки
        self.is_running = False         # Флаг состояния детекции
        self.faceNet = None             # Модель детекции лиц
        self.maskNet = None             # Модель классификации масок

        # Создание GUI компонентов
        self.create_widgets()

        # Загрузка предобученных моделей
        self.load_models()

        logger.info("Приложение инициализировано успешно")

    def create_widgets(self):
        """
        Создание виджетов пользовательского интерфейса.

        Создает следующие компоненты:
            - Заголовок приложения
            - Фрейм для видеопотока
            - Кнопки управления (Запуск, Остановка, Выход)
            - Строка состояния

        Паттерн: Builder - пошаговое создание сложного объекта (GUI)
        """
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Заголовок приложения
        title_label = ttk.Label(main_frame, text="Система обнаружения масок",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Фрейм для отображения видео
        self.video_frame = ttk.LabelFrame(main_frame, text="Видеопоток", padding="10")
        self.video_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))

        # Label для отображения кадров видео
        self.video_label = ttk.Label(self.video_frame, text="Нажмите 'Запуск' для начала работы",
                                     background="black", foreground="white")
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        # Панель управления с кнопками
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # Кнопка запуска детекции
        self.start_button = ttk.Button(button_frame, text="Запуск",
                                       command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=10)

        # Кнопка остановки детекции
        self.stop_button = ttk.Button(button_frame, text="Остановка",
                                      command=self.stop_detection,
                                      state="disabled")
        self.stop_button.grid(row=0, column=1, padx=10)

        # Кнопка выхода из приложения
        self.quit_button = ttk.Button(button_frame, text="Выход",
                                      command=self.quit_app)
        self.quit_button.grid(row=0, column=2, padx=10)

        # Строка состояния (статус-бар)
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief="sunken", anchor="w")
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Настройка масштабирования компонентов при изменении размера окна
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        self.video_frame.columnconfigure(0, weight=1)
        self.video_frame.rowconfigure(0, weight=1)

        logger.info("GUI компоненты созданы успешно")

    def load_models(self):
        """
        Загрузка предобученных моделей нейронных сетей.

        Загружает две модели:
            1. Face Detector (SSD): для обнаружения лиц на изображении
            2. Mask Classifier (MobileNetV2): для классификации наличия маски

        Raises:
            Exception: Если файлы моделей не найдены или повреждены

        Note:
            Пути к моделям определены в константах:
            - FACE_DETECTOR_PROTOTXT
            - FACE_DETECTOR_WEIGHTS
            - MASK_DETECTOR_MODEL
        """
        try:
            self.status_var.set("Загрузка моделей...")
            logger.info("Начало загрузки моделей...")

            # Загрузка модели детекции лиц (SSD на базе Caffe)
            logger.info(f"Загрузка Face Detector из {FACE_DETECTOR_PROTOTXT}")
            self.faceNet = cv2.dnn.readNet(
                str(FACE_DETECTOR_PROTOTXT),
                str(FACE_DETECTOR_WEIGHTS)
            )

            # Загрузка модели классификации масок (MobileNetV2)
            logger.info(f"Загрузка Mask Classifier из {MASK_DETECTOR_MODEL}")
            self.maskNet = load_model(str(MASK_DETECTOR_MODEL))

            self.status_var.set("Модели загружены успешно")
            logger.info("Все модели загружены успешно")

        except Exception as e:
            error_msg = f"Не удалось загрузить модели: {str(e)}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Ошибка", error_msg)
            self.status_var.set("Ошибка загрузки моделей")
            raise

    def detect_and_predict_mask(self, frame, faceNet, maskNet):
        """
        Обнаружение лиц и классификация масок на кадре.

        Алгоритм работы:
            1. Обнаружение лиц с помощью SSD (faceNet)
            2. Извлечение регионов с лицами (ROI)
            3. Предобработка для MobileNetV2
            4. Классификация наличия маски (maskNet)

        Args:
            frame (np.ndarray): Входной кадр в формате BGR
            faceNet (cv2.dnn.Net): Модель для детекции лиц
            maskNet (keras.Model): Модель для классификации масок

        Returns:
            tuple: (locations, predictions)
                - locations (list): Координаты обнаруженных лиц [(x1,y1,x2,y2), ...]
                - predictions (list): Вероятности классов [(mask_prob, no_mask_prob), ...]

        Note:
            Использует порог уверенности FACE_CONFIDENCE_THRESHOLD для фильтрации
            слабых детекций лиц.
        """
        # Получение размеров кадра
        (h, w) = frame.shape[:2]

        # Шаг 1: Подготовка blob для детекции лиц
        # blobFromImage выполняет: scale, resize, mean subtraction
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, FACE_INPUT_SIZE, FACE_MEAN_VALUES
        )
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # Инициализация списков для хранения результатов
        faces = []      # Предобработанные изображения лиц
        locs = []       # Координаты обнаруженных лиц
        preds = []      # Предсказания модели

        # Шаг 2: Обработка каждой детекции лица
        for i in range(0, detections.shape[2]):
            # Уверенность детекции (от 0 до 1)
            confidence = detections[0, 0, i, 2]

            # Фильтрация слабых детекций
            if confidence > FACE_CONFIDENCE_THRESHOLD:
                # Извлечение координат bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Обеспечение нахождения координат в пределах кадра
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # Шаг 3: Извлечение ROI (Region of Interest) - области лица
                face = frame[startY:endY, startX:endX]

                # Проверка валидности извлеченной области
                if face.shape[0] > 0 and face.shape[1] > 0:
                    # Предобработка для MobileNetV2:
                    # 1. Конвертация BGR -> RGB
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    # 2. Изменение размера
                    face = cv2.resize(face, MASK_INPUT_SIZE)
                    # 3. Конвертация в массив
                    face = img_to_array(face)
                    # 4. Нормализация (ImageNet preprocessing)
                    face = preprocess_input(face)

                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

        # Шаг 4: Батч-предсказание для всех обнаруженных лиц
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            # Предсказание вероятностей [mask, without_mask]
            preds = maskNet.predict(faces, batch_size=MASK_BATCH_SIZE, verbose=0)

        return (locs, preds)

    def start_detection(self):
        """
        Запуск процесса обнаружения масок в реальном времени.

        Инициализирует видеопоток с веб-камеры и запускает цикл обработки кадров.
        Паттерн: Command - инкапсуляция запроса как объекта
        """
        if not self.is_running:
            self.is_running = True
            # Изменение состояния кнопок
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_var.set("Запуск видеопотока...")

            # Инициализация видеопотока (src=0 - первая камера)
            logger.info("Инициализация VideoStream...")
            self.vs = VideoStream(src=0).start()

            # Пауза для прогрева камеры
            time.sleep(CAMERA_WARMUP_TIME)

            # Запуск цикла обновления кадров
            self.update_frame()
            logger.info("Детекция запущена")

    def stop_detection(self):
        """
        Остановка процесса обнаружения масок.

        Останавливает видеопоток и освобождает ресурсы.
        """
        self.is_running = False
        # Возврат кнопок в исходное состояние
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Остановлено")

        # Остановка видеопотока
        if self.vs:
            self.vs.stop()
            logger.info("VideoStream остановлен")

        # Очистка видео label
        self.video_label.config(image='', text="Нажмите 'Запуск' для начала работы")

    def quit_app(self):
        """
        Завершение работы приложения.

        Показывает диалог подтверждения и корректно закрывает все ресурсы.
        """
        if messagebox.askokcancel("Выход", "Вы уверены, что хотите выйти?"):
            logger.info("Завершение работы приложения")
            self.stop_detection()
            self.root.destroy()

    def update_frame(self):
        """
        Обновление кадра видео в реальном времени.

        Выполняет следующие операции в цикле:
            1. Захват кадра с камеры
            2. Обнаружение лиц на кадре
            3. Классификация масок
            4. Отрисовка результатов (bounding boxes + labels)
            5. Отображение в GUI

        Паттерн: Observer - автоматическое обновление при изменении данных

        Note:
            Метод рекурсивно вызывает сам себя через root.after() для создания
            непрерывного цикла обработки.
        """
        if self.is_running and self.vs:
            try:
                # Шаг 1: Захват кадра с камеры
                frame = self.vs.read()

                if frame is not None:
                    # Шаг 2: Изменение размера для оптимизации производительности
                    frame = imutils.resize(frame, width=VIDEO_WIDTH)

                    # Шаг 3: Обнаружение лиц и классификация масок
                    (locs, preds) = self.detect_and_predict_mask(
                        frame, self.faceNet, self.maskNet
                    )

                    # Шаг 4: Отрисовка результатов на каждом обнаруженном лице
                    for (box, pred) in zip(locs, preds):
                        # Распаковка координат и предсказаний
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred

                        # Определение класса и цвета
                        label = "Mask" if mask > withoutMask else "No Mask"
                        color = COLOR_MASK if label == "Mask" else COLOR_NO_MASK

                        # Форматирование метки с вероятностью
                        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                        # Отрисовка текста и bounding box
                        cv2.putText(
                            frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS
                        )
                        cv2.rectangle(
                            frame, (startX, startY), (endX, endY), color, BOX_THICKNESS
                        )

                    # Шаг 5: Конвертация для отображения в Tkinter
                    # OpenCV использует BGR, Tkinter/PIL - RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)

                    # Обновление label с изображением
                    self.video_label.configure(image=imgtk)
                    self.video_label.image = imgtk  # Сохранение ссылки для GC

                    # Обновление статуса с количеством обнаруженных лиц
                    self.status_var.set(f"Обнаружено лиц: {len(locs)}")

                # Шаг 6: Планирование следующего обновления
                if self.is_running:
                    self.root.after(FRAME_UPDATE_DELAY, self.update_frame)

            except Exception as e:
                error_msg = f"Ошибка: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.status_var.set(error_msg)
                self.stop_detection()


def main():
    """
    Точка входа приложения.

    Создает главное окно Tkinter и запускает приложение детекции масок.
    """
    logger.info("Запуск Mask Detection System...")

    # Создание главного окна
    root = tk.Tk()

    # Инициализация приложения
    app = MaskDetectionApp(root)

    # Обработка закрытия окна
    root.protocol("WM_DELETE_WINDOW", app.quit_app)

    # Запуск главного цикла событий
    root.mainloop()

    logger.info("Приложение завершено")


if __name__ == "__main__":
    main()