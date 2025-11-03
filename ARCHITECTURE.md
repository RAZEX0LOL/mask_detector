# Архитектура системы обнаружения масок

## ADR (Architecture Decision Records)

### ADR-001: Выбор архитектуры детекции лиц

**Статус:** Принято

**Контекст:**
Необходимо выбрать архитектуру для обнаружения лиц на изображениях в реальном времени.

**Варианты:**
1. Haar Cascades (OpenCV)
2. SSD (Single Shot Detector)
3. YOLO (You Only Look Once)
4. MTCNN (Multi-task Cascaded CNN)

**Решение:** SSD на базе ResNet-10

**Обоснование:**
- ✅ Хороший баланс точности и скорости (~20-25 FPS на CPU)
- ✅ Предобученная модель доступна в OpenCV
- ✅ Не требует GPU для real-time обработки
- ✅ Стабильная работа при различных условиях освещения
- ❌ Haar Cascades - низкая точность, много ложных срабатываний
- ❌ YOLO - требует GPU для реального времени
- ❌ MTCNN - более медленная, избыточная для нашей задачи

**Последствия:**
- Необходимость использования OpenCV DNN модуля
- Зависимость от предобученной модели Caffe
- Требуется confidence threshold для фильтрации

---

### ADR-002: Выбор архитектуры классификации масок

**Статус:** Принято

**Контекст:**
Необходимо выбрать архитектуру нейронной сети для бинарной классификации (маска/без маски).

**Варианты:**
1. Custom CNN
2. ResNet50 (Transfer Learning)
3. MobileNetV2 (Transfer Learning)
4. EfficientNet (Transfer Learning)

**Решение:** MobileNetV2 с Transfer Learning

**Обоснование:**
- ✅ Легковесная архитектура (3.4M параметров)
- ✅ Быстрая inference (~10ms на CPU)
- ✅ Оптимизирована для мобильных устройств
- ✅ Хорошо работает с небольшими датасетами через Transfer Learning
- ✅ Предобученные веса ImageNet
- ❌ Custom CNN - требует больше данных и времени на обучение
- ❌ ResNet50 - избыточная для бинарной классификации
- ❌ EfficientNet - более новая, меньше поддержки

**Последствия:**
- Зависимость от TensorFlow/Keras
- Необходимость fine-tuning на нашем датасете
- Требуется preprocessing совместимый с ImageNet

---

### ADR-003: Выбор GUI фреймворка

**Статус:** Принято

**Контекст:**
Необходимо создать графический интерфейс для демонстрации работы системы.

**Варианты:**
1. Tkinter (Built-in Python)
2. PyQt5
3. Streamlit
4. Веб-интерфейс (Flask/FastAPI)

**Решение:** Tkinter

**Обоснование:**
- ✅ Встроен в Python, не требует дополнительных зависимостей
- ✅ Простота разработки для базового GUI
- ✅ Кроссплатформенность
- ✅ Малый вес приложения
- ❌ PyQt5 - требует лицензию для коммерческого использования
- ❌ Streamlit - избыточен для локального приложения
- ❌ Веб - усложняет деплой и требует сервера

**Последствия:**
- Ограниченные возможности кастомизации UI
- Простой, но функциональный интерфейс
- Быстрая разработка

---

### ADR-004: Структура обработки видеопотока

**Статус:** Принято

**Контекст:**
Необходимо определить архитектуру обработки видеопотока в реальном времени.

**Варианты:**
1. Синхронная обработка (frame-by-frame)
2. Многопоточная обработка (threading)
3. Асинхронная обработка (asyncio)
4. Очередь задач (queue)

**Решение:** Синхронная обработка с VideoStream (threading под капотом)

**Обоснование:**
- ✅ Простота реализации
- ✅ imutils.VideoStream уже использует threading для чтения кадров
- ✅ Достаточная производительность для real-time (20+ FPS)
- ✅ Предсказуемое поведение
- ❌ Многопоточность - сложность синхронизации
- ❌ Asyncio - избыточна для данной задачи
- ❌ Queue - дополнительная сложность

**Последствия:**
- Зависимость от imutils
- Блокирующий GUI при обработке (решается через after())
- Простая отладка

---

## Компоненты системы

### 1. Face Detection Module

**Технология:** OpenCV DNN + Caffe Model

**Файлы:**
- `deploy.prototxt` - архитектура SSD
- `res10_300x300_ssd_iter_140000.caffemodel` - веса

**Алгоритм:**
```python
1. Input: BGR image (H x W x 3)
2. Preprocessing:
   - Создание blob: cv2.dnn.blobFromImage()
   - Scale: 1.0
   - Size: 300x300
   - Mean subtraction: (104, 177, 123)
3. Forward pass через SSD
4. Output: detections [1, 1, N, 7]
   - N - количество детекций
   - 7 - [image_id, label, confidence, x1, y1, x2, y2]
5. Filtering: confidence > 0.5
6. Bounding box coordinates normalization
```

**Параметры:**
- Input size: 300×300
- Confidence threshold: 0.5
- Output: List of (x1, y1, x2, y2) coordinates

---

### 2. Mask Classification Module

**Технология:** TensorFlow/Keras + MobileNetV2

**Архитектура:**
```
Input (224x224x3)
       ↓
MobileNetV2 (ImageNet pretrained)
  - Inverted Residual blocks
  - Depthwise Separable Convolutions
  - 53 layers
       ↓
GlobalAveragePooling2D
       ↓
Dense(128, activation='relu')
       ↓
Dropout(0.5)
       ↓
Dense(2, activation='softmax')
       ↓
Output: [P(mask), P(no_mask)]
```

**Обучение:**
- Optimizer: Adam (lr=1e-4)
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 20
- Batch size: 32
- Data augmentation: rotation, zoom, shift, flip

**Preprocessing:**
```python
1. BGR -> RGB conversion
2. Resize to 224x224
3. img_to_array()
4. preprocess_input() - ImageNet normalization
   - Scale: [0, 255] -> [-1, 1]
```

---

### 3. Video Stream Module

**Технология:** imutils.VideoStream (threading-based)

**Особенности:**
- Использует отдельный thread для чтения кадров
- Буферизация последнего кадра
- Автоматическое управление ресурсами

**Workflow:**
```python
# Инициализация
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Прогрев камеры

# Чтение кадров
while True:
    frame = vs.read()
    # Обработка...

# Остановка
vs.stop()
```

---

### 4. GUI Module (Tkinter)

**Компоненты:**

```
MainWindow (1000x700)
│
├── Title Label ("Система обнаружения масок")
│
├── Video Frame (LabelFrame)
│   └── Video Label (800x600)
│       └── Display area for video stream
│
├── Button Frame
│   ├── Start Button -> start_detection()
│   ├── Stop Button -> stop_detection()
│   └── Quit Button -> quit_app()
│
└── Status Bar (StringVar)
    └── "Обнаружено лиц: N"
```

**Паттерн обновления:**
```python
def update_frame():
    # 1. Обработка кадра
    # 2. Отображение результата
    # 3. Рекурсивный вызов
    if is_running:
        root.after(10, update_frame)  # 100 FPS max
```

---

## Поток данных (Data Flow)

### Полный цикл обработки

```
┌─────────────────┐
│  Веб-камера     │
│  (640x480 BGR)  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  VideoStream    │
│  • Threading    │
│  • Buffering    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Resize to 800  │
│  (imutils)      │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────┐
│  Face Detection (SSD)       │
│  • Blob creation (300x300)  │
│  • Forward pass             │
│  • Threshold filter (>0.5)  │
└────────┬────────────────────┘
         │
         ↓
┌─────────────────────────────┐
│  ROI Extraction             │
│  • For each detected face   │
│  • Crop to bounding box     │
│  • Validate dimensions      │
└────────┬────────────────────┘
         │
         ↓
┌─────────────────────────────┐
│  Preprocessing              │
│  • BGR -> RGB               │
│  • Resize to 224x224        │
│  • Normalize [-1, 1]        │
└────────┬────────────────────┘
         │
         ↓
┌─────────────────────────────┐
│  Mask Classification        │
│  • Batch prediction         │
│  • MobileNetV2 forward      │
│  • Softmax probabilities    │
└────────┬────────────────────┘
         │
         ↓
┌─────────────────────────────┐
│  Post-processing            │
│  • Determine class          │
│  • Select color (Green/Red) │
│  • Format label with %      │
└────────┬────────────────────┘
         │
         ↓
┌─────────────────────────────┐
│  Visualization              │
│  • Draw bounding boxes      │
│  • Put text labels          │
│  • BGR -> RGB for Tkinter   │
└────────┬────────────────────┘
         │
         ↓
┌─────────────────────────────┐
│  GUI Display (Tkinter)      │
│  • PhotoImage conversion    │
│  • Label update             │
│  • Status bar update        │
└─────────────────────────────┘
```

---

## Управление состоянием

### State Machine

```
┌─────────┐
│  IDLE   │ (начальное состояние)
└────┬────┘
     │ start_detection()
     ↓
┌─────────┐
│ RUNNING │ (обработка видео)
└────┬────┘
     │ stop_detection()
     ↓
┌─────────┐
│ STOPPED │
└────┬────┘
     │ start_detection()
     ↓
┌─────────┐
│ RUNNING │
└─────────┘
```

**Переменные состояния:**
```python
self.is_running = False      # Флаг активности детекции
self.vs = None               # VideoStream объект
self.current_frame = None    # Текущий кадр
self.faceNet = None          # Модель детекции лиц
self.maskNet = None          # Модель классификации масок
```

---

## Обработка ошибок

### Стратегия

1. **Загрузка моделей:**
   - Try-catch при загрузке
   - Messagebox с описанием ошибки
   - Логирование в console
   - Graceful degradation (приложение не падает)

2. **Обработка кадров:**
   - Try-catch в update_frame()
   - Автоматическая остановка при критической ошибке
   - Обновление status bar

3. **Камера:**
   - Проверка frame is not None
   - Warmup time перед началом обработки
   - Корректное освобождение ресурсов

### Логирование

```python
# Уровни логирования
logger.info()    # Информационные сообщения
logger.error()   # Ошибки с traceback
logger.warning() # Предупреждения

# Примеры
logger.info("Модели загружены успешно")
logger.error("Не удалось загрузить модели", exc_info=True)
```

---

## Оптимизация производительности

### Текущие оптимизации

1. **Resize входного кадра:**
   ```python
   frame = imutils.resize(frame, width=800)
   # Уменьшение размера -> меньше пикселей для обработки
   ```

2. **Batch prediction:**
   ```python
   preds = maskNet.predict(faces, batch_size=32)
   # Обработка всех лиц за один forward pass
   ```

3. **Threading в VideoStream:**
   - Параллельное чтение кадров
   - GUI не блокируется

4. **Константы вынесены наружу:**
   - Избегаем повторных вычислений
   - Легко настраивать параметры

### Потенциальные улучшения

1. **GPU ускорение:**
   ```python
   # TensorFlow
   with tf.device('/GPU:0'):
       preds = maskNet.predict(faces)
   ```

2. **Кэширование детекций:**
   - Обнаружение лиц реже (каждые N кадров)
   - Tracking между кадрами

3. **Quantization:**
   - TFLite модель для мобильных устройств
   - INT8 quantization

4. **Model pruning:**
   - Удаление неважных весов
   - Уменьшение размера модели

---

## Безопасность и приватность

### Текущие меры

1. **Локальная обработка:**
   - Все данные обрабатываются на устройстве
   - Нет отправки данных на сервер

2. **Не сохраняются кадры:**
   - Обработка в реальном времени
   - Кадры не записываются на диск

3. **Пользовательское согласие:**
   - Диалог подтверждения при выходе
   - Явный контроль старт/стоп

### Рекомендации для production

1. **Шифрование:**
   - Если добавляется сохранение кадров
   - Использовать encrypted storage

2. **Аудит логов:**
   - Не логировать персональные данные
   - Использовать только технические метрики

3. **Соответствие GDPR:**
   - Информирование пользователей
   - Право на удаление данных

---

## Масштабируемость

### Текущая архитектура
- Desktop приложение
- Single user
- Single camera

### Возможности расширения

#### 1. Многопользовательский режим
```
Client App <-> Server (FastAPI) <-> Database
                ↓
            ML Models
```

#### 2. Облачный деплой
```
Edge Device -> Message Queue (RabbitMQ) -> Worker Pool -> Results DB
```

#### 3. Мобильные приложения
- TFLite конвертация
- React Native / Flutter UI
- On-device inference

---

## Тестирование

### Unit тесты
```python
def test_face_detection():
    # Тест детекции лиц на тестовом изображении
    pass

def test_mask_classification():
    # Тест классификации на известных примерах
    pass

def test_model_loading():
    # Тест корректности загрузки моделей
    pass
```

### Integration тесты
```python
def test_full_pipeline():
    # Тест полного цикла: кадр -> результат
    pass
```

### Performance тесты
```python
def test_fps():
    # Измерение FPS на тестовом видео
    assert fps > 20
```

---

## Зависимости и версии

```
Python: 3.8+
├── TensorFlow: 2.0+
│   └── Keras: 2.3+
├── OpenCV: 4.2+
├── NumPy: 1.18+
├── Pillow: 8.0+
├── imutils: 0.5.3+
└── Tkinter: Built-in
```

**Критические зависимости:**
- TensorFlow/Keras для MobileNetV2
- OpenCV для SSD и видео

---

## Версионирование

- **v1.0** - Базовая функциональность
- **v2.0** - Улучшенный UI + логирование
- **v3.0** - Полная документация + рефакторинг (текущая)

---

**Дата создания:** 2025-11-03
**Последнее обновление:** 2025-11-03
**Авторы:** Хаттаев Расул, Замараева Ксения, Буров Владислав
