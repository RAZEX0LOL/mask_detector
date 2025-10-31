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


class MaskDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Detection System")
        self.root.geometry("1000x700")

        self.vs = None
        self.current_frame = None
        self.is_running = False

        self.create_widgets()
        self.load_models()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        title_label = ttk.Label(main_frame, text="Система обнаружения масок",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        self.video_frame = ttk.LabelFrame(main_frame, text="Видеопоток", padding="10")
        self.video_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))

        self.video_label = ttk.Label(self.video_frame, text="Нажмите 'Запуск' для начала работы",
                                     background="black", foreground="white")
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(button_frame, text="Запуск",
                                       command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = ttk.Button(button_frame, text="Остановка",
                                      command=self.stop_detection,
                                      state="disabled")
        self.stop_button.grid(row=0, column=1, padx=10)

        self.quit_button = ttk.Button(button_frame, text="Выход",
                                      command=self.quit_app)
        self.quit_button.grid(row=0, column=2, padx=10)

        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief="sunken", anchor="w")
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        self.video_frame.columnconfigure(0, weight=1)
        self.video_frame.rowconfigure(0, weight=1)

    def load_models(self):
        try:
            self.status_var.set("Загрузка моделей...")

            prototxtPath = r"face_detector\deploy.prototxt"
            weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
            self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

            self.maskNet = load_model("mask_detector.keras")

            self.status_var.set("Модели загружены успешно")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модели: {str(e)}")
            self.status_var.set("Ошибка загрузки моделей")

    def detect_and_predict_mask(self, frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)

    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_var.set("Запуск видеопотока...")

            self.vs = VideoStream(src=0).start()
            time.sleep(2.0)

            self.update_frame()

    def stop_detection(self):
        self.is_running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Остановлено")

        if self.vs:
            self.vs.stop()

        self.video_label.config(image='', text="Нажмите 'Запуск' для начала работы")

    def quit_app(self):
        if messagebox.askokcancel("Выход", "Вы уверены, что хотите выйти?"):
            self.stop_detection()
            self.root.destroy()

    def update_frame(self):
        if self.is_running and self.vs:
            try:
                frame = self.vs.read()
                if frame is not None:
                    frame = imutils.resize(frame, width=800)
                    (locs, preds) = self.detect_and_predict_mask(frame, self.faceNet, self.maskNet)

                    for (box, pred) in zip(locs, preds):
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred
                        label = "Mask" if mask > withoutMask else "No Mask"
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                        cv2.putText(frame, label, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)

                    self.video_label.configure(image=imgtk)
                    self.video_label.image = imgtk

                    self.status_var.set(f"Обнаружено лиц: {len(locs)}")

                if self.is_running:
                    self.root.after(10, self.update_frame)

            except Exception as e:
                self.status_var.set(f"Ошибка: {str(e)}")
                self.stop_detection()


def main():
    root = tk.Tk()
    app = MaskDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()