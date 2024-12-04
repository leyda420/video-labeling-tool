# =============================================================================
# ONE SCRIPT TO LABEL THEM ALL
# =============================================================================
# Author: [Your Name]
#
# Purpose:
#   Forged in the fires of Mount Code, this script is the One Script to Rule
#   Them All for video annotation and AI-assisted object detection. Wield it to:
#     - Label videos frame by frame with the precision of an elven archer.
#     - Save bounding boxes in YOLO format for training your own army of AI models.
#     - Harness the power of YOLOv11 to automatically detect objects as you traverse your video landscapes.
#     - Seamlessly switch between manual labeling and AI-assisted labeling modes.
#     - Control video playback, adjust detection confidence, and process batches of videos.
#
# How to Use:
#   1. **Load Your Video**: Click "Load Video" to select the video file—the palantír of your project.
#   2. **Set Output Directory**: Choose where your annotations and images will be stored.
#   3. **Choose Your Path**:
#      - **Manual Labeling**: Draw bounding boxes with the finesse of a master craftsman.
#        - Set the bounding box once, and it will be applied to every frame.
#        - Pause and adjust the box at any time, then continue.
#      - **AI-Assisted Labeling**: Toggle "Enable AI Mode" to enlist YOLOv11 in your quest.
#        - Adjust the confidence threshold on the fly.
#        - See prediction boxes overlaid on the video for instant feedback.
#        - Play or step through frames with AI detections displayed.
#   4. **Adjust Settings**: When a video is loaded, a settings window will pop up.
#      - Adjust playback speed, confidence threshold, and data splitting ratio.
#   5. **Save Your Progress**: Click "Save Current Frame" to save annotations, or let the script save them as you proceed.
#   6. **Generate Memes**: Need a laugh? Summon a random cat meme to lighten your journey.
#   7. **Batch Processing**: Use "Batch Process Videos" to let the script toil through multiple videos while you rest.
#   8. **Retrain the Model**: When ready, click "Retrain YOLO Model" to make your AI even wiser.
#
# Notes:
#   - May your labeling journey be swift and your models accurate.
#   - Remember: Even the smallest annotation can change the course of the future.
#
# "One script to rule them all, one script to find them,
# One script to bring them all, and in the data bind them."
# =============================================================================

import os
import cv2
import random
import requests
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QComboBox, QProgressBar, QMessageBox, QCheckBox, QFileDialog, QSlider,
    QDialog, QColorDialog, QInputDialog, QSpinBox, QTabWidget, QTextEdit, QMenu, QAction,
    QShortcut
)
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint, QSettings, QThread, pyqtSignal, QByteArray
from PyQt5.QtGui import QPixmap, QImage, QMovie, QPainter, QPen, QColor, QKeySequence, QIcon, QCursor
from ultralytics import YOLO  # Updated import
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import logging
import json
import zipfile
import shutil
import sys
import time
import tempfile

# Global variables
debug_mode = False

# Giphy API Key (Replace with your own key)
GIPHY_API_KEY = "your_giphy_api_key_here"

# Class colors (Extend this list if you have more classes)
class_colors = [
    Qt.red, Qt.green, Qt.blue, Qt.cyan, Qt.magenta, Qt.yellow,
    Qt.gray, Qt.darkRed, Qt.darkGreen, Qt.darkBlue, Qt.darkCyan, Qt.darkMagenta
]

# Supported YOLOv11 model sizes (e.g., 'n', 's', 'm', 'l', 'x')
model_sizes = ['n', 's', 'm', 'l', 'x']

# Set up logging
logging.basicConfig(
    filename='session.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
# -------------------
# Helper Functions
# -------------------

def download_random_gif(api_key):
    """
    Downloads a random GIF from Giphy and saves it locally.
    """
    try:
        url = f"https://api.giphy.com/v1/gifs/random?api_key={api_key}&tag=cat"
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (e.g., 404 or 500)

        data = response.json()

        # Ensure the data contains the expected keys
        if 'data' in data and 'images' in data['data'] and 'original' in data['data']['images']:
            gif_url = data['data']['images']['original']['url']
            gif_data = requests.get(gif_url).content

            gif_path = "random_meme.gif"
            with open(gif_path, 'wb') as f:
                f.write(gif_data)

            return gif_path
        else:
            raise ValueError("Unexpected API response structure")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch GIF: {e}")

def ensure_output_dirs(output_dir):
    """
    Ensures that the output directories for training and validation exist.
    """
    train_dir = os.path.join(output_dir, "train", "images")
    val_dir = os.path.join(output_dir, "val", "images")
    train_label_dir = os.path.join(output_dir, "train", "labels")
    val_label_dir = os.path.join(output_dir, "val", "labels")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

def yolo_format(box, img_width, img_height):
    """
    Converts bounding box coordinates to YOLO format.
    box: QRect object
    """
    x_center = (box.x() + box.width() / 2) / img_width
    y_center = (box.y() + box.height() / 2) / img_height
    width = box.width() / img_width
    height = box.height() / img_height
    return x_center, y_center, width, height

def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem using the Hungarian algorithm.
    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def coco_format(annotations, image_id, image_width, image_height):
    """
    Converts annotations to COCO format.
    annotations: List of label strings in YOLO format.
    """
    coco_annotations = []
    for ann_id, ann in enumerate(annotations):
        parts = ann.strip().split()
        if len(parts) != 5:
            continue  # Skip malformed lines
        cls_id, x_center, y_center, width, height = parts
        x_center = float(x_center) * image_width
        y_center = float(y_center) * image_height
        width = float(width) * image_width
        height = float(height) * image_height
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        coco_ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": int(cls_id),
            "bbox": [x_min, y_min, width, height],
            "area": width * height,
            "iscrowd": 0
        }
        coco_annotations.append(coco_ann)
    return coco_annotations

def get_class_distribution(output_dir):
    """
    Computes class distribution from the labels.
    """
    class_counts = {}
    for split in ['train', 'val']:
        label_dir = os.path.join(output_dir, split, 'labels')
        if not os.path.exists(label_dir):
            continue
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 0:
                            continue
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
    return class_counts
# -------------------
# Kalman Filter Class
# -------------------
class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bounding boxes.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.
        bbox: [x1, y1, x2, y2]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.P[4:, 4:] *= 1000.  # High uncertainty in the velocities
        self.kf.P *= 10.
        self.kf.R[2:, 2:] *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        x1, y1, x2, y2 = bbox
        self.kf.x[:4] = np.array([x1, y1, x2 - x1, y2 - y1]).reshape((4, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        x1, y1, x2, y2 = bbox
        z = np.array([x1, y1, x2 - x1, y2 - y1]).reshape((4, 1))
        self.kf.update(z)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        x, y, s, r = self.kf.x[:4].flatten()
        x1 = x
        y1 = y
        x2 = x + s
        y2 = y + r
        self.time_since_update += 1
        return [int(x1), int(y1), int(x2), int(y2)]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        x, y, s, r = self.kf.x[:4].flatten()
        x1 = x
        y1 = y
        x2 = x + s
        y2 = y + r
        return [int(x1), int(y1), int(x2), int(y2)]
# -------------------
# Settings Window
# -------------------
class SettingsWindow(QDialog):
    def __init__(self, parent=None, playback_speed=1.0, train_val_split=0.8, ai_mode=False):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setWindowModality(Qt.ApplicationModal)
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                font-size: 14px;
                padding: 8px;
            }
            QSlider {
                height: 30px;
            }
        """)

        self.playback_speed = playback_speed
        self.train_val_split = train_val_split
        self.ai_mode = ai_mode
        self.confidence_threshold = getattr(parent, 'confidence_threshold', 0.5)

        # Playback speed slider
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 50)  # 0.5x to 5x (multiplied by 10)
        self.speed_slider.setValue(int(self.playback_speed * 10))
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.valueChanged.connect(self.adjust_playback_speed)

        # Playback speed label
        self.speed_label = QLabel(f"Playback Speed: {self.playback_speed:.1f}x")

        # Data splitting ratio slider
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(int(self.train_val_split * 100))
        self.split_slider.setTickPosition(QSlider.TicksBelow)
        self.split_slider.setTickInterval(5)
        self.split_slider.valueChanged.connect(self.adjust_data_split)

        # Data splitting label
        self.split_label = QLabel(f"Train/Val Split: {self.train_val_split*100:.0f}% / {100 - self.train_val_split*100:.0f}%")

        # Confidence threshold slider (only if AI mode is enabled)
        if self.ai_mode:
            self.confidence_slider = QSlider(Qt.Horizontal)
            self.confidence_slider.setRange(0, 100)
            self.confidence_slider.setValue(int(self.confidence_threshold * 100))
            self.confidence_slider.setTickPosition(QSlider.TicksBelow)
            self.confidence_slider.setTickInterval(5)
            self.confidence_slider.valueChanged.connect(self.adjust_confidence_threshold)

            self.confidence_label = QLabel(f"Confidence Threshold: {self.confidence_threshold:.2f}")

        # Layouts
        layout = QVBoxLayout()
        layout.addWidget(self.speed_label)
        layout.addWidget(self.speed_slider)
        layout.addWidget(self.split_label)
        layout.addWidget(self.split_slider)

        if self.ai_mode:
            layout.addWidget(self.confidence_label)
            layout.addWidget(self.confidence_slider)

        # OK button
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def adjust_playback_speed(self, value):
        """
        Adjusts the playback speed based on the slider value.
        """
        self.playback_speed = value / 10  # Convert slider value to range 0.5x to 5x
        self.speed_label.setText(f"Playback Speed: {self.playback_speed:.1f}x")

    def adjust_data_split(self, value):
        """
        Adjusts the data splitting ratio based on the slider value.
        """
        self.train_val_split = value / 100.0
        self.split_label.setText(f"Train/Val Split: {self.train_val_split*100:.0f}% / {100 - self.train_val_split*100:.0f}%")

    def adjust_confidence_threshold(self, value):
        """
        Adjusts the confidence threshold based on the slider value.
        """
        self.confidence_threshold = value / 100.0
        self.confidence_label.setText(f"Confidence Threshold: {self.confidence_threshold:.2f}")
# -------------------
# Training Thread
# -------------------
class TrainingThread(QThread):
    progress = pyqtSignal(float)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, data_dir, epochs=10, batch_size=16, learning_rate=0.001, use_gpu=True):
        super().__init__()
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.stop_training = False

    def run(self):
        try:
            device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
            self.log.emit(f"Training started on {device}.")

            # Prepare the data
            data_yaml = os.path.join(self.data_dir, 'data.yaml')
            with open(data_yaml, 'w') as f:
                f.write(f"path: {self.data_dir}\n")
                f.write("train: train/images\n")
                f.write("val: val/images\n")
                f.write("nc: 1\n")  # Number of classes (modify as needed)
                f.write("names: ['object']\n")  # Class names (modify as needed)

            # Initialize the model
            model = YOLO('yolov8n.yaml')  # Using a small model for faster training

            # Start training
            model.train(
                data=data_yaml,
                epochs=self.epochs,
                batch=self.batch_size,
                lr0=self.learning_rate,
                device=device,
                imgsz=640,
                progress_callback=self.progress_callback
            )

            self.finished.emit(True)
            self.log.emit("Training completed successfully.")
        except Exception as e:
            self.log.emit(f"Training failed: {e}")
            self.finished.emit(False)

    def progress_callback(self, **kwargs):
        epoch = kwargs.get('epoch')
        epochs = kwargs.get('epochs')
        if epoch and epochs:
            self.progress.emit((epoch / epochs) * 100)
            self.log.emit(f"Epoch {epoch}/{epochs}")
# -------------------
# GUI Functionality
# -------------------
class LabelingApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.persistent_box = None  # Persistent bounding box
        self.current_class = 0      # Default class
        self.frame_number = 0
        self.paused = True
        self.drawing = False
        self.start_point = QPoint()
        self.video_path = None
        self.cap = None
        self.train_val_split = 0.8  # Default train/validation split
        self.output_dir = "output"  # Default output directory
        self.frame = None           # Current frame
        self.video_size = (640, 480)  # Default video size
        self.playback_speed = 1     # Default playback speed
        self.display_scale = 1.0
        self.x_offset = 0
        self.y_offset = 0
        self.settings = QSettings('YourCompany', 'LabelingApp')

        # AI Mode variables
        self.ai_mode = False
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trackers = []
        self.max_age = 1
        self.min_hits = 3
        self.confidence_threshold = 0.5  # Default confidence threshold

        # Initialize the model path with a default model
        self.model_path = 'best.pt'
        self.model = None  # The YOLO model

        # Recording flag
        self.recording_enabled = True

        # Zoom variables
        self.zoom_scale = 1.0

        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []

        # Multiple bounding boxes
        self.bounding_boxes = []  # List of dicts {'rect': QRect, 'class': int}

        # Pre-labeled suggestions
        self.suggested_boxes = []

        # Main layout setup
        self.setWindowTitle("Video Labeling and Image Data Collection Tool")
        self.setGeometry(100, 100, 1280, 800)
        self.setStyleSheet(self._load_styles())

        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(1280, 720)  # Fixed size for consistency

        # Buttons
        self.load_video_button = QPushButton("Load Video")
        self.set_output_button = QPushButton("Set Output Directory")
        self.play_button = QPushButton("Play")
        self.next_button = QPushButton("Next Frame")
        self.save_button = QPushButton("Save Current Frame")
        self.rewind_button = QPushButton("Rewind")
        self.fast_forward_button = QPushButton("Fast-Forward")
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_out_button = QPushButton("Zoom Out")
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.start_stop_recording_button = QPushButton("Stop Recording")
        self.loop_mode_button = QPushButton("Loop Mode")
        self.loop_mode_button.setCheckable(True)

        # Meme button
        self.generate_meme_button = QPushButton("Generate Random Meme")
        self.generate_meme_button.clicked.connect(self.generate_meme)

        # AI Mode button
        self.ai_mode_button = QPushButton("Enable AI Mode")
        self.ai_mode_button.setCheckable(True)
        self.ai_mode_button.toggled.connect(self.toggle_ai_mode)

        # Batch processing button
        self.batch_process_button = QPushButton("Batch Process Videos")
        self.batch_process_button.clicked.connect(self.batch_process_videos)

        # Retrain model button
        self.retrain_button = QPushButton("Train YOLO Model")
        self.retrain_button.clicked.connect(self.train_model)

        # Dataset health check button
        self.dataset_health_button = QPushButton("Dataset Health Check")
        self.dataset_health_button.clicked.connect(self.show_dataset_health)

        # Create the "Select Model" button
        self.select_model_button = QPushButton("Select Model")
        self.select_model_button.clicked.connect(self.select_model)

        # Create a label to display the selected model name
        self.model_label = QLabel(f"Model: {os.path.basename(self.model_path)}")

        # Create the "Export Dataset" button
        self.export_dataset_button = QPushButton("Export Dataset")
        self.export_dataset_button.clicked.connect(self.export_dataset)

        # Create the "Mark as Background" button
        self.mark_background_button = QPushButton("Mark as Background")
        self.mark_background_button.clicked.connect(self.mark_as_background)

        # Class selector
        self.class_selector = QComboBox()
        self.class_selector.addItems([f"Class {i+1}" for i in range(len(class_colors))])
        self.class_selector.currentIndexChanged.connect(self.set_current_class)

        # Color picker button
        self.color_picker_button = QPushButton("Pick Color")
        self.color_picker_button.clicked.connect(self.pick_color)

        # Theme selector
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Dark Mode", "Light Mode", "High Contrast"])
        self.theme_selector.currentIndexChanged.connect(self.change_theme)

        # Debug toggle
        self.debug_toggle = QCheckBox("Enable Debug Mode")
        self.debug_toggle.stateChanged.connect(self.toggle_debug_mode)

        # Recording button
        self.start_stop_recording_button.clicked.connect(self.toggle_recording)

        # Rewind and Fast-Forward buttons
        self.rewind_button.clicked.connect(self.rewind)
        self.fast_forward_button.clicked.connect(self.fast_forward)

        # Zoom buttons
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        # Undo/Redo buttons
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        # Loop Mode
        self.loop_mode = False
        self.loop_start = 0
        self.loop_end = 0

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.slider_moved)

        # Meme display area
        self.meme_label = QLabel()
        self.meme_label.setAlignment(Qt.AlignCenter)

        # Layouts
        self.controls_layout = QHBoxLayout()
        self.controls_layout.addWidget(self.load_video_button)
        self.controls_layout.addWidget(self.set_output_button)
        self.controls_layout.addWidget(self.play_button)
        self.controls_layout.addWidget(self.next_button)
        self.controls_layout.addWidget(self.rewind_button)
        self.controls_layout.addWidget(self.fast_forward_button)
        self.controls_layout.addWidget(self.save_button)
        self.controls_layout.addWidget(self.start_stop_recording_button)
        self.controls_layout.addWidget(self.generate_meme_button)
        self.controls_layout.addWidget(self.ai_mode_button)
        self.controls_layout.addWidget(self.batch_process_button)
        self.controls_layout.addWidget(self.retrain_button)
        self.controls_layout.addWidget(self.dataset_health_button)
        self.controls_layout.addWidget(self.select_model_button)
        self.controls_layout.addWidget(self.model_label)
        self.controls_layout.addWidget(self.export_dataset_button)
        self.controls_layout.addWidget(self.mark_background_button)
        self.controls_layout.addWidget(self.zoom_in_button)
        self.controls_layout.addWidget(self.zoom_out_button)
        self.controls_layout.addWidget(self.undo_button)
        self.controls_layout.addWidget(self.redo_button)
        self.controls_layout.addWidget(self.loop_mode_button)
        self.controls_layout.addWidget(QLabel("Class:"))
        self.controls_layout.addWidget(self.class_selector)
        self.controls_layout.addWidget(self.color_picker_button)
        self.controls_layout.addWidget(QLabel("Theme:"))
        self.controls_layout.addWidget(self.theme_selector)
        self.controls_layout.addWidget(self.debug_toggle)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.frame_slider)
        main_layout.addLayout(self.controls_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.meme_label)

        self.setLayout(main_layout)

        # Connect buttons
        self.load_video_button.clicked.connect(self.select_video)
        self.set_output_button.clicked.connect(self.set_output_directory)
        self.play_button.clicked.connect(self.toggle_play)
        self.next_button.clicked.connect(self.next_frame)
        self.save_button.clicked.connect(self.save_current_frame)

        # Keyboard shortcuts
        self.setup_shortcuts()

        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Mouse Interaction
        self.video_label.mousePressEvent = self.mouse_press
        self.video_label.mouseReleaseEvent = self.mouse_release
        self.video_label.mouseMoveEvent = self.mouse_move

        # Load user settings
        self.load_settings()
    def _load_styles(self):
        """
        Loads custom styles for the application.
        """
        return """
        QWidget {
            background-color: #2e3440;
            color: #d8dee9;
        }
        QLabel {
            font-size: 16px;
            color: #d8dee9;
        }
        QPushButton {
            font-size: 14px;
            padding: 8px;
            background-color: #5e81ac;
            color: #eceff4;
            border: none;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #81a1c1;
        }
        QComboBox {
            font-size: 14px;
            padding: 5px;
            background-color: #4c566a;
            color: #eceff4;
        }
        QCheckBox {
            font-size: 14px;
            padding: 5px;
        }
        QProgressBar {
            height: 20px;
            text-align: center;
            color: #eceff4;
        }
        QProgressBar::chunk {
            background-color: #5e81ac;
        }
        """

    def setup_shortcuts(self):
        """
        Sets up keyboard shortcuts.
        """
        QShortcut(QKeySequence("Space"), self, self.toggle_play)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_current_frame)
        QShortcut(QKeySequence("Right"), self, self.next_frame)
        QShortcut(QKeySequence("Left"), self, self.previous_frame)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)
        QShortcut(QKeySequence("+"), self, self.zoom_in)
        QShortcut(QKeySequence("-"), self, self.zoom_out)

    def load_settings(self):
        """
        Loads user settings.
        """
        self.output_dir = self.settings.value('output_dir', 'output')
        self.model_path = self.settings.value('model_path', 'best.pt')
        self.playback_speed = float(self.settings.value('playback_speed', 1.0))
        self.train_val_split = float(self.settings.value('train_val_split', 0.8))
        self.confidence_threshold = float(self.settings.value('confidence_threshold', 0.5))
        self.model_label.setText(f"Model: {os.path.basename(self.model_path)}")

    def save_settings(self):
        """
        Saves user settings.
        """
        self.settings.setValue('output_dir', self.output_dir)
        self.settings.setValue('model_path', self.model_path)
        self.settings.setValue('playback_speed', self.playback_speed)
        self.settings.setValue('train_val_split', self.train_val_split)
        self.settings.setValue('confidence_threshold', self.confidence_threshold)

    def show_settings(self):
        """
        Displays the settings window.
        """
        settings = SettingsWindow(
            parent=self,
            playback_speed=self.playback_speed,
            train_val_split=self.train_val_split,
            ai_mode=self.ai_mode
        )
        if settings.exec_() == QDialog.Accepted:
            self.playback_speed = settings.playback_speed
            self.train_val_split = settings.train_val_split
            if self.ai_mode:
                self.confidence_threshold = settings.confidence_threshold
            if not self.paused:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 30  # Fallback FPS
                interval = int(1000 / (fps * self.playback_speed))
                self.timer.setInterval(interval)
            if debug_mode:
                print("[DEBUG] Settings updated.")
            self.save_settings()
    def generate_meme(self):
        """
        Generates a random cat meme (GIF) using the Giphy API.
        """
        try:
            gif_path = download_random_gif(GIPHY_API_KEY)
            self.display_gif(gif_path)
            QMessageBox.information(self, "Meme Generated", f"Your random meme has been generated!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate meme: {str(e)}")
            if debug_mode:
                print(f"[DEBUG] Meme generation error: {str(e)}")
            logging.error(f"Meme generation error: {str(e)}")

    def display_gif(self, gif_path):
        """
        Displays the downloaded GIF in the meme area.
        """
        gif = QMovie(gif_path)
        self.meme_label.setMovie(gif)
        gif.start()

    def set_output_directory(self):
        """
        Opens a dialog for selecting the output directory.
        """
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            ensure_output_dirs(self.output_dir)
            QMessageBox.information(self, "Directory Selected", f"Output directory set to: {directory}")
            if debug_mode:
                print(f"[DEBUG] Output directory set to: {directory}")
            logging.info(f"Output directory set to: {directory}")
            self.save_settings()

    def select_video(self):
        """
        Opens a file dialog to select a video and initializes video capture.
        """
        self.video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", f"Unable to open video '{self.video_path}'.")
                self.video_path = None
                return
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start from first frame
                ret, self.frame = self.cap.read()
                if ret:
                    self.frame_number = 0
                    self.display_frame(self.frame)
                    self.progress_bar.setValue(0)
                    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.frame_slider.setRange(0, total_frames - 1)
                    self.show_settings()  # Show settings window
                    QMessageBox.information(self, "Video Loaded", f"Video '{self.video_path}' loaded successfully!")
                    if debug_mode:
                        print(f"[DEBUG] Video '{self.video_path}' loaded successfully.")
                    logging.info(f"Video '{self.video_path}' loaded successfully.")
                else:
                    QMessageBox.critical(self, "Error", f"Unable to read first frame of '{self.video_path}'.")
                    logging.error(f"Unable to read first frame of '{self.video_path}'.")
    def toggle_play(self):
        """
        Toggles between playing and pausing the video.
        """
        if self.cap is None or not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first!")
            return

        self.paused = not self.paused
        if not self.paused:
            self.play_button.setText("Pause")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Fallback FPS
            interval = int(1000 / (fps * self.playback_speed))
            self.timer.start(interval)
            if debug_mode:
                print("[DEBUG] Video playback started.")
            logging.info("Video playback started.")
        else:
            self.play_button.setText("Play")
            self.timer.stop()
            if debug_mode:
                print("[DEBUG] Video playback paused.")
            logging.info("Video playback paused.")

    def next_frame(self):
        """
        Moves to the next frame, saves current frame and labels.
        """
        if self.cap is None or not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first!")
            return

        # Save current frame and labels before moving
        if self.recording_enabled:
            self.save_frame_and_labels()

        ret, self.frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.play_button.setText("Play")
            QMessageBox.information(self, "End of Video", "End of video reached.")
            if debug_mode:
                print("[DEBUG] End of video reached.")
            logging.info("End of video reached.")
            return

        self.frame_number += 1
        self.frame_slider.setValue(self.frame_number)
        self.update_progress_bar()

        if self.ai_mode and self.model is not None:
            self.detections = self.run_detection(self.frame)
            self.apply_kalman_filters(self.detections)
            self.frame = self.draw_trackers(self.frame)

        else:
            # Apply persistent box to current frame
            pass  # Implement tracking if needed

        # Apply pre-labeled suggestions
        if self.ai_mode and self.model is not None:
            self.apply_suggestions()

        self.display_frame(self.frame)
        if debug_mode:
            print(f"[DEBUG] Moved to frame {self.frame_number}.")
        logging.info(f"Moved to frame {self.frame_number}.")

    def previous_frame(self):
        """
        Moves to the previous frame.
        """
        if self.cap is None or not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first!")
            return

        if self.frame_number > 0:
            self.frame_number -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            ret, self.frame = self.cap.read()
            if ret:
                self.display_frame(self.frame)
                self.frame_slider.setValue(self.frame_number)
                self.update_progress_bar()
                if debug_mode:
                    print(f"[DEBUG] Moved back to frame {self.frame_number}.")
                logging.info(f"Moved back to frame {self.frame_number}.")
    def save_current_frame(self):
        """
        Manually saves the current frame and labels.
        """
        if self.cap is None or not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first!")
            return

        self.save_frame_and_labels()
        QMessageBox.information(self, "Saved", f"Frame {self.frame_number} and labels have been saved.")
        if debug_mode:
            print(f"[DEBUG] Frame {self.frame_number} saved manually.")
        logging.info(f"Frame {self.frame_number} saved manually.")

    def save_frame_and_labels(self):
        """
        Saves the current frame and its bounding boxes in YOLO and COCO formats.
        """
        if self.frame is None:
            if debug_mode:
                print("[DEBUG] No frame to save.")
            return

        ensure_output_dirs(self.output_dir)

        is_train = random.random() < self.train_val_split
        split = "train" if is_train else "val"
        img_dir = os.path.join(self.output_dir, split, "images")
        label_dir = os.path.join(self.output_dir, split, "labels")
        coco_ann_dir = os.path.join(self.output_dir, split, "annotations")
        os.makedirs(coco_ann_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"frame_{self.frame_number:06d}.jpg")
        label_path = os.path.join(label_dir, f"frame_{self.frame_number:06d}.txt")
        coco_label_path = os.path.join(coco_ann_dir, f"frame_{self.frame_number:06d}.json")

        # Save the image
        cv2.imwrite(img_path, self.frame)
        if debug_mode:
            print(f"[DEBUG] Image saved to {img_path}")
        logging.info(f"Image saved to {img_path}")

        # Save the labels
        labels = []

        # Include manual bounding boxes
        for bbox_data in self.bounding_boxes:
            rect = bbox_data['rect']
            cls = bbox_data['class']
            x_center, y_center, width, height = yolo_format(rect, self.frame.shape[1], self.frame.shape[0])
            labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Include AI detections if in AI mode
        if self.ai_mode and self.model is not None:
            img_height, img_width = self.frame.shape[:2]
            for trk in self.trackers:
                if trk.hits >= self.min_hits or self.frame_number <= self.min_hits:
                    x1, y1, x2, y2 = trk.get_state()
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")  # Assuming class '0' for AI detections

        if labels:
            with open(label_path, "w") as f:
                f.write("\n".join(labels))
            if debug_mode:
                print(f"[DEBUG] Labels saved to {label_path}")
            logging.info(f"Labels saved to {label_path}")

            # Save in COCO format
            coco_annotations = coco_format(labels, self.frame_number, self.frame.shape[1], self.frame.shape[0])
            with open(coco_label_path, "w") as f:
                json.dump(coco_annotations, f)
            if debug_mode:
                print(f"[DEBUG] COCO annotations saved to {coco_label_path}")
            logging.info(f"COCO annotations saved to {coco_label_path}")
        else:
            # Create an empty label file if no bounding box
            open(label_path, 'a').close()
            if debug_mode:
                print(f"[DEBUG] Empty label file created at {label_path}")
            logging.info(f"Empty label file created at {label_path}")

    def update_frame(self):
        """
        Timer callback to update the frame during playback.
        """
        if self.cap is None or self.paused:
            return

        # Save current frame and labels before moving
        if self.recording_enabled:
            self.save_frame_and_labels()

        ret, self.frame = self.cap.read()
        if not ret:
            if self.loop_mode:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.loop_start)
                self.frame_number = self.loop_start
                ret, self.frame = self.cap.read()
                if not ret:
                    self.timer.stop()
                    self.play_button.setText("Play")
                    QMessageBox.information(self, "End of Video", "End of video reached.")
                    if debug_mode:
                        print("[DEBUG] End of video reached during playback.")
                    logging.info("End of video reached during playback.")
                    return
            else:
                self.timer.stop()
                self.play_button.setText("Play")
                QMessageBox.information(self, "End of Video", "End of video reached.")
                if debug_mode:
                    print("[DEBUG] End of video reached during playback.")
                logging.info("End of video reached during playback.")
                return

        self.frame_number += 1
        self.frame_slider.setValue(self.frame_number)
        self.update_progress_bar()

        if self.ai_mode and self.model is not None:
            self.detections = self.run_detection(self.frame)
            self.apply_kalman_filters(self.detections)
            self.frame = self.draw_trackers(self.frame)

        else:
            # Apply persistent box to current frame
            pass

        # Apply pre-labeled suggestions
        if self.ai_mode and self.model is not None:
            self.apply_suggestions()

        self.display_frame(self.frame)
        if debug_mode:
            print(f"[DEBUG] Playback moved to frame {self.frame_number}")
        logging.info(f"Playback moved to frame {self.frame_number}")

    def update_progress_bar(self):
        """
        Updates the progress bar based on the current frame number.
        """
        if self.cap is None:
            return
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames == 0:
            total_frames = 1  # Prevent division by zero
        progress = int((self.frame_number / total_frames) * 100)
        self.progress_bar.setValue(progress)
        if debug_mode:
            print(f"[DEBUG] Progress bar updated to {progress}%")
        logging.info(f"Progress bar updated to {progress}%")

    def display_frame(self, frame):
        """
        Displays the given frame with the bounding boxes overlay.
        """
        if frame is None:
            return

        temp_frame = frame.copy()
        rgb_image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        self.video_size = (w, h)
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Calculate scaling factors
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        scale_w = label_w / w
        scale_h = label_h / h
        self.display_scale = min(scale_w, scale_h)
        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)
        self.x_offset = (label_w - new_w) / 2
        self.y_offset = (label_h - new_h) / 2

        painter = QPainter(qt_image)

        # Draw manual bounding boxes
        for bbox_data in self.bounding_boxes:
            rect = bbox_data['rect']
            cls = bbox_data['class']
            pen = QPen(class_colors[cls], 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.drawText(rect.topLeft(), f"Class {cls+1}")

        # Draw AI detections
        if self.ai_mode and self.model is not None:
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)
            for trk in self.trackers:
                if trk.hits >= self.min_hits or self.frame_number <= self.min_hits:
                    x1, y1, x2, y2 = trk.get_state()
                    rect = QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    painter.drawRect(rect)
                    painter.drawText(int(x1), int(y1) - 10, f'ID {trk.id}')

        # Draw suggested boxes
        for suggestion in self.suggested_boxes:
            rect = suggestion['rect']
            cls = suggestion['class']
            pen = QPen(Qt.yellow, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.drawText(rect.topLeft(), f"Suggestion: Class {cls+1}")

        painter.end()

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)
    # -------------------
    # Bounding Box Management
    # -------------------

    def mouse_press(self, event):
        """
        Handles mouse press events to start drawing a bounding box or select existing ones.
        """
        if self.paused and self.frame is not None:
            if event.button() == Qt.LeftButton:
                pos = event.pos()
                adjusted_x = (pos.x() - self.x_offset) / self.display_scale
                adjusted_y = (pos.y() - self.y_offset) / self.display_scale
                clicked_point = QPoint(int(adjusted_x), int(adjusted_y))

                # Check if any existing box is selected
                for bbox_data in self.bounding_boxes:
                    if bbox_data['rect'].contains(clicked_point):
                        self.show_bbox_context_menu(bbox_data)
                        return

                # Start drawing new box
                self.start_point = clicked_point
                self.drawing = True
                if debug_mode:
                    print(f"[DEBUG] Started drawing bounding box at {self.start_point}")

    def mouse_release(self, event):
        """
        Handles mouse release events to finalize the bounding box.
        """
        if self.drawing and self.paused and self.frame is not None:
            pos = event.pos()
            adjusted_x = (pos.x() - self.x_offset) / self.display_scale
            adjusted_y = (pos.y() - self.y_offset) / self.display_scale
            end_point = QPoint(int(adjusted_x), int(adjusted_y))
            rect = QRect(self.start_point, end_point).normalized()

            # Ensure the bounding box has a minimum size
            min_size = 10
            if rect.width() < min_size or rect.height() < min_size:
                QMessageBox.warning(self, "Invalid Bounding Box", "Bounding box is too small. Please draw a larger box.")
                self.drawing = False
                if debug_mode:
                    print("[DEBUG] Bounding box too small and was discarded.")
                return

            # Add the bounding box to the list
            bbox_data = {'rect': rect, 'class': self.current_class}
            self.bounding_boxes.append(bbox_data)
            self.undo_stack.append(('add', bbox_data))
            self.redo_stack.clear()
            self.drawing = False
            if debug_mode:
                print(f"[DEBUG] Bounding box added: {rect}")
            logging.info(f"Bounding box added: {rect}")

            self.display_frame(self.frame)

    def mouse_move(self, event):
        """
        Handles mouse move events to draw the bounding box dynamically.
        """
        if self.drawing and self.paused and self.frame is not None:
            pos = event.pos()
            adjusted_x = (pos.x() - self.x_offset) / self.display_scale
            adjusted_y = (pos.y() - self.y_offset) / self.display_scale
            current_point = QPoint(int(adjusted_x), int(adjusted_y))

            # Create a temporary frame to display the drawing box
            temp_frame = self.frame.copy()
            rgb_image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            temp_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            painter = QPainter(temp_image)
            pen = QPen(class_colors[self.current_class], 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self.start_point, current_point).normalized()
            painter.drawRect(rect)
            painter.end()
            pixmap = QPixmap.fromImage(temp_image)
            scaled_pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
            if debug_mode:
                print(f"[DEBUG] Drawing bounding box: {rect}")

    def show_bbox_context_menu(self, bbox_data):
        """
        Displays a context menu for the selected bounding box with options to delete or edit.
        """
        menu = QMenu(self)
        delete_action = QAction('Delete', self)
        delete_action.triggered.connect(lambda: self.delete_bbox(bbox_data))
        edit_action = QAction('Edit Class', self)
        edit_action.triggered.connect(lambda: self.edit_bbox_class(bbox_data))
        menu.addAction(delete_action)
        menu.addAction(edit_action)
        menu.exec_(QCursor.pos())

    # -------------------
    # Bounding Box Management Methods
    # -------------------

    def delete_bbox(self, bbox_data):
        """
        Deletes a bounding box.
        """
        if bbox_data in self.bounding_boxes:
            self.bounding_boxes.remove(bbox_data)
            self.undo_stack.append(('delete', bbox_data))
            self.redo_stack.clear()
            self.display_frame(self.frame)
            if debug_mode:
                print(f"[DEBUG] Bounding box deleted: {bbox_data}")
            logging.info(f"Bounding box deleted: {bbox_data}")

    def edit_bbox_class(self, bbox_data):
        """
        Edits the class of a bounding box.
        """
        cls, ok = QInputDialog.getInt(
            self, "Edit Class", "Enter new class ID:", bbox_data['class'], 0, len(class_colors)-1, 1
        )
        if ok:
            old_class = bbox_data['class']
            bbox_data['class'] = cls
            self.undo_stack.append(('edit', bbox_data, old_class))
            self.redo_stack.clear()
            self.display_frame(self.frame)
            if debug_mode:
                print(f"[DEBUG] Bounding box class changed to {cls}.")
            logging.info(f"Bounding box class changed to {cls}.")

    def set_current_class(self):
        """
        Sets the current class for bounding boxes based on the selector.
        """
        self.current_class = self.class_selector.currentIndex()
        if debug_mode:
            print(f"[DEBUG] Current class set to: {self.current_class}")
        logging.info(f"Current class set to: {self.current_class}")

    def pick_color(self):
        """
        Opens a color picker dialog to set the color for the current class.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            class_colors[self.current_class] = color
            self.display_frame(self.frame)
            if debug_mode:
                print(f"[DEBUG] Color for class {self.current_class} changed to {color.name()}.")
            logging.info(f"Color for class {self.current_class} changed to {color.name()}.")
    # -------------------
    # AI Mode Integration
    # -------------------

    def toggle_ai_mode(self, state):
        """
        Toggles the AI Mode on or off.
        """
        if state:
            if self.model is None:
                success = self.load_yolo_model()
                if not success:
                    self.ai_mode_button.setChecked(False)
                    return
            self.ai_mode = True
            self.ai_mode_button.setText("Disable AI Mode")
            QMessageBox.information(self, "AI Mode Enabled", "AI Mode has been enabled.")
            self.show_settings()  # Show settings window with confidence slider
            logging.info("AI Mode enabled.")
        else:
            self.ai_mode = False
            self.ai_mode_button.setText("Enable AI Mode")
            QMessageBox.information(self, "AI Mode Disabled", "AI Mode has been disabled.")
            logging.info("AI Mode disabled.")

    def load_yolo_model(self):
        """
        Loads the YOLO model using the Ultralytics package.
        """
        try:
            self.model = YOLO(self.model_path)
            if debug_mode:
                print(f"[DEBUG] YOLO model '{self.model_path}' loaded successfully.")
            logging.info(f"YOLO model '{self.model_path}' loaded successfully.")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", f"An error occurred while loading the YOLO model: {str(e)}")
            self.model = None
            logging.error(f"Error loading YOLO model: {str(e)}")
            return False

    def run_detection(self, frame):
        """
        Runs the YOLO model on the current frame.
        """
        try:
            results = self.model.predict(frame, conf=self.confidence_threshold)
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), int(cls.item())])
            return detections
        except Exception as e:
            QMessageBox.critical(self, "Detection Error", f"An error occurred during detection: {str(e)}")
            logging.error(f"Detection error: {str(e)}")
            return []

    def apply_kalman_filters(self, detections):
        """
        Updates trackers using Kalman filters.
        """
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, self.trackers)

        # Update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t in matched:
                d = matched[t]
                trk.update(detections[d][:4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i][:4])
            self.trackers.append(trk)

        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

    def associate_detections_to_trackers(self, detections, trackers):
        """
        Assigns detections to tracked objects using IoU.
        """
        iou_threshold = 0.3
        if len(trackers) == 0:
            return {}, list(range(len(detections))), []

        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det[:4], trk.predict())

        matched_indices = linear_assignment(-iou_matrix)
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))

        matches = {}
        for m in matched_indices:
            if m[0] in unmatched_detections and m[1] in unmatched_trackers:
                if iou_matrix[m[0], m[1]] >= iou_threshold:
                    matches[m[1]] = m[0]
                    unmatched_detections.remove(m[0])
                    unmatched_trackers.remove(m[1])

        return matches, unmatched_detections, unmatched_trackers

    def iou(self, bbox1, bbox2):
        """
        Computes Intersection over Union (IoU) between two bounding boxes.
        """
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou

    def draw_trackers(self, frame):
        """
        Draws the bounding boxes from trackers on the frame.
        """
        for trk in self.trackers:
            if trk.hits >= self.min_hits or self.frame_number <= self.min_hits:
                bbox = trk.get_state()
                x1, y1, x2, y2 = bbox
                label = f'ID {trk.id}'
                color = (0, 255, 0)  # Green color for trackers
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def apply_suggestions(self):
        """
        Applies pre-labeled suggestions to the frame.
        """
        self.suggested_boxes = []
        detections = self.run_detection(self.frame)
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            rect = QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            suggestion = {'rect': rect, 'class': cls}
            self.suggested_boxes.append(suggestion)
    def batch_process_videos(self):
        """
        Processes all videos in a specified directory.
        """
        directory = QFileDialog.getExistingDirectory(self, "Select Videos Directory")
        if not directory:
            return

        video_files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(('.mp4', '.avi', '.mov'))
        ]

        if not video_files:
            QMessageBox.warning(self, 'Warning', 'No videos found in the selected directory.')
            return

        total_videos = len(video_files)
        video_progress = 0

        for video_file in video_files:
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                print(f'Failed to open {video_file}')
                logging.error(f"Failed to open {video_file}")
                continue

            self.frame_number = 0
            self.progress_bar.setValue(0)
            self.process_video()
            self.cap.release()
            video_progress += 1
            overall_progress = int((video_progress / total_videos) * 100)
            self.progress_bar.setValue(overall_progress)
            if debug_mode:
                print(f"[DEBUG] Processed video {video_progress}/{total_videos}: {video_file}")
            logging.info(f"Processed video {video_progress}/{total_videos}: {video_file}")

        QMessageBox.information(self, 'Batch Processing', 'Batch processing completed.')
        logging.info('Batch processing completed.')

    def process_video(self):
        """
        Processes the video without displaying frames.
        """
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            self.frame_number += 1
            if self.recording_enabled:
                if self.ai_mode and self.model is not None:
                    self.detections = self.run_detection(self.frame)
                    self.apply_kalman_filters(self.detections)
                self.save_frame_and_labels()
            self.update_progress_bar()

    def train_model(self):
        """
        Starts training the YOLO model.
        """
        epochs, ok = QInputDialog.getInt(
            self, "Training Parameters", "Enter number of epochs:", 10, 1, 1000, 1
        )
        if ok:
            self.training_thread = TrainingThread(self.output_dir, epochs=epochs)
            self.training_thread.progress.connect(self.update_training_progress)
            self.training_thread.log.connect(self.update_training_log)
            self.training_thread.finished.connect(self.training_finished)
            self.training_thread.start()
            self.training_progress_bar = QProgressBar()
            self.training_log = QTextEdit()
            self.training_log.setReadOnly(True)

            self.training_dialog = QDialog(self)
            self.training_dialog.setWindowTitle("Training Progress")
            layout = QVBoxLayout()
            layout.addWidget(self.training_progress_bar)
            layout.addWidget(self.training_log)
            self.training_dialog.setLayout(layout)
            self.training_dialog.show()
            logging.info("Training started.")

    def update_training_progress(self, value):
        """
        Updates the training progress bar.
        """
        self.training_progress_bar.setValue(int(value))

    def update_training_log(self, message):
        """
        Updates the training log.
        """
        self.training_log.append(message)
        logging.info(message)

    def training_finished(self, success):
        """
        Called when training is finished.
        """
        self.training_dialog.close()
        if success:
            QMessageBox.information(self, "Training Completed", "Model training completed successfully.")
            logging.info("Model training completed successfully.")
        else:
            QMessageBox.warning(self, "Training Failed", "Model training failed.")
            logging.error("Model training failed.")
    def zoom_in(self):
        """
        Zooms in on the video display.
        """
        self.zoom_scale += 0.1
        self.display_frame(self.frame)
        if debug_mode:
            print(f"[DEBUG] Zoomed in to scale {self.zoom_scale}.")
        logging.info(f"Zoomed in to scale {self.zoom_scale}.")

    def zoom_out(self):
        """
        Zooms out of the video display.
        """
        if self.zoom_scale > 0.1:
            self.zoom_scale -= 0.1
            self.display_frame(self.frame)
            if debug_mode:
                print(f"[DEBUG] Zoomed out to scale {self.zoom_scale}.")
            logging.info(f"Zoomed out to scale {self.zoom_scale}.")

    def undo(self):
        """
        Undoes the last action.
        """
        if self.undo_stack:
            action = self.undo_stack.pop()
            if action[0] == 'add':
                self.bounding_boxes.remove(action[1])
                self.redo_stack.append(action)
            elif action[0] == 'delete':
                self.bounding_boxes.append(action[1])
                self.redo_stack.append(action)
            elif action[0] == 'edit':
                action[1]['class'] = action[2]
                self.redo_stack.append(action)
            self.display_frame(self.frame)
            if debug_mode:
                print("[DEBUG] Undo performed.")
            logging.info("Undo performed.")

    def redo(self):
        """
        Redoes the last undone action.
        """
        if self.redo_stack:
            action = self.redo_stack.pop()
            if action[0] == 'add':
                self.bounding_boxes.append(action[1])
                self.undo_stack.append(action)
            elif action[0] == 'delete':
                self.bounding_boxes.remove(action[1])
                self.undo_stack.append(action)
            elif action[0] == 'edit':
                old_class = action[1]['class']
                action[1]['class'] = action[2]
                self.undo_stack.append(('edit', action[1], old_class))
            self.display_frame(self.frame)
            if debug_mode:
                print("[DEBUG] Redo performed.")
            logging.info("Redo performed.")

    def rewind(self):
        """
        Rewinds the video by 10 frames.
        """
        if self.cap is None or not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first!")
            return

        rewind_steps = 10
        new_frame_number = self.frame_number - rewind_steps
        if new_frame_number < 0:
            new_frame_number = 0
        self.frame_number = new_frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, self.frame = self.cap.read()
        if ret:
            self.display_frame(self.frame)
            self.frame_slider.setValue(self.frame_number)
            self.update_progress_bar()
            if debug_mode:
                print(f"[DEBUG] Rewound to frame {self.frame_number}.")
            logging.info(f"Rewound to frame {self.frame_number}.")

    def fast_forward(self):
        """
        Fast-forwards the video by 10 frames.
        """
        if self.cap is None or not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first!")
            return

        fast_forward_steps = 10
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        new_frame_number = self.frame_number + fast_forward_steps
        if new_frame_number >= total_frames:
            new_frame_number = total_frames - 1
        self.frame_number = new_frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, self.frame = self.cap.read()
        if ret:
            self.display_frame(self.frame)
            self.frame_slider.setValue(self.frame_number)
            self.update_progress_bar()
            if debug_mode:
                print(f"[DEBUG] Fast-forwarded to frame {self.frame_number}.")
            logging.info(f"Fast-forwarded to frame {self.frame_number}.")

    def slider_moved(self, value):
        """
        Handles the slider movement to jump to specific frames.
        """
        if self.cap is None or not self.video_path:
            return

        self.frame_number = value
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, self.frame = self.cap.read()
        if ret:
            self.display_frame(self.frame)
            self.update_progress_bar()
            if debug_mode:
                print(f"[DEBUG] Slider moved to frame {self.frame_number}.")
            logging.info(f"Slider moved to frame {self.frame_number}.")
# -------------------
# Main Entry Point
# -------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())
