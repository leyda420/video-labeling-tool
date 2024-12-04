# =============================================================================
# A SINGLE AI TO DETECT THEM
# =============================================================================
# Author: [Your Name]
# Purpose:
#   This script integrates a YOLOv11-powered AI module into the video labeling tool.
#   It provides real-time object detection with adjustable confidence thresholds,
#   box smoothing using Kalman filters, and supports batch processing of videos.
#
# Features:
#   - **Real-Time Object Detection**: Automatically label video frames using YOLOv11.
#   - **Interactive Threshold Adjustment**: Adjust detection confidence via GUI.
#   - **Box Smoothing**: Apply Kalman filters for smooth bounding box transitions.
#   - **Batch Processing**: Process multiple videos in a single command.
#   - **Feedback Loop**: Save detections for review and retraining.
#
# Requirements:
#   - Python 3.x
#   - Libraries: PyTorch, OpenCV, PyQt5, NumPy, filterpy
#   - NVIDIA RTX 3080 GPU with CUDA installed
#
# How to Use:
#   1. Place your videos in the `data/raw_videos/` directory.
#   2. Run the script to start the GUI or use batch processing mode.
#   3. Adjust the confidence threshold slider to filter detections.
#   4. Detections are saved to `data/detections/` for review.
#
# =============================================================================

import sys
import os
import glob
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QSlider, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from filterpy.kalman import KalmanFilter

# -------------------
# Helper Functions
# -------------------

def load_yolo_model(model_path):
    """
    Loads the YOLOv11 model from the specified path.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    return model, device

def preprocess_frame(frame):
    """
    Preprocesses the frame for YOLOv11 input.
    """
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0  # Normalize to [0, 1]
    return torch.from_numpy(img).unsqueeze(0)  # Add batch dimension

def postprocess_outputs(outputs, confidence_threshold, frame_shape):
    """
    Postprocesses the model outputs to extract detections above the confidence threshold.
    """
    detections = []
    # Assuming outputs are in [batch, num_boxes, 6] format: [x1, y1, x2, y2, conf, class]
    preds = outputs[0]  # Get predictions from the first batch
    for pred in preds:
        conf = pred[4]
        if conf >= confidence_threshold:
            x1 = int(pred[0])
            y1 = int(pred[1])
            x2 = int(pred[2])
            y2 = int(pred[3])
            cls = int(pred[5])
            detections.append([x1, y1, x2, y2, conf.item(), cls])
    return detections

def draw_detections(frame, detections):
    """
    Draws bounding boxes and labels on the frame.
    """
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f'Class {cls}: {conf:.2f}'
        color = (0, 255, 0)  # Green color for boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def ensure_output_dirs():
    """
    Ensures that the output directory for detections exists.
    """
    os.makedirs('data/detections/', exist_ok=True)

def save_detections(frame_number, frame, detections, video_name):
    """
    Saves the detection results to the detections directory.
    """
    detection_dir = os.path.join('data/detections/', video_name)
    os.makedirs(detection_dir, exist_ok=True)
    img_path = os.path.join(detection_dir, f'frame_{frame_number:06d}.jpg')
    label_path = os.path.join(detection_dir, f'frame_{frame_number:06d}.txt')

    # Save the image
    cv2.imwrite(img_path, frame)

    # Save the labels in YOLO format
    img_height, img_width = frame.shape[:2]
    with open(label_path, 'w') as f:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # Convert to YOLO format
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            f.write(f'{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

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
        Initialises a tracker using initial bounding box.
        bbox: [x1, y1, x2, y2]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.P[4:, 4:] *= 1000.  # Give high uncertainty to the unobservable initial velocities
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
# Main GUI Application
# -------------------

class YOLODetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('YOLOv11 Object Detection')
        self.setGeometry(100, 100, 1280, 800)

        # Initialize variables
        self.model_path = 'models/yolov11_base.pt'
        self.model, self.device = load_yolo_model(self.model_path)
        self.confidence_threshold = 0.5
        self.current_frame = None
        self.frame_number = 0
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.trackers = []
        self.max_age = 1  # Frames to keep alive without detection
        self.min_hits = 3  # Minimum detections before tracking

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(1280, 720)

        # Confidence threshold slider
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(int(self.confidence_threshold * 100))
        self.confidence_slider.valueChanged.connect(self.on_confidence_change)

        # Load video button
        self.load_button = QPushButton('Load Video')
        self.load_button.clicked.connect(self.load_video)

        # Play/Pause button
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.play_video)
        self.paused = True

        # Batch processing button
        self.batch_button = QPushButton('Batch Process Videos')
        self.batch_button.clicked.connect(self.batch_process_videos)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Layouts
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel('Confidence Threshold:'))
        slider_layout.addWidget(self.confidence_slider)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.batch_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(slider_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)

        central_widget.setLayout(main_layout)

    def on_confidence_change(self, value):
        """
        Updates the confidence threshold based on slider value.
        """
        self.confidence_threshold = value / 100.0

    def load_video(self):
        """
        Loads a video file for processing.
        """
        options = QFileDialog.Options()
        video_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Video', './data/raw_videos/', 'Video Files (*.mp4 *.avi *.mov)', options=options)
        if video_path:
            self.video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                QMessageBox.critical(self, 'Error', 'Failed to open video file.')
                return
            self.frame_number = 0
            self.progress_bar.setValue(0)
            self.play_video()

    def play_video(self):
        """
        Toggles video playback.
        """
        if self.video_capture is None:
            QMessageBox.warning(self, 'Warning', 'Please load a video first.')
            return

        if self.paused:
            self.play_button.setText('Pause')
            self.timer.start(30)  # Adjust based on video FPS
            self.paused = False
        else:
            self.play_button.setText('Play')
            self.timer.stop()
            self.paused = True

    def update_frame(self):
        """
        Reads the next frame, runs detection, and updates the display.
        """
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self.play_button.setText('Play')
            QMessageBox.information(self, 'End of Video', 'Reached the end of the video.')
            return

        self.frame_number += 1
        self.current_frame = frame.copy()
        detections = self.run_detection(frame)
        self.apply_kalman_filters(detections)
        frame = self.draw_trackers(frame)
        self.display_frame(frame)
        self.update_progress_bar()
        save_detections(self.frame_number, frame, detections, self.video_name)

    def run_detection(self, frame):
        """
        Runs the YOLOv11 model on the current frame.
        """
        input_tensor = preprocess_frame(frame).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        frame_shape = frame.shape
        detections = postprocess_outputs(outputs, self.confidence_threshold, frame_shape)
        return detections

    def apply_kalman_filters(self, detections):
        """
        Updates trackers using Kalman filters.
        """
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, self.trackers)

        # Update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
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
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out matches with low IoU
        matches = {}
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches[m[1]] = m[0]

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
                color = (255, 0, 0)  # Blue color for trackers
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def display_frame(self, frame):
        """
        Displays the current frame in the GUI.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)

    def update_progress_bar(self):
        """
        Updates the progress bar based on video progress.
        """
        total_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        progress = int((self.frame_number / total_frames) * 100)
        self.progress_bar.setValue(progress)

    def batch_process_videos(self):
        """
        Processes all videos in the raw_videos directory.
        """
        video_files = glob.glob('data/raw_videos/*.mp4') + \
                      glob.glob('data/raw_videos/*.avi') + \
                      glob.glob('data/raw_videos/*.mov')

        if not video_files:
            QMessageBox.warning(self, 'Warning', 'No videos found in raw_videos directory.')
            return

        for video_file in video_files:
            self.video_name = os.path.splitext(os.path.basename(video_file))[0]
            self.video_capture = cv2.VideoCapture(video_file)
            if not self.video_capture.isOpened():
                print(f'Failed to open {video_file}')
                continue

            self.frame_number = 0
            self.progress_bar.setValue(0)
            self.process_video()
            self.video_capture.release()

        QMessageBox.information(self, 'Batch Processing', 'Batch processing completed.')

    def process_video(self):
        """
        Processes the video without displaying frames.
        """
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            self.frame_number += 1
            detections = self.run_detection(frame)
            self.apply_kalman_filters(detections)
            save_detections(self.frame_number, frame, detections, self.video_name)
            self.update_progress_bar()

def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem using the Hungarian algorithm.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
        raise ImportError('Install scipy for linear assignment')

# -------------------
# Main Entry Point
# -------------------

if __name__ == '__main__':
    ensure_output_dirs()
    app = QApplication(sys.argv)
    window = YOLODetectionApp()
    window.show()
    sys.exit(app.exec_())
