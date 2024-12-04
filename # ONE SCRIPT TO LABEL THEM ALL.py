# =============================================================================
# ONE SCRIPT TO LABEL (AND CAPTURE THE CORRECT IMAGE TO MATCH) TO RULE THEM ALL
# =============================================================================
# Author: [Leyda]
# Purpose:
#   Forged in the fires of Mount Code, this script is the One Script to Rule 
#   Them All for video annotation and image data collection. Designed to 
#   combine precision, efficiency, and a touch of whimsy, this tool empowers you to:
#       - Label videos frame by frame with precision worthy of an elven smith.
#       - Save bounding boxes in YOLO format for training AI models to rival
#         the wisdom of the Istari.
#       - Control video playback speed, navigate frames seamlessly, and capture 
#         the right data for your machine learning quests.
#       - Summon random cat memes to lighten the burden of your great journey.

# Features:
#   - **Cat Memes of Power**: Harness the might of feline joy to banish the 
#     darkness of endless labeling tasks (powered by Giphy).
#   - **Debugging Magic**: Enable debug mode to reveal unseen paths and solve 
#     mysterious errors with ease.
#   - **Modular Design**: Expand or customize this script for your future 
#     adventures across the realms of machine learning.
#
# Requirements:
#   - Python 3.x
#   - Libraries: PyQt5, OpenCV, Requests
#   - A valid Giphy API key (for summoning memes).
#
# How to Use:
#   1. Load your video—the seeing-stone of your project.
#   2. Set an output directory to store your annotated frames and labels.
#   3. Play, pause, or step through frames, drawing bounding boxes as you go.
#   4. Click "Generate Random Meme" to bring a little light to the darkness.
#
# Notes:
#   - This tool is the ultimate companion for your labeling journey. May it 
#     guide you through Mordor-like datasets and lead you to victory.
#   - Remember: Even the smallest meme can change the course of your workflow.

# For every labeled frame, there is hope. 
# For every bounding box, there is precision.
# For every meme, there is laughter.

# "One script to rule them all, one script to find them,
# One script to bring them all, and in the data bind them."
# =============================================================================





import os
import cv2
import random
import requests
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QComboBox, QProgressBar, QMessageBox, QCheckBox, QFileDialog, QSlider
)
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QMovie, QPainter, QPen

# Global variables
debug_mode = False

# Giphy API Key (Replace with your own key)
GIPHY_API_KEY = "xCk9TIwrl1WQBBOQ65W95DmOTbVMk19w"

# Class colors
class_colors = [
    Qt.red, Qt.green, Qt.blue, Qt.cyan, Qt.magenta, Qt.yellow,
    Qt.gray, Qt.darkRed, Qt.darkGreen, Qt.darkBlue, Qt.darkCyan, Qt.darkMagenta
]

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
    """
    x_center = (box.x() + box.width() / 2) / img_width
    y_center = (box.y() + box.height() / 2) / img_height
    width = box.width() / img_width
    height = box.height() / img_height
    return x_center, y_center, width, height

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

        # Meme button
        self.generate_meme_button = QPushButton("Generate Random Meme")
        self.generate_meme_button.clicked.connect(self.generate_meme)

        # Connect buttons
        self.load_video_button.clicked.connect(self.select_video)
        self.set_output_button.clicked.connect(self.set_output_directory)
        self.play_button.clicked.connect(self.toggle_play)
        self.next_button.clicked.connect(self.next_frame)
        self.save_button.clicked.connect(self.save_current_frame)

        # Class selector
        self.class_selector = QComboBox()
        self.class_selector.addItems([f"Class {i+1}" for i in range(len(class_colors))])
        self.class_selector.currentIndexChanged.connect(self.set_current_class)

        # Debug toggle
        self.debug_toggle = QCheckBox("Enable Debug Mode")
        self.debug_toggle.stateChanged.connect(self.toggle_debug_mode)

        # Playback speed slider
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 50)  # 0.5x to 5x (multiplied by 10)
        self.speed_slider.setValue(10)     # Default to 1x speed
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.valueChanged.connect(self.adjust_playback_speed)

        # Playback speed label
        self.speed_label = QLabel("Playback Speed: 1.0x")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Meme display area
        self.meme_label = QLabel()
        self.meme_label.setAlignment(Qt.AlignCenter)

        # Layouts
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.load_video_button)
        controls_layout.addWidget(self.set_output_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.generate_meme_button)  # Add the meme button
        controls_layout.addWidget(QLabel("Class:"))
        controls_layout.addWidget(self.class_selector)
        controls_layout.addWidget(self.debug_toggle)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_slider)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(speed_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.meme_label)  # Add the meme label to the layout

        self.setLayout(main_layout)

        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Mouse Interaction
        self.video_label.mousePressEvent = self.mouse_press
        self.video_label.mouseReleaseEvent = self.mouse_release
        self.video_label.mouseMoveEvent = self.mouse_move

    def _load_styles(self):
        """
        Loads custom styles for the application.
        """
        return """
        QLabel {
            font-size: 16px;
        }
        QPushButton {
            font-size: 14px;
            padding: 8px;
        }
        QComboBox {
            font-size: 14px;
            padding: 5px;
        }
        QCheckBox {
            font-size: 14px;
            padding: 5px;
        }
        QSlider {
            height: 30px;
        }
        """

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
                    QMessageBox.information(self, "Video Loaded", f"Video '{self.video_path}' loaded successfully!")
                    if debug_mode:
                        print(f"[DEBUG] Video '{self.video_path}' loaded successfully.")
                else:
                    QMessageBox.critical(self, "Error", f"Unable to read first frame of '{self.video_path}'.")

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
        else:
            self.play_button.setText("Play")
            self.timer.stop()
            if debug_mode:
                print("[DEBUG] Video playback paused.")

    def next_frame(self):
        """
        Moves to the next frame, saves current frame and labels.
        """
        if self.cap is None or not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first!")
            return

        # Save current frame and labels before moving
        self.save_frame_and_labels()

        ret, self.frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.play_button.setText("Play")
            QMessageBox.information(self, "End of Video", "End of video reached.")
            if debug_mode:
                print("[DEBUG] End of video reached.")
            return

        self.frame_number += 1
        self.update_progress_bar()
        self.display_frame(self.frame)
        if debug_mode:
            print(f"[DEBUG] Moved to frame {self.frame_number}.")

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

    def save_frame_and_labels(self):
        """
        Saves the current frame and its bounding box in YOLO format.
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
        img_path = os.path.join(img_dir, f"frame_{self.frame_number:06d}.jpg")
        label_path = os.path.join(label_dir, f"frame_{self.frame_number:06d}.txt")

        # Save the image
        cv2.imwrite(img_path, self.frame)
        if debug_mode:
            print(f"[DEBUG] Image saved to {img_path}")

        # Save the label
        if self.persistent_box is not None:
            x_center, y_center, width, height = yolo_format(self.persistent_box, self.frame.shape[1], self.frame.shape[0])
            with open(label_path, "w") as f:
                f.write(f"{self.current_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            if debug_mode:
                print(f"[DEBUG] Label saved to {label_path}: Class {self.current_class}, Xc {x_center}, Yc {y_center}, W {width}, H {height}")
        else:
            # Create an empty label file if no bounding box
            open(label_path, 'a').close()
            if debug_mode:
                print(f"[DEBUG] Empty label file created at {label_path}")

    def adjust_playback_speed(self, value):
        """
        Adjusts the playback speed based on the slider value.
        """
        self.playback_speed = value / 10  # Convert slider value to range 0.5x to 5x
        self.speed_label.setText(f"Playback Speed: {self.playback_speed:.1f}x")
        if not self.paused:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Fallback FPS
            interval = int(1000 / (fps * self.playback_speed))
            self.timer.setInterval(interval)
            if debug_mode:
                print(f"[DEBUG] Playback speed adjusted to: {self.playback_speed:.1f}x")

    def update_frame(self):
        """
        Timer callback to update the frame during playback.
        """
        if self.cap is None or self.paused:
            return

        # Save current frame and labels before moving
        self.save_frame_and_labels()

        ret, self.frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.play_button.setText("Play")
            QMessageBox.information(self, "End of Video", "End of video reached.")
            if debug_mode:
                print("[DEBUG] End of video reached during playback.")
            return

        self.frame_number += 1
        self.update_progress_bar()
        self.display_frame(self.frame)
        if debug_mode:
            print(f"[DEBUG] Playback moved to frame {self.frame_number}")

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

    def display_frame(self, frame):
        """
        Displays the given frame with the persistent bounding box overlay.
        """
        if frame is None:
            return

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        if self.persistent_box is not None:
            pen = QPen(class_colors[self.current_class], 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.persistent_box)
            if debug_mode:
                print(f"[DEBUG] Drawing persistent bounding box: {self.persistent_box}")
        painter.end()

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)

    def mouse_press(self, event):
        """
        Handles mouse press events to start drawing a bounding box.
        """
        if self.paused and self.frame is not None:
            if event.button() == Qt.LeftButton:
                # Adjust mouse position to frame coordinates
                pos = event.pos()
                adjusted_x = (pos.x() - self.x_offset) / self.display_scale
                adjusted_y = (pos.y() - self.y_offset) / self.display_scale
                self.start_point = QPoint(int(adjusted_x), int(adjusted_y))
                self.drawing = True
                if debug_mode:
                    print(f"[DEBUG] Started drawing bounding box at {self.start_point}")

    def mouse_release(self, event):
        """
        Handles mouse release events to finalize the bounding box.
        """
        if self.drawing and self.paused and self.frame is not None:
            # Adjust mouse position to frame coordinates
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

            self.persistent_box = rect
            self.drawing = False
            if debug_mode:
                print(f"[DEBUG] Persistent bounding box updated: {rect}")

            self.display_frame(self.frame)

    def mouse_move(self, event):
        """
        Handles mouse move events to draw the bounding box dynamically.
        """
        if self.drawing and self.paused and self.frame is not None:
            # Adjust mouse position to frame coordinates
            pos = event.pos()
            adjusted_x = (pos.x() - self.x_offset) / self.display_scale
            adjusted_y = (pos.y() - self.y_offset) / self.display_scale
            current_point = QPoint(int(adjusted_x), int(adjusted_y))

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

    def set_current_class(self):
        """
        Sets the current class for bounding boxes based on the selector.
        """
        self.current_class = self.class_selector.currentIndex()
        if debug_mode:
            print(f"[DEBUG] Current class set to: {self.current_class}")

    def toggle_debug_mode(self, state):
        """
        Toggles debug mode on or off.
        """
        global debug_mode
        debug_mode = state == Qt.Checked
        if debug_mode:
            print("[DEBUG] Debug mode enabled.")
        else:
            print("[DEBUG] Debug mode disabled.")

# -------------------
# Main Entry Point
# -------------------

if __name__ == "__main__":
    app = QApplication([])
    window = LabelingApp()
    window.show()
    app.exec_()
