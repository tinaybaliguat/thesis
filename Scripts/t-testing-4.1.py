# t-testing-3_functional_tabs.py
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PyQt6.QtCore import (
    QTimer, QThread, pyqtSignal, Qt, QSize, QRect, QRectF, QPropertyAnimation,
    QEasingCurve, QPoint, QStandardPaths, QDateTime, QDate # Added QDateTime, QDate
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSlider, QFrame, QSpacerItem, QSizePolicy, QComboBox,
    QScrollArea, QGridLayout, QListWidget, QStackedLayout, QGraphicsOpacityEffect,
    QSplashScreen, QButtonGroup, QProgressBar, QStyle, QMessageBox, QTextBrowser,
    QLineEdit, QDateEdit # Added QLineEdit, QDateEdit
)
from PyQt6.QtGui import (
    QPixmap, QFont, QImage, QColor, QPainter, QBrush, QPen, QFontDatabase, QIcon, QTextOption
)
# --- Add PyQtChart ---
try:
    from PyQt5.QtChart import (
        QChart, QChartView, QBarSeries, QBarSet, QLineSeries,
        QValueAxis, QBarCategoryAxis, QDateTimeAxis, QSplineSeries # Added necessary chart components
        
    )
    PYQTCHART_AVAILABLE = True
except ImportError:
    print("Warning: PyQtChart not found. Analytics tab will be disabled.")
    print("Please install it: pip install PyQtChart")
    PYQTCHART_AVAILABLE = False
    # Dummy classes if PyQtChart is not available to avoid errors
    class QChartView(QWidget): pass
    class QChart: pass


import time
import os
import csv
import sqlite3 # Added for database
import json     # Added for storing detection lists
from datetime import datetime, timedelta # Added for date calculations

# --- Configuration ---
APP_FONT_FAMILY = "Arial"
ICON_DIR = "icons/"
DB_FILE = "detection_history.db" # SQLite database file
HISTORY_ITEMS_PER_PAGE = 12 # Items per page in the gallery

# --- Utility Functions ---
def get_device():
    """Determines the device to run the model on (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_icon(icon_name, fallback_style_enum=None):
    """Loads a custom icon, with fallback to a standard Qt icon."""
    if fallback_style_enum is None:
        fallback_style_enum = QStyle.StandardPixmap.SP_MessageBoxQuestion

    icon_path = os.path.join(ICON_DIR, icon_name)
    icon = QIcon(icon_path)
    if icon.isNull():
        if not os.path.exists(icon_path):
            print(f"Warning: Icon file '{icon_path}' does not exist. Using fallback.")
        else:
            print(f"Warning: Icon '{icon_name}' at '{icon_path}' could not be loaded. Using fallback.")
        if QApplication.instance():
            return QApplication.style().standardIcon(fallback_style_enum)
        else:
            print("Warning: QApplication instance not found for fallback icon. Returning empty QIcon.")
            return QIcon()
    return icon

# --- Database Functions ---
def init_database():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                source_type TEXT NOT NULL, -- 'file' or 'webcam'
                processing_time_ms REAL,
                confidence_threshold REAL,
                iou_threshold REAL,
                detected_objects TEXT -- JSON list of {'class': name, 'conf': value, 'box': [x1,y1,x2,y2]}
            )
        ''')
        conn.commit()
        conn.close()
        print(f"Database '{DB_FILE}' initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database Error: {e}")
        # Show critical error to user?
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText(f"Database Error: Failed to initialize database '{DB_FILE}'.\nHistory and Analytics will not work.\n\nError: {e}")
        msg_box.setWindowTitle("Database Initialization Failed")
        msg_box.exec()


def save_detection_to_db(image_path, source_type, proc_time, conf_thresh, iou_thresh, detections_list):
    """Saves a detection record to the database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detections_json = json.dumps(detections_list) # Convert list of dicts to JSON string

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (timestamp, image_path, source_type, processing_time_ms,
                                  confidence_threshold, iou_threshold, detected_objects)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, image_path, source_type, proc_time, conf_thresh, iou_thresh, detections_json))
        conn.commit()
        conn.close()
        # print(f"Saved detection from {source_type} to DB.") # Optional: for debugging
    except sqlite3.Error as e:
        print(f"Database Error: Could not save detection to DB. Error: {e}")


# --- Splash Screen (Same as before) ---
class SplashScreen(QSplashScreen):
    """Custom splash screen matching the mockup design."""
    def __init__(self):
        splash_width = 700
        splash_height = 500
        self.splash_pixmap = QPixmap(splash_width, splash_height)
        self.splash_pixmap.fill(QColor("#202430")) # Dark background from mockup

        with QPainter(self.splash_pixmap) as painter:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rect_color = QColor("#2A3040")
            rect_x = 50
            rect_y = 50
            rect_width = splash_width - 100
            rect_height = splash_height - 100
            painter.setBrush(QBrush(rect_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(rect_x, rect_y, rect_width, rect_height), 15, 15)

            logo_y_pos = rect_y + 40
            logo_icon = get_icon("app_logo_splash.png", QStyle.StandardPixmap.SP_ComputerIcon) # Use custom icon
            if not logo_icon.isNull():
                logo_pixmap_size = 70
                logo_x = rect_x + (rect_width - logo_pixmap_size) // 2
                logo_icon.paint(painter, QRect(logo_x, int(logo_y_pos), logo_pixmap_size, logo_pixmap_size))
                logo_y_pos += logo_pixmap_size + 15
            else: # Fallback drawing if icon still fails
                painter.setBrush(QColor("#4A90E2"))
                painter.drawEllipse(QRectF(rect_x + (rect_width - 70) // 2, logo_y_pos, 70, 70))
                logo_y_pos += 70 + 15

            painter.setPen(QColor("white"))
            title_font = QFont(APP_FONT_FAMILY, 32, QFont.Weight.Bold)
            painter.setFont(title_font)
            painter.drawText(QRect(rect_x, int(logo_y_pos), rect_width, 50), Qt.AlignmentFlag.AlignCenter, "YOLOv8-S")
            logo_y_pos += 45

            subtitle_font = QFont(APP_FONT_FAMILY, 18, QFont.Weight.Normal)
            painter.setFont(subtitle_font)
            painter.setPen(QColor("#A0A7B9"))
            painter.drawText(QRect(rect_x, int(logo_y_pos), rect_width, 30), Qt.AlignmentFlag.AlignCenter, "Plastic Waste Segregation")
            logo_y_pos += 45

            desc_font = QFont(APP_FONT_FAMILY, 11)
            painter.setFont(desc_font)
            painter.setPen(QColor("#C0C7D9"))
            desc_text = ("An advanced computer vision system designed to detect and "
                         "classify plastic waste materials in real-time. This application "
                         "helps improve recycling efficiency by accurately identifying "
                         "different types of plastic waste.")
            desc_rect = QRectF(rect_x + 40, logo_y_pos, rect_width - 80, 80)
            painter.drawText(desc_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, desc_text)
            logo_y_pos += desc_rect.height() + 30

            # Removed 'Launch Application' button drawing
            # Add 'Launching Model...' text
            loading_font = QFont(APP_FONT_FAMILY, 16, QFont.Weight.DemiBold)
            painter.setFont(loading_font)
            painter.setPen(QColor("#A3BE8C")) # Use a calming color like green
            loading_text = "Launching Model..."
            # Position text where button used to be
            loading_text_rect = QRectF(rect_x, logo_y_pos, rect_width, 45)
            painter.drawText(loading_text_rect, Qt.AlignmentFlag.AlignCenter, loading_text)
            logo_y_pos += 45 + 35 # Add space for version text below

            version_font = QFont(APP_FONT_FAMILY, 9)
            painter.setFont(version_font)
            painter.setPen(QColor("#707789"))
            version_text = "Version 1.1.0 • Powered by YOLOv8-S" #Updated version
            painter.drawText(QRect(rect_x, int(rect_y + rect_height - 30), rect_width, 20), Qt.AlignmentFlag.AlignCenter, version_text)

        super().__init__(self.splash_pixmap)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.opacity_animation.setDuration(1000)
        self.opacity_animation.setStartValue(0.0)
        self.opacity_animation.setEndValue(1.0)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def showEvent(self, event):
        super().showEvent(event)
        self.opacity_animation.start()

    def finish(self, widget):
        self.opacity_animation.setDirection(QPropertyAnimation.Direction.Backward)
        self.opacity_animation.finished.connect(lambda: super(SplashScreen, self).finish(widget))
        self.opacity_animation.start()

    def mousePressEvent(self, event):
        pass # Prevent splash screen from closing on click

# --- Model Loading Thread (Same as before) ---
class ModelLoadThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            self.progress.emit("Initializing...")
            time.sleep(0.2)
            self.progress.emit("Detecting hardware...")
            device = get_device()
            time.sleep(0.2)
            self.progress.emit(f"Loading YOLOv8 model on {device}...")
            model = YOLO(self.model_path).to(device)
            time.sleep(0.5)
            self.progress.emit("Finalizing setup...")
            time.sleep(0.2)
            self.finished.emit(model)
        except Exception as e:
            print(f"ModelLoadThread Error: {e}")
            self.progress.emit(f"Error: Failed to load model.")
            self.finished.emit(None)

# --- Main Application Window ---
class WasteDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.timeout.connect(self.process_current_image_if_loaded)

        self.current_image_index = -1
        self.image_paths = []
        self.processed_results_for_export = []
        self.webcam_running = False
        self.original_pixmap = None
        self.latest_detection_details = []

        # History tab state
        self.current_history_page = 1
        self.total_history_pages = 1

        # --- Initialize Database ---
        init_database()

        self.splash = SplashScreen()
        self.splash.show()
        QApplication.processEvents()

        self.setWindowTitle("YOLOv8-S Plastic Waste Segregation")
        self.setMinimumSize(1366, 768) # Increased minimum size slightly

        self.initUI()
        self.apply_stylesheet()

        model_file_path = "C:/Users/tina/Downloads/latest/thesis/weights/best.pt"
        self.model_thread = ModelLoadThread(model_file_path)
        self.model_thread.progress.connect(self.update_splash_message)
        self.model_thread.finished.connect(self.on_model_loaded)
        self.model_thread.start()

    def update_splash_message(self, message):
        self.splash.showMessage(message, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, QColor("white"))
        QApplication.processEvents()

    def on_model_loaded(self, model):
        if model is None:
            # ... (Error handling remains the same) ...
            self.splash.finish(self)
            self.show()
            QMessageBox.critical(self, "Model Load Error",
                                 "Failed to load YOLOv8 model. Please check the model path and dependencies in the console.")
            # Disable model-dependent features
            if hasattr(self, 'nav_buttons'):
                for btn_name in ["Detection", "Analytics", "History"]:
                    if btn_name in self.nav_buttons: self.nav_buttons[btn_name].setEnabled(False)
            if hasattr(self, 'webcam_btn'): self.webcam_btn.setEnabled(False)
            if hasattr(self, 'drop_frame'): self.drop_frame.setEnabled(False)
            if hasattr(self, 'export_btn'): self.export_btn.setEnabled(False)
            if hasattr(self, 'clear_btn'): self.clear_btn.setEnabled(False)
            return

        self.model = model
        if not hasattr(self.model, 'names') or not self.model.names:
             print("Warning: Model does not have class names. Using default placeholders.")
             self.model.names = {i: f"Class_{i}" for i in range(80)}

        self.update_webcam_list()
        self.setup_stat_cards() # Needs model.names

        # Enable controls
        if hasattr(self, 'conf_slider'): self.conf_slider.setEnabled(True)
        if hasattr(self, 'overlap_slider'): self.overlap_slider.setEnabled(True)
        if hasattr(self, 'drop_frame'): self.drop_frame.setEnabled(True)
        if hasattr(self, 'webcam_btn'): self.webcam_btn.setEnabled(True)
        if hasattr(self, 'export_btn'): self.export_btn.setEnabled(True)
        if hasattr(self, 'clear_btn'): self.clear_btn.setEnabled(True)
        # Enable Nav buttons
        if hasattr(self, 'nav_buttons'):
            for btn_name in ["Detection", "Analytics", "History"]:
                if btn_name in self.nav_buttons: self.nav_buttons[btn_name].setEnabled(True)

        QTimer.singleShot(500, lambda: self.splash.finish(self))
        self.show()

        # Set initial view to the first enabled button
        if hasattr(self, 'nav_button_group') and self.nav_button_group.buttons():
             for i, btn in enumerate(self.nav_button_group.buttons()):
                 if btn.isEnabled():
                     btn.setChecked(True)
                     self.handle_navigation(self.nav_button_group.id(btn)) # Use handler
                     break

    def initUI(self):
        overall_layout = QHBoxLayout(self)
        overall_layout.setContentsMargins(0, 0, 0, 0)
        overall_layout.setSpacing(0)

        self.nav_bar = self.create_navigation_bar()
        overall_layout.addWidget(self.nav_bar)

        main_content_widget = QWidget()
        main_content_layout = QVBoxLayout(main_content_widget)
        main_content_layout.setContentsMargins(0,0,0,0)
        main_content_layout.setSpacing(0)
        main_content_widget.setObjectName("mainContentWidget")

        self.header = self.create_header()
        main_content_layout.addWidget(self.header)

        self.stacked_layout_widget = QWidget()
        self.stacked_layout = QStackedLayout(self.stacked_layout_widget)

        # Create views
        self.detection_view = self.create_detection_view()
        self.analytics_view = self.create_analytics_view()
        self.history_view = self.create_history_view()
        self.definitions_view = self.create_definitions_view()

        # Add views to stacked layout
        self.stacked_layout.addWidget(self.detection_view) # Index 0
        self.stacked_layout.addWidget(self.analytics_view) # Index 1
        self.stacked_layout.addWidget(self.history_view)   # Index 2
        self.stacked_layout.addWidget(self.definitions_view) # Index 3

        main_content_layout.addWidget(self.stacked_layout_widget, 1)
        overall_layout.addWidget(main_content_widget, 1)

        # Initial view setting handled in on_model_loaded

    def create_navigation_bar(self):
        nav_widget = QWidget()
        nav_widget.setObjectName("navigationBar")
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(5, 20, 5, 20)
        nav_layout.setSpacing(15)

        self.nav_button_group = QButtonGroup(self)
        self.nav_button_group.setExclusive(True)
        # Connect the group's signal AFTER adding buttons
        self.nav_button_group.idClicked.connect(self.handle_navigation)

        buttons_config = [
            ("Detection", 0, "detection.svg", QStyle.StandardPixmap.SP_DesktopIcon),
            ("Analytics", 1, "analytics.svg", QStyle.StandardPixmap.SP_FileDialogDetailedView),
            ("History", 2, "history.svg", QStyle.StandardPixmap.SP_FileDialogContentsView),
            ("Definitions", 3, "info.svg", QStyle.StandardPixmap.SP_MessageBoxInformation)
        ]

        self.nav_buttons = {}
        for name, page_index, icon_svg_name, fallback_enum in buttons_config:
            icon = get_icon(icon_svg_name, fallback_enum)
            btn = QPushButton(icon, "")
            btn.setIconSize(QSize(28, 28)) # Slightly larger icons
            btn.setCheckable(True)
            btn.setToolTip(name)
            btn.setObjectName("navButton")
            nav_layout.addWidget(btn)
            self.nav_button_group.addButton(btn, page_index)
            self.nav_buttons[name] = btn
            # Don't connect individual clicked signals here, use the group signal

        # Disable model-dependent buttons initially
        if self.model is None:
            for btn_name in ["Detection", "Analytics", "History"]:
                 if btn_name in self.nav_buttons: self.nav_buttons[btn_name].setEnabled(False)

        nav_layout.addStretch(1)
        return nav_widget

    def handle_navigation(self, index):
        """Handles switching tabs and refreshing data if needed."""
        self.stacked_layout.setCurrentIndex(index)
        # Refresh data when switching TO these tabs
        if index == 1: # Analytics
            self.update_analytics_view()
        elif index == 2: # History
            self.update_history_view() # Load page 1 with default filters

    def create_header(self):
        # ... (Header creation remains the same) ...
        header_widget = QWidget()
        header_widget.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 0, 20, 0)

        app_logo_label = QLabel()
        logo_icon = get_icon("app_logo.png", QStyle.StandardPixmap.SP_ComputerIcon)
        app_logo_label.setPixmap(logo_icon.pixmap(QSize(30,30)))
        header_layout.addWidget(app_logo_label)
        header_layout.addSpacing(10)

        title_label = QLabel("YOLOv8-S Plastic Waste Segregation")
        title_label.setObjectName("headerTitleLabel")
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        exit_icon = get_icon("exit.png", QStyle.StandardPixmap.SP_DialogCloseButton)
        exit_btn = QPushButton(exit_icon, " Exit")
        exit_btn.setObjectName("headerExitButton")
        exit_btn.clicked.connect(self.close)
        header_layout.addWidget(exit_btn)
        return header_widget

    def create_detection_view(self):
        detection_widget = QWidget()
        overall_detection_layout = QHBoxLayout(detection_widget)
        overall_detection_layout.setContentsMargins(10,10,10,10)
        overall_detection_layout.setSpacing(10)

        # --- Left Panel (Controls - Mostly Same) ---
        left_panel = QFrame()
        left_panel.setObjectName("controlPanel")
        left_panel.setFixedWidth(280) # Slightly wider for better label spacing
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(15,15,15,15)
        left_panel_layout.setSpacing(20)

        # Model Config Group (Same)
        model_config_group = QWidget()
        model_config_layout = QVBoxLayout(model_config_group); model_config_layout.setContentsMargins(0,0,0,0); model_config_layout.setSpacing(8)
        config_title = QLabel("Model Configuration"); config_title.setObjectName("panelTitleLabel")
        model_config_layout.addWidget(config_title)
        self.conf_label = QLabel("Confidence Threshold: 50%")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal); self.conf_slider.setRange(1,100); self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        model_config_layout.addWidget(self.conf_label); model_config_layout.addWidget(self.conf_slider)
        self.overlap_label = QLabel("Overlap Threshold (IoU): 50%")
        self.overlap_slider = QSlider(Qt.Orientation.Horizontal); self.overlap_slider.setRange(1,100); self.overlap_slider.setValue(50)
        self.overlap_slider.valueChanged.connect(self.update_overlap)
        model_config_layout.addWidget(self.overlap_label); model_config_layout.addWidget(self.overlap_slider)
        left_panel_layout.addWidget(model_config_group)

        # Input Source Group (Word Wrap added)
        input_source_group = QWidget()
        input_source_layout = QVBoxLayout(input_source_group); input_source_layout.setContentsMargins(0,0,0,0); input_source_layout.setSpacing(8)
        input_title = QLabel("Input Source"); input_title.setObjectName("panelTitleLabel")
        input_source_layout.addWidget(input_title)
        self.drop_frame = QFrame(); self.drop_frame.setObjectName("dropFrame"); self.drop_frame.setAcceptDrops(True)
        self.drop_frame.dragEnterEvent = self.dragEnterEvent; self.drop_frame.dropEvent = self.dropEvent
        drop_layout = QVBoxLayout(self.drop_frame); drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter); drop_layout.setSpacing(10)
        upload_icon_label = QLabel()
        upload_svg_icon = get_icon("upload.svg", QStyle.StandardPixmap.SP_ArrowUp)
        upload_icon_label.setPixmap(upload_svg_icon.pixmap(QSize(40,40)))
        upload_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drag_label = QLabel("Drag and drop files here\nor click to browse"); drag_label.setAlignment(Qt.AlignmentFlag.AlignCenter); drag_label.setObjectName("dropFrameText")
        file_format_label = QLabel("Limit 200MB. JPG, PNG, BMP, WEBP"); file_format_label.setAlignment(Qt.AlignmentFlag.AlignCenter); file_format_label.setObjectName("dropFrameSubText")
        file_format_label.setWordWrap(True) # <<< Added Word Wrap fix
        drop_layout.addWidget(upload_icon_label); drop_layout.addWidget(drag_label); drop_layout.addWidget(file_format_label)
        self.drop_frame.mousePressEvent = lambda event: self.upload_images()
        input_source_layout.addWidget(self.drop_frame)
        left_panel_layout.addWidget(input_source_group)

        # Webcam Group (Same)
        webcam_group = QWidget()
        webcam_layout = QVBoxLayout(webcam_group); webcam_layout.setContentsMargins(0,0,0,0); webcam_layout.setSpacing(8)
        webcam_title_layout = QHBoxLayout(); webcam_title = QLabel("Webcam Input"); webcam_title.setObjectName("panelTitleLabel")
        self.webcam_status_indicator = QLabel("●"); self.webcam_status_indicator.setObjectName("webcamStatusOffline")
        webcam_title_layout.addWidget(webcam_title); webcam_title_layout.addStretch(); webcam_title_layout.addWidget(self.webcam_status_indicator)
        webcam_layout.addLayout(webcam_title_layout)
        self.webcam_dropdown = QComboBox(); self.webcam_dropdown.setToolTip("Select webcam device")
        webcam_layout.addWidget(self.webcam_dropdown)
        play_icon = get_icon("webcam_play.svg", QStyle.StandardPixmap.SP_MediaPlay)
        self.webcam_btn = QPushButton(play_icon, " Start Webcam Detection")
        self.webcam_btn.setObjectName("startWebcamButton"); self.webcam_btn.clicked.connect(self.toggle_webcam)
        webcam_layout.addWidget(self.webcam_btn)
        left_panel_layout.addWidget(webcam_group)
        left_panel_layout.addStretch(1)
        overall_detection_layout.addWidget(left_panel)

        # --- Right Panel (Display & Stats - Same Structure) ---
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(0,0,0,0)
        right_panel_layout.setSpacing(10)
        # Image Display Widget (Same)
        self.image_display_widget = QFrame()
        self.image_display_widget.setObjectName("imageDisplayWidget")
        display_widget_layout = QVBoxLayout(self.image_display_widget)
        display_widget_layout.setContentsMargins(5,5,5,5); display_widget_layout.setSpacing(5)
        self.image_scroll_area = QScrollArea(); self.image_scroll_area.setObjectName("imageScrollArea")
        self.image_scroll_area.setWidgetResizable(True); self.image_scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label = QLabel("Upload an image or start webcam to see output."); self.image_label.setObjectName("imageDisplayLabel")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.image_label.setMinimumSize(400, 300)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_scroll_area.setWidget(self.image_label)
        display_widget_layout.addWidget(self.image_scroll_area, 1)
        # Image Nav Layout (Same)
        image_nav_layout = QHBoxLayout(); image_nav_layout.setContentsMargins(0,0,0,0)
        prev_icon = get_icon("left-arrow.png", QStyle.StandardPixmap.SP_ArrowLeft)
        self.prev_btn = QPushButton(prev_icon, ""); self.prev_btn.setObjectName("imageNavButton"); self.prev_btn.clicked.connect(self.prev_image)
        self.image_count_label = QLabel("— / —"); self.image_count_label.setObjectName("imageCountLabel"); self.image_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        next_icon = get_icon("right-arrow.png", QStyle.StandardPixmap.SP_ArrowRight)
        self.next_btn = QPushButton(next_icon, ""); self.next_btn.setObjectName("imageNavButton"); self.next_btn.clicked.connect(self.next_image)
        image_nav_layout.addWidget(self.prev_btn); image_nav_layout.addStretch()
        image_nav_layout.addWidget(self.image_count_label); image_nav_layout.addStretch()
        image_nav_layout.addWidget(self.next_btn)
        display_widget_layout.addLayout(image_nav_layout)
        right_panel_layout.addWidget(self.image_display_widget, 1)
        # Stats Area Widget (Same structure, setup_stat_cards handles content)
        self.stats_area_widget = QWidget()
        stats_area_outer_layout = QVBoxLayout(self.stats_area_widget); stats_area_outer_layout.setContentsMargins(0,0,0,0)
        stats_content_frame = QFrame(); stats_content_frame.setObjectName("statsContentFrame")
        stats_content_frame_layout = QVBoxLayout(stats_content_frame); stats_content_frame_layout.setContentsMargins(15,15,15,15); stats_content_frame_layout.setSpacing(10)
        stats_header_layout = QHBoxLayout(); stats_title = QLabel("Detection Statistics"); stats_title.setObjectName("panelTitleLabel")
        stats_header_layout.addWidget(stats_title); stats_header_layout.addStretch()
        export_icon = get_icon("export.svg", QStyle.StandardPixmap.SP_DialogSaveButton)
        self.export_btn = QPushButton(export_icon, " Export"); self.export_btn.setObjectName("statsButton"); self.export_btn.clicked.connect(self.export_statistics)
        clear_icon = get_icon("clear.svg", QStyle.StandardPixmap.SP_TrashIcon)
        self.clear_btn = QPushButton(clear_icon, " Clear Current"); self.clear_btn.setObjectName("statsButtonSecondary"); self.clear_btn.clicked.connect(self.clear_current_detection_display) # Renamed for clarity
        stats_header_layout.addWidget(self.export_btn); stats_header_layout.addSpacing(8); stats_header_layout.addWidget(self.clear_btn)
        stats_content_frame_layout.addLayout(stats_header_layout)
        self.plastic_stats_cards_layout_row1 = QHBoxLayout(); self.plastic_stats_cards_layout_row1.setSpacing(10)
        self.plastic_stats_cards_layout_row2 = QHBoxLayout(); self.plastic_stats_cards_layout_row2.setSpacing(10)
        stats_content_frame_layout.addLayout(self.plastic_stats_cards_layout_row1); stats_content_frame_layout.addLayout(self.plastic_stats_cards_layout_row2)
        lower_stats_layout = QHBoxLayout(); lower_stats_layout.setSpacing(10)
        self.stat_total_items = self.create_info_card("Total Items", "0")
        self.stat_proc_time = self.create_info_card("Processing Time", "0ms")
        # Removed Accuracy card, as it's hard to determine without ground truth
        # self.stat_accuracy = self.create_info_card("Accuracy", "N/A")
        lower_stats_layout.addWidget(self.stat_total_items)
        lower_stats_layout.addWidget(self.stat_proc_time)
        # lower_stats_layout.addWidget(self.stat_accuracy)
        stats_content_frame_layout.addLayout(lower_stats_layout)
        stats_scroll_area = QScrollArea(); stats_scroll_area.setObjectName("statsArea"); stats_scroll_area.setWidgetResizable(True)
        stats_scroll_area.setWidget(stats_content_frame); stats_scroll_area.setMaximumHeight(180) # Increased max height slightly
        stats_area_outer_layout.addWidget(stats_scroll_area)
        right_panel_layout.addWidget(self.stats_area_widget)
        overall_detection_layout.addWidget(right_panel, 1)

        self.update_navigation_buttons()
        if self.model is None: # Initial state if model fails
            self.conf_slider.setEnabled(False); self.overlap_slider.setEnabled(False)
            self.drop_frame.setEnabled(False); self.webcam_btn.setEnabled(False)
            self.export_btn.setEnabled(False); self.clear_btn.setEnabled(False)

        return detection_widget

    def setup_stat_cards(self):
        # ... (Setup remains largely the same, ensure self.plastic_classes is populated) ...
         # Clear existing cards if any
        for layout in [self.plastic_stats_cards_layout_row1, self.plastic_stats_cards_layout_row2]:
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget: widget.deleteLater()

        self.plastic_classes = []
        if self.model and hasattr(self.model, 'names'):
            defined_plastics = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"]
            model_plastic_names = []
            if isinstance(self.model.names, dict):
                temp_names = sorted(list(self.model.names.values())) # Sort for consistency
                # Prioritize defined plastics if they exist in model names
                for dp in defined_plastics:
                    for name_val in temp_names:
                         if dp.lower() in name_val.lower() and dp not in model_plastic_names:
                              model_plastic_names.append(dp)
                # Add remaining model names if needed, up to a limit (e.g., 6 total)
                remaining_model_names = [name for name in temp_names if not any(dp.lower() in name.lower() for dp in defined_plastics)]
                needed = max(0, 6 - len(model_plastic_names))
                model_plastic_names.extend(remaining_model_names[:needed])

            if model_plastic_names:
                 self.plastic_classes = model_plastic_names # Use the derived list
            else: # Fallback if specific plastics not found or model names empty
                 self.plastic_classes = defined_plastics # Fall back to standard 6
                 if not self.plastic_classes: # Absolute fallback
                     self.plastic_classes = [f"Class_{i}" for i in range(min(6, len(self.model.names if self.model and hasattr(self.model, 'names') else {})))]
        else: # Fallback if model or model.names is not available
            self.plastic_classes = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"] # Default
            print("Warning: Model names not available for stat cards, using default plastic classes.")

        self.stat_cards = {}
        num_classes = len(self.plastic_classes)
        mid_point = (num_classes + 1) // 2

        for i, class_name in enumerate(self.plastic_classes):
            card = self.create_stat_card(class_name, "0", 0)
            self.stat_cards[class_name] = card
            layout_target = self.plastic_stats_cards_layout_row1 if i < mid_point else self.plastic_stats_cards_layout_row2
            layout_target.addWidget(card)

        self.plastic_stats_cards_layout_row1.update()
        self.plastic_stats_cards_layout_row2.update()
        self.clear_detection_statistics_display()


    def create_stat_card(self, title_text, value_text, progress_value):
        # ... (Remains the same, check height/font if cut-off persists) ...
        card = QFrame(); card.setObjectName("statCard")
        layout = QVBoxLayout(card); layout.setSpacing(4); layout.setContentsMargins(10, 7, 10, 7) # Slightly adjusted
        title = QLabel(title_text); title.setObjectName("statCardTitle")
        value_label = QLabel(value_text); value_label.setObjectName("statCardValue")
        progress = QProgressBar(); progress.setValue(progress_value); progress.setTextVisible(False)
        progress.setObjectName("statCardProgress")
        layout.addWidget(title); layout.addWidget(value_label); layout.addWidget(progress)
        card.setFixedHeight(85) # Adjusted height
        return card

    def create_info_card(self, title_text, value_text):
        # ... (Remains the same) ...
        card = QFrame(); card.setObjectName("statCard") # Use same style as stat card for consistency
        layout = QVBoxLayout(card); layout.setSpacing(5); layout.setContentsMargins(10, 8, 10, 8)
        title = QLabel(title_text); title.setObjectName("statCardTitle")
        value_label = QLabel(value_text); value_label.setObjectName("infoCardValue") # Larger font style
        layout.addWidget(title); layout.addWidget(value_label)
        # Let height be determined by content or set minimum
        card.setMinimumHeight(70) # Example minimum
        return card

    # --- Analytics View ---
    def create_analytics_view(self):
        """Creates the Analytics dashboard view widget."""
        analytics_widget = QWidget()
        analytics_widget.setObjectName("analyticsView")
        main_layout = QVBoxLayout(analytics_widget)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(15)

        # --- Top Bar: Title and Time Range ---
        top_bar_layout = QHBoxLayout()
        title_label = QLabel("📊 Analytics Dashboard")
        title_label.setObjectName("viewTitleLabel") # Use a specific style
        top_bar_layout.addWidget(title_label)
        top_bar_layout.addStretch()

        self.analytics_time_combo = QComboBox()
        self.analytics_time_combo.addItems(["Last 7 days", "Last 30 days", "All time"])
        self.analytics_time_combo.setFixedWidth(150)
        self.analytics_time_combo.setObjectName("timeFilterCombo")
        self.analytics_time_combo.currentTextChanged.connect(lambda: self.update_analytics_view()) # Update on change
        top_bar_layout.addWidget(self.analytics_time_combo)
        main_layout.addLayout(top_bar_layout)

        # --- Summary Cards ---
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(15)
        self.analytics_total_items_card = self.create_info_card("Total Items Detected", "0")
        self.analytics_avg_proc_time_card = self.create_info_card("Avg. Processing Time", "0ms")
        self.analytics_avg_conf_card = self.create_info_card("Avg. Confidence", "0%") # Example metric
        summary_layout.addWidget(self.analytics_total_items_card)
        summary_layout.addWidget(self.analytics_avg_proc_time_card)
        summary_layout.addWidget(self.analytics_avg_conf_card)
        main_layout.addLayout(summary_layout)

        # --- Charts ---
        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(15)

        # Chart 1: Detections per Day (Bar Chart)
        chart_detections_widget = QFrame()
        chart_detections_widget.setObjectName("chartFrame")
        chart_detections_layout = QVBoxLayout(chart_detections_widget)
        chart_title1 = QLabel("Daily Detections"); chart_title1.setObjectName("chartTitleLabel")
        self.detections_chart_view = QChartView()
        self.detections_chart_view.setMinimumHeight(250)
        chart_detections_layout.addWidget(chart_title1)
        chart_detections_layout.addWidget(self.detections_chart_view, 1)
        if not PYQTCHART_AVAILABLE: self.detections_chart_view.setVisible(False) # Hide if lib missing
        charts_layout.addWidget(chart_detections_widget, 1) # Equal stretch factor

        # Chart 2: Class Distribution (Could be Pie or Bar Chart)
        # For simplicity, let's reuse the structure - maybe show confidence trend?
        chart_confidence_widget = QFrame()
        chart_confidence_widget.setObjectName("chartFrame")
        chart_confidence_layout = QVBoxLayout(chart_confidence_widget)
        chart_title2 = QLabel("Confidence Trend"); chart_title2.setObjectName("chartTitleLabel")
        self.confidence_chart_view = QChartView()
        self.confidence_chart_view.setMinimumHeight(250)
        chart_confidence_layout.addWidget(chart_title2)
        chart_confidence_layout.addWidget(self.confidence_chart_view, 1)
        if not PYQTCHART_AVAILABLE: self.confidence_chart_view.setVisible(False)
        charts_layout.addWidget(chart_confidence_widget, 1) # Equal stretch factor

        main_layout.addLayout(charts_layout, 1) # Give charts vertical stretch

        if not PYQTCHART_AVAILABLE:
             warning_label = QLabel("PyQtChart library not found. Charts are disabled.\nPlease install it: pip install PyQtChart")
             warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
             warning_label.setStyleSheet("color: #D08770; font-size: 14px; font-weight: bold;")
             main_layout.addWidget(warning_label)

        return analytics_widget

    # --- History/Gallery View ---
    def create_history_view(self):
        """Creates the History/Gallery view widget."""
        history_widget = QWidget()
        history_widget.setObjectName("historyView")
        main_layout = QVBoxLayout(history_widget)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(10)

        # --- Filter Bar ---
        filter_bar_layout = QHBoxLayout()
        filter_bar_layout.setSpacing(10)

        search_icon = get_icon("search.svg", QStyle.StandardPixmap.SP_FileDialogContentsView) # Example icon
        search_label = QLabel() # Use label for icon
        search_label.setPixmap(search_icon.pixmap(QSize(18,18)))
        self.history_search_input = QLineEdit()
        self.history_search_input.setPlaceholderText("Search by image name or detected class...")
        self.history_search_input.setObjectName("searchInput")
        # Use timer for delayed search trigger
        self.history_search_timer = QTimer()
        self.history_search_timer.setSingleShot(True)
        self.history_search_timer.timeout.connect(lambda: self.update_history_view(page=1))
        self.history_search_input.textChanged.connect(lambda: self.history_search_timer.start(500)) # 500ms delay

        self.history_filter_combo = QComboBox()
        # Populate dynamically later or use fixed list for now
        self.history_filter_combo.addItem("Filter by type: All")
        # self.history_filter_combo.addItems(self.plastic_classes) # Populate after model load?
        self.history_filter_combo.setFixedWidth(180)
        self.history_filter_combo.currentIndexChanged.connect(lambda: self.update_history_view(page=1))

        # Date Range Filter
        self.history_date_start = QDateEdit()
        self.history_date_start.setCalendarPopup(True)
        self.history_date_start.setDisplayFormat("yyyy-MM-dd")
        self.history_date_start.setDate(QDate.currentDate().addMonths(-1)) # Default: 1 month ago
        self.history_date_start.dateChanged.connect(lambda: self.update_history_view(page=1))

        date_label = QLabel("to")

        self.history_date_end = QDateEdit()
        self.history_date_end.setCalendarPopup(True)
        self.history_date_end.setDisplayFormat("yyyy-MM-dd")
        self.history_date_end.setDate(QDate.currentDate()) # Default: today
        self.history_date_end.dateChanged.connect(lambda: self.update_history_view(page=1))


        filter_bar_layout.addWidget(search_label)
        filter_bar_layout.addWidget(self.history_search_input, 1) # Allow search to stretch
        filter_bar_layout.addWidget(self.history_filter_combo)
        filter_bar_layout.addWidget(QLabel("Date:"))
        filter_bar_layout.addWidget(self.history_date_start)
        filter_bar_layout.addWidget(date_label)
        filter_bar_layout.addWidget(self.history_date_end)
        main_layout.addLayout(filter_bar_layout)

        # --- Gallery Area ---
        self.history_scroll_area = QScrollArea()
        self.history_scroll_area.setWidgetResizable(True)
        self.history_scroll_area.setObjectName("galleryScrollArea")
        self.history_scroll_area.setFrameShape(QFrame.Shape.NoFrame) # No border for scroll area itself

        self.gallery_content_widget = QWidget() # Container for the grid
        self.gallery_grid_layout = QGridLayout(self.gallery_content_widget)
        self.gallery_grid_layout.setSpacing(15)
        self.gallery_grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft) # Align items top-left

        self.history_scroll_area.setWidget(self.gallery_content_widget)
        main_layout.addWidget(self.history_scroll_area, 1) # Allow gallery to stretch

        # --- Pagination Controls ---
        pagination_layout = QHBoxLayout()
        pagination_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pagination_layout.setSpacing(10)

        prev_icon = get_icon("left-arrow.png", QStyle.StandardPixmap.SP_ArrowLeft)
        self.history_prev_btn = QPushButton(prev_icon, " Previous")
        self.history_prev_btn.setObjectName("paginationButton")
        self.history_prev_btn.clicked.connect(self.history_prev_page)

        self.history_page_label = QLabel("Page 1 / 1")
        self.history_page_label.setObjectName("pageLabel")

        next_icon = get_icon("right-arrow.png", QStyle.StandardPixmap.SP_ArrowRight)
        self.history_next_btn = QPushButton("Next ") # Space for icon alignment
        self.history_next_btn.setIcon(next_icon)
        self.history_next_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft) # Icon on right
        self.history_next_btn.setObjectName("paginationButton")
        self.history_next_btn.clicked.connect(self.history_next_page)

        pagination_layout.addWidget(self.history_prev_btn)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.history_page_label)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.history_next_btn)
        main_layout.addLayout(pagination_layout)

        return history_widget

    # --- Definitions View (Remains the same) ---
    def create_definitions_view(self):
        # ... (Keep the existing definitions view code) ...
        widget = QWidget()
        widget.setStyleSheet("background-color: #2A3040; border-radius: 8px; padding:10px;")
        layout = QVBoxLayout(widget)

        title_label = QLabel("Plastic Resin Identification Codes")
        title_label.setObjectName("panelTitleLabel") # Use panel title style
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        text_browser = QTextBrowser()
        text_browser.setObjectName("definitionsBrowser")
        text_browser.setOpenExternalLinks(True)
        html_content = """
<html><body style='color: #E0E1E3; font-family: Arial, sans-serif; font-size: 13px;'>
<h2 style='color: #4A90E2;'>1. PET or PETE (Polyethylene Terephthalate)</h2>
<p><b>Common Uses:</b> Soft drink bottles, water bottles, food containers (like peanut butter jars), salad dressing bottles, some textile fibers.</p>
<p><b>Properties:</b> Clear, strong, lightweight, good barrier to gas and moisture. Often recycled.</p>
<hr>
<h2 style='color: #4A90E2;'>2. HDPE (High-Density Polyethylene)</h2>
<p><b>Common Uses:</b> Milk jugs, detergent bottles, shampoo bottles, grocery bags, some pipes, toys.</p>
<p><b>Properties:</b> Stiff, strong, good chemical resistance, relatively easy to process. Often recycled.</p>
<hr>
<h2 style='color: #4A90E2;'>3. PVC (Polyvinyl Chloride) or V (Vinyl)</h2>
<p><b>Common Uses:</b> Pipes, window frames, siding, flooring, some packaging (like blister packs), credit cards, some bottles (e.g., for chemicals).</p>
<p><b>Properties:</b> Versatile, can be rigid or flexible, good chemical resistance, durable. Recycling is less common and more complex.</p>
<hr>
<h2 style='color: #4A90E2;'>4. LDPE (Low-Density Polyethylene)</h2>
<p><b>Common Uses:</b> Plastic bags (bread bags, dry cleaning bags), shrink wrap, squeezable bottles, coatings for paper milk cartons.</p>
<p><b>Properties:</b> Flexible, tough, good moisture barrier. Sometimes recycled, often as plastic film.</p>
<hr>
<h2 style='color: #4A90E2;'>5. PP (Polypropylene)</h2>
<p><b>Common Uses:</b> Yogurt containers, margarine tubs, syrup bottles, bottle caps, straws, some car parts, ropes, carpets.</p>
<p><b>Properties:</b> Strong, good chemical resistance, high melting point. Often recycled.</p>
<hr>
<h2 style='color: #4A90E2;'>6. PS (Polystyrene)</h2>
<p><b>Common Uses:</b> Disposable cutlery and plates, CD cases, food containers (like egg cartons and meat trays), protective packaging (expanded polystyrene foam).</p>
<p><b>Properties:</b> Can be rigid or foamed, clear or colored, good insulator. Recycling can be challenging, especially for foam versions.</p>
<hr>
<p style='font-size:11px; color: #A0A7B9;'><i>Note: Recycling availability varies by region and facility. Always check with your local recycling program.</i></p>
</body></html>
"""
        text_browser.setHtml(html_content)
        layout.addWidget(text_browser)
        return widget


    def apply_stylesheet(self):
        # --- Add/Modify Styles for New Views ---
        self.setStyleSheet(f"""
            /* Existing styles remain here... */
            QWidget {{
                font-family: "{APP_FONT_FAMILY}", sans-serif;
                color: #E0E1E3;
            }}
            /* ... other styles ... */

            #controlPanel {{
                background-color: #2A3040;
                border-radius: 8px;
            }}
            QLabel#panelTitleLabel {{
                font-size: 15px; /* Reduced from 16px */
                font-weight: 600;
                color: white;
                margin-bottom: 8px;
            }}
            /* ... other styles ... */
             #dropFrame {{
                background-color: #202430;
                border: 1px dashed #444A58;
                border-radius: 8px;
                padding: 15px; /* Reduced padding slightly */
                min-height: 145px; /* Adjusted height */
            }}
            /* ... Drop frame text styles ... */
            #dropFrameText {{ font-size: 13px; color: #A0A7B9; }}
            #dropFrameSubText {{ font-size: 11px; color: #707789; }}

            /* ... Other styles ... */

            QFrame#statCard {{
                background-color: #202430; /* Darker cards */
                border-radius: 6px;
                padding: 8px 10px; /* Adjusted padding */
                /* min-height: 75px; Let fixed height control */
            }}
            QLabel#statCardTitle {{
                font-size: 11px; /* Smaller title */
                color: #A0A7B9;
                margin-bottom: 3px; /* Reduced margin */
                font-weight: 500;
            }}
            QLabel#statCardValue {{
                font-size: 16px; /* Smaller value */
                font-weight: bold;
                color: white;
                margin-bottom: 4px; /* Reduced margin */
            }}
             QLabel#infoCardValue {{ /* For larger info cards like total items */
                font-size: 22px; /* Larger font */
                font-weight: bold;
                color: white;
            }}
            QProgressBar#statCardProgress {{
                height: 6px; /* Slimmer bar */
                text-align: center;
                border-radius: 3px;
                background-color: #353A4C;
            }}
            QProgressBar::chunk#statCardProgress {{
                background-color: #4A90E2;
                border-radius: 3px;
            }}

             /* --- Styles for Analytics/History/Definitions --- */
            #analyticsView, #historyView {{
                background-color: #202430; /* Match main background */
            }}
            QLabel#viewTitleLabel {{
                font-size: 18px;
                font-weight: 600;
                color: white;
                margin-bottom: 10px;
            }}
             QComboBox#timeFilterCombo {{ /* Specific style for time filter */
                background-color: #353A4C;
                border-radius: 6px;
                padding: 6px 10px; /* Slightly smaller padding */
                border: 1px solid #202430;
                font-size: 12px;
                color: #E0E1E3;
            }}
            QComboBox#timeFilterCombo:hover {{ border-color: #4A90E2; }}
            QComboBox#timeFilterCombo::drop-down {{ border: none; width: 18px; }}
             QComboBox QAbstractItemView {{ /* Shared dropdown style */
                background-color: #353A4C;
                border: 1px solid #4A90E2;
                selection-background-color: #4A90E2;
                selection-color: white;
                color: #E0E1E3;
                padding: 5px; /* Smaller padding */
                outline: none;
            }}

            QFrame#chartFrame {{
                background-color: #2A3040;
                border-radius: 8px;
                padding: 15px;
            }}
            QLabel#chartTitleLabel {{
                font-size: 14px;
                font-weight: 500;
                color: #E0E1E3;
                margin-bottom: 8px;
                /* border-bottom: 1px solid #353A4C; */ /* Optional separator */
                /* padding-bottom: 5px; */
            }}
            QChartView {{
                background-color: transparent; /* Chart view itself transparent */
            }}

            QLineEdit#searchInput {{
                background-color: #2A3040;
                border: 1px solid #353A4C;
                border-radius: 6px;
                padding: 7px 10px;
                font-size: 13px;
                color: #E0E1E3;
            }}
            QLineEdit#searchInput:focus {{
                border-color: #4A90E2;
            }}
            QDateEdit {{
                background-color: #353A4C;
                border: 1px solid #202430;
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 12px;
                color: #E0E1E3;
            }}
             QDateEdit:hover {{ border-color: #4A90E2; }}
             QDateEdit::drop-down {{ border: none; width: 18px; }}
             /* Style calendar popup if needed */

            #galleryScrollArea {{
                background-color: #202430; /* Match background */
                border: none;
            }}
            #galleryContentWidget {{
                 background-color: transparent; /* Content widget transparent */
            }}

            QFrame#galleryItemFrame {{
                background-color: #2A3040;
                border-radius: 8px;
                /* Add subtle border? */
                /* border: 1px solid #353A4C; */
            }}
             QFrame#galleryItemFrame:hover {{
                 background-color: #353A4C; /* Highlight on hover */
            }}
            QLabel#galleryThumbLabel {{
                background-color: #1A1D25; /* Background for thumb area */
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-height: 120px; /* Ensure space for thumbnail */
                max-height: 120px;
            }}
            QLabel#galleryInfoLabel {{
                font-size: 12px;
                color: #E0E1E3;
                font-weight: 500;
                padding: 5px 8px 0px 8px; /* Top padding */
            }}
            QLabel#galleryDateLabel {{
                font-size: 10px;
                color: #A0A7B9;
                padding: 0px 8px 5px 8px; /* Bottom padding */
            }}
            QPushButton#galleryDetailsButton {{
                background-color: #4A90E2;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 11px;
                font-weight: 500;
                padding: 4px 8px;
                margin: 0px 8px 8px 8px; /* Margin around button */
            }}
             QPushButton#galleryDetailsButton:hover {{ background-color: #357ABD; }}

            QPushButton#paginationButton {{
                background-color: #353A4C;
                color: #E0E1E3;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
                padding: 7px 12px;
            }}
            QPushButton#paginationButton:hover {{ background-color: #4A5064; }}
            QPushButton#paginationButton:disabled {{
                background-color: #2A3040;
                color: #707789;
            }}
            QLabel#pageLabel {{
                font-size: 13px;
                color: #A0A7B9;
                font-weight: 500;
            }}

            #definitionsBrowser {{ /* Existing styles */ }}
            /* ... scrollbar styles ... */

        """)

    # --- Event Handlers (dragEnter, dropEvent - same) ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: event.ignore()

    def dropEvent(self, event):
        if not self.model: return
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        if image_files: self.load_dropped_images(image_files)

    # --- Image Loading/Processing (Modified to save to DB) ---
    def load_dropped_images(self, file_paths):
        if self.webcam_running: self.toggle_webcam()
        self.image_paths = file_paths
        self.current_image_index = -1
        if self.image_paths:
            self.current_image_index = 0
            self.run_model_on_image_path(self.image_paths[0]) # This will now save to DB
        self.update_navigation_buttons()
        self.update_image_count_label()

    def update_confidence(self, value):
        self.conf_label.setText(f"Confidence Threshold: {value}%")
        if self.model: self.slider_timer.start(300)

    def update_overlap(self, value):
        self.overlap_label.setText(f"Overlap Threshold (IoU): {value}%")
        if self.model: self.slider_timer.start(300)

    def update_webcam_list(self):
        # ... (Remains the same) ...
        self.webcam_dropdown.clear()
        available_cams = []
        for i in range(5): # Check first 5 indices
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # Use DSHOW backend for Windows if available
            if cap.isOpened():
                available_cams.append((f"Camera {i}", i))
                cap.release()
            else: # Try without DSHOW as fallback
                 cap = cv2.VideoCapture(i)
                 if cap.isOpened():
                     available_cams.append((f"Camera {i} (Default)", i))
                     cap.release()


        if available_cams:
            for name, idx in available_cams:
                self.webcam_dropdown.addItem(name, idx)
            if self.model and hasattr(self, 'webcam_btn'): self.webcam_btn.setEnabled(True)
        else:
            self.webcam_dropdown.addItem("No webcams found", -1)
            if hasattr(self, 'webcam_btn'): self.webcam_btn.setEnabled(False)

    def update_navigation_buttons(self):
        # ... (Remains the same) ...
        has_images = bool(self.image_paths) and not self.webcam_running
        can_navigate = has_images and self.model is not None

        if hasattr(self, 'prev_btn'):
             self.prev_btn.setEnabled(can_navigate and self.current_image_index > 0)
        if hasattr(self, 'next_btn'):
             self.next_btn.setEnabled(can_navigate and self.current_image_index < len(self.image_paths) - 1)

    def update_image_count_label(self):
        # ... (Remains the same) ...
        if self.image_paths and self.current_image_index != -1 and not self.webcam_running:
            self.image_count_label.setText(f"{self.current_image_index + 1} / {len(self.image_paths)}")
        else:
            self.image_count_label.setText("— / —")

    def upload_images(self):
        # ... (Remains the same) ...
        if not self.model:
            QMessageBox.warning(self, "Model Not Ready", "The model is not loaded yet. Please wait.")
            return
        if self.webcam_running: self.toggle_webcam()

        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_names:
            self.load_dropped_images(file_names) # Use the common loading function


    def process_current_image_if_loaded(self):
        if not self.webcam_running and self.image_paths and self.current_image_index >= 0 and self.model:
            self.run_model_on_image_path(self.image_paths[self.current_image_index])

    def run_model_on_image_path(self, file_path):
        """Runs model on a single image file and saves result to DB."""
        if not self.model:
            self.image_label.setText("Model not loaded. Cannot process image.")
            self.clear_detection_statistics_display()
            return
        try:
            confidence = self.conf_slider.value() / 100.0
            iou = self.overlap_slider.value() / 100.0

            # Use absolute path for storing in DB if file_path is relative
            abs_file_path = os.path.abspath(file_path) if not os.path.isabs(file_path) else file_path

            img_cv = cv2.imread(abs_file_path)
            if img_cv is None:
                self.image_label.setText(f"Error: Could not read image\n{abs_file_path}")
                self.original_pixmap = None
                self.display_scaled_image()
                self.clear_detection_statistics_display()
                return

            start_time = time.time()
            results = self.model(img_cv, conf=confidence, iou=iou)
            end_time = time.time()
            proc_time_ms = (end_time - start_time) * 1000

            # Extract detections in the required format for DB and drawing
            current_detections = []
            if results and results[0].boxes and self.model and hasattr(self.model, 'names'):
                 for box in results[0].boxes:
                     x1, y1, x2, y2 = map(int, box.xyxy[0])
                     cls_id = int(box.cls[0])
                     conf = float(box.conf[0])
                     class_name = self.model.names.get(cls_id, f"Class_{cls_id}")
                     current_detections.append({
                         "class": class_name,
                         "conf": round(conf, 4), # Store with precision
                         "box": [x1, y1, x2, y2]
                     })

            # Save to DB BEFORE drawing boxes (drawing modifies latest_detection_details)
            save_detection_to_db(abs_file_path, 'file', proc_time_ms, confidence, iou, current_detections)

            # Draw boxes using the extracted list
            annotated_img = self.draw_custom_boxes_from_list(img_cv.copy(), current_detections, os.path.basename(abs_file_path))

            # Update display
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            h, w, ch = annotated_img_rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(annotated_img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qt_img)

            self.display_scaled_image()
            # Use the extracted list for statistics update
            self.update_detection_statistics_from_list(current_detections, proc_time_ms)

        except Exception as e:
            self.image_label.setText(f"Error processing image:\n{str(e)}")
            print(f"Error in run_model_on_image_path: {e}")
            self.original_pixmap = None
            self.display_scaled_image()
            self.latest_detection_details = []
            self.clear_detection_statistics_display()


    # --- Drawing and Statistics (Modified to accept list) ---
    def draw_custom_boxes_from_list(self, image, detections_list, source_filename="image"):
        """Draws boxes based on a list of detection dictionaries."""
        img_h, img_w = image.shape[:2]
        class_colors = { # Example colors, align with self.plastic_classes if possible
            "PET": (60, 120, 216), "HDPE": (39, 174, 96), "PVC": (241, 196, 15),
            "LDPE": (231, 76, 60), "PP": (155, 89, 182), "PS": (26, 188, 156),
        }
        default_color = (200, 200, 200)
        annotated_image = image.copy()
        export_data_for_current_image = [] # For CSV export

        for i, det in enumerate(detections_list):
            x1, y1, x2, y2 = det['box']
            model_class_name_raw = det['class']
            conf = det['conf']

            # Determine display name and color (similar logic as before)
            display_class_name = model_class_name_raw
            color = default_color
            for target_class in self.plastic_classes:
                if target_class.lower() in model_class_name_raw.lower():
                    display_class_name = target_class
                    color = class_colors.get(target_class, default_color)
                    break

            export_data_for_current_image.append({
                "image_source": source_filename, "object_id": i + 1,
                "class_name": display_class_name, "confidence": f"{conf:.2f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

            label_text = f"{display_class_name} {conf:.2f}"
            font_scale = 0.5 # Reduced scale slightly
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            offset_from_box = 5
            background_padding_vertical = 2
            text_horizontal_padding = 3

            label_y_pos_baseline = y1 - offset_from_box
            label_bg_y1 = y1 - th - offset_from_box - background_padding_vertical
            label_bg_y2 = y1 - offset_from_box + baseline // 2

            if label_bg_y1 < 0:
                label_y_pos_baseline = y2 + th + offset_from_box
                label_bg_y1 = y2 + offset_from_box - baseline // 2
                label_bg_y2 = y2 + th + offset_from_box + background_padding_vertical

            label_bg_x1 = x1
            label_bg_x2 = x1 + tw + (2 * text_horizontal_padding)
            label_bg_x1 = max(0, label_bg_x1); label_bg_x2 = min(img_w, label_bg_x2)

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2) # Box border
            if label_bg_x1 < label_bg_x2 and label_bg_y1 < label_bg_y2: # Label background
                 cv2.rectangle(annotated_image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)

            actual_text_x_pos = max(x1, label_bg_x1) + text_horizontal_padding
            cv2.putText(annotated_image, label_text, (actual_text_x_pos, label_y_pos_baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)

        self.latest_detection_details = export_data_for_current_image # Used for export
        return annotated_image


    def clear_detection_statistics_display(self):
        # ... (Remains the same, but doesn't clear accuracy card) ...
        if not hasattr(self, 'stat_cards') or not self.stat_cards: return

        for class_name in self.plastic_classes:
            card = self.stat_cards.get(class_name)
            if card:
                value_label = card.findChild(QLabel, "statCardValue")
                progress_bar = card.findChild(QProgressBar, "statCardProgress")
                if value_label: value_label.setText("0")
                if progress_bar: progress_bar.setValue(0)

        if hasattr(self, 'stat_total_items') and self.stat_total_items:
            val_label = self.stat_total_items.findChild(QLabel, "infoCardValue")
            if val_label: val_label.setText("0")
        if hasattr(self, 'stat_proc_time') and self.stat_proc_time:
            val_label = self.stat_proc_time.findChild(QLabel, "infoCardValue")
            if val_label: val_label.setText("0ms")
        # No accuracy card to clear

    def clear_current_detection_display(self):
        """Clears only the current detection display, not history."""
        self.clear_detection_statistics_display()
        self.original_pixmap = None
        if hasattr(self, 'image_label'):
            self.image_label.clear()
            self.image_label.setText("Upload an image or start webcam to see output.")
            self.display_scaled_image()
        self.latest_detection_details = []
        print("Current detection display and stats cleared.")

    def update_detection_statistics_from_list(self, detections_list, inference_time_ms):
        """Updates stat cards based on a list of detection dictionaries."""
        if not self.model or not hasattr(self.model, 'names') or not self.stat_cards:
            self.clear_detection_statistics_display()
            return

        if not detections_list:
            self.clear_detection_statistics_display() # Clear counts
            if hasattr(self, 'stat_proc_time') and self.stat_proc_time: # Update time
                time_label = self.stat_proc_time.findChild(QLabel, "infoCardValue")
                if time_label: time_label.setText(f"{inference_time_ms:.1f}ms")
            return

        counts = {cls_name: 0 for cls_name in self.plastic_classes}
        total_detections = len(detections_list)

        for det in detections_list:
            model_class_name_raw = det['class'].lower()
            for target_class_name in self.plastic_classes:
                if target_class_name.lower() in model_class_name_raw:
                    counts[target_class_name] += 1
                    break

        for class_name, count in counts.items():
            card = self.stat_cards.get(class_name)
            if card:
                value_label = card.findChild(QLabel, "statCardValue")
                progress_bar = card.findChild(QProgressBar, "statCardProgress")
                if value_label: value_label.setText(str(count))
                progress_val = int((count / max(1, total_detections)) * 100) if total_detections > 0 else 0
                if progress_bar: progress_bar.setValue(progress_val)

        if hasattr(self, 'stat_total_items') and self.stat_total_items:
             total_label = self.stat_total_items.findChild(QLabel, "infoCardValue")
             if total_label: total_label.setText(str(total_detections))
        if hasattr(self, 'stat_proc_time') and self.stat_proc_time:
            time_label = self.stat_proc_time.findChild(QLabel, "infoCardValue")
            if time_label: time_label.setText(f"{inference_time_ms:.1f}ms")
        # No accuracy card to update


    # --- Display Scaling (Remains the same) ---
    def display_scaled_image(self):
        # ... (Keep existing display_scaled_image code) ...
        if not hasattr(self, 'image_label') or not self.image_label: return

        if self.original_pixmap is None or self.original_pixmap.isNull():
            self.image_label.clear()
            self.image_label.setText("Upload an image or start webcam to see output.")
            self.image_label.setMinimumSize(400,300) # Reset minimum size for placeholder
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return

        self.image_label.setMinimumSize(1,1) # Allow shrinking

        container_size = self.image_scroll_area.viewport().size() if hasattr(self, 'image_scroll_area') else self.image_label.size()

        if not container_size.isValid() or container_size.width() < 20 or container_size.height() < 20:
             scaled_pixmap = self.original_pixmap.scaled(max(200, self.original_pixmap.width()), max(150, self.original_pixmap.height()), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            scaled_pixmap = self.original_pixmap.scaled(container_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


    # --- Image Navigation (Remains the same) ---
    def prev_image(self):
        if not self.model: return
        if self.image_paths and self.current_image_index > 0:
            self.current_image_index -= 1
            self.run_model_on_image_path(self.image_paths[self.current_image_index])
            self.update_navigation_buttons(); self.update_image_count_label()

    def next_image(self):
        if not self.model: return
        if self.image_paths and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.run_model_on_image_path(self.image_paths[self.current_image_index])
            self.update_navigation_buttons(); self.update_image_count_label()

    # --- Webcam Handling (Modified to save to DB) ---
    def toggle_webcam(self):
        # ... (Stopping logic remains similar) ...
        if not self.model:
            self.image_label.setText("Model not loaded. Cannot start webcam.")
            return

        if self.webcam_running:
            self.webcam_running = False
            if hasattr(self, 'webcam_timer') and self.webcam_timer: self.webcam_timer.stop()
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened(): self.cap.release()

            play_icon = get_icon("webcam_play.svg", QStyle.StandardPixmap.SP_MediaPlay)
            self.webcam_btn.setText(" Start Webcam Detection"); self.webcam_btn.setIcon(play_icon)
            self.webcam_btn.setObjectName("startWebcamButton")
            self.webcam_status_indicator.setObjectName("webcamStatusOffline")
            self.webcam_btn.setStyleSheet(self.styleSheet()); self.webcam_status_indicator.setStyleSheet(self.styleSheet())

            self.image_label.setText("Webcam stopped.\nUpload an image or start webcam again.")
            self.original_pixmap = None
            self.display_scaled_image()
            self.latest_detection_details = [] # Clear export data
            self.update_navigation_buttons()
            self.webcam_dropdown.setEnabled(True)
            self.drop_frame.setEnabled(True)
            self.conf_slider.setEnabled(True); self.overlap_slider.setEnabled(True)
            self.update_image_count_label()
            self.clear_current_detection_display() # Clear stats from webcam session

        else: # Start webcam
            webcam_idx = self.webcam_dropdown.currentData()
            if webcam_idx is None or webcam_idx == -1 :
                self.image_label.setText("No webcam selected or found.")
                return

            self.image_paths = []; self.current_image_index = -1
            self.update_image_count_label(); self.update_navigation_buttons()

            # Try DSHOW first, then default
            self.cap = cv2.VideoCapture(webcam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print(f"Warning: Could not open webcam {webcam_idx} with DSHOW, trying default.")
                self.cap = cv2.VideoCapture(webcam_idx)
                if not self.cap.isOpened():
                     self.image_label.setText(f"Error: Could not open webcam {webcam_idx}")
                     self.cap = None
                     return

            self.webcam_running = True
            self.webcam_timer = QTimer(self)
            self.webcam_timer.timeout.connect(self.update_webcam_frame) # Will now save to DB
            self.webcam_timer.start(50) # Slightly slower frame rate (20 FPS)

            stop_icon = get_icon("webcam_stop.svg", QStyle.StandardPixmap.SP_MediaStop)
            self.webcam_btn.setText(" Stop Webcam Detection"); self.webcam_btn.setIcon(stop_icon)
            self.webcam_btn.setObjectName("stopWebcamButton")
            self.webcam_status_indicator.setObjectName("webcamStatusOnline")
            self.webcam_btn.setStyleSheet(self.styleSheet()); self.webcam_status_indicator.setStyleSheet(self.styleSheet())

            self.latest_detection_details = []
            self.webcam_dropdown.setEnabled(False); self.drop_frame.setEnabled(False)
            # Sliders remain enabled for webcam use

    def update_webcam_frame(self):
        """Processes a webcam frame, displays it, and saves result to DB."""
        if not self.webcam_running or not hasattr(self, 'cap') or not self.cap or not self.cap.isOpened():
             if self.webcam_running:
                 print("Webcam stream lost or not available.")
                 self.toggle_webcam()
             return

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            return

        confidence = self.conf_slider.value() / 100.0
        iou = self.overlap_slider.value() / 100.0

        start_time = time.time()
        results = self.model(frame, conf=confidence, iou=iou)
        end_time = time.time()
        proc_time_ms = (end_time - start_time) * 1000

        # Extract detections
        current_detections = []
        if results and results[0].boxes and self.model and hasattr(self.model, 'names'):
             for box in results[0].boxes:
                 x1, y1, x2, y2 = map(int, box.xyxy[0])
                 cls_id = int(box.cls[0])
                 conf = float(box.conf[0])
                 class_name = self.model.names.get(cls_id, f"Class_{cls_id}")
                 current_detections.append({
                     "class": class_name,
                     "conf": round(conf, 4),
                     "box": [x1, y1, x2, y2]
                 })

        # Save to DB (no image path for webcam)
        save_detection_to_db(None, 'webcam', proc_time_ms, confidence, iou, current_detections)

        # Draw boxes
        annotated_frame = self.draw_custom_boxes_from_list(frame.copy(), current_detections, "webcam_frame")

        # Update display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = annotated_frame_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(annotated_frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qt_img)

        self.display_scaled_image()
        self.update_detection_statistics_from_list(current_detections, proc_time_ms)


    # --- Export (Remains mostly the same, uses latest_detection_details) ---
    def export_statistics(self):
        # ... (Keep existing export logic using self.latest_detection_details) ...
         if not self.model:
            QMessageBox.information(self, "Export Data", "Model not loaded. No data to export.")
            return

         # Check if there's anything in the *current* view to export
         current_total_items_text = "0"
         if hasattr(self, 'stat_total_items') and self.stat_total_items:
              label = self.stat_total_items.findChild(QLabel, "infoCardValue")
              if label: current_total_items_text = label.text()

         if not self.latest_detection_details and current_total_items_text == "0":
             QMessageBox.information(self, "Export Data", "No detection data available in the current view to export.")
             return

         default_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
         timestamp = time.strftime("%Y%m%d_%H%M%S")
         source_info = "last_view" # Generic name for export from current view
         if self.webcam_running: source_info = "webcam_capture"
         elif self.image_paths and self.current_image_index != -1:
            try:
                 source_info = os.path.splitext(os.path.basename(self.image_paths[self.current_image_index]))[0]
            except Exception: pass # Ignore errors getting basename

         default_filename = os.path.join(default_dir, f"detection_export_{source_info}_{timestamp}.csv")

         filePath, _ = QFileDialog.getSaveFileName(self, "Save Export Data", default_filename, "CSV Files (*.csv)")
         if not filePath: return

         try:
            with open(filePath, 'w', newline='', encoding='utf-8') as csvfile: # Added encoding
                # Write detailed detections if available for the *last processed* frame/image
                if self.latest_detection_details:
                    fieldnames = ['image_source', 'object_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.latest_detection_details)
                    csvfile.write("\n")
                else:
                    csvfile.write("No detailed object detections found for the last processed image/frame.\n\n")

                # Always write summary statistics from the *current* display
                csvfile.write("Summary Statistics (Current View):\n")
                for class_name in self.plastic_classes:
                    card = self.stat_cards.get(class_name)
                    count = "0"
                    if card:
                        count_label = card.findChild(QLabel, "statCardValue")
                        if count_label: count = count_label.text()
                    csvfile.write(f"{class_name}: {count}\n")

                total_items_text = "0"
                if hasattr(self, 'stat_total_items') and self.stat_total_items:
                    total_label = self.stat_total_items.findChild(QLabel, "infoCardValue")
                    if total_label: total_items_text = total_label.text()
                csvfile.write(f"Total Items Detected: {total_items_text}\n")

                proc_time_text = "0ms"
                if hasattr(self, 'stat_proc_time') and self.stat_proc_time:
                    proc_label = self.stat_proc_time.findChild(QLabel, "infoCardValue")
                    if proc_label: proc_time_text = proc_label.text()
                csvfile.write(f"Processing Time: {proc_time_text}\n")

                csvfile.write("\nDetection Parameters (Current View):\n")
                csvfile.write(f"Confidence Threshold: {self.conf_slider.value()}%\n") # Use percent symbol directly
                csvfile.write(f"IoU Threshold: {self.overlap_slider.value()}%\n")

            QMessageBox.information(self, "Export Successful", f"Data exported to:\n{filePath}")
         except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not write file:\n{str(e)}")
            print(f"Error exporting data: {e}")


    # --- Resize Event (Remains the same) ---
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap and hasattr(self,'image_label') and self.image_label.isVisible():
            QTimer.singleShot(0, self.display_scaled_image)

    # --- Close Event (Remains the same) ---
    def closeEvent(self, event):
        if self.webcam_running: self.toggle_webcam()
        if hasattr(self, 'model_thread') and self.model_thread.isRunning():
            print("Waiting for model loading thread to finish...")
            self.model_thread.quit(); self.model_thread.wait(3000)
            if self.model_thread.isRunning():
                print("Model thread did not quit gracefully, terminating.")
                self.model_thread.terminate(); self.model_thread.wait()
        print("Application closing.")
        event.accept()


    # --- New Methods for Analytics ---
    def update_analytics_view(self):
        """Queries the DB and updates the analytics charts and cards."""
        if not PYQTCHART_AVAILABLE or not self.model:
            # Optionally display a message if charts are disabled
             if hasattr(self,'analytics_total_items_card'):
                  for card in [self.analytics_total_items_card, self.analytics_avg_proc_time_card, self.analytics_avg_conf_card]:
                       label = card.findChild(QLabel, "infoCardValue")
                       if label: label.setText("N/A")
             return

        time_filter = self.analytics_time_combo.currentText()
        end_date = datetime.now()
        if time_filter == "Last 7 days":
            start_date = end_date - timedelta(days=7)
        elif time_filter == "Last 30 days":
            start_date = end_date - timedelta(days=30)
        else: # All time
            start_date = datetime.min # Or query min date from DB if needed

        start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            # Query for summary stats
            query_summary = """
                SELECT COUNT(*), AVG(processing_time_ms), detected_objects
                FROM detections
                WHERE timestamp BETWEEN ? AND ?
            """
            cursor.execute(query_summary, (start_date_str, end_date_str))
            summary_result = cursor.fetchone()

            total_items_overall = 0
            avg_proc_time = 0
            all_confidences = []
            daily_counts = {} # {date_str: count}
            class_counts = {cls_name: 0 for cls_name in self.plastic_classes}

            if summary_result and summary_result[0] > 0:
                 total_records = summary_result[0]
                 avg_proc_time = summary_result[1] if summary_result[1] is not None else 0

                 # Need to iterate through all records again for confidences and daily/class counts
                 query_details = """
                     SELECT timestamp, detected_objects
                     FROM detections
                     WHERE timestamp BETWEEN ? AND ?
                 """
                 cursor.execute(query_details, (start_date_str, end_date_str))
                 for row in cursor.fetchall():
                     ts_str, objects_json = row
                     try:
                          timestamp_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                          date_str = timestamp_dt.strftime("%Y-%m-%d")
                          daily_counts[date_str] = daily_counts.get(date_str, 0)

                          detections = json.loads(objects_json)
                          total_items_overall += len(detections) # Count individual objects
                          daily_counts[date_str] += len(detections)

                          for det in detections:
                               all_confidences.append(det.get('conf', 0))
                               model_class_name_raw = det.get('class', '').lower()
                               for target_class_name in self.plastic_classes:
                                    if target_class_name.lower() in model_class_name_raw:
                                         class_counts[target_class_name] += 1
                                         break # Count each object only once for class totals

                     except (json.JSONDecodeError, ValueError) as e:
                          print(f"Error processing record data: {e} for record at {ts_str}")


            conn.close()

            # --- Update Summary Cards ---
            self.analytics_total_items_card.findChild(QLabel, "infoCardValue").setText(str(total_items_overall))
            self.analytics_avg_proc_time_card.findChild(QLabel, "infoCardValue").setText(f"{avg_proc_time:.1f}ms")
            avg_conf = (sum(all_confidences) / len(all_confidences)) * 100 if all_confidences else 0
            self.analytics_avg_conf_card.findChild(QLabel, "infoCardValue").setText(f"{avg_conf:.1f}%")


            # --- Update Detections Per Day Chart ---
            self.update_daily_detections_chart(daily_counts, start_date, end_date)

            # --- Update Confidence Trend Chart ---
            # This needs refinement - maybe plot avg confidence per day?
            self.update_confidence_trend_chart(all_confidences, start_date, end_date) # Placeholder


        except sqlite3.Error as e:
            print(f"Database Error updating analytics: {e}")
            QMessageBox.warning(self, "Analytics Error", f"Could not load analytics data from database.\nError: {e}")


    def update_daily_detections_chart(self, daily_counts, start_date, end_date):
        """Updates the bar chart showing detections per day."""
        if not PYQTCHART_AVAILABLE: return

        chart = QChart()
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        chart.setTheme(QChart.ChartTheme.ChartThemeDark) # Use dark theme
        chart.setBackgroundBrush(QBrush(QColor("#2A3040"))) # Match frame background
        chart.layout().setContentsMargins(0,0,0,0) # Remove chart padding
        chart.margins().setAll(10) # Set chart margins

        series = QBarSeries()

        # Generate all dates in the range to ensure gaps are shown
        categories = []
        date_map = {} # Store counts by date string
        current_date = start_date.date()
        end_date_only = end_date.date()
        while current_date <= end_date_only:
            date_str = current_date.strftime("%Y-%m-%d")
            short_date_str = current_date.strftime("%b %d") # e.g., "May 12"
            categories.append(short_date_str)
            date_map[date_str] = 0
            current_date += timedelta(days=1)

        # Fill counts from query results
        max_count = 0
        for date_str, count in daily_counts.items():
             if date_str in date_map:
                 date_map[date_str] = count
                 max_count = max(max_count, count)

        bar_set = QBarSet("Detections")
        bar_set.setColor(QColor("#4A90E2")) # Use accent color
        for date_str_full, count in date_map.items():
             bar_set.append(count) # Append counts in order of categories

        series.append(bar_set)
        chart.addSeries(series)

        # Axes
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_x.setLabelsColor(QColor("#A0A7B9"))
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, max(10, max_count * 1.1)) # Ensure axis shows at least 10
        axis_y.setLabelFormat("%d") # Integer labels
        axis_y.setLabelsColor(QColor("#A0A7B9"))
        axis_y.setGridLineVisible(True) # Show grid lines
        axis_y.setGridLineColor(QColor("#353A4C"))
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)

        chart.legend().setVisible(False) # Hide legend for single series

        self.detections_chart_view.setChart(chart)
        self.detections_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)


    def update_confidence_trend_chart(self, all_confidences, start_date, end_date):
        """Placeholder for confidence trend chart (e.g., line chart of avg conf)."""
        if not PYQTCHART_AVAILABLE: return

        chart = QChart()
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        chart.setBackgroundBrush(QBrush(QColor("#2A3040")))
        chart.layout().setContentsMargins(0,0,0,0); chart.margins().setAll(10)

        series = QLineSeries() # Use QLineSeries
        series.setName("Avg. Confidence")
        series.setColor(QColor("#A3BE8C")) # Different color

        # --- Data Processing (Example: Avg confidence per day) ---
        # This requires querying timestamp + detections again, grouping by day, calculating avg conf per day.
        # For simplicity now, let's just plot *all* confidences sequentially (not ideal)
        # A better approach involves more complex SQL or Python processing.

        # Simple placeholder: Plot first N confidences if available
        points_to_plot = min(100, len(all_confidences)) # Limit points for clarity
        for i in range(points_to_plot):
            series.append(i, all_confidences[i] * 100) # Plot confidence percentage

        chart.addSeries(series)

        # Axes
        axis_x = QValueAxis() # Simple value axis for now
        axis_x.setRange(0, max(10, points_to_plot -1))
        axis_x.setLabelFormat("%d")
        axis_x.setLabelsVisible(False) # Hide X labels for this simple plot
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, 100) # Confidence range 0-100%
        axis_y.setLabelFormat("%d%%")
        axis_y.setLabelsColor(QColor("#A0A7B9"))
        axis_y.setGridLineVisible(True)
        axis_y.setGridLineColor(QColor("#353A4C"))
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)

        chart.legend().setVisible(False)
        # chart.legend().setAlignment(Qt.AlignmentFlag.AlignTop)
        # chart.legend().setLabelColor(QColor("#E0E1E3"))

        self.confidence_chart_view.setChart(chart)
        self.confidence_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)


    # --- New Methods for History/Gallery ---
    def update_history_view(self, page=None):
        """Queries DB based on filters and updates the gallery grid."""
        if page is None:
             page = self.current_history_page # Use current page if not specified

        # Clear previous items
        while self.gallery_grid_layout.count():
            item = self.gallery_grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Get filter values
        search_term = self.history_search_input.text().strip()
        filter_type = self.history_filter_combo.currentText()
        start_date = self.history_date_start.date().toString("yyyy-MM-dd") + " 00:00:00"
        end_date = self.history_date_end.date().toString("yyyy-MM-dd") + " 23:59:59"

        # Build SQL query
        base_query = "FROM detections WHERE timestamp BETWEEN ? AND ?"
        count_query = "SELECT COUNT(*) " + base_query
        data_query = "SELECT id, timestamp, image_path, source_type, detected_objects " + base_query
        params = [start_date, end_date]

        # Add search filter (simple LIKE on path or detected objects JSON)
        if search_term:
            base_query += " AND (image_path LIKE ? OR detected_objects LIKE ?)"
            params.extend([f"%{search_term}%", f"%{search_term}%"])

        # Add type filter (checks if class name exists in JSON)
        if filter_type.startswith("Filter by type:") and filter_type != "Filter by type: All":
             class_name_filter = filter_type.split(": ")[1]
             base_query += " AND detected_objects LIKE ?"
             params.append(f'%"{class_name_filter}"%') # Simple check if class name string exists

        # Add order and pagination
        data_query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        offset = (page - 1) * HISTORY_ITEMS_PER_PAGE
        data_params = params + [HISTORY_ITEMS_PER_PAGE, offset]
        count_params = params # Count query doesn't need limit/offset

        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            # Get total count for pagination
            cursor.execute(count_query, count_params)
            total_items = cursor.fetchone()[0]
            self.total_history_pages = (total_items + HISTORY_ITEMS_PER_PAGE - 1) // HISTORY_ITEMS_PER_PAGE
            if self.total_history_pages == 0: self.total_history_pages = 1 # At least one page

            # Get data for current page
            cursor.execute(data_query, data_params)
            results = cursor.fetchall()
            conn.close()

            # Populate grid
            row, col = 0, 0
            items_in_row = 4 # Adjust based on desired grid width
            for record in results:
                item_widget = self.create_gallery_item_widget(record)
                if item_widget:
                     self.gallery_grid_layout.addWidget(item_widget, row, col)
                     col += 1
                     if col >= items_in_row:
                         col = 0
                         row += 1

            # Add spacer to push items up if grid isn't full
            # self.gallery_grid_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding), row + 1, 0, 1, items_in_row)


            # Update pagination controls
            self.current_history_page = page
            self.history_page_label.setText(f"Page {self.current_history_page} / {self.total_history_pages}")
            self.history_prev_btn.setEnabled(self.current_history_page > 1)
            self.history_next_btn.setEnabled(self.current_history_page < self.total_history_pages)

            # Scroll to top after update
            self.history_scroll_area.verticalScrollBar().setValue(0)


        except sqlite3.Error as e:
            print(f"Database Error updating history: {e}")
            QMessageBox.warning(self, "History Error", f"Could not load history data from database.\nError: {e}")
            # Reset pagination display on error
            self.current_history_page = 1; self.total_history_pages = 1
            self.history_page_label.setText("Page 1 / 1")
            self.history_prev_btn.setEnabled(False); self.history_next_btn.setEnabled(False)


    def create_gallery_item_widget(self, db_record):
        """Creates a widget for a single item in the history gallery."""
        try:
            record_id, timestamp_str, image_path, source_type, objects_json = db_record
            detections = json.loads(objects_json)

            item_frame = QFrame()
            item_frame.setObjectName("galleryItemFrame")
            item_frame.setCursor(Qt.CursorShape.PointingHandCursor) # Indicate clickable
            item_layout = QVBoxLayout(item_frame)
            item_layout.setContentsMargins(0, 0, 0, 0) # No margins for the frame itself
            item_layout.setSpacing(5)

            # Thumbnail Label
            thumb_label = QLabel()
            thumb_label.setObjectName("galleryThumbLabel")
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_size = 160 # Desired thumbnail width/height

            if image_path and os.path.exists(image_path):
                 pixmap = QPixmap(image_path)
                 if not pixmap.isNull():
                      scaled_pixmap = pixmap.scaled(thumb_size, thumb_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                      thumb_label.setPixmap(scaled_pixmap)
                 else:
                      thumb_label.setText("Invalid Image")
                      thumb_label.setStyleSheet("color: #BF616A;")
            elif source_type == 'webcam':
                 # Placeholder for webcam captures (maybe use an icon?)
                 webcam_icon = get_icon("webcam_play.svg", QStyle.StandardPixmap.SP_MediaPlay) # Re-use icon
                 thumb_label.setPixmap(webcam_icon.pixmap(QSize(64, 64)))
                 thumb_label.setStyleSheet("background-color: #353A4C;") # Darker bg for icon
            else:
                 thumb_label.setText("No Image")
                 thumb_label.setStyleSheet("color: #A0A7B9;")

            item_layout.addWidget(thumb_label)

            # Info Labels
            info_text = "N/A"
            if detections:
                 # Show primary detection or count
                 primary_class = detections[0]['class']
                 count = len(detections)
                 info_text = f"{primary_class}" + (f" (+{count-1})" if count > 1 else "")
                 # Limit text length
                 if len(info_text) > 25: info_text = info_text[:22] + "..."

            info_label = QLabel(info_text)
            info_label.setObjectName("galleryInfoLabel")
            item_layout.addWidget(info_label)

            date_label = QLabel(timestamp_str) # Show full timestamp
            date_label.setObjectName("galleryDateLabel")
            item_layout.addWidget(date_label)

            # Details Button (Optional - opens dialog later)
            details_btn = QPushButton("View Details")
            details_btn.setObjectName("galleryDetailsButton")
            details_btn.clicked.connect(lambda _, r_id=record_id: self.show_history_details(r_id))
            item_layout.addWidget(details_btn)

            item_frame.setFixedSize(thumb_size, thumb_size + 75) # Adjust height based on content below thumb

            return item_frame

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Error creating gallery item for record ID {db_record[0] if db_record else 'N/A'}: {e}")
            return None # Skip item on error

    def show_history_details(self, record_id):
        """Placeholder for showing detailed view of a history item."""
        # TODO: Implement a dialog
        # 1. Query DB for the full record using record_id
        # 2. Create a QDialog window
        # 3. Load the full image (if path exists)
        # 4. Draw bounding boxes from the 'detected_objects' JSON
        # 5. Display the image and other details (timestamp, conf, iou, etc.) in the dialog
        print(f"Details requested for record ID: {record_id}")
        QMessageBox.information(self, "Details", f"Detailed view for record {record_id} is not yet implemented.")


    def history_prev_page(self):
        if self.current_history_page > 1:
            self.update_history_view(page=self.current_history_page - 1)

    def history_next_page(self):
        if self.current_history_page < self.total_history_pages:
            self.update_history_view(page=self.current_history_page + 1)


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("YOLOv8S_PlasticSegregation")
    app.setOrganizationName("UserGroup")
    app.setApplicationVersion("1.1.0") # Updated version

    # Font loading (Optional - same as before)
    # ...

    # Create 'icons' directory (Same as before)
    if not os.path.exists(ICON_DIR):
        try:
            os.makedirs(ICON_DIR)
            print(f"Created '{ICON_DIR}' directory for icons. Please add your SVG files there.")
        except OSError as e:
            print(f"Could not create '{ICON_DIR}' directory: {e}. Please create it manually.")

    # Check PyQtChart Dependency
    if not PYQTCHART_AVAILABLE:
         msg_box = QMessageBox()
         msg_box.setIcon(QMessageBox.Icon.Warning)
         msg_box.setText("The PyQtChart library is required for the Analytics tab but was not found.\n\nCharts will be disabled.\n\nPlease install it using:\npip install PyQtChart")
         msg_box.setWindowTitle("Missing Dependency")
         msg_box.exec()


    window = WasteDetectionApp()
    # window.show() # show() is called after model load attempt
    sys.exit(app.exec())