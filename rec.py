import os, sys
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller .exe """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PyQt6.QtCore import (
    QTimer, QThread, pyqtSignal, Qt, QSize, QRect, QRectF, QPropertyAnimation,
    QEasingCurve, QPoint, QStandardPaths, QDateTime, QDate, Qt, QUrl, QTimer 
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLayout, QPushButton,
    QFileDialog, QSlider, QFrame, QSpacerItem, QSizePolicy, QComboBox, QToolButton,
    QScrollArea, QGridLayout, QListWidget, QStackedLayout, QGraphicsOpacityEffect,
    QSplashScreen, QButtonGroup, QProgressBar, QStyle, QMessageBox, QTextBrowser,
    QLineEdit, QDateEdit, QMainWindow
)
from PyQt6.QtGui import (
    QPixmap, QFont, QImage, QColor, QPainter, QBrush, QPen, QFontDatabase, QIcon, QTextOption, QScreen, QShortcut, QKeySequence, QDoubleValidator
)

from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
import time
import os
import csv
import json
from datetime import datetime, timedelta, date
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from Yolov7_StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT


# --- Configuration ---
APP_FONT_FAMILY = "Arial"
ICON_DIR = resource_path("icons/")
HISTORY_ITEMS_PER_PAGE = 16

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
            print(
                f"Warning: Icon file '{icon_path}' does not exist. Using fallback.")
        else:
            print(
                f"Warning: Icon '{icon_name}' at '{icon_path}' could not be loaded. Using fallback.")
        if QApplication.instance():
            return QApplication.style().standardIcon(fallback_style_enum)
        else:
            print(
                "Warning: QApplication instance not found for fallback icon. Returning empty QIcon.")
            return QIcon()
    return icon

# --- Splash Screen ---


class SplashScreen(QSplashScreen):
    def __init__(self):
        splash_width = 700
        splash_height = 500
        self.splash_pixmap = QPixmap(splash_width, splash_height)
        self.splash_pixmap.fill(QColor("#F4F6F8"))  

        with QPainter(self.splash_pixmap) as painter:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rect_color = QColor("#FFFFFF")
            rect_x = 50
            rect_y = 50
            rect_width = splash_width - 100
            rect_height = splash_height - 100
            painter.setBrush(QBrush(rect_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                QRectF(rect_x, rect_y, rect_width, rect_height), 15, 15)

            logo_y_pos = rect_y + 40
            logo_icon = get_icon(
                "app_logo_splash.png", QStyle.StandardPixmap.SP_ComputerIcon)
            if not logo_icon.isNull():
                logo_pixmap_size = 100
                logo_x = rect_x + (rect_width - logo_pixmap_size) // 2
                logo_icon.paint(
                    painter, QRect(logo_x, int(logo_y_pos), logo_pixmap_size, logo_pixmap_size))
                logo_y_pos += logo_pixmap_size + 15
            else:  # Fallback drawing
                painter.setBrush(QColor("#4A90E2"))
                painter.drawEllipse(
                    QRectF(rect_x + (rect_width - 70) // 2, logo_y_pos, 70, 70))
                logo_y_pos += 70 + 15

            painter.setPen(QColor("#212529"))
            title_font = QFont(APP_FONT_FAMILY, 32, QFont.Weight.Bold)
            painter.setFont(title_font)
            painter.drawText(QRect(rect_x, int(logo_y_pos), rect_width,
                             50), Qt.AlignmentFlag.AlignCenter, "YOLOv8-S")
            logo_y_pos += 45

            subtitle_font = QFont(APP_FONT_FAMILY, 18, QFont.Weight.Normal)
            painter.setFont(subtitle_font)
            painter.setPen(QColor("#495057"))
            painter.drawText(QRect(rect_x, int(logo_y_pos), rect_width, 30),
                             Qt.AlignmentFlag.AlignCenter, "Plastic Waste Segregation")
            logo_y_pos += 45

            desc_font = QFont(APP_FONT_FAMILY, 11)
            painter.setFont(desc_font)
            painter.setPen(QColor("#495057"))
            desc_text = ("An advanced computer vision system designed to detect and "
                         "classify plastic waste materials in real-time. This application "
                         "helps improve recycling efficiency by accurately identifying "
                         "different types of plastic waste.")
            desc_rect = QRectF(rect_x + 40, logo_y_pos, rect_width - 80, 80)
            painter.drawText(
                desc_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, desc_text)
            logo_y_pos += desc_rect.height() + 30

            loading_font = QFont(APP_FONT_FAMILY, 16, QFont.Weight.DemiBold)
            painter.setFont(loading_font)
            painter.setPen(QColor("#A3BE8C"))
            loading_text = "Launching Model..."
            loading_text_rect = QRectF(rect_x, logo_y_pos, rect_width, 45)
            painter.drawText(
                loading_text_rect, Qt.AlignmentFlag.AlignCenter, loading_text)
            logo_y_pos += 45 + 35

        super().__init__(self.splash_pixmap)
        self.setWindowFlags(
            Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_animation = QPropertyAnimation(
            self.opacity_effect, b"opacity")
        self.opacity_animation.setDuration(1000)
        self.opacity_animation.setStartValue(0.0)
        self.opacity_animation.setEndValue(1.0)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def showEvent(self, event):
        super().showEvent(event)
        self.opacity_animation.start()

    def finish(self, widget):
        self.opacity_animation.setDirection(
            QPropertyAnimation.Direction.Backward)
        self.opacity_animation.finished.connect(
            lambda: super(SplashScreen, self).finish(widget))
        self.opacity_animation.start()

    def mousePressEvent(self, event):
        pass  # Prevent splash screen from closing on click

class IntroVideoScreen(QWidget):
    def __init__(self, video_path, on_finished_callback):
        super().__init__()
        self.on_finished_callback = on_finished_callback
        self.ended = False  # Prevent multiple callbacks
        self.setWindowIcon(QIcon(resource_path("icons/app_logo.png")))


        self.setWindowTitle("Plastic Waste Awareness")
        self.setGeometry(300, 200, 800, 480)

        layout = QVBoxLayout(self)

        # Video widget
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)

        # Player setup
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

        # Load video
        self.player.setSource(QUrl.fromLocalFile(video_path))
        self.player.play()

        # Connect events
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        self.player.errorOccurred.connect(self._handle_error)

        # Buttons for skip
        button_layout = QHBoxLayout()
        names = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Others", "Skip Video"]
        seconds = [16, 52, 91, 129, 182, 216, 265, None]

        for name, sec in zip(names, seconds):
            btn = QPushButton(name)
            if sec is None:
                btn.clicked.connect(self._end_video)  # ✅ Proper skip
            else:
                btn.clicked.connect(lambda _, s=sec: self.player.setPosition(s * 1000))
            button_layout.addWidget(btn)

        layout.addLayout(button_layout)

        # Allow Esc to skip video
        QShortcut(QKeySequence("Escape"), self, activated=self._end_video)

    def _media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._end_video()

    def _handle_error(self, error):
        print(f"Video error: {error}")
        self._end_video()

    def _end_video(self):
        if self.ended:
            return
        self.ended = True
        print("Ending video...")
        try:
            self.player.stop()
            self.player.deleteLater()
        except Exception as e:
            print(f"Cleanup error: {e}")
        self.close()
        if callable(self.on_finished_callback):
            self.on_finished_callback()

# --- Model Loading Thread ---


class ModelLoadThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            self.progress.emit('<span style="color: black;">Initializing...</span>')
            time.sleep(0.2)
            self.progress.emit('<span style="color: black;">Detecting hardware...</span>')
            device = get_device()
            time.sleep(0.2)
            self.progress.emit(f'<span style="color: black;">Loading YOLOv8 model on {device}...</span>')
            model = YOLO(self.model_path).to(device)
            time.sleep(0.5)
            self.progress.emit('<span style="color: black;">Finalizing setup...</span>')
            time.sleep(0.2)
            self.finished.emit(model)
        except Exception as e:
            print(f"ModelLoadThread Error: {e}")
            self.progress.emit('<span style="color: black;">Error: Failed to load model.</span>')
            self.finished.emit(None)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setMinimumSize(300, 250) # Set a minimum size for the graph area
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def plot_data(self, data_dict):
        # Example: A bar chart of detected plastic types
        self.axes.clear() # Clear previous plot
        if data_dict:
            labels = list(data_dict.keys())
            values = list(data_dict.values())
            self.axes.bar(labels, values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
            self.axes.set_title('Detected Plastic Types')
            self.axes.set_ylabel('Count')
            self.axes.set_xticks(range(len(labels))) # Ensure all labels are shown
            self.axes.set_xticklabels(labels, rotation=45, ha='right') # Rotate for better readability
            self.axes.set_yticks(np.arange(0, max(values) + 1, 1)) # Ensure integer y-ticks
        else:
            self.axes.text(0.5, 0.5, "No data to display", horizontalalignment='center', verticalalignment='center', transform=self.axes.transAxes)
        self.draw() # Redraw the canvas
        
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=-1, hSpacing=-1, vSpacing=-1):
        super(FlowLayout, self).__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.m_hSpace = hSpacing
        self.m_vSpace = vSpacing
        self.m_itemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.m_itemList.append(item)

    def horizontalSpacing(self):
        if self.m_hSpace >= 0:
            return self.m_hSpace
        else:
            return self.smartSpacing(QStyle.PixelMetric.PM_LayoutHorizontalSpacing)

    def verticalSpacing(self):
        if self.m_vSpace >= 0:
            return self.m_vSpace
        else:
            return self.smartSpacing(QStyle.PixelMetric.PM_LayoutVerticalSpacing)

    def count(self):
        return len(self.m_itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.m_itemList):
            return self.m_itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.m_itemList):
            return self.m_itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0) # Not expanding

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self._doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.m_itemList:
            size = size.expandedTo(item.minimumSize())
        margin, _, _, _ = self.getContentsMargins() # In Qt6, getContentsMargins returns tuple
        size += QSize(2 * margin, 2 * margin)
        return size

    def _doLayout(self, rect, testOnly):
        left, top, right, bottom = self.getContentsMargins()
        effectiveRect = rect.adjusted(+left, +top, -right, -bottom)
        x = effectiveRect.x()
        y = effectiveRect.y()
        lineHeight = 0

        for item in self.m_itemList:
            wid = item.widget()
            spaceX = self.horizontalSpacing()
            if spaceX == -1:
                spaceX = wid.style().layoutSpacing(QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Horizontal)
            spaceY = self.verticalSpacing()
            if spaceY == -1:
                spaceY = wid.style().layoutSpacing(QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Vertical)

            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > effectiveRect.right() and lineHeight > 0:
                x = effectiveRect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y() + bottom

    def smartSpacing(self, pm):
        parent = self.parent()
        if parent is None:
            return -1
        elif parent.isWidgetType():
            return parent.style().pixelMetric(pm, None, parent)
        else:
            return parent.spacing()
        

# --- Main Application Window ---


class WasteDetectionApp(QWidget):
    def __init__(self, splash=None):
        super().__init__()
        self.setWindowIcon(QIcon(resource_path("icons/app_logo.png")))


        self.confidence_threshold = 0.50  # Changed to instance variable
        self.iou_threshold = 0.50 

        self.model = None
        self.splash = splash

        self.current_image_index = -1
        self.image_paths = []
        self.processed_results_for_export = []
        self.webcam_running = False
        self.original_pixmap = None
        self.latest_detection_details = []  # Export data for CURRENT view

        # --- In-Memory Storage ---
        self.detection_history_memory = []

        # --- Webcam Tracking State ---
        self.tracked_object_identities = {}

        # History tab state
        self.current_history_page = 1
        self.total_history_pages = 1

        self.open_accordion_frame = None  # Track the currently open frame
        self.open_accordion_button = None # Track the currently open button


        self.setWindowTitle("YOLOv8-S Plastic Waste Segregation")
        self.setMinimumSize(1366, 768)

        self.initUI()
        self.apply_stylesheet()

        model_file_path = resource_path("best.pt")
        # model_file_path = "C:/Users/mariel/thesis/weights/best.pt"
        self.model_thread = ModelLoadThread(model_file_path)
        self.model_thread.progress.connect(self.update_splash_message)
        self.model_thread.finished.connect(self.on_model_loaded)
        self.model_thread.start()

    def update_splash_message(self, message):
        if self.splash:
            self.splash.showMessage(
                message,
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                QColor("white")
            )
            QApplication.processEvents()
        else:
            print(f"[Splash] {message}")

    def on_model_loaded(self, model):
        if model is None:
        #    self.splash.finish(self)
            self.show()
            QMessageBox.critical(self, "Model Load Error",
                                 "Failed to load YOLOv8 model. Check path/console.")
            # Disable features
            if hasattr(self, 'nav_buttons'):
                for btn_name in ["Detection", "Analytics", "History"]:
                    if btn_name in self.nav_buttons:
                        self.nav_buttons[btn_name].setEnabled(False)
            if hasattr(self, 'webcam_btn'):
                self.webcam_btn.setEnabled(False)
            if hasattr(self, 'drop_frame'):
                self.drop_frame.setEnabled(False)
            if hasattr(self, 'export_btn'):
                self.export_btn.setEnabled(False)
            if hasattr(self, 'clear_btn'):
                self.clear_btn.setEnabled(False)
            if hasattr(self, 'clear_history_btn'):
                self.clear_history_btn.setEnabled(False)
            return

        self.model = model
        if not hasattr(self.model, 'names') or not self.model.names:
            print("Warning: Model has no class names. Using placeholders.")
            self.model.names = {i: f"Class_{i}" for i in range(80)}

        self.update_webcam_list()
        self.setup_stat_cards()  # Needs model.names
        self.populate_history_filter_combo()  # Populate history filter

        # Enable controls
        if hasattr(self, 'drop_frame'):
            self.drop_frame.setEnabled(True)
        if hasattr(self, 'webcam_btn'):
            self.webcam_btn.setEnabled(True)
        if hasattr(self, 'export_btn'):
            self.export_btn.setEnabled(True)
        if hasattr(self, 'clear_btn'):
            self.clear_btn.setEnabled(True)
        if hasattr(self, 'nav_buttons'):
            for btn_name in ["Detection", "Analytics", "History"]:
                if btn_name in self.nav_buttons:
                    self.nav_buttons[btn_name].setEnabled(True)
        if hasattr(self, 'clear_history_btn'):
            self.clear_history_btn.setEnabled(True)

    #    QTimer.singleShot(500, lambda: self.splash.finish(self) if self.splash else print("[Splash] Already closed or missing"))
        self.show()

        # Set initial view
        if hasattr(self, 'nav_button_group') and self.nav_button_group.buttons():
            for i, btn in enumerate(self.nav_button_group.buttons()):
                if btn.isEnabled():
                    btn.setChecked(True)
                    self.handle_navigation(self.nav_button_group.id(btn))
                    break

    def initUI(self):
        overall_layout = QHBoxLayout(self)
        overall_layout.setContentsMargins(0, 0, 0, 0)
        overall_layout.setSpacing(0)

        self.nav_bar = self.create_navigation_bar()
        overall_layout.addWidget(self.nav_bar)

        main_content_widget = QWidget()
        main_content_layout = QVBoxLayout(main_content_widget)
        main_content_layout.setContentsMargins(0, 0, 0, 0)
        main_content_layout.setSpacing(0)
        main_content_widget.setObjectName("mainContentWidget")

        self.header = self.create_header()
        main_content_layout.addWidget(self.header)

        self.stacked_layout_widget = QWidget()
        self.stacked_layout = QStackedLayout(self.stacked_layout_widget)

        # Create views
        self.detection_view = self.create_detection_view()
        self.analytics_view = self.create_analytics_view()  # No charts version
        self.history_view = self.create_history_view()

        # Add views to stacked layout
        self.stacked_layout.addWidget(self.detection_view)  # Index 0
        self.stacked_layout.addWidget(self.analytics_view)  # Index 1
        self.stacked_layout.addWidget(self.history_view)    # Index 2
        main_content_layout.addWidget(self.stacked_layout_widget, 1)
        overall_layout.addWidget(main_content_widget, 1)

    def create_navigation_bar(self):
        nav_widget = QWidget()
        nav_widget.setObjectName("navigationBar")
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(5, 70, 5, 20) #left buttons alignment!!!
        nav_layout.setSpacing(15)

        self.nav_button_group = QButtonGroup(self)
        self.nav_button_group.setExclusive(True)
        self.nav_button_group.idClicked.connect(self.handle_navigation)

        buttons_config = [
            ("Detection", 0, "detection.png",
             QStyle.StandardPixmap.SP_DesktopIcon),
            ("Analytics", 1, "analytics.png",
             QStyle.StandardPixmap.SP_FileDialogDetailedView),
            ("History", 2, "history.png",
             QStyle.StandardPixmap.SP_FileDialogContentsView)
        ]

        self.nav_buttons = {}
        for name, page_index, icon_svg_name, fallback_enum in buttons_config:
            icon = get_icon(icon_svg_name, fallback_enum)
            btn = QPushButton(icon, "")
            btn.setIconSize(QSize(28, 28))
            btn.setCheckable(True)
            btn.setToolTip(name)
            btn.setObjectName("navButton")
            nav_layout.addWidget(btn)
            self.nav_button_group.addButton(btn, page_index)
            self.nav_buttons[name] = btn

        if self.model is None:
            for btn_name in ["Detection", "Analytics", "History"]:
                if btn_name in self.nav_buttons:
                    self.nav_buttons[btn_name].setEnabled(False)

        nav_layout.addStretch(1)

        #clear_history_icon = get_icon(
        #    "clear.png", QStyle.StandardPixmap.SP_DialogResetButton)
        #self.clear_history_btn = QPushButton(clear_history_icon, "")
        #self.clear_history_btn.setObjectName("navButton")
        #self.clear_history_btn.setToolTip("Clear In-Memory History")
        #self.clear_history_btn.clicked.connect(self.clear_all_history)
        #self.clear_history_btn.setEnabled(
        #    self.model is not None)  # Enable only if model loaded
        #nav_layout.addWidget(self.clear_history_btn)

        return nav_widget

    def handle_navigation(self, index):
        self.stacked_layout.setCurrentIndex(index)
        if index == 1:
            self.update_analytics_view()
        elif index == 2:
            self.update_history_view(page=1)  # Reset to page 1 on tab switch

    def create_header(self):
        header_widget = QWidget()
        header_widget.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 0, 20, 0)

        app_logo_label = QLabel()
        logo_icon = get_icon("app_logo.png",
                             QStyle.StandardPixmap.SP_ComputerIcon)
        app_logo_label.setPixmap(logo_icon.pixmap(QSize(50, 50)))
        header_layout.addWidget(app_logo_label)
        header_layout.addSpacing(10)

        title_label = QLabel("YOLOv8-S Plastic Waste Segregation")
        title_label.setObjectName("headerTitleLabel")
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        exit_icon = get_icon("exit.png",
                             QStyle.StandardPixmap.SP_DialogCloseButton)
        exit_btn = QPushButton(exit_icon, " Exit")
        exit_btn.setObjectName("headerExitButton")
        exit_btn.clicked.connect(self.close)
        header_layout.addWidget(exit_btn)
        return header_widget

    def create_detection_view(self):
        detection_widget = QWidget()
        overall_detection_layout = QHBoxLayout(detection_widget)
        overall_detection_layout.setContentsMargins(10, 10, 10, 10)
        overall_detection_layout.setSpacing(10)

        # --- Left Panel (Controls) ---
        left_panel = QFrame()
        left_panel.setObjectName("controlPanel")
        left_panel.setFixedWidth(280)
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(15, 15, 15, 15)
        left_panel_layout.setSpacing(20)

        # Model Config Group
        model_config_group = QWidget()
        model_config_layout = QVBoxLayout(model_config_group)
        model_config_layout.setContentsMargins(0, 0, 0, 0)
        model_config_layout.setSpacing(8) # Corrected: Added spacing value
        config_title = QLabel("Model Configuration")
        config_title.setObjectName("panelTitleLabel")
        model_config_layout.addWidget(config_title)
        
        # CONFIGURED PLS CHECK
        # Confidence Threshold Input
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence Threshold:")
        conf_layout.addWidget(conf_label)

        self.conf_input = QLineEdit(f"{self.confidence_threshold:.2f}")
        self.conf_input.setValidator(QDoubleValidator(0.0, 1.0, 2))
        self.conf_input.setToolTip("Enter a value between 0.0 and 1.0")
        self.conf_input.editingFinished.connect(self.update_confidence_threshold_from_input)
        self.conf_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.conf_input.setFixedWidth(50)  # 👈 Small width, fits values like "0.99"
        conf_layout.addWidget(self.conf_input)
        model_config_layout.addLayout(conf_layout)

        # IoU Threshold Input
        iou_layout = QHBoxLayout()
        iou_label = QLabel("Overlap Threshold (IoU):")
        iou_layout.addWidget(iou_label)

        self.iou_input = QLineEdit(f"{self.iou_threshold:.2f}")
        self.iou_input.setValidator(QDoubleValidator(0.0, 1.0, 2))
        self.iou_input.setToolTip("Enter a value between 0.0 and 1.0")
        self.iou_input.editingFinished.connect(self.update_iou_threshold_from_input)
        self.iou_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.iou_input.setFixedWidth(50)  # 👈 Same width for consistency
        iou_layout.addWidget(self.iou_input)
        model_config_layout.addLayout(iou_layout)

        model_config_layout.addItem(
            QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        left_panel_layout.addWidget(model_config_group)



        # Input Source Group
        input_source_group = QWidget()
        input_source_layout = QVBoxLayout(input_source_group)
        input_source_layout.setContentsMargins(0, 0, 0, 0)
        input_source_layout.setSpacing(8) # Corrected: Added spacing value
        input_title = QLabel("Input Source")
        input_title.setObjectName("panelTitleLabel")
        input_source_layout.addWidget(input_title)
        self.drop_frame = QFrame()
        self.drop_frame.setObjectName("dropFrame")
        self.drop_frame.setAcceptDrops(True)
        self.drop_frame.dragEnterEvent = self.dragEnterEvent
        self.drop_frame.dropEvent = self.dropEvent
        drop_layout = QVBoxLayout(self.drop_frame)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.setSpacing(10)
        upload_icon_label = QLabel()
        upload_svg_icon = get_icon(
            "upload.png", QStyle.StandardPixmap.SP_ArrowUp)
        upload_icon_label.setPixmap(upload_svg_icon.pixmap(QSize(40, 40)))
        upload_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drag_label = QLabel("Drag and drop files here\nor click to browse")
        drag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drag_label.setObjectName("dropFrameText")
        file_format_label = QLabel("Limit 200MB. JPG, PNG, BMP, WEBP")
        file_format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_format_label.setObjectName("dropFrameSubText")
        file_format_label.setWordWrap(True)
        drop_layout.addWidget(upload_icon_label)
        drop_layout.addWidget(drag_label)
        drop_layout.addWidget(file_format_label)
        self.drop_frame.mousePressEvent = lambda event: self.upload_images()
        input_source_layout.addWidget(self.drop_frame)
        left_panel_layout.addWidget(input_source_group)

        # Webcam Group
        webcam_group = QWidget()
        webcam_layout = QVBoxLayout(webcam_group)
        webcam_layout.setContentsMargins(0, 0, 0, 0)
        webcam_layout.setSpacing(8) # Corrected: Added spacing value
        webcam_title_layout = QHBoxLayout()
        webcam_title = QLabel("Webcam Input")
        webcam_title.setObjectName("panelTitleLabel")
        self.webcam_status_indicator = QLabel("●")
        self.webcam_status_indicator.setObjectName("webcamStatusOffline")
        webcam_title_layout.addWidget(webcam_title)
        webcam_title_layout.addStretch()
        webcam_title_layout.addWidget(self.webcam_status_indicator)
        webcam_layout.addLayout(webcam_title_layout)
        self.webcam_dropdown = QComboBox()
        self.webcam_dropdown.setToolTip("Select webcam device")
        webcam_layout.addWidget(self.webcam_dropdown)
        play_icon = get_icon("webcam_play.svg",
                             QStyle.StandardPixmap.SP_MediaPlay)
        self.webcam_btn = QPushButton(play_icon, " Start Webcam Tracking")
        self.webcam_btn.setObjectName("startWebcamButton")
        self.webcam_btn.clicked.connect(self.toggle_webcam)
        # self.webcam_btn.setEnabled(False) # This line was removed as per the desired output
        webcam_layout.addWidget(self.webcam_btn)
        left_panel_layout.addWidget(webcam_group)

        left_panel_layout.addStretch(1)
        overall_detection_layout.addWidget(left_panel)

        # --- Image Display Widget (Middle Panel) ---
        self.image_display_widget = QFrame()
        self.image_display_widget.setObjectName("imageDisplayWidget")
        display_widget_layout = QVBoxLayout(self.image_display_widget)
        display_widget_layout.setContentsMargins(5, 5, 5, 5)
        display_widget_layout.setSpacing(5)
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setObjectName("imageScrollArea")
        self.image_scroll_area.setWidgetResizable(True)
        self.image_scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label = QLabel("Upload image or start webcam...")
        self.image_label.setObjectName("imageDisplayLabel")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_scroll_area.setWidget(self.image_label)
        display_widget_layout.addWidget(self.image_scroll_area, 1)

        # Image Nav Layout
        image_nav_layout = QHBoxLayout()
        image_nav_layout.setContentsMargins(0, 0, 0, 0)
        prev_icon = get_icon("left-arrow.png",
                             QStyle.StandardPixmap.SP_ArrowLeft)
        self.prev_btn = QPushButton(prev_icon, "")
        self.prev_btn.setObjectName("imageNavButton")
        self.prev_btn.clicked.connect(self.prev_image)
        self.image_count_label = QLabel("— / —")
        self.image_count_label.setObjectName("imageCountLabel")
        self.image_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        next_icon = get_icon("right-arrow.png",
                             QStyle.StandardPixmap.SP_ArrowRight)
        self.next_btn = QPushButton(next_icon, "")
        self.next_btn.setObjectName("imageNavButton")
        self.next_btn.clicked.connect(self.next_image)
        image_nav_layout.addWidget(self.prev_btn)
        image_nav_layout.addStretch()
        image_nav_layout.addWidget(self.image_count_label)
        image_nav_layout.addStretch()
        image_nav_layout.addWidget(self.next_btn)
        display_widget_layout.addLayout(image_nav_layout)
        
        # Add the image display widget directly to the overall layout
        overall_detection_layout.addWidget(self.image_display_widget, 1)

        # --- Stats Area Widget (Right Panel) ---
        self.stats_area_widget = QWidget()
        stats_area_outer_layout = QVBoxLayout(self.stats_area_widget)
        stats_area_outer_layout.setContentsMargins(0, 0, 0, 0)
        stats_content_frame = QFrame()
        stats_content_frame.setStyleSheet("""
            QFrame {
                background-color: "#F4F6F8";
            }
        """)
        stats_content_frame.setObjectName("statsContentFrame")
        stats_content_frame_layout = QVBoxLayout(stats_content_frame)
        stats_content_frame_layout.setContentsMargins(15, 15, 15, 15)
        stats_content_frame_layout.setSpacing(10)
        stats_header_layout = QHBoxLayout()
        stats_title = QLabel("Detection Statistics")
        stats_title.setObjectName("panelTitleLabel")
        stats_header_layout.addWidget(stats_title)
        stats_header_layout.addStretch()
        export_icon = get_icon("export.png",
                               QStyle.StandardPixmap.SP_DialogSaveButton)
        self.export_btn = QPushButton(export_icon, " Export")
        self.export_btn.setObjectName("statsButton")
        self.export_btn.clicked.connect(self.export_statistics)
        clear_icon = get_icon("clear_current.png", QStyle.StandardPixmap.SP_TrashIcon)
        self.clear_btn = QPushButton(clear_icon, " Clear Current")
        self.clear_btn.setObjectName("statsButtonSecondary")
        self.clear_btn.clicked.connect(self.clear_current_detection_display)
        stats_header_layout.addWidget(self.export_btn)
        stats_header_layout.addSpacing(8) # Corrected: Added spacing value
        stats_header_layout.addWidget(self.clear_btn)
        stats_content_frame_layout.addLayout(stats_header_layout)

        # MODIFIED: Use QGridLayout for plastic stats cards
        self.plastic_stats_grid_layout = QGridLayout()
        self.plastic_stats_grid_layout.setHorizontalSpacing(10)
        self.plastic_stats_grid_layout.setVerticalSpacing(10)
        # Add the grid layout to the content frame layout. It will expand.
        stats_content_frame_layout.addLayout(self.plastic_stats_grid_layout)

        emphasis_box = QFrame()
        emphasis_box.setStyleSheet("""
            QFrame {
                background-color: #FFF3CD;
                border: 1px solid #FFA000;
                border-radius: 5px;
            }
        """)
        emphasis_layout = QVBoxLayout(emphasis_box)
        emphasis_layout.setContentsMargins(10, 6, 10, 6)

        emphasis_label = QLabel("⚠️ To detect more objects, lower the threshold.")
        emphasis_label.setStyleSheet("font-weight: bold; color: #8a6d3b; font-size: 13px;")
        emphasis_layout.addWidget(emphasis_label)

        stats_content_frame_layout.addWidget(emphasis_box)
                # --- Plastic Legend ---

        legend_title = QLabel("Color Legends for Each Plastic Type:")
        legend_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        stats_content_frame_layout.addWidget(legend_title)
        legend_container = QWidget()
        legend_layout = QGridLayout(legend_container)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setHorizontalSpacing(30)
        legend_layout.setVerticalSpacing(10)

        plastic_colors = {
            "PET":  "#FFFF00",  # Yellow
            "HDPE": "#FFA500",  # Orange
            "LDPE": "#00FF00",  # Green
            "PVC":  "#FF0000",  # Red
            "PP":   "#808080",  # Gray
            "PS":   "#800080"   # Violet
        }

        row = 0
        col = 0
        for index, (plastic, color) in enumerate(plastic_colors.items()):
            # Create color box
            color_box = QLabel()
            color_box.setFixedSize(20, 20)
            color_box.setStyleSheet(f"background-color: {color}; border-radius: 3px;")

            # Label next to the color
            text_label = QLabel(plastic)
            text_label.setStyleSheet("font-size: 12px;")

            item = QWidget()
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(5)
            item_layout.addWidget(color_box)
            item_layout.addWidget(text_label)

            legend_layout.addWidget(item, row, col)

            row += 1
            if row == 3:  # move to next column after 3 items
                row = 0
                col += 1

        # Add the legend layout to the stats frame
        stats_content_frame_layout.addWidget(legend_container)


        lower_stats_layout = QHBoxLayout()
        lower_stats_layout.setSpacing(10)

        # CORRECT: Unpacking the tuple into card_widget and value_label
        self.stat_total_items_card, self.stat_total_items_value_label = self.create_info_card(
            "Total Items Detected", "0", "overallStatCard") # Assuming "overallStatCard" for the main one

        # CORRECT THIS LINE: Unpack the tuple here as well
        self.stat_proc_time_card, self.stat_proc_time_value_label = self.create_info_card(
            "Processing Time", "0ms", "statCard") # Assign an object name like "statCard"

        lower_stats_layout.addWidget(self.stat_total_items_card)

        # CORRECT THIS LINE: Add the _card variable, not the tuple
        lower_stats_layout.addWidget(self.stat_proc_time_card)

        stats_content_frame_layout.addLayout(lower_stats_layout)
                
        # MODIFIED: Removed stats_scroll_area and added stats_content_frame directly
        stats_area_outer_layout.addWidget(stats_content_frame, 1) # Add with stretch to take available vertical space
        
        # Add the stats area widget directly to the overall layout
        overall_detection_layout.addWidget(self.stats_area_widget) # Give it a default stretch

        self.update_navigation_buttons()
        if self.model is None:
            self.drop_frame.setEnabled(False)
            self.webcam_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)

        return detection_widget

    def update_confidence_threshold_from_input(self):
        try:
            value = float(self.conf_input.text())
            if 0.0 <= value <= 1.0:
                self.confidence_threshold = value
                # Optional: Re-format input field to always show 2 decimal places
                self.conf_input.setText(f"{self.confidence_threshold:.2f}")
                self.update_analytics_view()

                if hasattr(self, "current_image_path") and self.current_image_path:
                    self.run_model_on_image_path(self.current_image_path)
            else:
                QMessageBox.warning(self, "Invalid Input", "Confidence Threshold must be between 0.0 and 1.0.")
                self.conf_input.setText(f"{self.confidence_threshold:.2f}") # Revert to last valid
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for Confidence Threshold.")
            self.conf_input.setText(f"{self.confidence_threshold:.2f}") # Revert to last valid

    def update_iou_threshold_from_input(self):
        try:
            value = float(self.iou_input.text())
            if 0.0 <= value <= 1.0:
                self.iou_threshold = value
                # Optional: Re-format input field to always show 2 decimal places
                self.iou_input.setText(f"{self.iou_threshold:.2f}")
                self.update_analytics_view()
                if hasattr(self, "current_image_path") and self.current_image_path:
                    self.run_model_on_image_path(self.current_image_path)
            else:
                QMessageBox.warning(self, "Invalid Input", "IoU Threshold must be between 0.0 and 1.0.")
                self.iou_input.setText(f"{self.iou_threshold:.2f}") # Revert to last valid
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for IoU Threshold.")
            self.iou_input.setText(f"{self.iou_threshold:.2f}") # Revert to last valid
    
    def update_detection_stats_card(self):
        if self.detection_history_memory:
            latest_record = self.detection_history_memory[-1]
            latest_proc_time = latest_record.get("processing_time_ms", 0)
            latest_num_items = len(latest_record.get("detected_objects", []))

            self.stat_total_items_value_label.setText(str(latest_num_items))
            self.stat_proc_time_value_label.setText(f"{latest_proc_time:.1f}ms")

    def setup_stat_cards(self):
        # Clear existing cards from the grid layout
        if hasattr(self, 'plastic_stats_grid_layout') and self.plastic_stats_grid_layout is not None:
            while self.plastic_stats_grid_layout.count():
                item = self.plastic_stats_grid_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        self.plastic_classes = []
        # ... (your existing logic for determining self.plastic_classes) ...
        # This part is fine:
        if self.model and hasattr(self.model, 'names'):
            defined_plastics = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"]
            model_plastic_names = []
            if isinstance(self.model.names, dict):
                temp_names = sorted(list(self.model.names.values()))
                for dp in defined_plastics:
                    for name_val in temp_names:
                        if dp.lower() in name_val.lower() and dp not in model_plastic_names:
                            model_plastic_names.append(dp)
                            break
            if len(model_plastic_names) < 6:
                remaining_model_names = [
                    name for name in temp_names if not any(dp.lower() in name.lower() for dp in defined_plastics)]
                needed = max(0, 6 - len(model_plastic_names))
                model_plastic_names.extend(remaining_model_names[:needed])
            if model_plastic_names:
                self.plastic_classes = model_plastic_names
            else:
                self.plastic_classes = defined_plastics
        if not self.plastic_classes:
            self.plastic_classes = [f"Class_{i}" for i in range(min(6, len(
                self.model.names if self.model and hasattr(self.model, 'names') else {})))]
        else:
            self.plastic_classes = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"]
            print(
                "Warning: Model names not available for stat cards, using default plastic classes.")
        self.stat_cards = {}
        num_classes = len(self.plastic_classes)

        # MODIFIED: Add cards to QGridLayout with 2 columns
        for i, class_name in enumerate(self.plastic_classes):
            card = self.create_stat_card(class_name, "0", 0)
            self.stat_cards[class_name] = card
            row = i // 2
            col = i % 2
            self.plastic_stats_grid_layout.addWidget(card, row, col)

        # Add stretch to the grid layout to ensure cards occupy available space
        self.plastic_stats_grid_layout.setRowStretch(row + 1, 1)
        self.plastic_stats_grid_layout.setColumnStretch(0, 1)
        self.plastic_stats_grid_layout.setColumnStretch(1, 1)

        self.clear_detection_statistics_display()

    def create_stat_card(self, title_text, value_text, progress_value):
        card = QFrame()
        card.setObjectName("statCard")
        layout = QVBoxLayout(card)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 7, 10, 7)
        title = QLabel(title_text)
        title.setObjectName("statCardTitle")
        value_label = QLabel(value_text)
        value_label.setObjectName("statCardValue")
        progress = QProgressBar()
        progress.setValue(progress_value)
        progress.setTextVisible(False)
        progress.setObjectName("statCardProgress")
        layout.addWidget(title)
        layout.addWidget(value_label)
        layout.addWidget(progress)
        # MODIFIED: Removed fixed height to allow vertical expansion
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Allow both horizontal and vertical expansion
        return card

    # Place this method inside your WasteDetectionApp class
    def create_info_card(self, title_text, initial_value, card_object_name="statCard"):
        # Helper method to create a reusable info/stat card widget
        card_frame = QFrame()
        card_frame.setObjectName(card_object_name) # Used for QSS styling (e.g., "statCard", "overallStatCard")
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(15, 15, 15, 15) # Padding inside the card
        card_layout.setSpacing(5) # Space between title and value

        title_label = QLabel(title_text)
        title_label.setObjectName("statCardTitleLabel") # Used for QSS styling
        value_label = QLabel(initial_value)
        value_label.setObjectName("statCardValueLabel") # Used for QSS styling

        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)
        return card_frame, value_label # IMPORTANT: Return both the QFrame and the QLabel

    # --- Analytics View (No Charts) ---
    def create_analytics_view(self):
        """Creates the Analytics dashboard view widget with charts and improved summary."""
        analytics_widget = QWidget()
        analytics_widget.setObjectName("analyticsView")
        main_layout = QVBoxLayout(analytics_widget)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(15)  # Space between elements

        # --- Top Bar: Title and Time Range ---
        top_bar_layout = QHBoxLayout()
        title_label = QLabel("📊 Analytics Dashboard")
        title_label.setObjectName("viewTitleLabel")
        top_bar_layout.addWidget(title_label)
        top_bar_layout.addStretch()
        self.analytics_time_combo = QComboBox()
        self.analytics_time_combo.addItems(
            ["Current Session"])
        self.analytics_time_combo.setCurrentText("Current Session")
        self.analytics_time_combo.setFixedWidth(150)
        self.analytics_time_combo.setObjectName("timeFilterCombo")
        self.analytics_time_combo.currentTextChanged.connect(
            self.update_analytics_view)
        top_bar_layout.addWidget(self.analytics_time_combo)
        main_layout.addLayout(top_bar_layout)

        # --- Summary Cards ---
        # Store the returned value labels as instance variables for easy updating
        summary_layout_1 = QHBoxLayout()
        summary_layout_1.setSpacing(15)
        self.analytics_total_items_card, self.analytics_total_items_value = self.create_info_card(
            "Total Items Detected", "0")
        self.analytics_avg_proc_time_card, self.analytics_avg_proc_time_value = self.create_info_card(
            "Avg. Processing Time", "0ms")
        self.analytics_avg_conf_card, self.analytics_avg_conf_value = self.create_info_card(
            "Avg. Confidence", "0%")
        summary_layout_1.addWidget(self.analytics_total_items_card)
        summary_layout_1.addWidget(self.analytics_avg_proc_time_card)
        summary_layout_1.addWidget(self.analytics_avg_conf_card)
        main_layout.addLayout(summary_layout_1)

        summary_layout_2 = QHBoxLayout()
        summary_layout_2.setSpacing(15)
        self.analytics_num_records_card, self.analytics_num_records_value = self.create_info_card(
            "Total Records (Images/Frames)", "0")
        self.analytics_most_frequent_card, self.analytics_most_frequent_value = self.create_info_card(
            "Most Frequent Class", "N/A")
        summary_layout_2.addWidget(self.analytics_num_records_card)
        summary_layout_2.addWidget(self.analytics_most_frequent_card)
        summary_layout_2.addStretch(1)  # Keep cards aligned if fewer than 3
        main_layout.addLayout(summary_layout_2)

        # --- Detailed Breakdown Area / Chart ---
        # This frame will now hold the Matplotlib chart
        self.chart_frame_container = QFrame() # Renamed to avoid potential conflict with self.chart_frame if used elsewhere
        self.chart_frame_container.setObjectName("chartFrame")  # Reuse chart frame style
        chart_container_layout = QVBoxLayout(self.chart_frame_container)
        chart_container_layout.setContentsMargins(15, 10, 15, 15)
        chart_container_layout.setSpacing(10)

        chart_title = QLabel("Detection Class Distribution") # More appropriate title for a chart
        chart_title.setObjectName("chartTitleLabel")
        chart_container_layout.addWidget(chart_title)

        # Matplotlib Chart Integration
        self.figure = Figure(figsize=(5, 4), dpi=100) # You can adjust figsize and dpi as needed
        self.canvas = FigureCanvas(self.figure)
        chart_container_layout.addWidget(self.canvas)

        # Optional: Set transparent background for matplotlib figure and canvas for QSS to show through
        # You might need to import `plt` at the top of your file for `plt.style.use` if not already
        self.figure.patch.set_facecolor('none')
        self.canvas.setStyleSheet("background-color: transparent;")

        main_layout.addWidget(self.chart_frame_container, 1) # Allow this container to take vertical space

        return analytics_widget

    def create_info_card(self, title_text, initial_value, card_object_name="statCard"):
        # Helper method to create a reusable info/stat card widget
        card_frame = QFrame()
        card_frame.setObjectName(card_object_name) # Used for QSS styling (e.g., "statCard", "overallStatCard")
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(15, 15, 15, 15) # Padding inside the card
        card_layout.setSpacing(5) # Space between title and value

        title_label = QLabel(title_text)
        title_label.setObjectName("statCardTitleLabel") # Used for QSS styling
        value_label = QLabel(initial_value)
        value_label.setObjectName("statCardValueLabel") # Used for QSS styling

        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)
        return card_frame, value_label # Return both the card frame and the value label
    
    # --- History/Gallery View ---
    def create_history_view(self):
        history_widget = QWidget()
        history_widget.setStyleSheet(
            "background-color: #F4F6F8; border-radius: 8px; padding:10px;")
        history_widget.setObjectName("historyView")
        main_layout = QVBoxLayout(history_widget)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(10)  # Space between filter/gallery/pagination

        title_label = QLabel("🕓 Detection History")
        title_label.setObjectName("viewTitleLabel")
        main_layout.addWidget(title_label)

        # --- Filter Bar ---
        filter_bar_layout = QHBoxLayout()
        filter_bar_layout.setSpacing(10)
        search_icon = get_icon("search.svg", QStyle.StandardPixmap.SP_FileDialogContentsView)
        search_label = QLabel()
        search_label.setPixmap(search_icon.pixmap(QSize(18, 18)))
        self.history_search_input = QLineEdit()
        self.history_search_input.setPlaceholderText(
            "Search by image name or detected class...")
        self.history_search_input.setObjectName("searchInput")
        self.history_search_timer = QTimer()
        self.history_search_timer.setSingleShot(True)
        self.history_search_timer.timeout.connect(
            lambda: self.update_history_view(page=1))
        self.history_search_input.textChanged.connect(
            lambda: self.history_search_timer.start(500))  # 500ms delay
        self.history_filter_combo = QComboBox()
        self.history_filter_combo.addItem("Filter by type: All")
        self.history_filter_combo.setFixedWidth(180)
        self.history_filter_combo.currentIndexChanged.connect(
            lambda: self.update_history_view(page=1))
        filter_bar_layout.addWidget(search_label)
        filter_bar_layout.addWidget(self.history_search_input, 1)  # Stretch
        filter_bar_layout.addWidget(self.history_filter_combo)
        main_layout.addLayout(filter_bar_layout)

        # --- Gallery Area ---
        self.history_scroll_area = QScrollArea()
        self.history_scroll_area.setWidgetResizable(True)
        self.history_scroll_area.setObjectName("galleryScrollArea")
        self.history_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.gallery_content_widget = QWidget()  # Container for the items
   
        # MODIFIED: Use FlowLayout instead of QGridLayout
        self.gallery_flow_layout = FlowLayout(self.gallery_content_widget, margin=10, hSpacing=15, vSpacing=15)
        # self.gallery_content_widget.setLayout(self.gallery_flow_layout) # This line is not needed as FlowLayout takes parent

        self.history_scroll_area.setWidget(self.gallery_content_widget)
        main_layout.addWidget(self.history_scroll_area, 1) # Stretch gallery

        self.history_scroll_area.setWidget(self.gallery_content_widget)
        main_layout.addWidget(self.history_scroll_area, 1) # Stretch gallery

        # --- Pagination Controls ---
        pagination_layout = QHBoxLayout()
        pagination_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pagination_layout.setSpacing(10)

        prev_icon = get_icon("left-arrow.png",
                             QStyle.StandardPixmap.SP_ArrowLeft)
        self.history_prev_btn = QPushButton(prev_icon, " Previous")
        self.history_prev_btn.setObjectName("paginationButton")
        self.history_prev_btn.clicked.connect(self.history_prev_page)

        self.history_page_label = QLabel("Page 1 / 1")
        self.history_page_label.setObjectName("pageLabel")

        next_icon = get_icon("right-arrow.png",
                             QStyle.StandardPixmap.SP_ArrowRight)
        self.history_next_btn = QPushButton("Next ") # Space for icon
        self.history_next_btn.setIcon(next_icon)
        # Icon on right
        self.history_next_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.history_next_btn.setObjectName("paginationButton")
        self.history_next_btn.clicked.connect(self.history_next_page)

        pagination_layout.addWidget(self.history_prev_btn)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.history_page_label)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.history_next_btn)
        main_layout.addLayout(pagination_layout)

        return history_widget

    def populate_history_filter_combo(self):
        if hasattr(self, 'history_filter_combo') and self.plastic_classes:
            current_text = self.history_filter_combo.currentText()
            # Clear existing items except "All"
            while self.history_filter_combo.count() > 1:
                self.history_filter_combo.removeItem(1)
            # Add determined classes
            for class_name in self.plastic_classes:
                self.history_filter_combo.addItem(f"Filter by type: {class_name}")
            # Restore selection if possible
            index = self.history_filter_combo.findText(current_text)
            if index != -1:
                self.history_filter_combo.setCurrentIndex(index)
            else:
                self.history_filter_combo.setCurrentIndex(0)


    def handle_accordion_toggle(self, checked, current_frame, current_button):
        # Collapse the previously open accordion, if any
        if self.open_accordion_frame and self.open_accordion_frame != current_frame:
            self.open_accordion_frame.setMaximumHeight(0)
            self.open_accordion_frame.hide()
            # Uncheck the previous button without triggering its toggle signal again
            self.open_accordion_button.blockSignals(True)
            self.open_accordion_button.setChecked(False)
            self.open_accordion_button.blockSignals(False)


        if checked:
            # Expand the current accordion
            current_frame.setMaximumHeight(current_frame.sizeHint().height() + 50) # Add some buffer
            current_frame.show()
            self.open_accordion_frame = current_frame
            self.open_accordion_button = current_button
        else:
            # Collapse the current accordion if it was the one currently open
            if self.open_accordion_frame == current_frame:
                current_frame.setMaximumHeight(0)
                current_frame.hide()
                self.open_accordion_frame = None
                self.open_accordion_button = None

    def apply_stylesheet(self):
        BG_PRIMARY = "#F9FAFC"
        BG_SECONDARY = "#FFFFFF"
        BG_CONTENT_PRIMARY = "#FFFFFF"
        BG_CONTENT_SECONDARY = "#F3F5F7"

        TEXT_PRIMARY = "#2C3E50"
        TEXT_SECONDARY = "#7F8C8D"
        TEXT_PLACEHOLDER = "#BDC3C7"
        TEXT_ON_BUTTON_GREEN = "#FFFFFF"
        TEXT_ON_HEADER = "#FFFFFF"
        TEXT_GREEN_TITLE = "#27AE60"

        BUTTON_GREEN = "#27AE60"
        BUTTON_GREEN_HOVER = "#2ECC71"
        BUTTON_GREEN_PRESSED = "#229954"

        BORDER_COLOR = "#E0E6ED"
        SEPARATOR_COLOR = "#D1D9E0"
        DISABLED_BG_COLOR = "#EAECEF"
        DISABLED_TEXT_COLOR = "#AAB7BF"

        SIDEBAR_BG = "#FFFFFF"
        SIDEBAR_BORDER = "#E0E6ED"
        SIDEBAR_TEXT = "#2C3E50"
        SIDEBAR_ICON_COLOR = "#7F8C8D"
        SIDEBAR_BUTTON_HOVER_BG = "#808080"
        SIDEBAR_BUTTON_HOVER_TEXT = "#27AE60"
        SIDEBAR_BUTTON_CHECKED_BG = "#22892d"
        SIDEBAR_BUTTON_CHECKED_TEXT = "#27AE60"
        SIDEBAR_BUTTON_CHECKED_BORDER = "#27AE60"

        HEADER_BG_COLOR = "#F0F5F8" # A slightly off-white/light gray for header distinction

        stylesheet = f"""
                QWidget {{
                    font-family: "{APP_FONT_FAMILY}";
                }}
                WasteDetectionApp {{
                    background-color: {BG_PRIMARY};
                }}
                SplashScreen {{
                    background-color: {BG_SECONDARY};
                    border: 2px solid {BUTTON_GREEN};
                }}
                #headerBar {{
                    background-color: {HEADER_BG_COLOR}; /* Changed from BG_SECONDARY */
                    border-bottom: 1px solid {BORDER_COLOR};
                }}
                #headerTitleLabel {{
                    color: {TEXT_GREEN_TITLE};
                    font-size: 17px;
                    font-weight: bold;
                }}
                #headerExitButton {{
                    color: {TEXT_SECONDARY};
                    background-color: transparent; border: none;
                    padding: 8px 12px; font-size: 13px;
                }}
                #headerExitButton:hover {{ background-color: {DISABLED_BG_COLOR}; }}
                #navBar {{
                    background-color: {SIDEBAR_BG};
                    border-right: 1px solid {SIDEBAR_BORDER};
                    padding-top: 50px;
                }}
                QPushButton#navButton {{
                    background-color: {BUTTON_GREEN};
                    color: {BORDER_COLOR};
                    border: none;

                    width: 60px;
                    height: 60px;
                    min-width: 60px;
                    max-width: 60px;
                    min-height: 60px;
                    max-height: 60px;

                    text-align: center;
                    padding: 0px;

                    qproperty-iconSize: 32px 32px;

                    border-radius: 5px;
                }}
                QPushButton#navButton:hover {{
                    background-color: {SIDEBAR_BUTTON_HOVER_BG};
                    color: {SIDEBAR_BUTTON_HOVER_TEXT};
                }}
                QPushButton#navButton:checked {{
                    background-color: {SIDEBAR_BUTTON_CHECKED_BG};
                    color: {SIDEBAR_BUTTON_CHECKED_TEXT};

                    border: 2px solid {SIDEBAR_BUTTON_CHECKED_BORDER};
                }}
                #navBarAppNameLabel {{
                    color: {SIDEBAR_TEXT}; font-size: 15px; font-weight: bold;
                    padding: 15px 10px; border-bottom: 1px solid {SIDEBAR_BORDER};
                }}
                #mainContentWidget {{ background-color: {BG_PRIMARY}; padding: 0px; }}
                #controlPanel, #imageDisplayWidget, #statsArea,
                #analyticsViewWidgetContainer, #historyViewWidgetContainer {{
                    background-color: {BG_CONTENT_PRIMARY}; border-radius: 8px;
                    border: 1px solid {BORDER_COLOR};
                }}
                #controlPanel {{ padding: 15px; }}
                #imageDisplayWidget {{ padding: 0px; }}
                #statsArea {{ padding: 15px; }}
                #analyticsViewWidgetContainer {{ padding: 15px; }}
                #historyViewWidgetContainer {{ padding: 15px; }}
                QLabel {{ color: {TEXT_PRIMARY}; font-size: 13px; background-color: transparent; }}
                #panelTitleLabel, .panelSectionTitleLabel {{
                    color: {TEXT_GREEN_TITLE};
                    font-size: 18px;
                    font-weight: bold;
                    padding-bottom: 8px;
                    border-bottom: 2px solid {BUTTON_GREEN};
                    margin-bottom: 15px;
                }}
                #imageDisplayLabel {{
                    background-color: {BG_CONTENT_SECONDARY}; color: {TEXT_PLACEHOLDER};
                    border-radius: 6px;
                    border: 1px solid {BORDER_COLOR};
                }}
                #imageScrollArea {{ border: none; background-color: transparent; }}
                QPushButton {{
                    color: {TEXT_ON_BUTTON_GREEN}; background-color: {BUTTON_GREEN};
                    border: none; padding: 10px 18px; border-radius: 6px;
                    font-size: 14px; font-weight: 600;
                    min-width: 80px;
                }}
                QPushButton:hover {{ background-color: {BUTTON_GREEN_HOVER}; }}
                QPushButton:pressed {{ background-color: {BUTTON_GREEN_PRESSED}; }}
                QPushButton:disabled {{
                    background-color: {DISABLED_BG_COLOR}; color: {DISABLED_TEXT_COLOR};
                }}
                #imageNavButton {{
                    background-color: {BG_SECONDARY}; color: {TEXT_PRIMARY};
                    border: 1px solid {BORDER_COLOR}; padding: 8px 10px;
                    border-radius: 5px;
                }}
                #imageNavButton:hover {{ background-color: {DISABLED_BG_COLOR}; }}
                #imageNavButton:disabled {{
                    background-color: {DISABLED_BG_COLOR}; color: {DISABLED_TEXT_COLOR};
                    border: 1px solid {DISABLED_BG_COLOR};
                }}
                #statsButtonSecondary {{
                    background-color: {BG_PRIMARY};
                    color: {TEXT_PRIMARY};
                    border: 1px solid {BORDER_COLOR};
                }}
                #statsButtonSecondary:hover {{ background-color: {DISABLED_BG_COLOR}; }}
                QSlider::groove:horizontal {{
                    border: 1px solid {BORDER_COLOR}; height: 8px;
                    background: {DISABLED_BG_COLOR}; border-radius: 4px;
                }}
                QSlider::handle:horizontal {{
                    background: {BUTTON_GREEN}; border: 1px solid {BUTTON_GREEN};
                    width: 16px; height: 16px; margin: -4px 0;
                    border-radius: 8px;
                }}
                QComboBox {{
                    border: 1px solid {BORDER_COLOR}; border-radius: 6px;
                    padding: 7px 10px; background-color: {BG_SECONDARY};
                    color: {TEXT_PRIMARY}; min-height: 24px;
                }}
                QComboBox::drop-down {{
                    subcontrol-origin: padding; subcontrol-position: top right; width: 25px;
                    border-left: 1px solid {BORDER_COLOR};
                }}
                QComboBox::down-arrow {{ image: url({ICON_DIR}down_arrow_dark.png); width: 12px; height: 12px; }}
                QComboBox QAbstractItemView {{
                    border: 1px solid {BORDER_COLOR}; background-color: {BG_SECONDARY};
                    color: {TEXT_PRIMARY}; selection-background-color: {BUTTON_GREEN};
                    selection-color: {TEXT_ON_BUTTON_GREEN};
                    padding: 5px;
                }}
                QFrame#dropFrame {{
                    border: 2px dashed {BORDER_COLOR}; border-radius: 8px;
                    background-color: {BG_CONTENT_SECONDARY};
                }}
                #dropFrame QLabel {{ color: {TEXT_PLACEHOLDER}; background-color: transparent; }}
                QFrame#statCard, QFrame#overallStatCard {{
                    background-color: {BG_SECONDARY}; border-radius: 8px;
                    border: 1px solid {BORDER_COLOR}; padding: 15px;
                    box-shadow: 2px 2px 5px 0px rgba(0,0,0,0.05);
                }}
                .statCardTitleLabel {{
                    font-weight: bold; color: {TEXT_SECONDARY}; font-size: 13px;
                }}
                .statCardValueLabel {{
                    font-size: 20px; font-weight: bold;
                    color: {BUTTON_GREEN};
                }}
                .statCardPercentageLabel {{
                    font-size: 12px; color: {TEXT_PLACEHOLDER};
                }}
                #overallStatCard .statCardTitleLabel {{
                    color: {TEXT_SECONDARY};
                }}
                #overallStatCard .statCardValueLabel {{
                    font-size: 18px; color: {TEXT_PRIMARY};
                }}
                QScrollBar:vertical, QScrollBar:horizontal {{
                    border: none; background: {BG_PRIMARY};
                    width: 12px; height: 12px;
                }}
                QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                    background: {SEPARATOR_COLOR}; min-height: 30px; min-width: 30px;
                    border-radius: 6px;
                }}
                QScrollBar::handle:hover {{ background: {TEXT_PLACEHOLDER}; }}
                QScrollBar::add-line, QScrollBar::sub-line,
                QScrollBar::up-arrow, QScrollBar::down-arrow, QScrollBar::left-arrow, QScrollBar::right-arrow,
                QScrollBar::add-page, QScrollBar::sub-page {{
                    background: none; border: none; height:0px; width:0px;
                }}
                #analyticsViewWidgetContainer QTextBrowser,
                #historyViewWidgetContainer QTextBrowser {{
                    background-color: {BG_SECONDARY}; color: {TEXT_PRIMARY};
                    border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 12px;
                    font-size: 14px;
                }}
                #analyticsViewWidgetContainer QChartView {{
                    border: 1px solid {BORDER_COLOR}; border-radius: 6px;
                    background-color: {BG_SECONDARY};
                }}
                #historyViewWidgetContainer QLineEdit,
                #historyViewWidgetContainer QDateEdit {{
                    border: 1px solid {BORDER_COLOR}; border-radius: 6px;
                    padding: 7px 10px; background-color: {BG_SECONDARY};
                    color: {TEXT_PRIMARY}; min-height: 24px;
                }}
                #historyViewWidgetContainer QDateEdit::drop-down {{
                    image: url({ICON_DIR}calendar_dark.png);
                    border-left: 1px solid {BORDER_COLOR}; padding: 3px;
                }}
                #historyGridContainer {{ background-color: transparent; }}
                QFrame.historyItemFrame {{
                    background-color: {BG_SECONDARY}; border: 1px solid {BORDER_COLOR};
                    border-radius: 8px; padding:15px;
                    box-shadow: 1px 1px 3px 0px rgba(0,0,0,0.03);
                }}
                QFrame.historyItemFrame:hover {{ border-color: {BUTTON_GREEN}; }}
                QLabel.historyItemImageLabel {{
                    background-color: {BG_CONTENT_SECONDARY}; border-radius: 6px;
                    min-width: 80px; max-width: 80px;
                    min-height: 80px; max-height: 80px;
                }}
                .historyItemDetailsTitleLabel {{
                    font-size: 13px; font-weight: bold; color: {TEXT_SECONDARY};
                    background-color: transparent;
                }}
                .historyItemDetailsValueLabel {{
                    font-size: 13px; color: {TEXT_PRIMARY};
                    background-color: transparent;
                }}
                .historyItemTimestampLabel {{
                    font-size: 11px; color: {TEXT_PLACEHOLDER};
                    background-color: transparent;
                }}
                #historyPageLabel {{ font-weight: bold; font-size: 14px; color: {TEXT_PRIMARY}; }}
                #historyViewWidgetContainer .navigationButton {{
                    background-color: {BG_SECONDARY}; color: {BUTTON_GREEN};
                    border: 1px solid {BUTTON_GREEN};
                    padding: 8px 12px; font-weight: normal;
                    border-radius: 5px;
                }}
                #historyViewWidgetContainer .navigationButton:hover {{
                    background-color: {BUTTON_GREEN}; color: {TEXT_ON_BUTTON_GREEN};
                    border-color: {BUTTON_GREEN_HOVER};
                }}
                #historyViewWidgetContainer .navigationButton:disabled {{
                    background-color: {DISABLED_BG_COLOR}; color: {DISABLED_TEXT_COLOR};
                    border: 1px solid {DISABLED_BG_COLOR};
                }}
                QProgressBar {{
                    border: 1px solid {BORDER_COLOR}; border-radius: 6px; text-align: center;
                    background-color: {DISABLED_BG_COLOR}; color: {TEXT_PRIMARY};
                    font-size: 13px; height: 20px;
                }}
                QProgressBar::chunk {{
                    background-color: {BUTTON_GREEN}; border-radius: 5px;
                }}
                #analyticsView, #historyView, #definitionsView {{
                    background-color: {BG_PRIMARY};
                }}
                #viewTitleLabel {{
                    font-size: 24px;
                    font-weight: bold;
                    color: {TEXT_PRIMARY};
                }}
                #timeFilterCombo {{
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 6px;
                    padding: 6px 12px;
                    background-color: {BG_SECONDARY};
                    color: {TEXT_PRIMARY};
                }}
                #chartFrame {{
                    background-color: {BG_SECONDARY};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 2px 2px 8px 0px rgba(0,0,0,0.08);
                }}
                #chartTitleLabel {{
                    font-size: 17px;
                    font-weight: bold;
                    color: {TEXT_PRIMARY};
                }}
                #definitionsBrowser {{
                    background-color: {BG_SECONDARY};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 8px;
                    padding: 18px;
                    color: {TEXT_PRIMARY};
                    font-size: 14px;
                }}
                #definitionsBrowser a {{
                    color: {BUTTON_GREEN};
                    text-decoration: none;
                }}
                #definitionsBrowser a:hover {{
                    text-decoration: underline;
                }}
                #searchInput {{
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 6px;
                    padding: 7px 12px;
                    background-color: {BG_SECONDARY};
                    color: {TEXT_PRIMARY};
                }}
                #searchInput::placeholder {{
                    color: {TEXT_PLACEHOLDER};
                }}
                QDateEdit {{
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 6px;
                    padding: 6px 10px;
                    background-color: {BG_SECONDARY};
                    color: {TEXT_PRIMARY};
                }}
                QDateEdit::drop-down {{
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 25px;
                    border-left-width: 1px;
                    border-left-color: {BORDER_COLOR};
                    border-left-style: solid;
                    border-top-right-radius: 5px;
                    border-bottom-right-radius: 5px;
                }}
                QDateEdit::down-arrow {{
                    image: url(icons/calendar.svg);
                    width: 14px;
                    height: 14px;
                }}
                QListWidget {{
                    background-color: {BG_PRIMARY};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 10px;
                    padding: 8px;
                    outline: none;
                }}
                QListWidget::item {{
                    background-color: {BG_SECONDARY};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 8px;
                    padding: 12px;
                    min-height: 85px;
                }}
                QListWidget::item:hover {{
                    background-color: {DISABLED_BG_COLOR};
                }}
                QListWidget::item:selected {{
                    background-color: {BUTTON_GREEN};
                    color: white;
                    border: 1px solid {BUTTON_GREEN_HOVER};
                }}
                #paginationButton {{
                    background-color: {BG_SECONDARY};
                    color: {BUTTON_GREEN};
                    border: 1px solid {BUTTON_GREEN};
                    border-radius: 5px;
                    padding: 6px 15px;
                }}
                #paginationButton:hover {{
                    background-color: {BUTTON_GREEN};
                    color: {TEXT_ON_BUTTON_GREEN};
                    border-color: {BUTTON_GREEN_HOVER};
                }}
                #paginationButton:pressed {{
                    background-color: {BUTTON_GREEN_PRESSED};
                    color: {TEXT_ON_BUTTON_GREEN};
                }}
                #paginationButton:disabled {{
                    opacity: 0.6;
                    background-color: {DISABLED_BG_COLOR};
                    color: {DISABLED_TEXT_COLOR};
                    border: 1px solid {DISABLED_BG_COLOR};
                }}
                #currentPageLabel {{
                    color: {TEXT_PRIMARY};
                    font-weight: bold;
                    font-size: 16px;
                }}
                """
        self.setStyleSheet(stylesheet)
        self.style().unpolish(self)
        self.style().polish(self)
        for child_widget in self.findChildren(QWidget):
            child_widget.style().unpolish(child_widget)
            child_widget.style().polish(child_widget)
            child_widget.update()

    # --- Event Handlers ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.model:
            return
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        image_files = [f for f in files if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        if image_files:
            self.load_dropped_images(image_files)

    # --- Image Loading/Processing ---
    def load_dropped_images(self, file_paths):
        if self.webcam_running:
            self.toggle_webcam()
        self.image_paths = file_paths
        self.current_image_index = -1
        if self.image_paths:
            self.current_image_index = 0
            self.current_image_path = self.image_paths[0]  
            self.run_model_on_image_path(self.current_image_path)
        self.update_navigation_buttons()
        self.update_image_count_label()

    def update_webcam_list(self):
        self.webcam_dropdown.clear()
        available_cams = []
        for i in range(5):  # Check first 5 indices
            # Try DirectShow backend first (often better on Windows)
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cams.append((f"Camera {i}", i))
                cap.release()
            else:  # Try default backend as fallback
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cams.append((f"Camera {i} (Default)", i))
                    cap.release()

        if available_cams:
            for name, idx in available_cams:
                self.webcam_dropdown.addItem(name, idx)
            if self.model and hasattr(self, 'webcam_btn'):
                self.webcam_btn.setEnabled(True)
        else:
            self.webcam_dropdown.addItem("No webcams found", -1)
            if hasattr(self, 'webcam_btn'):
                self.webcam_btn.setEnabled(False)

    def update_navigation_buttons(self):
        has_images = bool(self.image_paths) and not self.webcam_running
        can_navigate = has_images and self.model is not None

        if hasattr(self, 'prev_btn'):
            self.prev_btn.setEnabled(
                can_navigate and self.current_image_index > 0)
        if hasattr(self, 'next_btn'):
            self.next_btn.setEnabled(
                can_navigate and self.current_image_index < len(self.image_paths) - 1)

    def update_image_count_label(self):
        if self.image_paths and self.current_image_index != -1 and not self.webcam_running:
            self.image_count_label.setText(
                f"{self.current_image_index + 1} / {len(self.image_paths)}")
        else:
            self.image_count_label.setText("— / —")

    def upload_images(self):
        if not self.model:
            QMessageBox.warning(self, "Model Not Ready",
                                "Please wait for the model to load.")
            return
        if self.webcam_running:
            self.toggle_webcam()

        file_names, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_names:
            self.load_dropped_images(file_names)

    def run_model_on_image_path(self, file_path):
        """Runs model on a single image file and saves result to memory."""
        if not self.model:
            self.image_label.setText("Model not loaded.")
            self.clear_detection_statistics_display()
            return
        try:
            confidence = self.confidence_threshold
            iou = self.iou_threshold
            abs_file_path = os.path.abspath(file_path) if not os.path.isabs(file_path) else file_path

            img_cv = cv2.imread(abs_file_path)
            if img_cv is None:
                self.image_label.setText(f"Error reading image\n{abs_file_path}")
                self.original_pixmap = None
                self.display_scaled_image()
                self.clear_detection_statistics_display()
                return

            # Check if this image is already in memory
            existing_record = next((r for r in self.detection_history_memory if r['image_path'] == abs_file_path), None)

            if existing_record:
                # Use stored raw detections and processing time
                current_detections = existing_record['detected_objects']
                proc_time_ms = existing_record['processing_time_ms']
            else:
                start_time = time.time()
                # Set conf to 0.01 to get ALL detections
                results = self.model(img_cv, conf=0.01, iou=iou)
                end_time = time.time()
                proc_time_ms = (end_time - start_time) * 1000

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

                # Save unfiltered detections to memory
                history_record = {
                    "id": len(self.detection_history_memory) + 1,
                    "timestamp": datetime.now(),
                    "image_path": abs_file_path,
                    "source_type": 'file',
                    "processing_time_ms": proc_time_ms,
                    "confidence_threshold": confidence,
                    "iou_threshold": iou,
                    "detected_objects": current_detections
                }
                self.detection_history_memory.append(history_record)
                self.update_detection_stats_card()

            # Filter for display (apply current threshold)
            display_detections = [
                det for det in current_detections if det.get("conf", 0) >= self.confidence_threshold
            ]

            # Draw only filtered boxes
            annotated_img = self.draw_custom_boxes_from_list(
                img_cv.copy(), display_detections, os.path.basename(abs_file_path)
            )

            # Convert to QImage for display
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            h, w, ch = annotated_img_rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(annotated_img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qt_img)

            # Show image
            self.display_scaled_image()

            # Update UI stats from filtered detections
            self.update_detection_statistics_from_list(display_detections, proc_time_ms)

        except Exception as e:
            self.image_label.setText(f"Error processing image:\n{str(e)}")
            print(f"Error in run_model_on_image_path: {e}")
            self.original_pixmap = None
            self.display_scaled_image()
            self.latest_detection_details = []
            self.clear_detection_statistics_display()


    # --- Drawing and Statistics ---
    def draw_custom_boxes_from_list(self, image, detections_list, source_filename="image"):
        """Draws boxes and updates latest_detection_details. Now handles optional track_id."""
        img_h, img_w = image.shape[:2]
        class_colors = {
            "PET": (0, 255, 255),   # Yellow (B=0, G=255, R=255)
            "HDPE": (0, 165, 255),  # Orange (B=0, G=165, R=255)
            "LDPE": (0, 255, 0),    # Green  (B=0, G=255, R=0)
            "PVC": (0, 0, 255),     # Red    (B=0, G=0, R=255)
            "PP": (128, 128, 128),  # Gray   (B=128, G=128, R=128)
            "PS": (128, 0, 128),    # Purple (B=128, G=0, R=128)
        }
        default_color = (200, 200, 200)
        annotated_image = image.copy()
        export_data_for_current_image = []

        for i, det in enumerate(detections_list):
            x1, y1, x2, y2 = det['box']
            model_class_name_raw = det['class']
            conf = det['conf']
            track_id = det.get('track_id') # Get track_id if it exists
            
            display_class_name = model_class_name_raw  # Default
            color = default_color

            if display_class_name in ["PET", "HDPE", "LDPE"]:
                font_color = (0, 0, 0)  # Black font for these classes
            else:
                font_color = (255, 255, 255)
            # Match standard abbreviations
            for target_class in self.plastic_classes:
                if target_class.lower() in model_class_name_raw.lower():
                    display_class_name = target_class
                    color = class_colors.get(target_class, default_color)
                    break
            
            export_obj = {
                "image_source": source_filename, "object_id": i + 1,
                "class_name": display_class_name, "confidence": f"{conf:.2f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            }
            if track_id is not None:
                export_obj["track_id"] = track_id
            export_data_for_current_image.append(export_obj)

            label_text = f"{display_class_name} {conf:.2f}"
            if conf < 0.5:
                label_text += " (low confidence)"
            if track_id is not None:
                label_text = f"ID {track_id}: {label_text}"
                
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            offset_from_box = 5
            background_padding_vertical = 2
            text_horizontal_padding = 3

            label_y_pos_baseline = y1 - offset_from_box
            label_bg_y1 = y1 - th - offset_from_box - background_padding_vertical
            label_bg_y2 = y1 - offset_from_box + baseline // 2

            # Adjust label position if it goes off the top
            if label_bg_y1 < 0:
                label_y_pos_baseline = y2 + th + offset_from_box
                label_bg_y1 = y2 + offset_from_box - baseline // 2
                label_bg_y2 = y2 + th + offset_from_box + background_padding_vertical

            label_bg_x1 = x1
            label_bg_x2 = x1 + tw + (2 * text_horizontal_padding)
            # Clamp label background to image bounds
            label_bg_x1 = max(0, label_bg_x1)
            label_bg_x2 = min(img_w, label_bg_x2)

            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            # Draw label background
            if label_bg_x1 < label_bg_x2 and label_bg_y1 < label_bg_y2:
                cv2.rectangle(annotated_image, (label_bg_x1, label_bg_y1),
                              (label_bg_x2, label_bg_y2), color, -1)
            # Draw label text
            actual_text_x_pos = max(x1, label_bg_x1) + text_horizontal_padding
            actual_text_y_pos = max(th, label_y_pos_baseline)
            cv2.putText(annotated_image, label_text, (actual_text_x_pos, actual_text_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, cv2.LINE_AA)
            

        # Update instance variable for export
        self.latest_detection_details = export_data_for_current_image
        return annotated_image

    def clear_detection_statistics_display(self):
        if not hasattr(self, 'stat_cards') or not self.stat_cards:
            return
        # Reset class counts
        for class_name in self.plastic_classes:
            card = self.stat_cards.get(class_name)
            if card:
                value_label = card.findChild(QLabel, "statCardValue")
                progress_bar = card.findChild(QProgressBar, "statCardProgress")
                if value_label:
                    value_label.setText("0")
                if progress_bar:
                    progress_bar.setValue(0)
        # Reset total and time
        if hasattr(self, 'stat_total_items') and self.stat_total_items:
            val_label = self.stat_total_items.findChild(QLabel, "infoCardValue")
            if val_label:
                val_label.setText("0")
        if hasattr(self, 'stat_proc_time') and self.stat_proc_time:
            val_label = self.stat_proc_time.findChild(QLabel, "infoCardValue")
            if val_label:
                val_label.setText("0ms")

    def clear_current_detection_display(self):
        """Clears only the current detection display, not history."""
        self.clear_detection_statistics_display()
        self.original_pixmap = None
        if hasattr(self, 'image_label'):
            self.image_label.clear()
            self.image_label.setText("Upload image or start webcam...")
            self.display_scaled_image()  # Update display to show placeholder
        self.latest_detection_details = []  # Clear details for export
        print("Current detection display cleared.")

    def clear_all_history(self):
        """Clears the in-memory history and updates relevant views."""
        reply = QMessageBox.question(self, 'Clear History',
                                     "Clear all detection history for this session?\nThis cannot be undone.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.detection_history_memory = []
            self.latest_detection_details = []  # Clear current view details too
            print("In-memory history cleared.")
            # Update views if they are currently active
            if self.stacked_layout.currentIndex() == 1:  # Analytics
                self.update_analytics_view()
            if self.stacked_layout.currentIndex() == 2:  # History
                self.update_history_view(page=1)
            # Also clear current detection display
            self.clear_current_detection_display()

    def update_detection_statistics_from_list(self, detections_list, inference_time_ms):
        """Updates stat cards based on a list of detection dictionaries."""
        if not self.model or not hasattr(self.model, 'names') or not self.stat_cards:
            self.clear_detection_statistics_display()
            return

        if not detections_list:
            self.clear_detection_statistics_display()  # Clear counts
            if hasattr(self, 'stat_proc_time') and self.stat_proc_time:
                time_label = self.stat_proc_time.findChild(QLabel, "infoCardValue")
                if time_label:
                    time_label.setText(f"{inference_time_ms:.1f}ms")
            return

        counts = {cls_name: 0 for cls_name in self.plastic_classes}
        total_detections = len(detections_list)

        for det in detections_list:
            model_class_name_raw = det['class'].lower()
            # Match against the defined plastic classes
            for target_class_name in self.plastic_classes:
                if target_class_name.lower() in model_class_name_raw:
                    counts[target_class_name] += 1
                    break  # Count only the first match

        # Update class cards
        for class_name, count in counts.items():
            card = self.stat_cards.get(class_name)
            if card:
                value_label = card.findChild(QLabel, "statCardValue")
                progress_bar = card.findChild(QProgressBar, "statCardProgress")
                if value_label:
                    value_label.setText(str(count))
                progress_val = int((count / max(1, total_detections)) * 100) if total_detections > 0 else 0
                if progress_bar:
                    progress_bar.setValue(progress_val)

        # Update total and time cards
        if hasattr(self, 'stat_total_items_value_label'):
            self.stat_total_items_value_label.setText(str(total_detections))
        if hasattr(self, 'stat_proc_time_value_label'):
            self.stat_proc_time_value_label.setText(f"{inference_time_ms:.1f}ms")

    # --- Display Scaling ---
    def display_scaled_image(self):
        if not hasattr(self, 'image_label') or not self.image_label:
            return

        if self.original_pixmap is None or self.original_pixmap.isNull():
            self.image_label.clear()
            self.image_label.setText("Upload image or start webcam...")
            # Reset minimum size for placeholder
            self.image_label.setMinimumSize(400, 300)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return

        # Allow shrinking below original size
        self.image_label.setMinimumSize(1, 1)

        # Get container size (viewport of scroll area)
        container_size = self.image_scroll_area.viewport().size() if hasattr(
            self, 'image_scroll_area') else self.image_label.size()

        if not container_size.isValid() or container_size.width() < 20 or container_size.height() < 20:
            # Fallback size if container is invalid or too small
            scaled_pixmap = self.original_pixmap.scaled(max(200, self.original_pixmap.width()), max(150, self.original_pixmap.height()),
                                                        Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            scaled_pixmap = self.original_pixmap.scaled(
                container_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)
        # Keep label centered within its area
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # --- Image Navigation ---
    def prev_image(self):
        if not self.model:
            return
        if self.image_paths and self.current_image_index > 0:
            self.current_image_index -= 1
            self.current_image_path = self.image_paths[self.current_image_index]  
            self.run_model_on_image_path(self.current_image_path)
            self.update_navigation_buttons()
            self.update_image_count_label()

    def next_image(self):
        if not self.model:
            return
        if self.image_paths and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.current_image_path = self.image_paths[self.current_image_index]  
            self.run_model_on_image_path(self.current_image_path)
            self.update_navigation_buttons()
            self.update_image_count_label()

    # --- Webcam Handling ---
    def toggle_webcam(self):
        checkpoint_path = resource_path("Yolov7_StrongSORT_OSNet/strong_sort/deep/checkpoint/osnet_x0_25_market1501.pt")

        self.strongsort = StrongSORT(
            model_weights=checkpoint_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            fp16=False
        )
        if not self.model:
            self.image_label.setText("Model not loaded.")
            return

        if self.webcam_running:
            self.webcam_running = False
            if hasattr(self, 'webcam_timer') and self.webcam_timer:
                self.webcam_timer.stop()
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None # Ensure it's cleared

            play_icon = get_icon("webcam_play.svg",
                                 QStyle.StandardPixmap.SP_MediaPlay)
            self.webcam_btn.setText(" Start Webcam Tracking")
            self.webcam_btn.setIcon(play_icon)
            self.webcam_btn.setObjectName("startWebcamButton")
            self.webcam_status_indicator.setObjectName("webcamStatusOffline")
            # Reapply stylesheet to update button/indicator styles
            self.webcam_btn.setStyleSheet(self.styleSheet())
            self.webcam_status_indicator.setStyleSheet(self.styleSheet())

            self.image_label.setText("Webcam stopped.")
            self.original_pixmap = None
            self.display_scaled_image()
            self.latest_detection_details = []  # Clear export details
            self.update_navigation_buttons()
            self.webcam_dropdown.setEnabled(True)
            self.drop_frame.setEnabled(True)
            self.update_image_count_label()
            # Optionally clear stats display here if desired
            # self.clear_current_detection_display()

        else:  # Start webcam
            webcam_idx = self.webcam_dropdown.currentData()
            if webcam_idx is None or webcam_idx == -1:
                self.image_label.setText("No webcam selected.")
                return

            # Clear previous state
            self.image_paths = []
            self.current_image_index = -1
            self.update_image_count_label()
            self.update_navigation_buttons()
            self.clear_current_detection_display() # Clear stats/image

            # --- MODIFIED: Reset webcam tracking state on start ---
            self.tracked_object_identities = {}

            # Try opening webcam
            self.cap = cv2.VideoCapture(webcam_idx, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                print(
                    f"Warning: DSHOW failed for webcam {webcam_idx}, trying default.")
                self.cap = cv2.VideoCapture(webcam_idx)
                if not self.cap or not self.cap.isOpened():
                    self.image_label.setText(
                        f"Error opening webcam {webcam_idx}")
                    self.cap = None
                    return

            self.webcam_running = True
            self.webcam_timer = QTimer(self)
            self.webcam_timer.timeout.connect(self.update_webcam_frame)
            self.webcam_timer.start(30)  # Target ~30 FPS for smoother tracking

            stop_icon = get_icon("webcam_stop.svg",
                                 QStyle.StandardPixmap.SP_MediaStop)
            self.webcam_btn.setText(" Stop Webcam Tracking")
            self.webcam_btn.setIcon(stop_icon)
            self.webcam_btn.setObjectName("stopWebcamButton")
            self.webcam_status_indicator.setObjectName("webcamStatusOnline")
            # Reapply stylesheet
            self.webcam_btn.setStyleSheet(self.styleSheet())
            self.webcam_status_indicator.setStyleSheet(self.styleSheet())

            self.latest_detection_details = []
            self.webcam_dropdown.setEnabled(False)
            self.drop_frame.setEnabled(False)

    def update_webcam_frame(self):
        """Processes a webcam frame using StrongSORT tracking and handles cases with no detections smoothly."""
        if not self.webcam_running or not hasattr(self, 'cap') or not self.cap or not self.cap.isOpened():
            if self.webcam_running:
                print("Webcam stream lost or unavailable.")
                self.toggle_webcam()
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Failed to grab frame from webcam.")
            return

        try:
            confidence = self.confidence_threshold
            iou = self.iou_threshold

            start_time = time.time()
            results = self.model.predict(frame, conf=confidence, iou=iou)[0]
            end_time = time.time()
            proc_time_ms = (end_time - start_time) * 1000

            tracks = np.empty((0, 7))
            current_detections_for_display = []
            newly_detected_objects_for_history = []

            if results.boxes and results.boxes.xyxy.numel() > 0:
                boxes_xyxy = results.boxes.xyxy.detach().cpu().numpy()
                confs = results.boxes.conf.detach().cpu().numpy()
                class_ids = results.boxes.cls.detach().cpu().numpy().astype(int)

                bbox_xywh = []
                for x1, y1, x2, y2 in boxes_xyxy:
                    xc = (x1 + x2) / 2
                    yc = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    bbox_xywh.append([xc, yc, w, h])
                bbox_xywh = np.array(bbox_xywh)

                tracks = self.strongsort.update(bbox_xywh, confs, class_ids, frame)
            else:
                self.strongsort.increment_ages()

            # PRUNE old track identities if needed
            current_ids = set(track[4] for track in tracks) if len(tracks) else set()
            self.tracked_object_identities = {
                tid: name for tid, name in self.tracked_object_identities.items() if tid in current_ids
            }

            for track in tracks:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                class_id = int(track[5])
                conf = float(track[6])

                current_class_name = self.model.names.get(class_id, f"Class_{class_id}")
                display_class_name = ""

                if track_id in self.tracked_object_identities:
                    display_class_name = self.tracked_object_identities[track_id]
                else:
                    display_class_name = current_class_name
                    self.tracked_object_identities[track_id] = display_class_name
                    print(f"New object tracked: ID {track_id} as {display_class_name}")
                    newly_detected_objects_for_history.append({
                        "class": display_class_name, "conf": round(conf, 4), "box": [x1, y1, x2, y2]
                    })

                if hasattr(self, 'locked_ids') and track_id in self.locked_ids:
                    print(f">>> LOCKED OBJECT {track_id} is on screen at {[x1, y1, x2, y2]}")

                current_detections_for_display.append({
                    "class": display_class_name,
                    "conf": round(conf, 4),
                    "box": [x1, y1, x2, y2],
                    "track_id": track_id
                })

            if newly_detected_objects_for_history:
                history_record = {
                    "id": len(self.detection_history_memory) + 1,
                    "timestamp": datetime.now(),
                    "image_path": None,
                    "source_type": 'webcam_tracked',
                    "processing_time_ms": proc_time_ms,
                    "confidence_threshold": confidence,
                    "iou_threshold": iou,
                    "detected_objects": newly_detected_objects_for_history
                }
                self.detection_history_memory.append(history_record)

            # Always draw — even if no tracks
            if len(current_detections_for_display) == 0:
                annotated_frame = frame.copy()
                text = "No plastics detected"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_scale = 1.0
                text_thickness = 2
                text_size = cv2.getTextSize(text, font, text_scale, text_thickness)[0]
                text_x = int((annotated_frame.shape[1] - text_size[0]) / 2)
                text_y = 50

                cv2.putText(
                    annotated_frame,
                    text,
                    (text_x, text_y),
                    font,
                    text_scale,
                    (0, 0, 255),   # Red text
                    text_thickness,
                    cv2.LINE_AA
                )
            else:
                annotated_frame = self.draw_custom_boxes_from_list(
                    frame.copy(), current_detections_for_display, "webcam_tracked_frame")

            # Convert for Qt display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = annotated_frame_rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(annotated_frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qt_img)
            self.display_scaled_image()

            # Update stats
            self.update_detection_statistics_from_list(current_detections_for_display, proc_time_ms)

        except Exception as e:
            print(f"Error processing webcam frame: {e}")



    # --- Export ---
    def export_statistics(self):
        """Exports the statistics shown in the CURRENT detection view."""
        if not self.model:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("Export Data")
            msg_box.setText("Model not loaded.")
            msg_box.setStyleSheet("QMessageBox QLabel { color: white; }")  # HERE
            msg_box.exec()
            return

        # Check if there's anything in the current view's stats
        current_total_items_text = "0"
        if hasattr(self, 'stat_total_items') and self.stat_total_items:
            label = self.stat_total_items.findChild(QLabel, "infoCardValue")
            if label:
                current_total_items_text = label.text()

        # Use self.latest_detection_details for detailed rows
        if not self.latest_detection_details and current_total_items_text == "0":
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("Export Data")
            msg_box.setText("No detection data in the current view to export.")
            msg_box.setStyleSheet("QMessageBox QLabel { color: white; }")  # HERE
            msg_box.exec()
            return

        default_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        source_info = "last_view"  # Default name
        if self.webcam_running:
            source_info = "webcam_capture"
        elif self.image_paths and self.current_image_index != -1:
            try:
                source_info = os.path.splitext(os.path.basename(
                    self.image_paths[self.current_image_index]))[0]
            except Exception:
                pass  # Ignore errors getting basename

        default_filename = os.path.join(
            default_dir, f"detection_export_{source_info}_{timestamp}.csv")

        filePath, _ = QFileDialog.getSaveFileName(
            self, "Save Export Data", default_filename, "CSV Files (*.csv)")
        if not filePath:
            return

        try:
            with open(filePath, 'w', newline='', encoding='utf-8') as csvfile:
                # Write detailed detections for the last processed frame/image
                if self.latest_detection_details:
                    fieldnames = ['image_source', 'object_id', 'track_id',
                                  'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']
                    # Filter fieldnames to only include columns present in the data
                    actual_fieldnames = [f for f in fieldnames if f in self.latest_detection_details[0]]
                    writer = csv.DictWriter(csvfile, fieldnames=actual_fieldnames)
                    writer.writeheader()
                    writer.writerows(self.latest_detection_details)
                    csvfile.write("\n")
                else:
                    csvfile.write(
                        "No detailed object detections for the last view.\n\n")

                # Write summary statistics from the current display
                csvfile.write("Summary Statistics (Current View):\n")
                for class_name in self.plastic_classes:
                    card = self.stat_cards.get(class_name)
                    count = "0"
                    if card:
                        count_label = card.findChild(QLabel, "statCardValue")
                        if count_label:
                            count = count_label.text()
                    csvfile.write(f"{class_name}: {count}\n")

                total_items_text = "0"
                if hasattr(self, 'stat_total_items') and self.stat_total_items:
                    total_label = self.stat_total_items.findChild(
                        QLabel, "infoCardValue")
                    if total_label:
                        total_items_text = total_label.text()
                csvfile.write(f"Total Items Detected: {total_items_text}\n")

                proc_time_text = "0ms"
                if hasattr(self, 'stat_proc_time') and self.stat_proc_time:
                    proc_label = self.stat_proc_time.findChild(
                        QLabel, "infoCardValue")
                    if proc_label:
                        proc_time_text = proc_label.text()
                csvfile.write(f"Processing Time: {proc_time_text}\n")

                csvfile.write("\nDetection Parameters (Current View):\n")
                csvfile.write(f"Confidence Threshold: {int(self.confidence_threshold * 100)}%\n")
                csvfile.write(f"IoU Threshold: {int(self.iou_threshold * 100)}%\n")

            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("Export Successful")
            msg_box.setText(f"Data exported to:\n{filePath}")
            msg_box.setStyleSheet("QMessageBox QLabel { color: white; }")  # HERE
            msg_box.exec()
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Could not write file:\n{str(e)}")
            print(f"Error exporting data: {e}")

    # --- Resize Event ---
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Use timer to delay image rescaling until resize finished
        if hasattr(self, '_resize_timer'):
            self._resize_timer.stop()
        else:
            self._resize_timer = QTimer()
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._handle_resize_end)
        self._resize_timer.start(100) # 100ms delay

    def _handle_resize_end(self):
        # Rescale image after resize pause
        if self.original_pixmap and hasattr(self, 'image_label') and self.image_label.isVisible():
            self.display_scaled_image()

    # --- Close Event ---
    def closeEvent(self, event):
        if self.webcam_running:
            self.toggle_webcam() # Stop webcam if running

        if hasattr(self, 'model_thread') and self.model_thread.isRunning():
            print("Waiting for model loading thread...")
            self.model_thread.quit()
            self.model_thread.wait(3000) # Wait 3 seconds
            if self.model_thread.isRunning():
                print("Model thread did not quit gracefully, terminating.")
                self.model_thread.terminate()
                self.model_thread.wait()

        print("Application closing.")
        # Stop timers
        if hasattr(self, 'history_search_timer'): self.history_search_timer.stop()
        if hasattr(self, '_resize_timer'): self._resize_timer.stop()

        event.accept()

    # --- Methods for Analytics (No Charts) ---
    def update_analytics_view(self):
        """Queries in-memory history and updates analytics cards and the chart."""
        if not self.model:
            # Clear chart if model is not available
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Model not loaded.\nAnalytics unavailable.",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='gray', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        filtered_history = self.detection_history_memory

        # Calculate Aggregates
        total_items_overall = 0
        total_proc_time = 0
        all_confidences = []
        class_counts = Counter()
        num_records = len(filtered_history) # Number of processed frames/images

        if not filtered_history:
            # Reset cards if no data
            self.analytics_total_items_value.setText("0")
            self.analytics_avg_proc_time_value.setText("0ms")
            self.analytics_avg_conf_value.setText("0%")
            self.analytics_num_records_value.setText("0")
            self.analytics_most_frequent_value.setText("N/A")

            # Clear chart and display 'No data' message
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available for the selected period.",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='gray', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        # Use the current confidence threshold
        current_threshold = self.confidence_threshold

        for record in filtered_history:
            filtered_detections = [
                det for det in record['detected_objects']
                if det.get('conf', 0) >= current_threshold
            ]

            total_items_overall += len(filtered_detections)
            total_proc_time += record.get('processing_time_ms', 0)

            for det in filtered_detections:
                conf = det.get('conf', 0)
                all_confidences.append(conf)

                model_class_name_raw = det.get('class', '').lower()
                matched_class = None

                # Try to match with model class
                for class_id, class_name in self.model.names.items():
                    if class_name.lower() == model_class_name_raw:
                        matched_class = class_name
                        break

                # Fallback: match against known plastic classes
                if not matched_class:
                    for target_class_name in self.plastic_classes:
                        if target_class_name.lower() in model_class_name_raw:
                            matched_class = target_class_name
                            break

                class_counts[matched_class if matched_class else det.get('class', 'Unknown')] += 1


        # Calculate Averages and Most Frequent
        avg_proc_time = (total_proc_time / num_records) if num_records else 0
        avg_conf = (sum(all_confidences) / len(all_confidences)
                            ) * 100 if all_confidences else 0
        most_frequent = class_counts.most_common(1)
        most_frequent_class_str = f"{most_frequent[0][0]} ({most_frequent[0][1]})" if most_frequent else "N/A"

        # Update Summary Cards (using the stored QLabel references)
        self.analytics_total_items_value.setText(str(total_items_overall))
        self.analytics_avg_proc_time_value.setText(f"{avg_proc_time:.1f}ms")
        self.analytics_avg_conf_value.setText(f"{avg_conf:.1f}%")
        self.analytics_num_records_value.setText(str(num_records))
        self.analytics_most_frequent_value.setText(most_frequent_class_str)

        # --- Update Chart ---
        self.figure.clear() # Clear the previous plot
        ax = self.figure.add_subplot(111)

        # Prepare data for the chart
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())

        if labels and sum(sizes) > 0: # Only draw chart if there's data to plot
            label_to_color = {
                "PET": "#FFFF00",   # Yellow
                "HDPE": "#FFA500",     # Orange
                "LDPE": "#00FF00",     # Green
                "PVC": "#FF0000",     # Red
                "PP": "#808080",     # Gray
                "PS": "#800080",   # Purple
            }

            # Use default color if label not in dict
            colors_to_use = [label_to_color.get(label, "#000000") for label in labels]

            # Create the pie chart
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%', # Format percentage text
                startangle=90,
                colors=colors_to_use,
                pctdistance=0.85 # Distance of percentage labels from center
            )
            
            # Make sure percentage labels are visible (e.g., white color)
            for autotext in autotexts:
                autotext.set_color('white')
            for text in texts:
                text.set_color('black') # Color for the class labels

            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title("Distribution of Detected Classes", fontsize=14, color='black') # You can customize color

        else:
            ax.text(0.5, 0.5, "No class data to display.",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='gray', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        self.canvas.draw() # Redraw the canvas with the new plot

    # --- Methods for History/Gallery (Using Memory) ---
    def update_history_view(self, page=None):
        if page is None:
            page = self.current_history_page

        # Clear previous items
        while self.gallery_flow_layout.count():
            item = self.gallery_flow_layout.takeAt(0) # FlowLayout's takeAt
            widget = item.widget()
            if widget:
                # No need to call layout.removeWidget(widget) with FlowLayout's takeAt
                widget.deleteLater()

        # Get filter values
        search_term = self.history_search_input.text().strip().lower()
        filter_type_full = self.history_filter_combo.currentText()

        # Filter the in-memory list
        filtered_data = []
        for record in self.detection_history_memory:
            # Type Filter
            if filter_type_full.startswith("Filter by type:") and filter_type_full != "Filter by type: All":
                class_name_filter = filter_type_full.split(": ")[1].lower()
                found_class = any(class_name_filter in det.get('class', '').lower(
                ) for det in record['detected_objects'])
                if not found_class:
                    continue
            # Search Term Filter
            if search_term:
                path_match = record['image_path'] and search_term in record['image_path'].lower()
                class_match = any(search_term in det.get(
                    'class', '').lower() for det in record['detected_objects'])
                if not (path_match or class_match):
                    continue
            # If passes all filters
            filtered_data.append(record)

        # Sort (newest first)
        filtered_data.sort(key=lambda x: x['timestamp'], reverse=True)

        # Paginate
        total_items = len(filtered_data)
        self.total_history_pages = (
            total_items + HISTORY_ITEMS_PER_PAGE - 1) // HISTORY_ITEMS_PER_PAGE
        if self.total_history_pages == 0:
            self.total_history_pages = 1
        self.current_history_page = max(1, min(page, self.total_history_pages))
        start_index = (self.current_history_page - 1) * HISTORY_ITEMS_PER_PAGE
        end_index = start_index + HISTORY_ITEMS_PER_PAGE
        paginated_data = filtered_data[start_index:end_index]

        # Populate grid
        if not paginated_data:
            no_history_label = QLabel("No history items match filters." if self.detection_history_memory else "No history recorded yet.")
            no_history_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_history_label.setStyleSheet("color: #A0A7B9; font-size: 14px; margin-top: 20px;")
            self.gallery_flow_layout.addWidget(no_history_label) # Add directly
        else:
            for record in paginated_data:
                item_widget = self.create_gallery_item_widget(record)
                if item_widget:
                    self.gallery_flow_layout.addWidget(item_widget)

        # Update pagination controls
        self.history_page_label.setText(
            f"Page {self.current_history_page} / {self.total_history_pages}")
        self.history_prev_btn.setEnabled(self.current_history_page > 1)
        self.history_next_btn.setEnabled(
            self.current_history_page < self.total_history_pages)

        # Scroll to top after update
        self.history_scroll_area.verticalScrollBar().setValue(0)

    def create_gallery_item_widget(self, history_record):
        """Creates a widget for a single history item."""
        try:
            record_id = history_record['id']
            timestamp_dt = history_record['timestamp']
            timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
            image_path = history_record['image_path']
            source_type = history_record['source_type']
            detections = history_record['detected_objects']

            item_frame = QFrame()
            item_frame.setObjectName("galleryItemFrame") 
            item_frame.setStyleSheet("background-color: #FFFFFF; border: 1px solid black; ")
            item_layout = QVBoxLayout(item_frame)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(5) # Space between thumb/labels/button

            # Thumbnail Label
            thumb_label = QLabel()
            thumb_label.setObjectName("galleryThumbLabel")
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_size = 160 # Keep fixed for grid consistency

            if image_path and os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        thumb_size, thumb_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    thumb_label.setPixmap(scaled_pixmap)
                else:
                    thumb_label.setText("Invalid Image")
                    thumb_label.setStyleSheet("color: #BF616A;")
            elif 'webcam' in source_type:
                webcam_icon = get_icon(
                    "webcam_play.svg", QStyle.StandardPixmap.SP_MediaPlay)
                thumb_label.setPixmap(webcam_icon.pixmap(QSize(64, 64)))
                thumb_label.setStyleSheet("background-color: #353A4C;")
            else:
                thumb_label.setText("No Image")
                thumb_label.setStyleSheet("color: #A0A7B9;")
            item_layout.addWidget(thumb_label)

            # Info Labels
            info_text = "N/A"
            if detections:
                primary_class = "Unknown"
                if detections[0].get('class'):
                    primary_class_raw = detections[0]['class']
                    primary_class = primary_class_raw
                    # Map to standard name
                    for target_class in self.plastic_classes:
                        if target_class.lower() in primary_class_raw.lower():
                            primary_class = target_class
                            break
                count = len(detections)
                info_text = f"{primary_class}" + \
                    (f" (+{count-1})" if count > 1 else "")
                # Truncate if too long
                max_len = 22
                if len(info_text) > max_len:
                    info_text = info_text[:max_len-3] + "..."
            info_label = QLabel(info_text)
            info_label.setObjectName("galleryInfoLabel")
            item_layout.addWidget(info_label)

            date_label = QLabel(timestamp_str)
            date_label.setObjectName("galleryDateLabel")
            item_layout.addWidget(date_label)

            # Details Button
            details_btn = QPushButton("View Details")
            details_btn.setStyleSheet("background-color: #A9A9A9;")
            details_btn.setObjectName("galleryDetailsButton")
            details_btn.clicked.connect(
                lambda checked=False, rec=history_record: self.show_history_details(rec))
            item_layout.addWidget(details_btn)

            # Calculate fixed height
            item_height = thumb_size + 65 # Adjust based on label/button heights
            item_frame.setFixedSize(thumb_size, item_height)

            return item_frame

        except Exception as e:
            print(f"Error creating gallery item ID {history_record.get('id', 'N/A')}: {e}")
            # Return an error placeholder widget
            error_frame = QFrame()
            error_frame.setObjectName("galleryItemFrame")
            error_layout = QVBoxLayout(error_frame)
            error_label = QLabel(f"Error\nID: {history_record.get('id', 'N/A')}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setStyleSheet(
                "color: #BF616A; background-color: #1A1D25; border-radius: 8px; padding: 10px;")
            error_layout.addWidget(error_label)
            thumb_size = 160; item_height = thumb_size + 65
            error_frame.setFixedSize(thumb_size, item_height)
            return error_frame

    def show_history_details(self, history_record):
        """Shows detailed view of a history item using a MessageBox."""
        record_id = history_record.get('id', 'N/A')
        details_text = f"<b>Record ID: {record_id}</b><br>" # Use HTML for basic formatting
        details_text += f"Timestamp: {history_record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
        details_text += f"Source: {history_record['source_type']}<br>"
        if history_record['image_path']:
            # Make path selectable but maybe not clickable
            details_text += f"Path: {history_record['image_path']}<br>"
        details_text += f"Proc Time: {history_record['processing_time_ms']:.1f}ms<br>"
        details_text += f"Confidence Thresh: {history_record['confidence_threshold']:.2f}<br>"
        details_text += f"IOU Thresh: {history_record['iou_threshold']:.2f}<br><br>"
        details_text += "<b>Detected Objects:</b><br>"

        if history_record['detected_objects']:
            details_text += "<ul style='margin-left: 0px; padding-left: 15px;'>" # Basic list
            for i, det in enumerate(history_record['detected_objects']):
                details_text += f"<li>{det['class']} (Conf: {det['conf']:.2f}) @ Box: {det['box']}</li>"
            details_text += "</ul>"
        else:
            details_text += "<i>None</i><br>"

        # Use a QMessageBox that allows rich text
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(f"History Details - Record {record_id}")
        msg_box.setStyleSheet("QMessageBox QLabel { color: white; }")
        msg_box.setTextFormat(Qt.TextFormat.RichText) # Enable HTML/Rich Text
        msg_box.setText(details_text)
        msg_box.setIcon(QMessageBox.Icon.Information)

        # Try to add image preview
        pixmap_preview = None
        if history_record['image_path'] and os.path.exists(history_record['image_path']):
            pixmap = QPixmap(history_record['image_path'])
            if not pixmap.isNull():
                # Scale for preview (consider drawing boxes here later if needed)
                scaled_pixmap = pixmap.scaled(
                    200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                pixmap_preview = scaled_pixmap

        if pixmap_preview:
            msg_box.setIconPixmap(pixmap_preview) # Set image as icon

        msg_box.exec()

    def history_prev_page(self):
        if self.current_history_page > 1:
            self.update_history_view(page=self.current_history_page - 1)

    def history_next_page(self):
        if self.current_history_page < self.total_history_pages:
            self.update_history_view(page=self.current_history_page + 1)


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.main_window = None

    # Track global components
    splash = SplashScreen()
    splash.show()
    video_screen = None
    video_finished = False

    def cleanup_video():
        """Stop and clean up video resources."""
        global video_screen
        if video_screen:
            video_screen.close()
            video_screen.deleteLater()
            video_screen = None
        print("Video resources cleaned up")

    def start_main_ui():
        """Launch the main WasteDetectionApp after video."""
        global splash, video_screen
        print("Starting main UI...")

        # Ensure video is closed
        cleanup_video()

        try:
            from rec import WasteDetectionApp
            app.main_window = WasteDetectionApp()
            app.main_window.show()

            if splash:
                splash.finish(app.main_window)
                splash = None

            print("Main UI loaded successfully")
        except Exception as e:
            import traceback
            print(f"Error loading main UI: {e}")
            traceback.print_exc()
            app.quit()

    def on_video_finished():
        """Handle when intro video ends."""
        global video_finished
        if video_finished:
            print("Video already finished, ignoring duplicate signal")
            return
        video_finished = True
        print("Video finished, moving to main UI...")
        start_main_ui()

    def start_video():
        """Show intro video after splash."""
        global video_screen
        splash.hide() 

        try:
            video_path = os.path.abspath(resource_path("icons/plastic.mp4"))
            if os.path.exists(video_path):
                print(f"Loading intro video: {video_path}")
                video_screen = IntroVideoScreen(video_path, on_video_finished)
                video_screen.show()
            else:
                print(f"Video not found: {video_path}, skipping...")
                start_main_ui()
        except Exception as e:
            print(f"Error starting video: {e}")
            start_main_ui()

    # Sequence: show splash → wait → start video
    print("Application started. Showing splash...")
    QTimer.singleShot(3000, start_video)

    sys.exit(app.exec())