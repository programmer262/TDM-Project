import sys
from PySide6.QtSvgWidgets import QSvgWidget
import os
from Login import init_database, LoginDialog
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QFileDialog, QLabel, QComboBox, QSizePolicy, QFrame,
                              QSlider, QProgressBar, QStyleFactory, QTabWidget, QGridLayout, QToolTip,
                              QSpacerItem, QGraphicsDropShadowEffect, QScrollArea, QDialog, QTextBrowser,QCheckBox,QMessageBox)
from PySide6.QtCore import (QPropertyAnimation, QEasingCurve, Qt, QTimer, 
                           QSize, QRect, QPoint, QSequentialAnimationGroup, QParallelAnimationGroup,
                           QThread, Signal)
from PySide6.QtGui import QColor, QIcon, QPixmap, QFont, QPalette, QLinearGradient, QGradient, QBrush
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import butter, filtfilt
from scipy.fftpack import dct, idct
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from huffman import HuffmanCoder
from LZW import LZWCoder
from psychoacousticmodel import PsychoacousticModel
import warnings

warnings.filterwarnings('ignore')

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QSplashScreen
from PySide6.QtGui import QPixmap, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import Qt, QTimer, QRectF

from PySide6.QtWidgets import QSplashScreen
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QRect

# Color Schemes
DARK_COLORS = {
    'primary_dark': '#121212',
    'secondary_dark': '#1E1E2E',
    'panel_bg': '#1A2333',
    'accent': '#00CED1',
    'accent_light': '#A4B8FF',
    'accent_dark': '#5A6FB5',
    'text_primary': '#E0E0E0',
    'text_secondary': '#B0C4DE',
    'success': '#4CAF50',
    'warning': '#FFC107',
    'error': '#FF5252',
    'highlight': '#BB86FC',
}

LIGHT_COLORS = {
    'primary_dark': '#F5F5F5',
    'secondary_dark': '#E0E0E0',
    'panel_bg': '#FFFFFF',
    'accent': '#00CED1',
    'accent_light': '#4A90E2',
    'accent_dark': '#2A6DB5',
    'text_primary': '#242424',
    'text_secondary': '#686868',
    'success': '#4CAF50',
    'warning': '#FFC107',
    'error': '#FF5252',
    'highlight': '#BB86FC',
}

# Global mode flag
current_mode = 'dark'
COLORS = DARK_COLORS

class SvgSplashScreen(QSplashScreen):
    def __init__(self, svg_path, text="Loading...", padding=20, bg_color=COLORS['secondary_dark']):
        temp_svg = QSvgWidget(svg_path)
        svg_size = temp_svg.renderer().defaultSize()
        w, h = svg_size.width(), svg_size.height()
        text_height = 30
        total_w = w + 2 * padding
        total_h = h + text_height + 2 * padding
        pixmap = QPixmap(total_w, total_h)
        pixmap.fill(QColor(bg_color))
        painter = QPainter(pixmap)
        svg = QSvgWidget(svg_path)
        target_rect = QRect(padding, padding, w, h)
        svg.renderer().render(painter, target_rect)
        painter.setPen(QColor(COLORS['text_primary']))
        painter.setFont(QFont("Roboto", 12))
        text_rect = QRect(0, padding + h, total_w, text_height)
        painter.drawText(text_rect, Qt.AlignCenter, text)
        painter.end()
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

# Global variables
audio = None
file_path = ""
original_file = "original.wav"
reconstructed_file = "reconstructed.wav"
quantization_factor = 0.02
frame_size = 1024
active_threads = []
compressed_file = f""

# Worker Threads
class AudioLoadWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(bool)
    error = Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        global audio
        try:
            self.progress.emit(10, "Chargement du fichier audio...")
            audio = AudioSegment.from_file(self.file_path)
            self.finished.emit(True)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False)

class CompressionWorker(QThread):
    progress = Signal(int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, compression_type):
        super().__init__()
        self.compression_type = compression_type

    def run(self):
        global audio, file_path
        try:
            audio.export(original_file, format="wav")
            fs = audio.frame_rate
            signal = audio_to_signal(audio)
            signal = centrer_signal(signal)
            signal = normaliser_signal(signal)
            self.progress.emit(10, "Transformation de l'audio...")
            coeffs = mdct(signal, frame_size=frame_size)
            q = quantization_factor
            if self.compression_type in ["Huffman + LZW + Masquage", "Huffman + Masquage", "LZW + Masquage"]:
                psycho_model = PsychoacousticModel(fs)
                weights = psycho_model.perceptual_bit_allocation(coeffs, frame_size)
                qcoeffs, q_factors = quantifier_perceptual(coeffs, weights, q)
            else:
                qcoeffs = quantifier(coeffs, q=q)
            self.progress.emit(30, "Compression des données...")
            compressed_data = self.compress_data(qcoeffs)
            with open(compressed_file, 'wb') as f:
                write_irm_header(f, fs, frame_size, qcoeffs.shape, self.compression_type)
                f.write(compressed_data)
            self.progress.emit(50, "Décompression des données...")
            with open(compressed_file, 'rb') as f:
                header_info = read_irm_header(f)
                fs_loaded = header_info['fs']
                frame_size_loaded = header_info['frame_size']
                rows, cols = header_info['shape']
                compressed_data = f.read()
            self.progress.emit(80, "Reconstruction de l'audio...")
            coeffs_recon = self.decompress_data(compressed_data, header_info, rows, cols, q)
            signal_recon = imdct(coeffs_recon, frame_size=frame_size_loaded)
            signal_recon = normaliser_signal(signal_recon)
            audio_recon = signal_to_audio(signal_recon, fs_loaded)
            audio_recon.export(reconstructed_file, format="wav")
            self.progress.emit(100, "Compression terminée !")
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def compress_data(self, qcoeffs):
        if self.compression_type == "LZW only":
            lzw = LZWCoder()
            return lzw.compress(qcoeffs.flatten().tobytes())
        elif self.compression_type == "Huffman only":
            huffman = HuffmanCoder(precision=0)
            return huffman.compress(qcoeffs.flatten().tolist())
        elif self.compression_type == "Huffman + LZW":
            huffman = HuffmanCoder(precision=0)
            huffman_compressed = huffman.compress(qcoeffs.flatten().tolist())
            lzw = LZWCoder()
            return lzw.compress(huffman_compressed)
        elif self.compression_type == "Huffman + LZW + Masquage":
            huffman = HuffmanCoder(precision=0)
            huffman_compressed = huffman.compress(qcoeffs.flatten().tolist())
            lzw = LZWCoder()
            return lzw.compress(huffman_compressed)
        elif self.compression_type == "Huffman + Masquage":
            huffman = HuffmanCoder(precision=0)
            return huffman.compress(qcoeffs.flatten().tolist())
        elif self.compression_type == "LZW + Masquage":
            lzw = LZWCoder()
            return lzw.compress(qcoeffs.flatten().tobytes())

    def decompress_data(self, compressed_data, header_info, rows, cols, q):
        compression_type = header_info['compression_type_str']
        if compression_type == "LZW only":
            lzw = LZWCoder()
            decompressed_data = lzw.decompress(compressed_data)
            qcoeffs_loaded = np.frombuffer(decompressed_data, dtype=np.int16).reshape(rows, cols)
            return dequantifier(qcoeffs_loaded, q=q)
        elif compression_type == "Huffman only":
            huffman = HuffmanCoder(precision=0)
            flat_qcoeffs = huffman.decompress(compressed_data)
            qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
            return dequantifier(qcoeffs_loaded, q=q)
        elif compression_type == "Huffman + LZW":
            lzw = LZWCoder()
            huffman_data = lzw.decompress(compressed_data)
            huffman = HuffmanCoder(precision=0)
            flat_qcoeffs = huffman.decompress(huffman_data)
            qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
            return dequantifier(qcoeffs_loaded, q=q)
        else:  # Masquage cases
            if compression_type == "Huffman + LZW + Masquage":
                lzw = LZWCoder()
                huffman_data = lzw.decompress(compressed_data)
                huffman = HuffmanCoder(precision=0)
                flat_qcoeffs = huffman.decompress(huffman_data)
            elif compression_type == "Huffman + Masquage":
                huffman = HuffmanCoder(precision=0)
                flat_qcoeffs = huffman.decompress(compressed_data)
            else:
                lzw = LZWCoder()
                decompressed_data = lzw.decompress(compressed_data)
                flat_qcoeffs = np.frombuffer(decompressed_data, dtype=np.int16)
            qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
            return dequantifier(qcoeffs_loaded, q=q)

class PlaybackWorker(QThread):
    finished = Signal()
    error = Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            audio_segment = AudioSegment.from_file(self.file_path)
            play(audio_segment)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

# Help Dialog Class
class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Aide - Compresseur Audio")
        self.setFixedSize(500, 400)
        self.setStyleSheet(f"""
            background: {COLORS['panel_bg']};
            color: {COLORS['text_primary']};
            border-radius: 8px;
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        self.setLayout(layout)
        help_text = QTextBrowser()
        help_text.setStyleSheet(f"""
            background: {COLORS['secondary_dark']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['accent_dark']};
            border-radius: 4px;
            padding: 10px;
            font-size: 13px;
        """)
        help_text.setOpenExternalLinks(True)
        help_text.setHtml("""
           <h2 style='color: #7289DA;'>Bienvenue dans IRM</h2>
            <p><b>Objectif :</b> Cette application GUI permet la compression et la décompression de fichiers audio à l'aide d'algorithmes avancés pour réduire la taille des fichiers tout en maintenant leur qualité.</p>
            <p><b>Fonctionnalités :</b></p>
            <ul>
                <li>Charger des fichiers audio (WAV, MP3, FLAC, OGG) via glisser-déposer ou sélection de fichier.</li>
                <li>Appliquer la compression en utilisant diverses méthodes.</li>
                <li>Visualiser les formes d'onde originales et compressées.</li>
                <li>Lire les fichiers audio originaux et compressés pour comparaison.</li>
                <li>Afficher les statistiques de compression (réduction de taille, pourcentage).</li>
            </ul>
            <p><b>Algorithmes Utilisés :</b></p>
            <ul>
                <li><b>Codage de Huffman :</b> Une méthode de compression sans perte qui attribue des codes de longueur variable aux données en fonction de leur fréquence, réduisant ainsi les redondances.</li>
                <li><b>Compression LZW :</b> Un algorithme basé sur un dictionnaire qui construit des modèles pour remplacer les données répétées par des codes plus courts, efficace pour les séquences répétitives.</li>
            </ul>
            <p>Sélectionnez une méthode de compression dans le menu déroulant pour combiner ces algorithmes avec une modélisation psychoacoustique pour des résultats optimaux.</p>
        """)
        layout.addWidget(help_text)
        close_button = QPushButton("Fermer")
        close_button.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent']};
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {COLORS['accent_light']};
            }}
        """)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

# Drop Area Class
class DropArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Déposez le fichier audio ici")
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setStyleSheet(f"""
            border: 2px dashed {COLORS['accent']};
            border-radius: 8px;
            padding: 20px;
            background: {COLORS['panel_bg']};
            color: {COLORS['text_secondary']};
            font-size: 15px;
        """)
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                border: 2px dashed {COLORS['highlight']};
                border-radius: 8px;
                padding: 20px;
                background: {COLORS['secondary_dark']};
                color: {COLORS['text_primary']};
                font-size: 15px;
            """)
    def dragLeaveEvent(self, event):
        self.setStyleSheet(f"""
            border: 2px dashed {COLORS['accent']};
            border-radius: 8px;
            padding: 20px;
            background: {COLORS['panel_bg']};
            color: {COLORS['text_secondary']};
            font-size: 15px;
        """)
    def dropEvent(self, event):
        global file_path
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                load_audio_file(file_path)
            self.setStyleSheet(f"""
                border: 2px dashed {COLORS['accent']};
                border-radius: 8px;
                padding: 20px;
                background: {COLORS['panel_bg']};
                color: {COLORS['text_secondary']};
                font-size: 15px;
            """)

# Styled Components
class StyledButton(QPushButton):
    def __init__(self, text, icon_path=None, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Roboto", 11))
        self.setCursor(Qt.PointingHandCursor)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        if icon_path:
            self.setIcon(QIcon(QPixmap(icon_path)))
            self.setIconSize(QSize(18, 18))
        self.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['primary_dark']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 6px;
                padding: 10px 18px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 {COLORS['accent']}, stop:1 {COLORS['accent_dark']});
                color: {COLORS['text_primary']};
            }}
            QPushButton:pressed {{
                background: {COLORS['accent_dark']};
                padding-top: 11px;
                padding-left: 19px;
                padding-bottom: 9px;
                padding-right: 17px;
            }}
            QPushButton:disabled {{
                background: {COLORS['secondary_dark']};
                color: {COLORS['text_secondary']};
            }}
        """)

class InfoCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(layout)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-size: 15px;
            font-weight: bold;
            color: {COLORS['accent']};
            padding-bottom: 6px;
            border-bottom: 1px solid {COLORS['accent_dark']};
        """)
        layout.addWidget(title_label)
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(5)
        layout.addLayout(self.content_layout)
        self.setStyleSheet(f"""
            background: {COLORS['panel_bg']};
            border-radius: 8px;
        """)
    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
    def add_layout(self, layout):
        self.content_layout.addLayout(layout)

class EnhancedProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(True)
        self.setFormat("%p% - Ready")
        self.setMinimumHeight(22)
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {COLORS['accent']};
                border-radius: 4px;
                text-align: center;
                background: {COLORS['panel_bg']};
                color: {COLORS['text_secondary']};
                font-size: 12px;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                stop:0 {COLORS['accent_dark']}, stop:1 {COLORS['accent']});
                border-radius: 3px;
            }}
        """)
    def update_progress(self, value, message=""):
        self.setValue(value)
        if message:
            self.setFormat(f"{value}% - {message}")
        else:
            self.setFormat(f"{value}%")

# Animation Functions
def pulse_effect(widget):
    pulse_anim = QPropertyAnimation(widget, b"geometry")
    pulse_anim.setDuration(200)
    orig_geo = widget.geometry()
    pulse_geo = QRect(orig_geo.x()-5, orig_geo.y()-5,orig_geo.width()+10, orig_geo.height()+10)
    pulse_anim.setStartValue(orig_geo)
    pulse_anim.setEndValue(pulse_geo)
    pulse_anim.setEasingCurve(QEasingCurve.OutQuad)
    pulse_anim_back = QPropertyAnimation(widget, b"geometry")
    pulse_anim_back.setDuration(200)
    pulse_anim_back.setStartValue(pulse_geo)
    pulse_anim_back.setEndValue(orig_geo)
    pulse_anim_back.setEasingCurve(QEasingCurve.InQuad)
    group = QSequentialAnimationGroup()
    group.addAnimation(pulse_anim)
    group.addAnimation(pulse_anim_back)
    group.start()

def highlight_widget(widget, duration=500):
    original_style = widget.styleSheet()
    highlight_style = f"""
        border: 2px solid {COLORS['highlight']};
        background-color: rgba(187, 134, 252, 0.15);
        border-radius: 6px;
    """
    widget.setProperty("originalStyle", original_style)
    merged_style = original_style + highlight_style
    animation = QPropertyAnimation(widget, b"styleSheet")
    animation.setDuration(duration)
    animation.setStartValue(original_style)
    animation.setEndValue(merged_style)
    animation.setEasingCurve(QEasingCurve.OutCubic)
    QTimer.singleShot(duration + 1000, lambda: widget.setStyleSheet(original_style))
    animation.start()

# Core Audio Functions
def centrer_signal(signal):
    return signal - np.mean(signal)

def normaliser_signal(signal):
    max_abs = np.max(np.abs(signal))
    return signal / max_abs if max_abs != 0 else signal

def audio_to_signal(audio):
    signal = np.array(audio.get_array_of_samples()).astype(np.float32)
    return signal / max(abs(signal))

def mdct(signal, frame_size=1024):
    N = frame_size
    overlap = N // 2
    nb_frames = (len(signal) - overlap) // overlap
    mdct_coeffs = []
    window = np.sin(np.pi / N * (np.arange(N) + 0.5))
    for i in range(nb_frames):
        frame = signal[i * overlap : i * overlap + N]
        if len(frame) < N:
            break
        windowed = frame * window
        coeffs = dct(windowed, type=2, norm='ortho')
        mdct_coeffs.append(coeffs)
    return np.array(mdct_coeffs)

def imdct(coeffs, frame_size=1024):
    N = frame_size
    overlap = N // 2
    window = np.sin(np.pi / N * (np.arange(N) + 0.5))
    signal = np.zeros(overlap * (len(coeffs) + 1))
    for i, c in enumerate(coeffs):
        frame = idct(c, type=2, norm='ortho') * window
        start = i * overlap
        signal[start:start+N] += frame
    return signal

def quantifier_perceptual(coeffs, weights, base_q=0.02):
    q_factors = base_q / weights
    q_factors = np.clip(q_factors, base_q, base_q * 20)
    qcoeffs = np.zeros_like(coeffs, dtype=np.int16)
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            qcoeffs[i, j] = int(round(coeffs[i, j] / q_factors[i, j]))
    return qcoeffs, q_factors

def dequantifier_perceptual(qcoeffs, q_factors):
    coeffs_recon = np.zeros_like(qcoeffs, dtype=np.float32)
    for i in range(qcoeffs.shape[0]):
        for j in range(qcoeffs.shape[1]):
            coeffs_recon[i, j] = qcoeffs[i, j] * q_factors[i, j]
    return coeffs_recon

def quantifier(coeffs, q=0.02):
    return np.round(coeffs / q).astype(np.int16)

def dequantifier(qcoeffs, q=0.02):
    return qcoeffs.astype(np.float32) * q

def signal_to_audio(signal, fs):
    signal = np.clip(signal, -1, 1)
    pcm = (signal * 32767).astype(np.int16)
    audio = AudioSegment(pcm.tobytes(), frame_rate=fs, sample_width=2, channels=1)
    return audio

def write_irm_header(file, fs, frame_size, coeffs_shape, compression_type_str, version="1.0"):
    compression_map = {
        "LZW only": 0,
        "Huffman only": 1,
        "Huffman + LZW": 2,
        "Huffman + LZW + Masquage": 3,
        "Huffman + Masquage": 4,
        "LZW + Masquage": 5
    }
    compression_type = compression_map.get(compression_type_str, 0)
    file.write(b'IRM\x00')
    major, minor = map(int, version.split('.'))
    file.write(bytes([major, minor]))
    file.write(bytes([compression_type]))
    file.write(fs.to_bytes(4, byteorder='little'))
    file.write(frame_size.to_bytes(2, byteorder='little'))
    rows, cols = coeffs_shape
    file.write(rows.to_bytes(4, byteorder='little'))
    file.write(cols.to_bytes(4, byteorder='little'))

def read_irm_header(file):
    signature = file.read(4)
    if signature != b'IRM\x00':
        raise ValueError("Ce fichier n'est pas au format IRM valide")
    version_bytes = file.read(2)
    version = f"{version_bytes[0]}.{version_bytes[1]}"
    compression_type = file.read(1)[0]
    fs = int.from_bytes(file.read(4), byteorder='little')
    frame_size = int.from_bytes(file.read(2), byteorder='little')
    rows = int.from_bytes(file.read(4), byteorder='little')
    cols = int.from_bytes(file.read(4), byteorder='little')
    compression_types = [
        "LZW only",
        "Huffman only",
        "Huffman + LZW",
        "Huffman + LZW + Masquage",
        "Huffman + Masquage",
        "LZW + Masquage"
    ]
    compression_type_str = compression_types[compression_type] if compression_type < len(compression_types) else "Unknown"
    return {
        'version': version,
        'compression_type': compression_type,
        'compression_type_str': compression_type_str,
        'fs': fs,
        'frame_size': frame_size,
        'shape': (rows, cols)
    }

def load_audio_file(file_path):
    global audio, active_threads, compressed_file
    file_label.setText(f"Chargement: {os.path.basename(file_path)}...")
    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)
    original_filename = os.path.splitext(os.path.basename(file_path))[0]
    compressed_file = os.path.join(downloads_dir, f"{original_filename}.irm")
    worker = AudioLoadWorker(file_path)
    active_threads.append(worker)
    worker.progress.connect(progress_bar.update_progress)
    worker.finished.connect(lambda success: on_audio_load_finished(success, file_path, worker))
    worker.error.connect(lambda error: file_label.setText(f"Error: {error}"))
    worker.start()
    progress_bar.show()

def on_audio_load_finished(success, file_path, worker):
    global active_threads
    if success:
        file_label.setText(f"Fichier chargé: {os.path.basename(file_path)}")
        duration_label.setText(f"{len(audio)/1000:.2f} seconds")
        channels_label.setText(f"{audio.channels}")
        sample_rate_label.setText(f"{audio.frame_rate} Hz")
        bit_depth_label.setText(f"{audio.sample_width * 8} bits")
        file_name = os.path.basename(file_path)
        if recent_files_combo.findText(file_name) == -1:
            recent_files_combo.addItem(file_name)
        compress_button.setEnabled(True)
        play_original_button.setEnabled(True)
        highlight_widget(file_info_card)
        plot_original_waveform()
    else:
        file_label.setText("Failed to load file")
    progress_bar.hide()
    if worker in active_threads:
        active_threads.remove(worker)
    worker.deleteLater()

def compress_audio():
    global audio, active_threads
    if not audio:
        file_label.setText("Aucun fichier audio chargé")
        return
    compression_type = compression_combo.currentText()
    progress_bar.show()
    progress_bar.update_progress(0, "Début de la compression...")
    compress_button.setEnabled(False)
    load_button.setEnabled(False)
    worker = CompressionWorker(compression_type)
    active_threads.append(worker)
    worker.progress.connect(progress_bar.update_progress)
    worker.finished.connect(lambda: on_compression_finished(compression_type, worker))
    worker.error.connect(lambda error: file_label.setText(f"Error: {error}"))
    worker.start()

def on_compression_finished(compression_type, worker):
    global active_threads
    update_stats()
    plot_comparison()
    compress_button.setEnabled(True)
    load_button.setEnabled(True)
    play_compressed_button.setEnabled(True)
    pulse_effect(stats_frame)
    file_label.setText(f"Compressé par {compression_type}\nOutput: {os.path.basename(compressed_file)}")
    QTimer.singleShot(2000, lambda: progress_bar.hide())
    if worker in active_threads:
        active_threads.remove(worker)
    worker.deleteLater()

def play_compressed_audio():
    global active_threads
    if os.path.exists(reconstructed_file):
        play_compressed_button.setEnabled(False)
        worker = PlaybackWorker(reconstructed_file)
        active_threads.append(worker)
        worker.finished.connect(lambda: on_playback_finished(play_compressed_button, worker))
        worker.error.connect(lambda error: file_label.setText(f"Error: {error}"))
        worker.start()
    else:
        file_label.setText("Aucun fichier compressé disponible")

def play_original_audio():
    global active_threads
    if audio:
        play_original_button.setEnabled(False)
        worker = PlaybackWorker(original_file)
        active_threads.append(worker)
        worker.finished.connect(lambda: on_playback_finished(play_original_button, worker))
        worker.error.connect(lambda error: file_label.setText(f"Error: {error}"))
        worker.start()
    else:
        file_label.setText("Aucun fichier audio chargé")

def on_playback_finished(button, worker):
    global active_threads
    button.setEnabled(True)
    highlight_widget(button)
    if worker in active_threads:
        active_threads.remove(worker)
    worker.deleteLater()

def cleanup_threads():
    global active_threads
    for thread in active_threads:
        thread.quit()
        thread.wait()
        thread.deleteLater()
    active_threads.clear()

def show_size_difference():
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        original_size = os.path.getsize(original_file) / 1024
        compressed_size = os.path.getsize(compressed_file) / 1024
        diff = original_size - compressed_size
        stats_labels["différence"].setText(f"{diff:.2f} KB")
        highlight_widget(stats_frame)
    else:
        file_label.setText("Fichiers requis non disponibles")

def show_compression_percentage():
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        original_size = os.path.getsize(original_file) / 1024
        compressed_size = os.path.getsize(compressed_file) / 1024
        if original_size > 0:
            percentage = (1 - compressed_size / original_size) * 100
            stats_labels["pourcentage"].setText(f"{percentage:.2f}%")
            highlight_widget(stats_frame)
        else:
            file_label.setText("Taille originale invalide")
    else:
        file_label.setText("Fichiers requis non disponibles")

def update_stats():
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        original_size = os.path.getsize(original_file) / 1024
        compressed_size = os.path.getsize(compressed_file) / 1024
        diff = original_size - compressed_size
        percentage = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        stats_labels["original"].setText(f"{original_size:.2f} KB")
        stats_labels["compresseé"].setText(f"{compressed_size:.2f} KB")
        stats_labels["différence"].setText(f"{diff:.2f} KB")
        stats_labels["pourcentage"].setText(f"{percentage:.2f}%")

def plot_original_waveform():
    if audio:
        signal = normaliser_signal(audio_to_signal(audio))
        for i in reversed(range(waveform_preview_layout.count())):
            widget = waveform_preview_layout.itemAt(i).widget()
            if widget:
                waveform_preview_layout.removeWidget(widget)
                widget.deleteLater()
        fig, ax = plt.subplots(figsize=(6, 2))
        plt.style.use('default' if current_mode == 'light' else 'dark_background')
        ax.plot(signal, color=COLORS['accent_light'], linewidth=0.8)
        ax.set_title("Aperçu de la forme d'onde audio", color=COLORS['text_primary'])
        ax.set_facecolor(COLORS['panel_bg'])
        fig.patch.set_facecolor(COLORS['panel_bg'])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5000)
        ax.set_ylim(-1, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['accent_dark'])
        ax.spines['left'].set_color(COLORS['accent_dark'])
        ax.tick_params(axis='x', colors=COLORS['text_secondary'])
        ax.tick_params(axis='y', colors=COLORS['text_secondary'])
        canvas = FigureCanvas(fig)
        waveform_preview_layout.addWidget(canvas)
        window._canvases.append(canvas)

def plot_comparison():
    global visualization_layout
    for i in reversed(range(visualization_layout.count())):
        widget = visualization_layout.itemAt(i).widget()
        if widget:
            visualization_layout.removeWidget(widget)
            widget.deleteLater()
    signal = normaliser_signal(audio_to_signal(audio))
    signal_recon = normaliser_signal(audio_to_signal(AudioSegment.from_file(reconstructed_file)))
    fig1, ax1 = plt.subplots(figsize=(6, 2))
    plt.style.use('default' if current_mode == 'light' else 'dark_background')
    ax1.plot(signal, label="Original", color=COLORS['accent_light'])
    ax1.set_title("Signal Original", color=COLORS['text_primary'])
    ax1.set_facecolor(COLORS['panel_bg'])
    fig1.patch.set_facecolor(COLORS['panel_bg'])
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    plt.style.use('default' if current_mode == 'light' else 'dark_background')
    ax2.plot(signal_recon, label="Reconstructed", color=COLORS['highlight'])
    ax2.set_title("Signal Reconstruit", color=COLORS['text_primary'])
    ax2.set_facecolor(COLORS['panel_bg'])
    fig2.patch.set_facecolor(COLORS['panel_bg'])
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    canvas1 = FigureCanvas(fig1)
    canvas2 = FigureCanvas(fig2)
    visualization_layout.addWidget(canvas1, 0, 0)
    visualization_layout.addWidget(canvas2, 0, 1)
    window._canvases.extend([canvas1, canvas2])

def update_ui_colors():
    global COLORS
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(COLORS['primary_dark']))
    dark_palette.setColor(QPalette.WindowText, QColor(COLORS['text_primary']))
    dark_palette.setColor(QPalette.Base, QColor(COLORS['secondary_dark']))
    dark_palette.setColor(QPalette.AlternateBase, QColor(COLORS['panel_bg']))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(COLORS['accent']))
    dark_palette.setColor(QPalette.ToolTipText, QColor(COLORS['text_primary']))
    dark_palette.setColor(QPalette.Text, QColor(COLORS['text_primary']))
    dark_palette.setColor(QPalette.Button, QColor(COLORS['panel_bg']))
    dark_palette.setColor(QPalette.ButtonText, QColor(COLORS['text_primary']))
    dark_palette.setColor(QPalette.Link, QColor(COLORS['accent']))
    dark_palette.setColor(QPalette.Highlight, QColor(COLORS['accent']))
    dark_palette.setColor(QPalette.HighlightedText, QColor(COLORS['text_primary']))
    app.setPalette(dark_palette)
    app.setStyleSheet(f"""
        QToolTip {{
            background-color: {COLORS['panel_bg']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
            padding: 5px;
        }}
    """)
    for widget in window.findChildren(QWidget):
        if hasattr(widget, 'setStyleSheet'):
            current_style = widget.styleSheet()
            new_style = current_style
            for key, value in COLORS.items():
                new_style = new_style.replace(DARK_COLORS.get(key, ''), value).replace(LIGHT_COLORS.get(key, ''), value)
            widget.setStyleSheet(new_style)
    if audio:
        plot_original_waveform()
        if os.path.exists(reconstructed_file):
            plot_comparison()

def toggle_theme():
    global current_mode, COLORS, header_logo_label
    current_mode = 'light' if current_mode == 'dark' else 'dark'
    COLORS = LIGHT_COLORS if current_mode == 'light' else DARK_COLORS
    
    # Update logo based on theme
    logo_path = "icons/path61.svg" if current_mode == 'light' else "icons/Logo.svg"
    header_logo_label.load(logo_path)
    header_logo_label.renderer().setAspectRatioMode(Qt.KeepAspectRatio)  # Reapply aspect ratio
    header_logo_label.setFixedSize(120, 60)  # Reset size (optional, but ensures consistency)
    
    mode_toggle_button.setText("🌙" if current_mode == 'dark' else "☀")
    update_ui_colors()
    
def restrict_ui_by_role(profession):
    stats_tab_index = tabs.indexOf(settings_tab)
    tabs.setTabEnabled(stats_tab_index, False)
    size_diff_button.setVisible(False)
    percentage_button.setVisible(False)
    add_algorithm_button.setVisible(False)
    file_info_card.setVisible(True)
    stats_card.setVisible(True)
    compression_types = [
        "LZW only",
        "Huffman only",
        "Huffman + LZW",
        "Huffman + LZW + Masquage",
        "Huffman + Masquage",
        "LZW + Masquage"
    ]
    compression_combo.clear()
    if profession == "Utilisateur Standard":
        tabs.setTabVisible(stats_tab_index, False)
        compression_combo.addItems(compression_types)
        file_info_card.setVisible(False)
        stats_card.setVisible(False)
    elif profession == "Ingénieur du son":
        compression_combo.addItems(compression_types)
        tabs.setTabEnabled(stats_tab_index, True)
    elif profession == "Chercheur":
        compression_combo.addItems(compression_types)
        tabs.setTabEnabled(stats_tab_index, True)
        size_diff_button.setVisible(True)
        percentage_button.setVisible(True)
    elif profession == "Développeur":
        compression_combo.addItems(compression_types)
        tabs.setTabEnabled(stats_tab_index, True)
        size_diff_button.setVisible(True)
        percentage_button.setVisible(True)
        add_algorithm_button.setVisible(True)

# Main Application Setup
app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
app.setStyle(QStyleFactory.create("Fusion"))
app.setFont(QFont("Roboto", 11))

window = QWidget()
window._canvases = []
window.setWindowTitle("Compresseur Audio IRM")
window.setGeometry(100, 100, 1400, 900)
window.setMinimumSize(1000, 700)

main_layout = QVBoxLayout()
main_layout.setContentsMargins(15, 15, 15, 15)
window.setLayout(main_layout)

header_frame = QFrame()
header_layout = QHBoxLayout()
header_layout.setContentsMargins(0, 0, 0, 8)
header_frame.setLayout(header_layout)

logo_label = QLabel()
logo_pixmap = QPixmap("icons/L.png")
if not logo_pixmap.isNull():
    logo_label.setPixmap(logo_pixmap.scaled(35, 35, Qt.KeepAspectRatio, Qt.SmoothTransformation))
else:
    logo_label.setText("")
    logo_label.setStyleSheet(f"font-size: 28px; color: {COLORS['accent']};")
header_layout.addWidget(logo_label)

header_logo_label = QSvgWidget("icons/Logo.svg")
header_logo_label.setFixedSize(120, 60)
header_logo_label.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
if header_logo_label.renderer().isValid():
    header_logo_label.setStyleSheet(f"background: transparent;")
else:
    header_logo_label = QLabel("Logo")
    header_logo_label.setStyleSheet(f"font-size: 24px; color: {COLORS['accent']};")
header_layout.addWidget(header_logo_label, alignment=Qt.AlignLeft)

mode_toggle_button = StyledButton("🌙", None)
mode_toggle_button.setFixedSize(28, 28)
mode_toggle_button.setStyleSheet(f"""
    QPushButton {{
        background: {COLORS['accent']};
        color: {COLORS['text_primary']};
        border-radius: 14px;
        font-weight: bold;
        font-size: 16px;
    }}
    QPushButton:hover {{
        background: {COLORS['accent_light']};
    }}
""")
mode_toggle_button.setToolTip("Toggle Dark/Light Mode")
mode_toggle_button.clicked.connect(toggle_theme)
header_layout.addWidget(mode_toggle_button)

help_button = StyledButton("?", None)
help_button.setFixedSize(28, 28)
help_button.setStyleSheet(f"""
    QPushButton {{
        background: {COLORS['accent']};
        color: {COLORS['text_primary']};
        border-radius: 14px;
        font-weight: bold;
        font-size: 16px;
    }}
    QPushButton:hover {{
        background: {COLORS['accent_light']};
    }}
""")
help_button.setToolTip("Afficher l'aide et la documentation")
help_button.clicked.connect(lambda: HelpDialog(window).exec())
header_layout.addWidget(help_button)

main_layout.addWidget(header_frame)

tabs = QTabWidget()
tabs.setStyleSheet(f"""
    QTabWidget::pane {{
        border: 1px solid {COLORS['accent']};
        border-radius: 6px;
        margin-top: -1px;
    }}
    QTabBar::tab {{
        background: {COLORS['panel_bg']};
        color: {COLORS['text_secondary']};
        padding: 8px 18px;
        margin: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-size: 13px;
    }}
    QTabBar::tab:selected {{
        background: {COLORS['accent']};
        color: {COLORS['text_primary']};
        font-weight: bold;
    }}
    QTabBar::tab:hover:!selected {{
        background: {COLORS['secondary_dark']};
        color: {COLORS['text_primary']};
    }}
""")
main_layout.addWidget(tabs)

compression_tab = QWidget()
compression_layout = QHBoxLayout()
compression_layout.setSpacing(15)
compression_tab.setLayout(compression_layout)
left_panel = QScrollArea()
left_panel.setWidgetResizable(True)
left_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
left_panel.setStyleSheet(f"""
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollBar:vertical {{
        background: {COLORS['panel_bg']};
        width: 8px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLORS['accent']};
        min-height: 20px;
        border-radius: 4px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
""")
left_widget = QWidget()
left_layout = QHBoxLayout()
left_layout.setSpacing(10)
left_layout.setAlignment(Qt.AlignCenter)
left_widget.setLayout(left_layout)
left_layout.addStretch()
controls_card = InfoCard("Contrôles de compression")
drop_area = DropArea()
controls_card.add_widget(drop_area)
load_button = StyledButton("Sélectionner un fichier", "icons/load.png")
load_button.clicked.connect(lambda: load_audio_file(QFileDialog.getOpenFileName(None, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)")[0]))
controls_card.add_widget(load_button)
recent_files_label = QLabel("Fichiers récents:")
recent_files_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
controls_card.add_widget(recent_files_label)
recent_files_combo = QComboBox()
recent_files_combo.setStyleSheet(f"""
    QComboBox {{
        background: {COLORS['primary_dark']};
        color: {COLORS['text_primary']};
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 25px;
    }}
    QComboBox QAbstractItemView {{
        background: {COLORS['primary_dark']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['accent']};
    }}
""")
recent_files_combo.setToolTip("Sélectionner un fichier récemment utilisé")
controls_card.add_widget(recent_files_combo)
compression_label = QLabel("Méthode de compression:")
compression_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; margin-top: 10px;")
controls_card.add_widget(compression_label)
compression_combo = QComboBox()
compression_combo.addItems([
    "LZW only",
    "Huffman only",
    "Huffman + LZW",
    "Huffman + LZW + Masquage",
    "Huffman + Masquage",
    "LZW + Masquage"
])
compression_combo.setStyleSheet(f"""
    QComboBox {{
        background: {COLORS['primary_dark']};
        color: {COLORS['text_primary']};
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 25px;
    }}
    QComboBox QAbstractItemView {{
        background: {COLORS['primary_dark']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['accent']};
    }}
""")
compression_combo.setToolTip("Choisir une méthode de compression")
controls_card.add_widget(compression_combo)
compress_button = StyledButton("Compresser", "icons/compress.png")
compress_button.clicked.connect(compress_audio)
compress_button.setEnabled(False)
controls_card.add_widget(compress_button)
add_algorithm_button = StyledButton("Ajouter Algorithme", "icons/add.png")
add_algorithm_button.clicked.connect(lambda: QMessageBox.information(window, "Info", "Fonctionnalité à implémenter dans le code : Ajouter un nouvel algorithme de compression"))
add_algorithm_button.setVisible(False)
controls_card.add_widget(add_algorithm_button)
playback_layout = QHBoxLayout()
playback_layout.setSpacing(10)
play_original_button = StyledButton("Play Original", "icons/play.png")
play_original_button.clicked.connect(play_original_audio)
play_original_button.setEnabled(False)
playback_layout.addWidget(play_original_button)
play_compressed_button = StyledButton("Play Compressé", "icons/play.png")
play_compressed_button.clicked.connect(play_compressed_audio)
play_compressed_button.setEnabled(False)
playback_layout.addWidget(play_compressed_button)
controls_card.add_layout(playback_layout)
progress_bar = EnhancedProgressBar()
progress_bar.hide()
controls_card.add_widget(progress_bar)
left_layout.addWidget(controls_card)
file_info_card = InfoCard("Informations sur le fichier audio")
file_label = QLabel("Aucun fichier sélectionné")
file_label.setStyleSheet(f"""
    font-size: 13px; 
    padding: 8px; 
    background: {COLORS['secondary_dark']}; 
    border-radius: 4px; 
    color: {COLORS['text_secondary']};
""")
file_label.setAlignment(Qt.AlignCenter)
file_info_card.add_widget(file_label)
file_details_layout = QGridLayout()
file_details_layout.setColumnStretch(1, 1)
file_details_layout.setVerticalSpacing(5)
file_details_layout.addWidget(QLabel("Durée:"), 0, 0)
duration_label = QLabel("0.00 seconds")
duration_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
file_details_layout.addWidget(duration_label, 0, 1)
file_details_layout.addWidget(QLabel("Canaux:"), 1, 0)
channels_label = QLabel("0")
channels_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
file_details_layout.addWidget(channels_label, 1, 1)
file_details_layout.addWidget(QLabel("Fréquence d'échantillonnage:"), 2, 0)
sample_rate_label = QLabel("0 Hz")
sample_rate_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
file_details_layout.addWidget(sample_rate_label, 2, 1)
file_details_layout.addWidget(QLabel("Profondeur:"), 3, 0)
bit_depth_label = QLabel("0 bits")
bit_depth_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
file_details_layout.addWidget(bit_depth_label, 3, 1)
file_info_card.add_layout(file_details_layout)
left_layout.addWidget(file_info_card)
stats_card = InfoCard("Statistiques de compression")
stats_frame = QFrame()
stats_layout = QGridLayout()
stats_layout.setVerticalSpacing(5)
stats_frame.setLayout(stats_layout)
stats_labels = {}
labels = ["Taille Originale :", "Taille Compressée:", "Différence:", "Taux de Compression:"]
keys = ["original", "compresseé", "différence", "pourcentage"]
for i, (label, key) in enumerate(zip(labels, keys)):
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
    stats_layout.addWidget(lbl, i, 0)
    stats_labels[key] = QLabel("0.00")
    stats_labels[key].setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px;")
    stats_layout.addWidget(stats_labels[key], i, 1)
stats_card.add_widget(stats_frame)
stats_buttons_layout = QHBoxLayout()
stats_buttons_layout.setSpacing(10)
size_diff_button = StyledButton("Afficher différence", "icons/size.png")
size_diff_button.clicked.connect(show_size_difference)
stats_buttons_layout.addWidget(size_diff_button)
percentage_button = StyledButton("Afficher %", "icons/percentage.png")
percentage_button.clicked.connect(show_compression_percentage)
stats_buttons_layout.addWidget(percentage_button)
stats_card.add_layout(stats_buttons_layout)
left_layout.addWidget(stats_card)
left_layout.addStretch()
left_widget.setLayout(left_layout)
left_panel.setWidget(left_widget)
compression_layout.addWidget(left_panel)
tabs.addTab(compression_tab, "Compression")
settings_tab = QWidget()
settings_layout = QVBoxLayout(settings_tab)
settings_layout.setContentsMargins(15, 15, 15, 15)
settings_layout.setSpacing(10)
waveform_card = InfoCard("Aperçu de la forme d'onde")
waveform_preview_layout = QVBoxLayout()
waveform_card.add_layout(waveform_preview_layout)
visualization_card = InfoCard("Analyse de compression")
visualization_layout = QGridLayout()
visualization_card.add_layout(visualization_layout)
settings_layout.addWidget(waveform_card)
settings_layout.addWidget(visualization_card)
tabs.addTab(settings_tab, "Stats")
window.closeEvent = lambda event: [cleanup_threads(), event.accept()]
init_database()
splash = SvgSplashScreen("icons/Logo.svg", text="Starting IRM Audio Compressor........", padding=90)
splash.show()

def center_window(win):
    fg = win.frameGeometry()
    screen_geom = app.primaryScreen().geometry()
    fg.moveCenter(screen_geom.center())
    win.move(fg.topLeft())

screen = app.primaryScreen()
avail = screen.availableGeometry()
w = int(avail.width())
h = int(avail.height())

def start_main_app():
    splash.close()
    login_dialog = LoginDialog(colors=COLORS)
    if login_dialog.exec():
        profession = login_dialog.profession
        restrict_ui_by_role(profession)
        window.resize(w, h)
        window.setWindowIcon(QIcon('icons/Logo.ico'))
        window.setMinimumSize(int(avail.width() * 0.5), int(avail.height() * 0.5))
        center_window(window)
        window.show()
    else:
        app.quit()

QTimer.singleShot(2000, start_main_app)
sys.exit(app.exec())