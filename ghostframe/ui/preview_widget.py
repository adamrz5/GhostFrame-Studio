"""
Preview widget — renders a single frame (with censure applied) inside a QLabel.
Handles both vertical (9:16) and horizontal (16:9) videos gracefully.
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QEvent, QPoint, Qt
from PyQt5.QtGui import QImage, QPixmap


def frame_to_pixmap(frame: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    if frame is None or frame.size == 0:
        img = QImage(max_w, max_h, QImage.Format_RGB888)
        img.fill(Qt.black)
        return QPixmap.fromImage(img)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if w <= 0 or h <= 0:
        img = QImage(max_w, max_h, QImage.Format_RGB888)
        img.fill(Qt.black)
        return QPixmap.fromImage(img)
    # Guardar el bytes en variable local — QImage NO copia el buffer,
    # solo guarda un puntero. Si tobytes() devuelve un temporal anónimo,
    # el GC lo destruiría antes de que fromImage() termine de leer los datos.
    raw = rgb.tobytes()
    qi = QImage(raw, w, h, 3 * w, QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(qi)
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom: float = 1.0
        self._zoom_center: QPoint = QPoint(0, 0)
        self._last_frame: np.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("Sin preview — carga un vídeo y analiza las caras")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setMinimumSize(280, 200)
        self.label.setStyleSheet(
            "background: #050505; color: #ffffff; border: 1px solid #202020; border-radius: 6px; font-size: 12px;"
        )
        layout.addWidget(self.label)
        self.label.installEventFilter(self)

    def show_frame(self, frame: np.ndarray):
        # Guardar copia propia — el caller puede seguir mutando su frame
        # (p.ej. apply_censure in-place) sin corromper el zoom display
        self._last_frame = frame.copy() if frame is not None else None
        self._render()

    def _render(self):
        frame = self._last_frame
        if frame is None:
            return
        w = self.label.width() - 4
        h = self.label.height() - 4
        if self._zoom > 1.0 and frame is not None and frame.size:
            frame = self._zoomed_frame(frame, max(w, 100), max(h, 100))
        pix = frame_to_pixmap(frame, max(w, 100), max(h, 100))
        self.label.setPixmap(pix)

    def _zoomed_frame(self, frame: np.ndarray, view_w: int, view_h: int) -> np.ndarray:
        fh, fw = frame.shape[:2]
        if fw <= 1 or fh <= 1:
            return frame
        scale = min(view_w / fw, view_h / fh)
        if scale < 1.0:
            display_w = int(fw * scale)
            display_h = int(fh * scale)
        else:
            display_w = fw
            display_h = fh
        offset_x = max(0, (view_w - display_w) // 2)
        offset_y = max(0, (view_h - display_h) // 2)
        local_x = min(display_w, max(0, self._zoom_center.x() - offset_x))
        local_y = min(display_h, max(0, self._zoom_center.y() - offset_y))
        cx_ratio = min(1.0, max(0.0, local_x / max(1, display_w)))
        cy_ratio = min(1.0, max(0.0, local_y / max(1, display_h)))
        crop_w = max(1, int(fw / self._zoom))
        crop_h = max(1, int(fh / self._zoom))
        cx = int(cx_ratio * fw)
        cy = int(cy_ratio * fh)
        x1 = min(max(0, cx - crop_w // 2), max(0, fw - crop_w))
        y1 = min(max(0, cy - crop_h // 2), max(0, fh - crop_h))
        return frame[y1:y1 + crop_h, x1:x1 + crop_w]

    def eventFilter(self, watched, event):
        if watched is self.label:
            if event.type() == QEvent.Wheel:
                self.wheelEvent(event)
                return True
            if event.type() == QEvent.MouseButtonDblClick:
                self.mouseDoubleClickEvent(event)
                return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        if delta:
            self._zoom = max(1.0, min(8.0, self._zoom * (1.15 ** delta)))
            self._zoom_center = event.pos()
            self._render()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self._zoom = 1.0
        self._zoom_center = QPoint(0, 0)
        self._render()
        event.accept()

    def clear(self):
        self._last_frame = None
        self._zoom = 1.0
        self.label.clear()
        self.label.setText("Sin preview")
