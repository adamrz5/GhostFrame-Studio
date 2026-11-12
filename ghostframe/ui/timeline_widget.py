"""
Timeline scrubber widget with formatted time display.
Emits frame_selected(int) when the user drags or clicks.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSlider, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


class ScrubSlider(QSlider):
    scrub_started = pyqtSignal()

    def mousePressEvent(self, event):
        self.scrub_started.emit()
        super().mousePressEvent(event)


class TimelineWidget(QWidget):
    frame_selected  = pyqtSignal(int)   # emitido en cada movimiento del slider
    scrub_started   = pyqtSignal()      # emitido al empezar a arrastrar el slider
    scrub_released  = pyqtSignal(int)   # emitido al soltar el slider (mouseRelease)

    def __init__(self, total_frames: int, fps: float, parent=None):
        super().__init__(parent)
        self.total_frames = max(1, total_frames)
        self.fps = fps if fps > 0 else 25.0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        row = QHBoxLayout()

        end_sec = (self.total_frames - 1) / self.fps
        # Width from the longer of the two labels so both are consistent.
        lbl_sample = QLabel(_fmt(end_sec))
        lbl_sample.setStyleSheet("color: #ffffff; font-size: 11px;")
        lbl_w = max(68, lbl_sample.sizeHint().width() + 12)

        self.lbl_start = QLabel(_fmt(0))
        self.lbl_start.setStyleSheet("color: #ffffff; font-size: 11px;")
        self.lbl_start.setFixedWidth(lbl_w)
        self.lbl_start.setAlignment(Qt.AlignCenter)
        row.addWidget(self.lbl_start)

        self.slider = ScrubSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        self.slider.scrub_started.connect(self.scrub_started.emit)
        self.slider.valueChanged.connect(self._on_value)
        self.slider.sliderReleased.connect(self._on_released)
        row.addWidget(self.slider, stretch=1)

        self.lbl_end = QLabel(_fmt(end_sec))
        self.lbl_end.setStyleSheet("color: #ffffff; font-size: 11px;")
        self.lbl_end.setFixedWidth(lbl_w)
        self.lbl_end.setAlignment(Qt.AlignCenter)
        row.addWidget(self.lbl_end)

        layout.addLayout(row)

        self.lbl_pos = QLabel("00:00.00  (Frame 0)")
        self.lbl_pos.setAlignment(Qt.AlignCenter)
        self.lbl_pos.setStyleSheet("color: #ffffff; font-size: 11px;")
        layout.addWidget(self.lbl_pos)

    def _on_value(self, fi: int):
        t = fi / self.fps
        self.lbl_pos.setText(f"{_fmt(t)}  (Frame {fi})")
        self.frame_selected.emit(fi)

    def _on_released(self):
        self.scrub_released.emit(self.slider.value())

    def current_frame(self) -> int:
        return self.slider.value()

    def set_frame(self, fi: int):
        fi = max(0, min(fi, self.total_frames - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(fi)
        self.slider.blockSignals(False)
        t = fi / self.fps
        self.lbl_pos.setText(f"{_fmt(t)}  (Frame {fi})")

    def update_total_frames(self, total_frames: int):
        """Corrige el total de frames si probe_video() sobreestimó el recuento.

        Llamar una vez al finalizar el análisis o la reproducción completa, cuando
        se conoce el frame real más alto. Actualiza el slider y la etiqueta de fin.
        """
        total_frames = max(1, total_frames)
        if total_frames == self.total_frames:
            return
        self.total_frames = total_frames
        # Reajustar rango del slider
        current = min(self.slider.value(), total_frames - 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, total_frames - 1)
        self.slider.setValue(current)
        self.slider.blockSignals(False)
        # Actualizar etiqueta de duración total
        end_sec = (total_frames - 1) / self.fps
        self.lbl_end.setText(_fmt(end_sec))
