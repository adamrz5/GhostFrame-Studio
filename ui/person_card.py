"""
PersonCard — tarjeta de persona con:
- Miniatura de cara
- Barra visual de apariciones (muestra en qué frames está presente)
- Nombre editable con doble clic
- Controles de censura (efecto, intensidad, margen, rango de tiempo)
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QComboBox, QSlider, QGroupBox, QFrame, QSizePolicy, QInputDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor

from core import settings as cfg


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bgr_to_pixmap(bgr: np.ndarray, size: int = 96) -> QPixmap:
    bg = QColor(21, 21, 21)
    if bgr is None or bgr.size == 0:
        img = QImage(size, size, QImage.Format_RGB888)
        img.fill(bg)
        return QPixmap.fromImage(img)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]
    scale = min(size / max(1, w0), size / max(1, h0))
    nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), (bg.red(), bg.green(), bg.blue()), dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = rgb
    raw = canvas.tobytes()
    qi = QImage(raw, size, size, 3 * size, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qi)


EFFECT_OPTIONS = [
    ("Pixelado",       "pixelate"),
    ("Gaussian Blur",  "blur"),
    ("Caja negra",     "blackbox"),
]

PERSON_COLORS = [
    QColor(85,  136, 238),   # azul
    QColor(238, 100,  85),   # rojo
    QColor(80,  200, 110),   # verde
    QColor(220, 180,  50),   # amarillo
    QColor(180,  85, 220),   # morado
    QColor(60,  190, 210),   # cian
    QColor(240, 130,  40),   # naranja
    QColor(150, 200,  80),   # lima
]


# ─── Barra visual de apariciones ─────────────────────────────────────────────

class PersonTimelineBar(QWidget):
    """
    Raya horizontal que dibuja un tick por cada frame en que se detectó
    a esta persona. Imprescindible para saber qué rango usar en split/merge.
    """
    def __init__(self, person, total_frames: int, color: QColor, parent=None):
        super().__init__(parent)
        self.person = person
        self.total_frames = max(1, total_frames)
        self.color = color
        self.setFixedHeight(8)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Frames donde aparece esta persona (color de la persona = detectado)")

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        radius = min(3, h // 2)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(21, 21, 21))
        painter.drawRoundedRect(0, 1, w, max(1, h - 2), radius, radius)
        if not self.person.frame_data:
            return
        bar_color = QColor(self.color)
        bar_color.setAlpha(150)
        painter.setBrush(bar_color)

        frames = sorted(self.person.frame_data)
        start = prev = frames[0]
        segments = []
        for fi in frames[1:]:
            if fi <= prev + 1:
                prev = fi
                continue
            segments.append((start, prev))
            start = prev = fi
        segments.append((start, prev))

        for start, end in segments:
            x1 = int(start / self.total_frames * w)
            x2 = int((end + 1) / self.total_frames * w)
            painter.drawRoundedRect(x1, 1, max(2, x2 - x1), max(1, h - 2), radius, radius)


# ─── PersonCard ───────────────────────────────────────────────────────────────

class PersonCard(QWidget):
    config_changed = pyqtSignal()
    person_renamed = pyqtSignal()   # emitida tras renombrar — para que main_window auto-guarde

    def __init__(self, person, fps: float, total_frames: int, display_index: int | None = None, parent=None):
        super().__init__(parent)
        self.person = person
        self.fps = fps
        self.total_frames = max(1, total_frames)
        self.display_index = display_index
        color_idx = (display_index - 1) if display_index is not None else person.person_id
        self.color = PERSON_COLORS[color_idx % len(PERSON_COLORS)]
        # Etiqueta personalizable — por defecto "Persona N"
        self._custom_label: str | None = getattr(person, "_custom_label", None)
        self._build_ui()

    @property
    def display_name(self) -> str:
        if self._custom_label:
            return self._custom_label
        idx = self.display_index if self.display_index is not None else self.person.person_id + 1
        return f"Persona {idx}"

    def _build_ui(self):
        self.setMinimumWidth(210)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        cx = self.color.name()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #3c3c44; border-radius: 6px; background: #151515; }"
        )
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(5)

        # Miniatura
        thumb = QLabel()
        thumb.setAlignment(Qt.AlignCenter)
        thumb.setPixmap(_bgr_to_pixmap(self.person.thumbnail))
        thumb.setStyleSheet("background: #151515; border: 1px solid #3c3c44; border-radius: 4px; padding: 2px;")
        inner.addWidget(thumb)

        # Nombre editable con doble clic
        self.lbl_name = QLabel(self.display_name)
        self.lbl_name.setAlignment(Qt.AlignCenter)
        self.lbl_name.setStyleSheet(
            f"font-weight: bold; font-size: 13px; color: {cx};"
            " background: #1d1d22; border: 1px solid #3c3c44; border-radius: 3px; padding: 2px 4px;"
        )
        self.lbl_name.setToolTip("Doble clic para renombrar")
        self.lbl_name.mouseDoubleClickEvent = self._rename_person
        inner.addWidget(self.lbl_name)

        frames_lbl = QLabel(f"{self.person.frame_count} frames detectados")
        frames_lbl.setAlignment(Qt.AlignCenter)
        frames_lbl.setStyleSheet("color: #ffffff; font-size: 10px;")
        inner.addWidget(frames_lbl)

        # Barra visual de apariciones
        self.timeline_bar = PersonTimelineBar(self.person, self.total_frames, self.color)
        inner.addWidget(self.timeline_bar)

        # Toggle censura
        self.chk_enable = QCheckBox("Censurar esta persona")
        self.chk_enable.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                background: #151515;
                border: none;
                padding: 2px 0;
                spacing: 6px;
            }
            QCheckBox:focus {
                border: none;
                outline: none;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                background: #f4f4f4;
                border: 1px solid #9a9aa2;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                background: #5f8cff;
                border-color: #ffffff;
            }
        """)
        self.chk_enable.toggled.connect(self._on_toggle)
        inner.addWidget(self.chk_enable)

        # Controles editables siempre; la casilla solo decide si se aplican al render.
        self.ctrl_group = QGroupBox()
        self.ctrl_group.setEnabled(True)
        self.ctrl_group.setStyleSheet("""
            QGroupBox { border: none; margin: 0; padding: 0; background: #151515; color: #ffffff; }
            QLabel { color: #ffffff; background: transparent; border: none; }
            QComboBox { background: #242429; color: #ffffff; border: 1px solid #555; }
            QSlider {
                background: #151515;
                border: none;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #151515;
                border: none;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #5f8cff;
                border-radius: 2px;
            }
            QSlider::add-page:horizontal {
                background: #24242a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 10px;
                height: 10px;
                background: #ffffff;
                border-radius: 5px;
                margin: -3px 0;
            }
        """)
        ctrl = QVBoxLayout(self.ctrl_group)
        ctrl.setContentsMargins(0, 4, 0, 0)
        ctrl.setSpacing(6)

        # Efecto
        eff_row = QHBoxLayout()
        eff_row.addWidget(QLabel("Efecto:"))
        self.cmb_effect = QComboBox()
        for lbl, _ in EFFECT_OPTIONS:
            self.cmb_effect.addItem(lbl)
        default_eff = cfg.get("default_effect")
        self.cmb_effect.setCurrentIndex(
            next((i for i, (_, k) in enumerate(EFFECT_OPTIONS) if k == default_eff), 0)
        )
        self.cmb_effect.currentIndexChanged.connect(lambda _: self._emit_if_enabled())
        eff_row.addWidget(self.cmb_effect)
        ctrl.addLayout(eff_row)

        # Intensidad
        ctrl.addWidget(QLabel("Intensidad:"))
        int_row = QHBoxLayout()
        self.sld_intensity = QSlider(Qt.Horizontal)
        self.sld_intensity.setRange(1, 10)
        self.sld_intensity.setValue(cfg.get("default_intensity"))
        self.sld_intensity.valueChanged.connect(lambda _: self._emit_if_enabled())
        self.lbl_int = QLabel(str(self.sld_intensity.value()))
        self.lbl_int.setFixedWidth(18)
        self.sld_intensity.valueChanged.connect(lambda v: self.lbl_int.setText(str(v)))
        int_row.addWidget(self.sld_intensity)
        int_row.addWidget(self.lbl_int)
        ctrl.addLayout(int_row)

        # Margen
        ctrl.addWidget(QLabel("Margen alrededor (%):"))
        pad_row = QHBoxLayout()
        self.sld_padding = QSlider(Qt.Horizontal)
        self.sld_padding.setRange(0, 50)
        self.sld_padding.setValue(cfg.get("default_padding_pct"))
        self.sld_padding.valueChanged.connect(lambda _: self._emit_if_enabled())
        self.lbl_pad = QLabel(f"{self.sld_padding.value()}%")
        self.lbl_pad.setFixedWidth(30)
        self.sld_padding.valueChanged.connect(lambda v: self.lbl_pad.setText(f"{v}%"))
        pad_row.addWidget(self.sld_padding)
        pad_row.addWidget(self.lbl_pad)
        ctrl.addLayout(pad_row)

        # Rango de tiempo
        ctrl.addWidget(QLabel("Rango de tiempo:"))
        self.cmb_range = QComboBox()
        self.cmb_range.addItems(["Todo el vídeo", "Personalizado"])
        self.cmb_range.currentIndexChanged.connect(self._on_range_changed)
        ctrl.addWidget(self.cmb_range)

        self.custom_range = QWidget()
        cr = QVBoxLayout(self.custom_range)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.setSpacing(2)

        s_row = QHBoxLayout()
        s_row.addWidget(QLabel("Inicio:"))
        self.sld_start = QSlider(Qt.Horizontal)
        self.sld_start.setRange(0, self.total_frames - 1)
        self.sld_start.setValue(0)
        self.sld_start.valueChanged.connect(self._on_start_changed)
        self.lbl_start_t = QLabel(self._ft(0))
        self.lbl_start_t.setFixedWidth(58)
        s_row.addWidget(self.sld_start)
        s_row.addWidget(self.lbl_start_t)
        cr.addLayout(s_row)

        e_row = QHBoxLayout()
        e_row.addWidget(QLabel("  Fin:"))
        self.sld_end = QSlider(Qt.Horizontal)
        self.sld_end.setRange(0, self.total_frames - 1)
        self.sld_end.setValue(self.total_frames - 1)
        self.sld_end.valueChanged.connect(self._on_end_changed)
        self.lbl_end_t = QLabel(self._ft(self.total_frames - 1))
        self.lbl_end_t.setFixedWidth(58)
        e_row.addWidget(self.sld_end)
        e_row.addWidget(self.lbl_end_t)
        cr.addLayout(e_row)

        self.custom_range.setVisible(False)
        ctrl.addWidget(self.custom_range)

        inner.addWidget(self.ctrl_group)
        outer.addWidget(frame)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ft(self, fi: int) -> str:
        fps = self.fps if self.fps > 0 else 25.0
        t = fi / fps
        m, s = divmod(t, 60)
        return f"{int(m):02d}:{s:04.1f}"

    def _on_toggle(self, checked: bool):
        self.config_changed.emit()

    def _emit_if_enabled(self):
        if self.chk_enable.isChecked():
            self.config_changed.emit()

    def _on_range_changed(self, idx: int):
        self.custom_range.setVisible(idx == 1)
        self._emit_if_enabled()

    def _on_start_changed(self, val: int):
        if val > self.sld_end.value():
            self.sld_end.blockSignals(True)
            self.sld_end.setValue(val)
            self.sld_end.blockSignals(False)
            self.lbl_end_t.setText(self._ft(val))
        self.lbl_start_t.setText(self._ft(val))
        self._emit_if_enabled()

    def _on_end_changed(self, val: int):
        if val < self.sld_start.value():
            self.sld_start.blockSignals(True)
            self.sld_start.setValue(val)
            self.sld_start.blockSignals(False)
            self.lbl_start_t.setText(self._ft(val))
        self.lbl_end_t.setText(self._ft(val))
        self._emit_if_enabled()

    def _rename_person(self, _event=None):
        """Doble clic en el nombre → diálogo para renombrar."""
        new_name, ok = QInputDialog.getText(
            self, "Renombrar persona",
            "Nuevo nombre:",
            text=self._custom_label or self.display_name,
        )
        if ok and new_name.strip():
            self._custom_label = new_name.strip()
            self.person._custom_label = self._custom_label
            self.lbl_name.setText(self._custom_label)
            self.person_renamed.emit()

    def get_config(self) -> dict:
        effect = EFFECT_OPTIONS[self.cmb_effect.currentIndex()][1]
        if self.cmb_range.currentIndex() == 0:
            start_frame, end_frame = 0, -1
        else:
            start_frame = self.sld_start.value()
            end_frame   = max(self.sld_start.value(), self.sld_end.value())
        return {
            "person_id":   self.person.person_id,
            "label":       self.display_name,
            "enabled":     self.chk_enable.isChecked(),
            "effect":      effect,
            "intensity":   self.sld_intensity.value(),
            "padding_pct": self.sld_padding.value() / 100.0,
            "start_frame": start_frame,
            "end_frame":   end_frame,
        }

    def apply_config(self, config: dict) -> None:
        """
        Aplica una configuración de censura completa (incluye rango temporal).
        Bloquea señales durante la aplicación para emitir config_changed
        una sola vez al final, evitando previsualizaciones intermedias incorrectas.
        """
        widgets = [
            self.chk_enable, self.cmb_effect, self.sld_intensity,
            self.sld_padding, self.cmb_range, self.sld_start, self.sld_end,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.chk_enable.setChecked(bool(config.get("enabled", False)))
            effect = config.get("effect", cfg.get("default_effect"))
            effect_idx = next((i for i, (_, key) in enumerate(EFFECT_OPTIONS) if key == effect), 0)
            self.cmb_effect.setCurrentIndex(effect_idx)
            self.sld_intensity.setValue(int(config.get("intensity", self.sld_intensity.value())))
            padding = int(round(float(config.get("padding_pct", self.sld_padding.value() / 100.0)) * 100))
            self.sld_padding.setValue(max(self.sld_padding.minimum(), min(self.sld_padding.maximum(), padding)))
            # Restaurar rango temporal si está presente en el config importado
            if "start_frame" in config or "end_frame" in config:
                start = int(config.get("start_frame", 0))
                end   = int(config.get("end_frame", -1))
                if start == 0 and end == -1:
                    self.cmb_range.setCurrentIndex(0)        # Todo el vídeo
                    self.custom_range.setVisible(False)
                else:
                    self.cmb_range.setCurrentIndex(1)        # Personalizado
                    self.custom_range.setVisible(True)
                    max_frame = self.total_frames - 1
                    self.sld_start.setValue(max(0, min(start, max_frame)))
                    real_end = end if end >= 0 else max_frame
                    self.sld_end.setValue(max(self.sld_start.value(), min(real_end, max_frame)))
                    self.lbl_start_t.setText(self._ft(self.sld_start.value()))
                    self.lbl_end_t.setText(self._ft(self.sld_end.value()))
        finally:
            for w in widgets:
                w.blockSignals(False)
        # Emitir una vez con el estado final completo
        self.config_changed.emit()
