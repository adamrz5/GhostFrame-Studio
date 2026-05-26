"""
Diálogo de configuración completo — expone todos los parámetros de core/settings.py.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QLineEdit, QFileDialog,
    QGroupBox, QFormLayout, QDialogButtonBox, QCheckBox, QMessageBox,
)
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal

import os
import ctypes
import json
import subprocess

from core import settings as cfg
from core.ffmpeg_utils import find_ffmpeg, find_ffprobe, get_ffmpeg_version, is_ffmpeg_executable
from ui.window_chrome import apply_dark_title_bar


class RecommendationWorker(QObject):
    finished = pyqtSignal(dict)

    def run(self):
        self.finished.emit(_hardware_recommendation())


def _ram_gb() -> float:
    if not hasattr(ctypes, "windll"):
        return 0.0

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return stat.ullTotalPhys / (1024 ** 3)
    return 0.0


def _detect_gpu_vram_registry() -> int:
    """
    Lee la VRAM real desde el registro de Windows (QWORD HardwareInformation.MemorySize).
    Devuelve el valor máximo encontrado en bytes, o 0 si no está disponible.
    Se usa cuando Win32_VideoController.AdapterRAM devuelve el valor capeado de 4 GB
    (DWORD de 32 bits que no puede representar más de 4 294 967 296 bytes).
    """
    try:
        ps_cmd = (
            "$sizes = (Get-ChildItem "
            "'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
            "{4d36e968-e325-11ce-bfc1-08002be10318}' "
            "-ErrorAction SilentlyContinue) | ForEach-Object { "
            "(Get-ItemProperty $_.PSPath "
            "'HardwareInformation.MemorySize' "
            "-ErrorAction SilentlyContinue).'HardwareInformation.MemorySize' "
            "} | Where-Object { $_ -gt 4294967296 }; "
            "if ($sizes) { ($sizes | Measure-Object -Maximum).Maximum } else { 0 }"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        val = result.stdout.strip()
        if val and val.replace(".", "", 1).isdigit():
            return int(float(val))
    except Exception:
        pass
    return 0


def _detect_gpus() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
        gpus = []
        # Detectar si alguna GPU muestra el valor capeado (4 GB exactos = overflow DWORD).
        # Si es así, intentar obtener el valor real desde el registro (QWORD).
        _DWORD_CAP = 4_294_967_296   # 4 GB en bytes — máximo representable en DWORD
        needs_registry = any(int(item.get("AdapterRAM") or 0) in (0, _DWORD_CAP)
                             for item in data if item.get("Name"))
        registry_vram = _detect_gpu_vram_registry() if needs_registry else 0
        for item in data:
            name = str(item.get("Name") or "").strip()
            ram = int(item.get("AdapterRAM") or 0)
            if not name:
                continue
            # Si AdapterRAM es el valor capeado o cero, usar valor de registro si es mayor.
            if ram in (0, _DWORD_CAP) and registry_vram > _DWORD_CAP:
                ram = registry_vram
            gpus.append({"name": name, "vram_gb": ram / (1024 ** 3) if ram > 0 else 0.0})
        return gpus
    except Exception:
        return []


def _available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


def _hardware_recommendation() -> dict:
    cpu_threads = os.cpu_count() or 1
    ram = _ram_gb()
    gpus = _detect_gpus()
    providers = _available_onnx_providers()
    gpu_names = ", ".join(g["name"] for g in gpus) or "No detectada"
    has_nvidia = any("nvidia" in g["name"].lower() for g in gpus)
    has_gpu = bool(gpus)
    max_vram = max((g["vram_gb"] for g in gpus), default=0.0)

    if has_nvidia and "CUDAExecutionProvider" in providers:
        provider = "cuda"
        provider_reason = "CUDA disponible: es lo más rápido en NVIDIA."
    elif has_gpu and "DmlExecutionProvider" in providers:
        provider = "directml"
        provider_reason = "DirectML disponible: usa la GPU aunque no sea CUDA."
    elif "CUDAExecutionProvider" in providers:
        provider = "cuda"
        provider_reason = "CUDA está instalado y ONNX Runtime lo ofrece."
    else:
        provider = "auto"
        provider_reason = "Auto elegirá el mejor proveedor instalado."

    step = 2 if has_nvidia or cpu_threads >= 8 else 5
    det_size = 320
    model = "auto"
    preset = "fast" if cpu_threads >= 6 else "veryfast"
    crf = 18

    summary = (
        f"Detectado: {cpu_threads} hilos CPU, {ram:.0f} GB RAM, GPU: {gpu_names}.\n"
        f"Recomendado: proveedor {provider.upper()}, analizar cada {step} frames, "
        f"detector {det_size}, encoder hardware activado, CRF {crf}, preset {preset}.\n"
        f"Motivo: {provider_reason}"
    )
    if max_vram >= 10:
        summary += "\nTu VRAM es alta; si hay caras pequeñas puedes probar detector 640, pero 320 suele ir mejor para entrevistas."

    return {
        "summary": summary,
        "analysis_step": step,
        "similarity_threshold": 0.65,
        "iou_threshold": 0.40,
        "max_interp_gap": 30,
        "min_det_score": 0.50,
        "checkpoint_every": 500,
        "execution_provider": provider,
        "model_name": model,
        "det_size": det_size,
        "use_hw_encode": True,
        "crf": crf,
        "encode_preset": preset,
    }


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración — GhostFrame Studio")
        self.setMinimumWidth(520)
        self.setModal(True)
        apply_dark_title_bar(self)
        self._recommendation = {
            "summary": "Calculando recomendación para este PC...",
            "analysis_step": cfg.get("analysis_step"),
            "similarity_threshold": cfg.get("similarity_threshold"),
            "iou_threshold": cfg.get("iou_threshold"),
            "max_interp_gap": cfg.get("max_interp_gap"),
            "min_det_score": cfg.get("min_det_score"),
            "checkpoint_every": cfg.get("checkpoint_every"),
            "execution_provider": cfg.get("execution_provider"),
            "model_name": cfg.get("model_name"),
            "det_size": cfg.get("det_size"),
            "use_hw_encode": cfg.get("use_hw_encode"),
            "crf": cfg.get("crf"),
            "encode_preset": cfg.get("encode_preset"),
        }
        self._rec_thread: QThread | None = None
        self._rec_worker: RecommendationWorker | None = None
        self._pending_done: str | None = None
        self._build_ui()
        self._load()
        self._start_recommendation_worker()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Recomendación automática ─────────────────────────────────────────
        grp_rec = QGroupBox("Recomendación para este PC")
        vr = QVBoxLayout(grp_rec)
        self.lbl_recommendation = QLabel(self._recommendation["summary"])
        self.lbl_recommendation.setWordWrap(True)
        self.lbl_recommendation.setStyleSheet("color: #b8d7ff; font-size: 11px;")
        vr.addWidget(self.lbl_recommendation)
        self.btn_apply_recommended = QPushButton("Aplicar configuración recomendada")
        self.btn_apply_recommended.setEnabled(False)
        self.btn_apply_recommended.setToolTip("Rellena estos ajustes con una configuración equilibrada para tu PC.")
        self.btn_apply_recommended.clicked.connect(self._apply_recommended)
        vr.addWidget(self.btn_apply_recommended, alignment=Qt.AlignRight)
        layout.addWidget(grp_rec)

        # ── Análisis ─────────────────────────────────────────────────────────
        grp_a = QGroupBox("Análisis facial")
        fa = QFormLayout(grp_a)

        self.spn_step = QSpinBox()
        self.spn_step.setRange(1, 30)
        self.spn_step.setSuffix(" frames")
        self.spn_step.setToolTip("Cada cuántos frames buscar caras. Menos = más preciso. Más = más rápido.")
        fa.addRow("Analizar cada:", self.spn_step)

        self.spn_threshold = QDoubleSpinBox()
        self.spn_threshold.setRange(0.25, 0.90)
        self.spn_threshold.setSingleStep(0.05)
        self.spn_threshold.setDecimals(2)
        self.spn_threshold.setToolTip(
            "Decide si dos caras son la misma persona.\n"
            "Sube si mezcla personas distintas. Baja si separa demasiado a la misma persona."
        )
        fa.addRow("Umbral de similitud:", self.spn_threshold)

        self.spn_iou = QDoubleSpinBox()
        self.spn_iou.setRange(0.10, 0.80)
        self.spn_iou.setSingleStep(0.05)
        self.spn_iou.setDecimals(2)
        self.spn_iou.setToolTip(
            "Ayuda a seguir una cara entre frames cercanos.\n"
            "Normalmente 0,40 está bien."
        )
        fa.addRow("Umbral IoU (in-scene):", self.spn_iou)

        self.spn_gap = QSpinBox()
        self.spn_gap.setRange(5, 120)
        self.spn_gap.setSuffix(" frames")
        self.spn_gap.setToolTip("Rellena pequeños huecos si una cara desaparece durante pocos frames.")
        fa.addRow("Máx. interpolación:", self.spn_gap)

        self.spn_score = QDoubleSpinBox()
        self.spn_score.setRange(0.10, 0.95)
        self.spn_score.setSingleStep(0.05)
        self.spn_score.setDecimals(2)
        self.spn_score.setToolTip(
            "Confianza mínima para aceptar una cara.\n"
            "Sube si aparecen falsos positivos. Baja si no detecta algunas caras reales."
        )
        fa.addRow("Confianza mínima:", self.spn_score)

        self.spn_checkpoint = QSpinBox()
        self.spn_checkpoint.setRange(100, 2000)
        self.spn_checkpoint.setSuffix(" frames")
        self.spn_checkpoint.setToolTip(
            "Guarda el progreso del análisis cada cierto tiempo para poder recuperarlo."
        )
        fa.addRow("Checkpoint cada:", self.spn_checkpoint)

        layout.addWidget(grp_a)

        # ── Modelo / Hardware ─────────────────────────────────────────────────
        grp_m = QGroupBox("Modelo e inferencia (Windows)")
        fm = QFormLayout(grp_m)

        self.cmb_model = QComboBox()
        self.cmb_model.addItems([
            "auto  (buffalo_l en Windows)",
            "buffalo_l  — alta precisión (~500 MB)",
            "buffalo_s  — ligero y rápido",
        ])
        fm.addRow("Modelo InsightFace:", self.cmb_model)

        self.cmb_provider = QComboBox()
        self.cmb_provider.addItems([
            "auto  (CUDA > DirectML > CPU)",
            "cpu   — siempre disponible",
            "cuda  — NVIDIA con CUDA vía pip",
            "directml  — GPU universal Windows (pip install onnxruntime-directml)",
        ])
        self.cmb_provider.setToolTip(
            "Dónde se ejecuta la detección de caras.\n"
            "CUDA es lo mejor en NVIDIA. DirectML sirve para otras GPU. CPU es el modo seguro."
        )
        fm.addRow("Proveedor ONNX:", self.cmb_provider)

        self.cmb_det_size = QComboBox()
        self.cmb_det_size.addItems([
            "320  — rápido (recomendado para entrevistas)",
            "640  — máxima precisión (caras pequeñas o lejanas)",
        ])
        self.cmb_det_size.setToolTip(
            "320 es rápido y va bien para entrevistas.\n"
            "640 detecta caras más pequeñas, pero tarda bastante más."
        )
        fm.addRow("Tamaño detector:", self.cmb_det_size)

        layout.addWidget(grp_m)

        # ── Renderizado ───────────────────────────────────────────────────────
        grp_r = QGroupBox("Renderizado")
        fr = QFormLayout(grp_r)

        self.chk_hw_encode = QCheckBox("Usar encoder hardware si está disponible")
        self.chk_hw_encode.setToolTip(
            "Usa la GPU para guardar el vídeo si hay encoder compatible.\n"
            "Si falla, se usa CPU automáticamente."
        )
        fr.addRow("", self.chk_hw_encode)

        self.spn_crf = QSpinBox()
        self.spn_crf.setRange(0, 51)
        self.spn_crf.setToolTip(
            "Calidad si se renderiza por CPU.\n"
            "18 se ve muy bien. Más alto pesa menos, pero pierde calidad."
        )
        fr.addRow("CRF software (0=lossless):", self.spn_crf)

        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"])
        self.cmb_preset.setToolTip("Velocidad del render por CPU. 'fast' suele ser buen equilibrio.")
        fr.addRow("Preset software:", self.cmb_preset)

        self.chk_open_folder = QCheckBox("Abrir carpeta de destino al terminar")
        fr.addRow("", self.chk_open_folder)

        layout.addWidget(grp_r)

        # ── FFmpeg ────────────────────────────────────────────────────────────
        grp_ff = QGroupBox("FFmpeg")
        fff = QFormLayout(grp_ff)

        ff_row = QHBoxLayout()
        self.txt_ffmpeg = QLineEdit()
        self.txt_ffmpeg.setPlaceholderText("Dejar vacío para auto-detectar en PATH")
        ff_row.addWidget(self.txt_ffmpeg)
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self._browse_ffmpeg)
        ff_row.addWidget(btn_browse)
        fff.addRow("Ruta ffmpeg:", ff_row)

        # Mostrar el FFmpeg que realmente usará el renderer:
        # si el usuario configuró una ruta explícita y es válida, esa tiene prioridad.
        configured = cfg.get("ffmpeg_path") or ""
        ff_bin  = configured if configured and os.path.isfile(configured) else find_ffmpeg()
        ffp_bin = find_ffprobe(ff_bin)
        ff_ver  = get_ffmpeg_version(ff_bin) if ff_bin else "—"
        ok      = ff_bin is not None
        ico     = "OK" if ok else "FALLO"
        color   = "#88cc88" if ok else "#cc5555"
        status  = f"{ico}  {ff_bin or 'ffmpeg NO encontrado'}"
        if ffp_bin:
            status += "  |  ffprobe: OK"
        status += f"  [{ff_ver}]"
        lbl_st = QLabel(status)
        lbl_st.setStyleSheet(f"color: {color}; font-size: 10px;")
        lbl_st.setWordWrap(True)
        fff.addRow("Estado:", lbl_st)

        layout.addWidget(grp_ff)

        # ── Botones ───────────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Guardar")
        btns.button(QDialogButtonBox.Cancel).setText("Cancelar")
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _start_recommendation_worker(self):
        if self._rec_thread and self._rec_thread.isRunning():
            return
        worker = RecommendationWorker()
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_recommendation_ready)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_recommendation_finished)
        self._rec_worker = worker
        self._rec_thread = thread
        thread.start()
        # Safety: if the worker hangs (e.g. blocked subprocess) re-enable the dialog
        # after 8 s so the user is never permanently locked out.
        QTimer.singleShot(8000, self._timeout_recommendation)

    def _on_recommendation_ready(self, recommendation: dict):
        self._recommendation = recommendation
        self.lbl_recommendation.setText(recommendation["summary"])
        self.btn_apply_recommended.setEnabled(True)

    def _on_recommendation_finished(self):
        self._rec_worker = None
        self._rec_thread = None
        pending = self._pending_done
        self._pending_done = None
        if pending == "accept":
            self.accept()
        elif pending == "reject":
            self.reject()

    def _timeout_recommendation(self):
        """Safety fallback: if recommendation thread hangs past 8 s, terminate it."""
        if not (self._rec_thread and self._rec_thread.isRunning()):
            return
        print("[Settings] Timeout del diagnóstico de PC — terminando hilo.")
        self._rec_thread.quit()
        if not self._rec_thread.wait(500):
            self._rec_thread.terminate()
        pending = self._pending_done
        self._pending_done = None
        self._rec_worker = None
        self._rec_thread = None
        self.setEnabled(True)
        if pending == "accept":
            super().accept()
        elif pending == "reject":
            super().reject()

    def _defer_done_if_needed(self, action: str) -> bool:
        if self._rec_thread and self._rec_thread.isRunning():
            self.lbl_recommendation.setText("Terminando diagnóstico del PC...")
            self.setEnabled(False)
            self._pending_done = action
            return True
        return False

    def accept(self):
        if self._defer_done_if_needed("accept"):
            return
        super().accept()

    def reject(self):
        if self._defer_done_if_needed("reject"):
            return
        super().reject()

    def closeEvent(self, event):
        if self._defer_done_if_needed("reject"):
            event.ignore()
            return
        event.accept()

    def _load(self):
        self.spn_step.setValue(cfg.get("analysis_step"))
        self.spn_threshold.setValue(cfg.get("similarity_threshold"))
        self.spn_iou.setValue(cfg.get("iou_threshold"))
        self.spn_gap.setValue(cfg.get("max_interp_gap"))
        self.spn_score.setValue(cfg.get("min_det_score"))
        self.spn_checkpoint.setValue(cfg.get("checkpoint_every"))
        self.spn_crf.setValue(cfg.get("crf"))
        self.chk_hw_encode.setChecked(cfg.get("use_hw_encode"))
        self.chk_open_folder.setChecked(cfg.get("open_folder_after_render"))
        self.txt_ffmpeg.setText(cfg.get("ffmpeg_path") or "")

        model_map = {"auto": 0, "buffalo_l": 1, "buffalo_s": 2}
        self.cmb_model.setCurrentIndex(model_map.get(cfg.get("model_name"), 0))

        prov_map  = {"auto": 0, "cpu": 1, "cuda": 2, "directml": 3}
        self.cmb_provider.setCurrentIndex(prov_map.get(cfg.get("execution_provider"), 0))

        det_map = {320: 0, 640: 1}
        self.cmb_det_size.setCurrentIndex(det_map.get(cfg.get("det_size"), 0))

        presets = [self.cmb_preset.itemText(i) for i in range(self.cmb_preset.count())]
        p = cfg.get("encode_preset")
        if p in presets:
            self.cmb_preset.setCurrentIndex(presets.index(p))

    def _apply_recommended(self):
        rec = self._recommendation
        self.spn_step.setValue(rec["analysis_step"])
        self.spn_threshold.setValue(rec["similarity_threshold"])
        self.spn_iou.setValue(rec["iou_threshold"])
        self.spn_gap.setValue(rec["max_interp_gap"])
        self.spn_score.setValue(rec["min_det_score"])
        self.spn_checkpoint.setValue(rec["checkpoint_every"])
        self.spn_crf.setValue(rec["crf"])
        self.chk_hw_encode.setChecked(rec["use_hw_encode"])

        model_map = {"auto": 0, "buffalo_l": 1, "buffalo_s": 2}
        prov_map = {"auto": 0, "cpu": 1, "cuda": 2, "directml": 3}
        det_map = {320: 0, 640: 1}
        self.cmb_model.setCurrentIndex(model_map.get(rec["model_name"], 0))
        self.cmb_provider.setCurrentIndex(prov_map.get(rec["execution_provider"], 0))
        self.cmb_det_size.setCurrentIndex(det_map.get(rec["det_size"], 0))

        presets = [self.cmb_preset.itemText(i) for i in range(self.cmb_preset.count())]
        if rec["encode_preset"] in presets:
            self.cmb_preset.setCurrentIndex(presets.index(rec["encode_preset"]))

        QMessageBox.information(
            self,
            "Recomendación aplicada",
            "Se han aplicado los ajustes recomendados. Pulsa Guardar para conservarlos.",
        )

    def _browse_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar ffmpeg.exe", "",
            "Ejecutable (*.exe);;Todos (*)"
        )
        if path:
            self.txt_ffmpeg.setText(path)

    def _save(self):
        model_keys = ["auto", "buffalo_l", "buffalo_s"]
        prov_keys  = ["auto", "cpu", "cuda", "directml"]
        presets    = [self.cmb_preset.itemText(i) for i in range(self.cmb_preset.count())]
        ffmpeg_path = self.txt_ffmpeg.text().strip()
        if ffmpeg_path and not os.path.isfile(ffmpeg_path):
            QMessageBox.warning(
                self,
                "FFmpeg no válido",
                f"La ruta configurada no existe:\n{ffmpeg_path}"
            )
            return
        if ffmpeg_path and not is_ffmpeg_executable(ffmpeg_path):
            QMessageBox.warning(
                self,
                "FFmpeg no válido",
                f"La ruta configurada no parece ser ffmpeg.exe:\n{ffmpeg_path}"
            )
            return

        s = cfg.load()
        s["analysis_step"]          = self.spn_step.value()
        s["similarity_threshold"]   = round(self.spn_threshold.value(), 2)
        s["iou_threshold"]          = round(self.spn_iou.value(), 2)
        s["max_interp_gap"]         = self.spn_gap.value()
        s["min_det_score"]          = round(self.spn_score.value(), 2)
        s["checkpoint_every"]       = self.spn_checkpoint.value()
        s["crf"]                    = self.spn_crf.value()
        s["use_hw_encode"]          = self.chk_hw_encode.isChecked()
        s["open_folder_after_render"] = self.chk_open_folder.isChecked()
        s["ffmpeg_path"]            = ffmpeg_path
        s["model_name"]             = model_keys[min(self.cmb_model.currentIndex(), 2)]
        s["execution_provider"]     = prov_keys[min(self.cmb_provider.currentIndex(), 3)]
        s["det_size"]               = [320, 640][min(self.cmb_det_size.currentIndex(), 1)]
        _idx = self.cmb_preset.currentIndex()
        s["encode_preset"]          = (
            presets[_idx]
            if 0 <= _idx < len(presets)
            else cfg.DEFAULTS.get("encode_preset", presets[0])
        )
        cfg.save(s)
        self.accept()
