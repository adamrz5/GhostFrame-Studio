"""
Batch processing dialog — apply the same censure config to all videos in a folder.
Runs each video sequentially in a background thread.
"""
from __future__ import annotations

import os
import cv2
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QProgressBar, QFileDialog, QMessageBox, QTextEdit,
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject

from core import settings as cfg
from ui.window_chrome import apply_dark_title_bar


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp", ".ts", ".flv"}
SCENE_CUT_THRESHOLD = 0.45


class BatchWorker(QObject):
    file_started  = pyqtSignal(str, int, int)   # (path, current, total)
    file_done     = pyqtSignal(str)             # output path del render terminado
    file_skipped  = pyqtSignal(str)             # input path del vídeo omitido (sin rostros)
    file_error    = pyqtSignal(str, str)
    all_done      = pyqtSignal(bool, bool)   # (cancelado, con_errores)
    log_line      = pyqtSignal(str)

    def __init__(self, video_paths: list[str], persons_config: list[dict], analysis_step: int):
        super().__init__()
        self.video_paths = video_paths
        self.persons_config = persons_config
        self.analysis_step = max(1, int(analysis_step or 1))
        self._cancel = False
        self._had_errors = False

    def cancel(self):
        self._cancel = True

    def _build_persons_config(self, tracker) -> list[dict]:
        """
        Construye un persons_config para el tracker de UN vídeo batch concreto.

        Los person_id del tracker de cada vídeo nuevo (0, 1, 2…) no tienen
        relación con los del vídeo principal — no es posible reutilizar
        self.persons_config directamente.  En su lugar se censuran TODOS los
        rostros detectados, usando el efecto/intensidad de la primera persona
        habilitada del vídeo principal como ajustes por defecto.
        """
        # Tomar los ajustes de efecto de la primera persona habilitada del vídeo principal
        ref = next((c for c in self.persons_config if c.get("enabled")), {})
        effect      = ref.get("effect",    cfg.get("default_effect"))
        intensity   = ref.get("intensity", cfg.get("default_intensity"))
        padding_pct = ref["padding_pct"] if "padding_pct" in ref else cfg.get("default_padding_pct") / 100.0

        return [
            {
                "person_id":   p.person_id,
                "enabled":     True,
                "effect":      effect,
                "intensity":   intensity,
                "padding_pct": padding_pct,
                "start_frame": 0,
                "end_frame":   -1,
            }
            for p in tracker.persons
        ]

    def run(self):
        from core.face_detector import detect_faces
        from core.face_tracker import FaceTracker
        from core.video_processor import VideoReader
        from core.renderer import render_video
        from core import settings as cfg
        from core import session as session_mod
        from core.ffmpeg_utils import RenderCancelled, probe_video

        total = len(self.video_paths)
        for i, vpath in enumerate(self.video_paths):
            if self._cancel:
                self.log_line.emit("Proceso cancelado por el usuario.")
                break
            self.file_started.emit(vpath, i + 1, total)
            self.log_line.emit(f"[{i+1}/{total}] Analizando: {os.path.basename(vpath)}")
            reader = None
            try:
                info = probe_video(vpath, cfg.get("ffmpeg_path") or None)
                loaded_from_cache = False
                try:
                    tracker, cached_info, _ = session_mod.load_session(vpath)
                    info.update(cached_info or {})
                    loaded_from_cache = True
                    self.log_line.emit("  Caché de sesión cargada.")
                except Exception:
                    reader = VideoReader(
                        vpath,
                        rotation=info.get("rotation", 0),
                        frame_count_hint=info.get("frame_count"),
                    )
                    tracker = FaceTracker(
                        similarity_threshold=cfg.get("similarity_threshold"),
                        iou_threshold=cfg.get("iou_threshold"),
                        max_interp_gap=cfg.get("max_interp_gap"),
                    )
                    last_checkpoint = 0
                    chk_every = cfg.get("checkpoint_every") or 500
                    prev_hist = None
                    for fi, frame in reader.iter_frames(step=self.analysis_step):
                        if self._cancel:
                            break
                        # Scene cut detection: mirror AnalysisWorker._producer()
                        # so interpolation does not cross hard cuts in batch mode.
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                        cv2.normalize(hist, hist)
                        if prev_hist is not None:
                            dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                            if dist > SCENE_CUT_THRESHOLD:
                                tracker.scene_cuts.add(fi)
                        prev_hist = hist

                        dets = detect_faces(frame, min_det_score=cfg.get("min_det_score"))
                        tracker.process_frame(fi, dets, frame)
                        # Checkpoint parcial — si el análisis se interrumpe la próxima
                        # ejecución cargará la caché en lugar de empezar desde cero.
                        if fi - last_checkpoint >= chk_every * self.analysis_step:
                            try:
                                session_mod.save_session(vpath, tracker, info, self.analysis_step)
                                last_checkpoint = fi
                            except Exception:
                                pass
                    reader.release()
                    reader = None

                if self._cancel:
                    break

                if not loaded_from_cache:
                    tracker.interpolate_bboxes()
                    tracker.consolidate_persons(sim_threshold=cfg.get("similarity_threshold"))
                    tracker.prune_persons(min_real_frames=5)
                    try:
                        session_mod.save_session(vpath, tracker, info, self.analysis_step)
                    except Exception as exc:
                        self.log_line.emit(f"  Aviso: no se pudo guardar caché ({exc})")

                n_persons = len(tracker.persons)
                if n_persons == 0:
                    self.log_line.emit(f"  Sin rostros detectados — vídeo omitido.")
                    self.file_skipped.emit(vpath)   # no cuenta como render exitoso
                    continue

                # Construir config para ESTE vídeo — los person_id son propios del
                # tracker local y no coinciden con los del vídeo principal.
                persons_config = self._build_persons_config(tracker)
                frame_data = {p.person_id: p.frame_data for p in tracker.persons}
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                renders_dir = os.path.join(project_root, "renders")
                os.makedirs(renders_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(vpath))[0]
                out_path = os.path.join(renders_dir, f"{base}_censored.mp4")
                # Evitar sobrescribir renders anteriores: añadir contador incremental
                if os.path.exists(out_path):
                    counter = 1
                    while True:
                        candidate = os.path.join(renders_dir, f"{base}_censored_{counter:03d}.mp4")
                        if not os.path.exists(candidate):
                            out_path = candidate
                            break
                        counter += 1
                self.log_line.emit(
                    f"  {n_persons} persona(s) detectada(s) — Renderizando → {os.path.basename(out_path)}"
                )

                render_video(
                    input_path=vpath,
                    output_path=out_path,
                    persons_config=persons_config,
                    frame_data=frame_data,
                    video_info=info,
                    cancel_callback=lambda: self._cancel,
                )
                self.file_done.emit(out_path)
                self.log_line.emit(f"  OK Listo: {os.path.basename(out_path)}")
            except RenderCancelled:
                self.log_line.emit("  Render cancelado por el usuario.")
                self._cancel = True   # propagar cancelación para salir del bucle
                break
            except Exception as e:
                self._had_errors = True
                self.file_error.emit(vpath, str(e))
                self.log_line.emit(f"  FALLO Error: {e}")
            finally:
                if reader is not None:
                    reader.release()

        self.all_done.emit(self._cancel, self._had_errors)


class BatchDialog(QDialog):
    _last_folder: str = ""   # persiste entre aperturas del diálogo en la misma sesión

    def __init__(self, persons_config: list[dict], analysis_step: int, parent=None):
        super().__init__(parent)
        self.persons_config = persons_config
        self.analysis_step = analysis_step
        self._thread = None
        self._worker = None
        self._close_after_batch_cancel: str | None = None
        self._processed_count = 0
        self._skipped_count = 0

        self.setWindowTitle("Modo Batch — GhostFrame")
        self.setMinimumSize(580, 480)
        apply_dark_title_bar(self)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Selecciona los vídeos que quieres procesar.\n"
            "Se detectarán TODOS los rostros de cada vídeo y se censurarán con el\n"
            "efecto de la sesión principal. No se identifica si son las mismas\n"
            "personas que en el vídeo principal — en batch se censura todo."
        ))

        # File list
        list_row = QHBoxLayout()
        self.lst_files = QListWidget()
        self.lst_files.setAlternatingRowColors(True)
        list_row.addWidget(self.lst_files)

        btn_col = QVBoxLayout()
        btn_add = QPushButton("+ Añadir vídeos")
        btn_add.clicked.connect(self._add_files)
        btn_folder = QPushButton("+ Carpeta entera")
        btn_folder.clicked.connect(self._add_folder)
        btn_remove = QPushButton("X Quitar seleccion")
        btn_remove.clicked.connect(self._remove_selected)
        btn_col.addWidget(btn_add)
        btn_col.addWidget(btn_folder)
        btn_col.addWidget(btn_remove)
        btn_col.addStretch()
        list_row.addLayout(btn_col)
        layout.addLayout(list_row)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.lbl_status = QLabel("Listo para procesar.")
        layout.addWidget(self.progress)
        layout.addWidget(self.lbl_status)

        # Log
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(120)
        self.txt_log.setStyleSheet("background: #111; color: #aaa; font-size: 11px; font-family: monospace;")
        layout.addWidget(self.txt_log)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Iniciar batch")
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar vídeos", BatchDialog._last_folder,
            "Vídeos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.3gp *.ts *.flv);;Todos (*)"
        )
        if paths:
            BatchDialog._last_folder = os.path.dirname(paths[0])
        for p in paths:
            if p not in self._all_paths():
                self.lst_files.addItem(p)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta", BatchDialog._last_folder
        )
        if folder:
            BatchDialog._last_folder = folder
            for fname in sorted(os.listdir(folder)):
                if os.path.splitext(fname)[1].lower() in VIDEO_EXTS:
                    full = os.path.join(folder, fname)
                    if full not in self._all_paths():
                        self.lst_files.addItem(full)

    def _remove_selected(self):
        for item in self.lst_files.selectedItems():
            self.lst_files.takeItem(self.lst_files.row(item))

    def _all_paths(self) -> list[str]:
        return [self.lst_files.item(i).text() for i in range(self.lst_files.count())]

    def _start(self):
        paths = self._all_paths()
        if not paths:
            QMessageBox.information(self, "Sin archivos", "Añade al menos un vídeo.")
            return
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.txt_log.clear()
        self._processed_count = 0
        self._skipped_count = 0

        worker = BatchWorker(paths, self.persons_config, self.analysis_step)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.file_started.connect(self._on_file_started)
        worker.file_done.connect(self._on_file_done)
        worker.file_skipped.connect(self._on_file_skipped)
        worker.file_error.connect(self._on_file_error)
        worker.log_line.connect(self._log)
        worker.all_done.connect(self._on_all_done)
        worker.all_done.connect(lambda *_: thread.quit())
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.start()
        self._thread = thread
        self._worker = worker

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)

    def _on_file_started(self, path: str, current: int, total: int):
        self._batch_current = current
        self._batch_total   = total
        pct = int(100 * (current - 1) / total)
        self.progress.setValue(pct)
        self.lbl_status.setText(f"[{current}/{total}] {os.path.basename(path)}")

    def _on_file_done(self, _path: str):
        self._processed_count += 1
        total   = getattr(self, "_batch_total",   1) or 1
        current = getattr(self, "_batch_current", 1)
        self.progress.setValue(int(100 * current / total))

    def _on_file_skipped(self, _path: str):
        """Vídeo omitido (sin rostros): avanzar barra pero sin contar como éxito."""
        self._skipped_count += 1
        total   = getattr(self, "_batch_total",   1) or 1
        current = getattr(self, "_batch_current", 1)
        self.progress.setValue(int(100 * current / total))

    def _on_file_error(self, path: str, error: str):
        """Error procesando un vídeo: avanzar barra igualmente para reflejar progreso real."""
        self._log(f"FALLO {os.path.basename(path)}: {error}")
        total   = getattr(self, "_batch_total",   1) or 1
        current = getattr(self, "_batch_current", 1)
        self.progress.setValue(int(100 * current / total))

    def _on_all_done(self, was_cancelled: bool, had_errors: bool):
        if not was_cancelled and not had_errors:
            self.progress.setValue(100)
        if was_cancelled:
            self.lbl_status.setText(
                f"Batch cancelado. {self._processed_count} procesados, "
                f"{self._skipped_count} saltados."
            )
        elif had_errors:
            self.lbl_status.setText(
                f"Batch terminado con errores. {self._processed_count} procesados, "
                f"{self._skipped_count} saltados (sin caras detectadas)."
            )
        else:
            self.lbl_status.setText(
                f"Completado: {self._processed_count} procesados, "
                f"{self._skipped_count} saltados (sin caras detectadas)."
            )
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _on_thread_finished(self):
        self._worker = None
        self._thread = None
        pending = self._close_after_batch_cancel
        self._close_after_batch_cancel = None
        if pending == "accept":
            super().accept()
        elif pending == "reject":
            super().reject()

    def accept(self):
        if self._worker and self._thread and self._thread.isRunning():
            self._worker.cancel()
            self._close_after_batch_cancel = "accept"
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelando batch... se cerrará al terminar el vídeo actual.")
            return
        super().accept()

    def reject(self):
        if self._worker and self._thread and self._thread.isRunning():
            self._worker.cancel()
            self._close_after_batch_cancel = "reject"
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelando batch... se cerrará al terminar el vídeo actual.")
            return
        super().reject()

    def closeEvent(self, event):
        """Asegura que el hilo de batch se detenga antes de cerrar el diálogo."""
        if self._worker and self._thread and self._thread.isRunning():
            self._worker.cancel()
            self._close_after_batch_cancel = "reject"
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelando batch... se cerrará al terminar el vídeo actual.")
            event.ignore()
            return
        event.accept()

    def _log(self, text: str):
        self.txt_log.append(text)
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum()
        )
