"""
Ventana principal — GhostFrame Studio.

Novedades respecto a la versión anterior:
- Play preview secuencial en QThread (sin seeks por frame, compatible con HEVC/iPhone)
- Checkpoint de análisis: guarda .gfs parcial cada N frames para no perder trabajo
- Export frame: guardar el frame actual como imagen
- Barra de info de vídeo detallada (VFR, rotación iPhone, codec, tamaño)
- Detección de hardware encoder en status bar
- Soporte DirectML (GPU universal Windows) como proveedor ONNX
"""
from __future__ import annotations

import copy
import os
import queue as _queue
import sys
import threading
import time

import cv2
from PyQt5.QtCore import Qt, QEventLoop, QThread, QTimer, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAction, QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QCheckBox, QLineEdit, QScrollArea, QShortcut, QSpinBox, QSplitter, QStatusBar,
    QStyle, QVBoxLayout, QTextBrowser, QWidget,
)

from core import session as session_mod
from core import settings as cfg
from core.face_detector import (
    active_model, active_provider, detect_faces, init_detector, gpu_fallback_occurred,
)
from core.face_tracker import FaceTracker
from core.ffmpeg_utils import assert_ffmpeg, find_ffmpeg, probe_video, detect_hardware_encoders
from core.video_processor import VideoReader, apply_censure, censure_roi_inplace
from ui.person_card import PERSON_COLORS, PersonCard
from ui.preview_widget import PreviewWidget
from ui.timeline_widget import TimelineWidget
from ui.window_chrome import apply_dark_title_bar


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fmt_eta(secs: int) -> str:
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ─── Scrub worker (frame seek en hilo propio, evita congelar la UI) ──────────

class ScrubWorker(QObject):
    """
    Maintains its own VideoCapture so scrub seeks never block the UI thread.
    Runs in a dedicated QThread; receives frame requests via queued signals.
    """
    frame_ready = pyqtSignal(object, int)   # (frame_bgr np.ndarray, frame_idx)

    def __init__(self):
        super().__init__()
        self._cap:        cv2.VideoCapture | None = None
        self._video_path: str | None              = None
        self._rotation:   int                     = 0

    @pyqtSlot(str, int)
    def open_video(self, path: str, rotation: int):
        """Open (or switch to) a video file. Safe to call from any thread via queued connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._video_path = path
        self._rotation   = rotation
        self._cap        = cv2.VideoCapture(path)

    @pyqtSlot(int)
    def read_frame(self, fi: int):
        """Seek to fi and emit the decoded frame. Silently drops the request if not open."""
        if self._cap is None or not self._cap.isOpened():
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self.frame_ready.emit(VideoReader._rotate(frame, self._rotation), fi)

    @pyqtSlot()
    def close_video(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._video_path = None

    @pyqtSlot()
    def stop(self):
        """
        Libera el VideoCapture inmediatamente.
        Llamar antes de thread.quit() para que cap.read() en curso retorne rápido
        y no deje el hilo esperando hasta 3 s en closeEvent.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._video_path = None


# ─── Playback worker (audio mpv + OpenCV video) ──────────────────────────────

class PlaybackWorker(QObject):
    """
    Lee frames con OpenCV usando el audio (mpv) como reloj maestro.

    Si python-mpv o libmpv no están disponibles, usa un reloj monotónico
    interno (mismo comportamiento de antes: sin audio).

    Algoritmo de sincronización
    ───────────────────────────
    Cada iteración:
      1. Pedir time_pos al gestor de audio.
      2. Calcular frame objetivo = int(time_pos * fps).
      3. Comparar con el último frame leído:
         - gap == 0 → dormir medio frame y volver.
         - 1 ≤ gap ≤ SEQUENTIAL_THRESHOLD → grab() los intermedios (sin decodificar)
           y read() solo el objetivo.
         - gap > SEQUENTIAL_THRESHOLD → seek directo en OpenCV.
         - gap < 0 → vídeo va adelantado al audio; reducir wait.
      4. Rotar el frame si es iPhone/vertical.
      5. Emitir frame_ready(frame, fi).

    El gestor de audio también sirve de fuente de EOF: con mpv, time_pos vuelve
    a None cuando termina el archivo. Con InternalClock se detecta por frame_count.
    """

    # Número máximo de frames que se leen secuencialmente antes de hacer seek.
    SEQUENTIAL_THRESHOLD = 4
    # Si vamos más de este número de frames por detrás del audio, saltamos.
    MAX_LAG_FRAMES = 8
    # Tiempo máximo de espera para que mpv empiece a reproducir (segundos).
    MPV_START_TIMEOUT = 3.0
    # Tiempo máximo de espera para que mpv confirme un seek (segundos).
    MPV_SEEK_TIMEOUT = 1.5

    frame_ready = pyqtSignal(object, int)   # (frame_bgr np.ndarray, frame_index)
    finished    = pyqtSignal()

    def __init__(
        self,
        video_path:  str,
        start_frame: int,
        fps:         float,
        speed:       float,
        rotation:    int = 0,
        frame_count: int = 0,
        has_audio:   bool = True,
    ):
        super().__init__()
        self.video_path  = video_path
        self.start_frame = start_frame
        self.fps         = max(0.1, fps)
        self.speed       = max(0.05, speed)
        self.rotation    = rotation % 360
        self.frame_count = frame_count  # 0 = desconocido
        self.has_audio   = has_audio
        self._stop       = False
        # Thread-safe seek queue: UI escribe, worker lee. Queue(maxsize=1) conserva
        # solo el seek más reciente (si el worker no llega a tiempo, el viejo se descarta).
        self._seek_queue: _queue.Queue = _queue.Queue(maxsize=1)
        self._audio      = None   # AudioPlaybackManager; creado en run()

    def stop(self) -> None:
        self._stop = True
        if self._audio is not None:
            self._audio.stop()

    def seek_to_frame(self, frame_idx: int) -> None:
        """Thread-safe: llamable desde cualquier hilo."""
        fi = max(0, int(frame_idx))
        # Descartar seeks pendientes anteriores; guardar solo el más reciente
        while not self._seek_queue.empty():
            try:
                self._seek_queue.get_nowait()
            except _queue.Empty:
                break
        try:
            self._seek_queue.put_nowait(fi)
        except _queue.Full:
            pass

    def run(self) -> None:
        from ui.playback_manager import AudioPlaybackManager

        fps   = self.fps
        speed = self.speed

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            cap.release()
            self.finished.emit()
            return

        # Construir AudioPlaybackManager DENTRO de un guard: si su __init__
        # lanza (p.ej. mpv no encontrado, pyaudio falla) nos aseguramos de
        # liberar `cap` antes de salir.
        try:
            audio = AudioPlaybackManager(self.video_path, fps, speed, prefer_mpv=self.has_audio)
        except Exception as exc:
            cap.release()
            print(f"[PlaybackWorker] AudioPlaybackManager init falló: {exc}")
            self.finished.emit()
            return
        self._audio = audio

        try:
            start_pos  = self.start_frame / fps
            half_frame = 1.0 / (fps * 2.0 * speed)

            # ── FASE 1: arrancar mpv PAUSADO, esperar primer time_pos ─────────────
            # play() inicia mpv en pausa → sin flash de audio antes del seek.
            audio.play(start_pos)

            seen_first_pos = False
            deadline = time.monotonic() + self.MPV_START_TIMEOUT

            while not self._stop and time.monotonic() < deadline:
                t = audio.time_pos
                if t is not None:
                    seen_first_pos = True
                    break
                time.sleep(0.02)

            if not seen_first_pos:
                # mpv arrancó pero nunca devolvió time_pos → sin audio (pista ausente,
                # formato no soportado, etc.). Cambiar a InternalClock sin interrumpir.
                audio.fallback_to_clock(start_pos)

            # ── FASE 2: seek de inicio MIENTRAS está pausado, luego despausar ─────
            if audio.using_mpv:
                seek_ok = True
                if self.start_frame > 0:
                    seek_ok = False
                    if not audio.seek(start_pos):
                        # seek() lanzó excepción interna → fallback directo a reloj.
                        audio.fallback_to_clock(start_pos)
                        seek_ok = True   # InternalClock ya está corriendo desde start_pos
                    else:
                        # Confirmar el seek (margen ≤ 0.15 s para no empezar desincronizado)
                        seek_deadline = time.monotonic() + self.MPV_SEEK_TIMEOUT
                        while not self._stop and time.monotonic() < seek_deadline:
                            t = audio.time_pos
                            if t is not None and abs(t - start_pos) < 0.15:
                                seek_ok = True
                                break
                            time.sleep(0.015)
                if seek_ok and audio.using_mpv:
                    # Despausar: el audio empieza a sonar exactamente en start_pos
                    audio.resume()
                elif audio.using_mpv:
                    # mpv no confirmó el salto → preview en silencio para no desincronizar
                    audio.fallback_to_clock(start_pos)

            # Pre-posicionar OpenCV al mismo punto
            if self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            last_fi     = self.start_frame - 1
            none_streak = 0  # iteraciones consecutivas con time_pos == None

            # ── FASE 3: bucle principal de reproducción ───────────────────────────
            while not self._stop:
                # Seek externo solicitado desde el hilo UI (thread-safe via queue)
                try:
                    target_seek = self._seek_queue.get_nowait()
                    if self.frame_count > 0:
                        target_seek = min(target_seek, self.frame_count - 1)
                    target_pos = target_seek / fps
                    seek_ok = audio.seek(target_pos)
                    if not seek_ok and audio.using_mpv:
                        # seek() lanzó excepción → sincronizar con reloj interno.
                        audio.fallback_to_clock(target_pos)
                    elif audio.using_mpv:
                        seek_deadline = time.monotonic() + self.MPV_SEEK_TIMEOUT
                        _seek_confirmed = False
                        while not self._stop and time.monotonic() < seek_deadline:
                            confirmed_pos = audio.time_pos
                            if confirmed_pos is not None and abs(confirmed_pos - target_pos) < 0.15:
                                _seek_confirmed = True
                                break
                            time.sleep(0.015)
                        if not _seek_confirmed and audio.using_mpv:
                            # mpv aceptó el seek pero nunca confirmó posición → desync.
                            # Fallback a reloj para mantener imagen sincronizada.
                            print(f"[PlaybackWorker] seek a {target_pos:.2f}s no confirmado — fallback a reloj")
                            audio.fallback_to_clock(target_pos)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_seek)
                    last_fi     = target_seek - 1
                    none_streak = 0
                    continue
                except _queue.Empty:
                    pass

                t_pos = audio.time_pos

                if t_pos is None:
                    if seen_first_pos and audio.using_mpv:
                        # time_pos puede ser None brevemente durante buffering o seek interno.
                        # Solo es EOF real si persiste más de ~200 ms (10 ciclos × 20 ms).
                        none_streak += 1
                        if none_streak >= 10:
                            break  # EOF confirmado
                        time.sleep(0.02)
                        continue
                    # InternalClock: nunca None después de play(); mpv aún no arrancó.
                    time.sleep(half_frame)
                    continue

                none_streak    = 0
                seen_first_pos = True
                target_fi      = int(t_pos * fps)

                # EOF por frame_count (para InternalClock que no reporta None al final)
                if self.frame_count > 0 and target_fi >= self.frame_count:
                    break

                gap = target_fi - last_fi

                if gap <= 0:
                    # El reloj de audio aún no avanzó lo suficiente para el siguiente frame
                    time.sleep(half_frame)
                    continue

                if gap > self.MAX_LAG_FRAMES:
                    # Muy retrasados: seek directo, saltamos frames intermedios
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
                elif gap > self.SEQUENTIAL_THRESHOLD:
                    # Salto moderado: seek OpenCV
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
                else:
                    # Lectura secuencial: grab() sin decodificar los frames intermedios
                    for _ in range(gap - 1):
                        if not cap.grab():
                            break

                ret, frame = cap.read()
                if not ret:
                    break

                frame   = VideoReader._rotate(frame, self.rotation)
                last_fi = target_fi

                if not self._stop:
                    self.frame_ready.emit(frame.copy(), target_fi)

        finally:
            # Garantizar limpieza aunque se lance una excepción en el bucle
            audio.terminate()
            cap.release()
        self.finished.emit()


# ─── Workers de análisis y render ────────────────────────────────────────────

class WarmupWorker(QObject):
    done  = pyqtSignal(str)
    error = pyqtSignal(str)

    def run(self):
        try:
            prov = init_detector(cfg.get("model_name"), cfg.get("execution_provider"))
            # Pre-warm the encoder cache here (worker thread) so that _on_warmup_done
            # can read it from cache without blocking the UI thread (verify_encoder can
            # take several seconds the first time it spawns ffmpeg test processes).
            try:
                from core.ffmpeg_utils import best_encoder, find_ffmpeg as _ff
                ff = _ff()
                if ff:
                    best_encoder(ff)
            except Exception:
                pass
            self.done.emit(prov)
        except Exception as e:
            self.error.emit(str(e))


class LoadVideoWorker(QObject):
    finished = pyqtSignal(str, object, object)  # path, info, reader
    error = pyqtSignal(str, str)

    def __init__(self, path: str, ffmpeg_path: str | None):
        super().__init__()
        self.path = path
        self.ffmpeg_path = ffmpeg_path

    def run(self):
        try:
            info = probe_video(self.path, self.ffmpeg_path)
            # Pasar frame_count_hint para evitar la corrección seek-and-read
            # duplicada: probe_video() ya la ejecutó, VideoReader no debe repetirla.
            reader = VideoReader(
                self.path,
                rotation=info.get("rotation", 0),
                frame_count_hint=info.get("frame_count"),
            )
            self.finished.emit(self.path, info, reader)
        except Exception as exc:
            self.error.emit(self.path, str(exc))


class DiagnosticsWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, builder):
        super().__init__()
        self.builder = builder

    def run(self):
        try:
            self.finished.emit(self.builder())
        except Exception as exc:
            self.error.emit(str(exc))


class AnalysisWorker(QObject):
    progress  = pyqtSignal(int, int)
    eta       = pyqtSignal(str)    # "N.N frames/s · ETA Xm Ys"
    finished  = pyqtSignal(object, dict)
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()
    checkpoint= pyqtSignal(int)   # frame_idx donde se guardó checkpoint

    def __init__(self, video_path: str, step: int, min_score: float, chk_every: int):
        super().__init__()
        self.video_path = video_path
        self.step       = step
        self.min_score  = min_score
        self.chk_every  = chk_every
        self._cancel    = False

    def cancel(self):
        self._cancel = True

    def run(self):
        import queue as _queue_mod
        reader = None
        producer_thread = None
        _stop_producer = threading.Event()
        try:
            info   = probe_video(self.video_path, cfg.get("ffmpeg_path") or None)
            # frame_count_hint evita la doble corrección seek-and-read:
            # probe_video() ya la ejecutó, VideoReader no debe repetirla.
            reader = VideoReader(
                self.video_path,
                rotation=info.get("rotation", 0),
                frame_count_hint=info.get("frame_count"),
            )
            tracker = FaceTracker(
                similarity_threshold=cfg.get("similarity_threshold"),
                iou_threshold       =cfg.get("iou_threshold"),
                max_interp_gap      =cfg.get("max_interp_gap"),
            )

            # ── Pipeline produce-consume ──────────────────────────────────────
            # Productor: itera frames con OpenCV (I/O bound).
            # Consumidor (este hilo): detect_faces() + tracker (GPU/CPU bound).
            # La queue de tamaño 4 solapa lectura con inferencia sin acumular RAM.
            frame_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=4)
            # Para scene cut detection necesitamos el histograma del frame anterior;
            # usamos una lista de un elemento (mutable closure compartida entre hilos).
            _prev_hist_box: list = [None]

            def _producer():
                try:
                    for fi, frame in reader.iter_frames(step=self.step):
                        if _stop_producer.is_set():
                            break
                        # ── Detección de corte de escena (Bhattacharyya) ──────
                        # Compara el histograma de luminancia del frame actual con
                        # el anterior. Un valor > 0.45 indica cambio brusco de escena.
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                        cv2.normalize(hist, hist)
                        prev = _prev_hist_box[0]
                        if prev is not None:
                            diff = cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)
                            if diff > 0.45:
                                tracker.scene_cuts.add(fi)
                        _prev_hist_box[0] = hist
                        frame_queue.put((fi, frame))
                finally:
                    frame_queue.put(None)   # centinela EOF

            producer_thread = threading.Thread(
                target=_producer, daemon=True, name="FrameProducer"
            )
            producer_thread.start()

            t_start     = time.perf_counter()
            frames_done = 0
            total_iters = max(1, reader.frame_count // max(1, self.step))
            last_checkpoint = 0

            while True:
                if self._cancel:
                    _stop_producer.set()
                    self.cancelled.emit()
                    return

                try:
                    item = frame_queue.get(timeout=0.1)
                except _queue_mod.Empty:
                    continue

                if item is None:   # centinela EOF del productor
                    break

                fi, frame = item
                dets = detect_faces(frame, min_det_score=self.min_score)
                tracker.process_frame(fi, dets, frame)
                frames_done += 1
                self.progress.emit(fi, reader.frame_count)

                elapsed = time.perf_counter() - t_start
                if elapsed > 2.0 and frames_done > 1:
                    rate = frames_done / elapsed
                    remaining = max(0, total_iters - frames_done)
                    eta_s = int(remaining / rate)
                    self.eta.emit(f"Analizando… {rate:.1f} fr/s · ETA {_fmt_eta(eta_s)}")

                # Checkpoint parcial
                if fi - last_checkpoint >= self.chk_every * self.step:
                    try:
                        session_mod.save_session(self.video_path, tracker, info, self.step)
                        self.checkpoint.emit(fi)
                        last_checkpoint = fi
                    except Exception as e:
                        self.eta.emit(f"No se pudo guardar checkpoint: {e}")

            # Asegurarse de que el productor termina antes de interpolate_bboxes()
            # para que scene_cuts esté completo al usarlo.
            producer_thread.join(timeout=5)
            producer_thread = None
            tracker.interpolate_bboxes()

            # ── Post-procesado: consolida identidades fragmentadas y descarta fantasmas ──
            n_before = len(tracker.persons)
            merges  = tracker.consolidate_persons(sim_threshold=cfg.get("similarity_threshold"))
            removed = tracker.prune_persons(min_real_frames=5)
            n_after = len(tracker.persons)
            if merges or removed:
                self.eta.emit(
                    f"Post-procesado: {merges} fusiones, {removed} fantasmas eliminados "
                    f"({n_before} → {n_after} personas)"
                )
            try:
                session_mod.save_session(self.video_path, tracker, info, self.step)
            except Exception as e:
                self.eta.emit(f"Aviso: no se pudo guardar la sesión ({e})")

            self.finished.emit(tracker, info)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Señalar al productor que debe parar y esperar a que termine
            # antes de liberar el reader (el productor usa reader.iter_frames).
            _stop_producer.set()
            if producer_thread is not None:
                producer_thread.join(timeout=5)
            if reader is not None:
                reader.release()


class RenderWorker(QObject):
    progress         = pyqtSignal(int, int)
    finished         = pyqtSignal(str)
    error            = pyqtSignal(str)
    cancelled        = pyqtSignal()
    warning          = pyqtSignal(str)
    encoder_selected = pyqtSignal(str)   # emite el nombre del encoder antes de empezar

    def __init__(self, input_path, output_path, persons_config, frame_data, video_info):
        super().__init__()
        self.input_path     = input_path
        self.output_path    = output_path
        self.persons_config = persons_config
        self.frame_data     = frame_data
        self.video_info     = video_info
        self._cancel        = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from core.renderer import render_video
            from core.ffmpeg_utils import RenderCancelled, assert_ffmpeg, best_encoder

            # ── Detectar y anunciar el encoder ANTES de empezar ───────────────
            ff = assert_ffmpeg(cfg.get("ffmpeg_path") or None)
            if ff and cfg.get("use_hw_encode"):
                codec, _ = best_encoder(ff)           # llama verify_encoder internamente
                is_hw = codec not in ("libx264", "libx265")
                label = f"{codec.upper()} (GPU)" if is_hw else "libx264 (CPU — no se detectó encoder HW)"
            else:
                label = f"libx264 CRF {cfg.get('crf')} (CPU)"
            self.encoder_selected.emit(label)

            out = render_video(
                input_path      =self.input_path,
                output_path     =self.output_path,
                persons_config  =self.persons_config,
                frame_data      =self.frame_data,
                video_info      =self.video_info,
                progress_callback=lambda c, t: self.progress.emit(c, t),
                cancel_callback =lambda: self._cancel,
                warning_callback=lambda msg: self.warning.emit(msg),
            )
            self.finished.emit(out)
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class RegroupWorker(QObject):
    finished = pyqtSignal(object, dict, int, int)
    error = pyqtSignal(str)

    def __init__(self, tracker, video_path: str, video_info: dict, step: int, sim_threshold: float):
        super().__init__()
        self.tracker = tracker
        self.video_path = video_path
        self.video_info = dict(video_info)
        self.step = step
        self.sim_threshold = sim_threshold

    def run(self):
        try:
            tracker = self.tracker
            tracker.interpolate_bboxes()
            merges = tracker.consolidate_persons(self.sim_threshold)
            removed = tracker.prune_persons(min_real_frames=5)
            session_mod.save_session(self.video_path, tracker, self.video_info, self.step)
            self.finished.emit(tracker, self.video_info, merges, removed)
        except Exception as e:
            self.error.emit(str(e))


# ─── Diálogo de fusión ───────────────────────────────────────────────────────

class MergePersonsDialog(QDialog):
    """
    Diálogo visual para fusionar dos personas.
    Muestra miniatura + nombre + nº de frames de cada candidato.
    Verde = persona BASE que se conserva. Rojo = persona que desaparece.
    """

    _CARD_A = (
        "QFrame { border: 1px solid #48ff70; border-radius: 8px; background: #202026; }"
    )
    _CARD_B = (
        "QFrame { border: 1px solid #ff6666; border-radius: 8px; background: #202026; }"
    )
    _CARD_SAME = (
        "QFrame { border: 2px solid #888; border-radius: 8px; background: #1e1e22; }"
    )

    def __init__(self, tracker, parent=None):
        super().__init__(parent)
        self.tracker = tracker
        self._persons = sorted(tracker.persons, key=lambda p: (-p.frame_count, p.person_id))
        self._pid_a: int | None = None
        self._pid_b: int | None = None
        self.setWindowTitle("Fusionar personas")
        self.setMinimumWidth(500)
        self.setModal(True)
        apply_dark_title_bar(self)
        self.setStyleSheet(
            "QDialog  { background: #1b1b20; }"
            "QLabel   { color: #e0e0e0; background: transparent; border: none; }"
            "QComboBox { background: #24242a; border: 1px solid #4a4a54; "
            "  color: #f2f2f2; border-radius: 5px; padding: 5px 8px; font-size: 12px; }"
            "QComboBox:hover { border-color: #707078; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #25252a; color: #e0e0e0; "
            "  selection-background-color: #3a3a60; }"
        )
        self._build_ui()

    @staticmethod
    def _pname(p, idx: int) -> str:
        return getattr(p, "_custom_label", None) or f"Persona {idx + 1}"

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(9)
        root.setContentsMargins(16, 14, 16, 14)

        # ── Header ─────────────────────────────────────────────────────────────
        lbl_title = QLabel("Fusionar personas")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        root.addWidget(lbl_title)

        lbl_hint = QLabel("Elige qué identidad se conserva y cuál se absorbe en la fusión.")
        lbl_hint.setStyleSheet("font-size: 11px; color: #a8a8b0;")
        root.addWidget(lbl_hint)

        # ── Two-column panel ───────────────────────────────────────────────────
        cols = QHBoxLayout()
        cols.setSpacing(12)

        # Column A ─ keep
        col_a = QVBoxLayout()
        col_a.setSpacing(6)
        lbl_a = QLabel("SE CONSERVA")
        lbl_a.setStyleSheet("font-size: 10px; font-weight: bold; color: #e8e8ee; letter-spacing: 1px;")
        col_a.addWidget(lbl_a)
        self._cmb_a = QComboBox()
        for i, p in enumerate(self._persons):
            self._cmb_a.addItem(f"{self._pname(p, i)}  ({p.frame_count} frames)")
        col_a.addWidget(self._cmb_a)
        self._card_a = QFrame()
        self._card_a.setFixedHeight(132)
        self._card_a.setStyleSheet(self._CARD_A)
        self._thumb_a, self._name_a, self._frames_a = self._build_card(self._card_a)
        col_a.addWidget(self._card_a)

        # Center arrow
        col_c = QVBoxLayout()
        col_c.setSpacing(6)
        col_c.addSpacing(28)
        lbl_arrow = QLabel("→")
        lbl_arrow.setAlignment(Qt.AlignCenter)
        lbl_arrow.setFixedSize(48, 132)
        lbl_arrow.setStyleSheet(
            "font-size: 34px; color: #b7bac5; font-weight: bold; "
            "background: transparent; border: none;"
        )
        col_c.addWidget(lbl_arrow)

        # Column B ─ discard
        col_b = QVBoxLayout()
        col_b.setSpacing(6)
        lbl_b = QLabel("SE ELIMINA")
        lbl_b.setStyleSheet("font-size: 10px; font-weight: bold; color: #e8e8ee; letter-spacing: 1px;")
        col_b.addWidget(lbl_b)
        self._cmb_b = QComboBox()
        for i, p in enumerate(self._persons):
            self._cmb_b.addItem(f"{self._pname(p, i)}  ({p.frame_count} frames)")
        col_b.addWidget(self._cmb_b)
        self._card_b = QFrame()
        self._card_b.setFixedHeight(132)
        self._card_b.setStyleSheet(self._CARD_B)
        self._thumb_b, self._name_b, self._frames_b = self._build_card(self._card_b)
        col_b.addWidget(self._card_b)

        cols.addLayout(col_a)
        cols.addLayout(col_c)
        cols.addLayout(col_b)
        root.addLayout(cols)

        # ── Warning label ──────────────────────────────────────────────────────
        self._lbl_warn = QLabel()
        self._lbl_warn.setWordWrap(True)
        self._lbl_warn.setStyleSheet(
            "color: #ffcc44; font-size: 11px; "
            "background: #2a2200; border: 1px solid #554400; "
            "border-radius: 5px; padding: 6px 8px;"
        )
        self._lbl_warn.setVisible(False)
        root.addWidget(self._lbl_warn)

        # ── Buttons ────────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setFixedHeight(32)
        btn_cancel.setStyleSheet(
            "QPushButton { background:#25252a; border:1px solid #555; color:#aaa; "
            "border-radius:5px; padding:5px 18px; }"
            "QPushButton:hover { background:#35353b; color:#fff; }"
        )
        self._btn_merge = QPushButton("  Fusionar  ")
        self._btn_merge.setFixedHeight(32)
        self._btn_merge.setDefault(True)
        self._btn_merge.setStyleSheet(
            "QPushButton { background:#123018; border:1px solid #55ff78; color:#fff; "
            "border-radius:5px; padding:5px 18px; font-weight:bold; }"
            "QPushButton:hover { background:#184020; }"
            "QPushButton:disabled { background:#25252a; border-color:#3a3a40; color:#555; }"
        )
        btn_cancel.clicked.connect(self.reject)
        self._btn_merge.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self._btn_merge)
        root.addLayout(btn_row)

        # ── Initial state ──────────────────────────────────────────────────────
        self._cmb_a.currentIndexChanged.connect(self._refresh)
        self._cmb_b.currentIndexChanged.connect(self._refresh)
        if len(self._persons) > 1:
            self._cmb_b.blockSignals(True)
            self._cmb_b.setCurrentIndex(1)
            self._cmb_b.blockSignals(False)
        self._refresh()

    @staticmethod
    def _build_card(frame: QFrame):
        """Populate a card frame and return (thumb_lbl, name_lbl, frames_lbl)."""
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(8, 7, 8, 6)
        lay.setSpacing(2)
        thumb = QLabel()
        thumb.setFixedSize(72, 72)
        thumb.setAlignment(Qt.AlignCenter)
        thumb.setStyleSheet("border: none;")
        lay.addWidget(thumb, alignment=Qt.AlignCenter)
        name_lbl = QLabel()
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff; border: none;")
        lay.addWidget(name_lbl)
        frames_lbl = QLabel()
        frames_lbl.setAlignment(Qt.AlignCenter)
        frames_lbl.setStyleSheet("font-size: 10px; color: #888; border: none;")
        lay.addWidget(frames_lbl)
        return thumb, name_lbl, frames_lbl

    def _fill_card(self, thumb: QLabel, name_lbl: QLabel, frames_lbl: QLabel,
                   p, idx: int) -> None:
        from ui.person_card import _bgr_to_pixmap
        thumb.setPixmap(_bgr_to_pixmap(p.thumbnail, size=72))
        name_lbl.setText(self._pname(p, idx))
        frames_lbl.setText(f"{p.frame_count} frames  ·  ID {p.person_id}")

    def _refresh(self):
        idx_a = self._cmb_a.currentIndex()
        idx_b = self._cmb_b.currentIndex()
        pa = self._persons[idx_a] if 0 <= idx_a < len(self._persons) else None
        pb = self._persons[idx_b] if 0 <= idx_b < len(self._persons) else None

        if pa:
            self._fill_card(self._thumb_a, self._name_a, self._frames_a, pa, idx_a)
        if pb:
            self._fill_card(self._thumb_b, self._name_b, self._frames_b, pb, idx_b)

        self._pid_a = pa.person_id if pa else None
        self._pid_b = pb.person_id if pb else None
        same = (self._pid_a is not None and self._pid_a == self._pid_b)

        self._card_a.setStyleSheet(self._CARD_SAME if same else self._CARD_A)
        self._card_b.setStyleSheet(self._CARD_SAME if same else self._CARD_B)
        self._btn_merge.setEnabled(not same)

        if same:
            self._lbl_warn.setStyleSheet(
                "color: #ff8888; font-size: 11px; "
                "background: #2a1010; border: 1px solid #552222; "
                "border-radius: 4px; padding: 6px;"
            )
            self._lbl_warn.setText("⚠  Selecciona dos personas distintas.")
            self._lbl_warn.setVisible(True)
        elif pa and pb and self.tracker._has_temporal_conflict(pa, pb):
            self._lbl_warn.setStyleSheet(
                "color: #ffcc44; font-size: 11px; "
                "background: #2a2200; border: 1px solid #554400; "
                "border-radius: 4px; padding: 6px;"
            )
            self._lbl_warn.setText(
                "⚠  Estas personas aparecen en el mismo frame real. "
                "Fusionarlas puede mezclar identidades distintas — "
                "si continúas la fusión se forzará."
            )
            self._lbl_warn.setVisible(True)
        else:
            self._lbl_warn.setVisible(False)

    def result_pids(self) -> tuple[int, int, bool]:
        """Returns (pid_keep, pid_discard, force_merge)."""
        idx_a = self._cmb_a.currentIndex()
        idx_b = self._cmb_b.currentIndex()
        pa = self._persons[idx_a] if 0 <= idx_a < len(self._persons) else None
        pb = self._persons[idx_b] if 0 <= idx_b < len(self._persons) else None
        force = bool(pa and pb and self.tracker._has_temporal_conflict(pa, pb))
        return self._pid_a, self._pid_b, force


# ─── Main Window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    scrub_open_requested = pyqtSignal(str, int)
    scrub_frame_requested = pyqtSignal(int)
    _last_video_dir: str = ""   # persists across instances within the same session

    STYLE = """
        QMainWindow, QDialog { background: #1f1f23; }
        QWidget { background: #1f1f23; color: #f4f4f4; font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; }
        QPushButton {
            background: #2b2b31; color: #ffffff; border: 1px solid #74747c; border-radius: 4px;
            padding: 5px 12px;
        }
        QPushButton:hover   { background: #34343b; border-color: #ffffff; }
        QPushButton:pressed { background: #1a1a1e; color: #ffffff; }
        QPushButton:disabled { color: #8a8a8f; border-color: #3a3a40; background: #25252a; }
        QComboBox {
            background: #2a2a30; color: #ffffff; border: 1px solid #666; border-radius: 3px;
            padding: 3px 6px; min-height: 22px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { background: #2a2a30; color: #ffffff; selection-background-color: #3a6aee; }
        QSlider::groove:horizontal { height: 5px; background: #3a3a40; border-radius: 2px; }
        QSlider::handle:horizontal {
            width: 15px; height: 15px; background: #ffffff;
            border-radius: 7px; margin: -5px 0;
        }
        QSlider::sub-page:horizontal { background: #3a6aee; border-radius: 2px; }
        QProgressBar {
            background: #2a2a30; color: #ffffff; border: 1px solid #666; border-radius: 3px;
            text-align: center; font-size: 11px;
        }
        QProgressBar::chunk { background: #5588ee; border-radius: 2px; }
        QScrollArea { background: #25252a; border: none; }
        QScrollArea > QWidget > QWidget { background: #25252a; }
        QScrollBar {
            background: #202026;
            border: none;
        }
        QScrollBar:vertical {
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #6d6d76;
            border-radius: 5px;
            min-height: 28px;
        }
        QScrollBar::handle:vertical:hover {
            background: #8a8a94;
        }
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {
            background: #202026;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
            background: #202026;
            border: none;
        }
        QLabel { color: #ffffff; }
        QCheckBox { color: #ffffff; spacing: 6px; }
        QGroupBox { border: 1px solid #4a4a52; border-radius: 4px; margin-top: 6px; padding-top: 4px; color: #ffffff; }
        QGroupBox::title { color: #ffffff; subcontrol-origin: margin; left: 8px; }
        QMenuBar { background: #18181c; color: #ffffff; border-bottom: 1px solid #3f3f46; }
        QMenuBar::item { padding: 4px 8px; }
        QMenuBar::item:selected { background: #181818; }
        QMenu { background: #26262c; color: #ffffff; border: 1px solid #666; }
        QMenu::item:selected { background: #3a6aee; }
        QStatusBar { background: #18181c; color: #ffffff; border-top: 1px solid #3f3f46; font-size: 11px; }
        QSplitter::handle { background: #25252a; }
        QLineEdit { background: #2a2a30; color: #ffffff; border: 1px solid #666; border-radius: 3px; padding: 3px 6px; }
        QSpinBox, QDoubleSpinBox { background: #2a2a30; color: #ffffff; border: 1px solid #666; border-radius: 3px; padding: 2px 4px; }
        QToolTip {
            background: #2b2b31; color: #ffffff; border: 1px solid #8a8a92;
            padding: 4px 7px; opacity: 255;
        }
        QPushButton#topbarButton {
            background: #1f1f25;
            border-top: 1px solid #8a8a92;
            border-left: 1px solid #686872;
            border-right: 1px solid #686872;
            border-bottom: 2px solid #111116;
            color: #ffffff;
        }
        QPushButton#topbarButton:hover {
            background: #303038;
            border-top-color: #ffffff;
            border-left-color: #ffffff;
            border-right-color: #ffffff;
        }
        QPushButton#topbarButton:pressed {
            background: #17171c;
            border-top: 2px solid #0d0d10;
            border-bottom: 1px solid #777780;
            color: #ffffff;
        }
        QCheckBox#topbarCheck {
            background: #1f1f25;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px 8px;
            color: #ffffff;
        }
        QCheckBox#topbarCheck:hover {
            background: #303038;
            border-color: #ffffff;
        }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GhostFrame")
        apply_dark_title_bar(self)
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets",
            "ghostframe-app-icon.ico",
        )
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(1120, 720)
        self.setAcceptDrops(True)
        self.setStyleSheet(self.STYLE)

        # Estado
        self.video_path: str | None = None
        self.video_info: dict = {}
        self.tracker: FaceTracker | None = None
        self.person_cards: list[PersonCard] = []
        self._reader: VideoReader | None = None
        self._timeline: TimelineWidget | None = None
        self._show_detections = False
        self._undo_stack: list[FaceTracker] = []
        self._redo_stack: list[FaceTracker] = []
        self._history_disabled_reason: str = ""

        # Reproducción de preview — worker OpenCV con audio mpv como reloj maestro
        self._playing     = False
        self._was_playing = False   # True si estaba reproduciendo antes de un scrub manual
        self._playback_worker: "PlaybackWorker | None" = None
        self._playback_thread: "QThread | None" = None
        self._last_raw_frame = None   # último frame BGR crudo recibido del worker
        self._close_after_playback_stop = False
        self._user_scrubbing = False
        self._pending_playback_seek_fi: int | None = None
        self._pending_playback_seek_token = 0
        self._scrub_preview_pending = False

        # Scrub worker — lectura de frames en hilo propio (evita congelar la UI con HEVC)
        self._scrub_target_fi: int = 0
        self._scrub_debounce = QTimer(self)
        self._scrub_debounce.setSingleShot(True)
        self._scrub_debounce.timeout.connect(self._do_scrub_frame_read)

        self._scrub_worker = ScrubWorker()
        self._scrub_thread = QThread(self)
        self._scrub_worker.moveToThread(self._scrub_thread)
        self.scrub_open_requested.connect(self._scrub_worker.open_video)
        self.scrub_frame_requested.connect(self._scrub_worker.read_frame)
        self._scrub_worker.frame_ready.connect(self._on_scrub_frame_received)
        self._scrub_thread.finished.connect(self._scrub_worker.deleteLater)
        self._scrub_thread.start()

        # Threads (guardados como attrs para evitar GC → crash)
        self._warmup_thread: QThread | None = None
        self._warmup_worker: WarmupWorker | None = None
        self._load_video_thread: QThread | None = None
        self._load_video_worker: LoadVideoWorker | None = None
        self._diagnostics_thread: QThread | None = None
        self._diagnostics_worker: DiagnosticsWorker | None = None
        self._analysis_thread: QThread | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._regroup_thread: QThread | None = None
        self._regroup_worker: RegroupWorker | None = None
        self._render_thread: QThread | None = None
        self._render_worker: RenderWorker | None = None
        self._active_encoder: str = ""  # encoder activo durante el render
        self._close_after_render_cancel = False
        self._close_after_analysis_cancel = False
        self._close_after_regroup = False

        self._check_ffmpeg()
        self._build_menu()
        self._build_ui()
        self._build_shortcuts()
        self._update_action_state()
        self._warmup_model()
        self._check_mpv_preview_audio()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _check_ffmpeg(self):
        if not find_ffmpeg():
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(600, self._warn_ffmpeg)

    def _warn_ffmpeg(self):
        QMessageBox.warning(
            self, "FFmpeg no encontrado",
            "FFmpeg no está instalado o no se encontró en el PATH.\n\n"
            "Sin FFmpeg no podrás renderizar el vídeo final.\n\n"
            "1. Descarga FFmpeg: https://ffmpeg.org/download.html\n"
            "2. Descomprime y añade la carpeta bin\\ al PATH de Windows,\n"
            "   o configura la ruta en Herramientas → Configuración → FFmpeg.\n\n"
            "Versión recomendada: ffmpeg-release-essentials (win64)"
        )

    def _check_mpv_preview_audio(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isfile(os.path.join(root, "libmpv-2.dll")):
            self.status_bar.showMessage(
                "Aviso: libmpv-2.dll no está junto a main.py; el preview puede reproducirse sin audio. El render conserva audio.",
                12000,
            )

    def _warmup_model(self):
        worker = WarmupWorker()
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_warmup_done)
        worker.error.connect(lambda e: self.status_bar.showMessage(
            f"Aviso InsightFace: {e[:100]}"
        ))
        worker.done.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_warmup_finished)
        thread.start()
        self._warmup_thread = thread
        self._warmup_worker = worker

    def _on_warmup_finished(self):
        self._warmup_thread = None
        self._warmup_worker = None

    def _on_warmup_done(self, prov: str):
        print(f"[Inicio] InsightFace listo · proveedor: {prov}")
        hw_info = "  ·  encoder: ver Ayuda > Diagnóstico"

        # Construir etiqueta de proveedor de análisis
        prov_label = prov.upper()
        if prov == "cuda":
            prov_label = "CUDA OK GPU"
        elif prov == "directml":
            prov_label = "DirectML OK GPU"
        elif prov == "cpu":
            if gpu_fallback_occurred():
                prov_label = "CPU (AVISO: GPU solicitada pero no disponible)"
            else:
                prov_label = "CPU"

        self.status_bar.showMessage(
            f"InsightFace listo  [{active_model()} / {prov_label}]{hw_info}"
        )

        # Si el análisis cayó a CPU sin que el usuario lo pidiera, mostrar aviso
        if gpu_fallback_occurred():
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(1500, self._warn_gpu_fallback)

    def _warn_gpu_fallback(self):
        QMessageBox.warning(
            self, "Análisis en CPU (GPU no disponible)",
            "Se intentó iniciar InsightFace con GPU pero falló. "
            "El análisis de caras correrá en CPU.\n\n"
            "Para usar GPU:\n"
            "  • GPU NVIDIA (más rápido):\n"
            "    pip uninstall onnxruntime -y\n"
            "    pip install onnxruntime-gpu==1.19.2\n\n"
            "  • GPU NVIDIA/AMD/Intel (DirectML):\n"
            "    pip uninstall onnxruntime -y\n"
            "    pip install onnxruntime-directml==1.19.2\n\n"
            "Luego en Herramientas → Configuración → Proveedor ONNX\n"
            "selecciona 'cuda' o 'directml' y reinicia el programa."
        )

    # ── Menú ─────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        fm = mb.addMenu("Archivo")
        self._act_export_cfg = None
        self._act_import_cfg = None
        for label, shortcut, slot in [
            ("Abrir vídeo…", "Ctrl+O", self._open_video_dialog),
            ("Exportar log de detección…", "", self._export_log),
            ("Exportar configuración…", "", self._export_censure_config),
            ("Importar configuración…", "", self._import_censure_config),
            ("Borrar caché de sesión", "", self._delete_cache),
        ]:
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            fm.addAction(a)
            if slot == self._export_censure_config:
                self._act_export_cfg = a
            elif slot == self._import_censure_config:
                self._act_import_cfg = a

        em = mb.addMenu("Editar")
        self._act_undo = QAction("Deshacer", self)
        self._act_undo.triggered.connect(self._undo)
        em.addAction(self._act_undo)
        self._act_redo = QAction("Rehacer", self)
        self._act_redo.triggered.connect(self._redo)
        em.addAction(self._act_redo)

        tm = mb.addMenu("Herramientas")
        # Guardamos referencias a las acciones que se deshabilitan durante el render
        self._act_settings = None
        self._act_batch    = None
        self._act_regroup  = None
        for label, shortcut, slot in [
            ("Fusionar personas…", "", self._merge_persons),
            ("Dividir persona…",   "", self._split_person),
            ("Re-agrupar personas", "", self._start_regroup),
            ("Guardar frame actual…", "Ctrl+S", self._export_frame),
            ("—", "", None),
            ("Modo batch (varios vídeos)…", "", self._open_batch),
            ("Configuración…", "Ctrl+,", self._open_settings),
        ]:
            if label == "—":
                tm.addSeparator()
                continue
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            tm.addAction(a)
            if slot == self._open_settings:
                self._act_settings = a
            elif slot == self._open_batch:
                self._act_batch = a
            elif slot == self._start_regroup:
                self._act_regroup = a

        hm = mb.addMenu("Ayuda")
        act_manual = QAction("Manual y atajos", self)
        act_manual.setShortcut("F1")
        act_manual.triggered.connect(self._show_manual)
        hm.addAction(act_manual)
        act_diag = QAction("Diagnóstico", self)
        act_diag.triggered.connect(self._show_diagnostics)
        hm.addAction(act_diag)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._redo)
        sc_play = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc_play.activated.connect(self._toggle_play_from_shortcut)

    @staticmethod
    def _deepcopy_async(obj, timeout_s: float = 5.0):
        """
        Deepcopy en hilo de fondo mientras se bombean eventos Qt.
        Evita congelar la UI durante la copia de trackers grandes.
        Devuelve la copia o lanza la excepción original si falla.
        Si el deepcopy tarda más de timeout_s, lanza MemoryError para que el
        caller deshabilite el historial en lugar de bloquear indefinidamente.
        """
        result: list = [None]
        exc:    list = [None]

        def _do():
            try:
                result[0] = copy.deepcopy(obj)
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        elapsed = 0.0
        chunk   = 0.05
        while t.is_alive() and elapsed < timeout_s:
            t.join(timeout=chunk)
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents | QEventLoop.ExcludeSocketNotifiers)
            elapsed += chunk
        if t.is_alive():
            raise MemoryError(
                f"deepcopy tardó más de {timeout_s:.0f}s — tracker demasiado grande para historial."
            )
        if exc[0] is not None:
            raise exc[0]
        return result[0]

    @staticmethod
    def _estimate_tracker_mb(tracker) -> float:
        """Estimación rápida del tamaño en MB de un FaceTracker."""
        total = 0
        for p in tracker.persons:
            total += sys.getsizeof(p) + sys.getsizeof(p.frame_data) + sys.getsizeof(p.embeddings)
            for emb in p.embeddings:
                total += getattr(emb, "nbytes", 512 * 4)
            # frame_data almacena dict + bbox/list por frame; usar margen alto para no
            # permitir copias que luego disparen RAM real.
            total += len(p.frame_data) * 700
            if p.thumbnail is not None:
                total += p.thumbnail.nbytes
        return total / (1024 * 1024)

    def _push_undo(self):
        if not self.tracker:
            return
        estimated_mb = self._estimate_tracker_mb(self.tracker)
        if estimated_mb > 120:
            self._history_disabled_reason = (
                f"Historial desactivado: el análisis ocupa ~{estimated_mb:.0f} MB por copia."
            )
            self.status_bar.showMessage(self._history_disabled_reason)
            self._update_action_state()
            return
        try:
            snapshot = self._deepcopy_async(self.tracker)
        except MemoryError:
            self.status_bar.showMessage("Historial desactivado: RAM insuficiente para guardar el estado.")
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._history_disabled_reason = "Historial desactivado: RAM insuficiente para guardar el estado."
            self._update_action_state()
            return
        self._history_disabled_reason = ""
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 15:
            self._undo_stack = self._undo_stack[-15:]
        self._redo_stack.clear()
        self._update_action_state()

    def _undo(self):
        if not self._undo_stack or not self.tracker:
            return
        # Guardar snapshot del estado actual como redo — solo si el tracker cabe en RAM
        estimated_mb = self._estimate_tracker_mb(self.tracker)
        if estimated_mb <= 120:
            try:
                snap = self._deepcopy_async(self.tracker)
                self._redo_stack.append(snap)
                if len(self._redo_stack) > 15:
                    self._redo_stack = self._redo_stack[-15:]
            except MemoryError:
                self.status_bar.showMessage("No hay RAM suficiente para guardar estado de rehacer.")
                self._redo_stack.clear()
        self.tracker = self._undo_stack.pop()
        self._populate_persons()
        self._autosave_session("Deshacer")
        self._update_action_state()

    def _redo(self):
        if not self._redo_stack or not self.tracker:
            return
        # Guardar snapshot del estado actual como undo — solo si el tracker cabe en RAM
        estimated_mb = self._estimate_tracker_mb(self.tracker)
        if estimated_mb <= 120:
            try:
                snap = self._deepcopy_async(self.tracker)
                self._undo_stack.append(snap)
                if len(self._undo_stack) > 15:
                    self._undo_stack = self._undo_stack[-15:]
            except MemoryError:
                self.status_bar.showMessage("No hay RAM suficiente para guardar estado de deshacer.")
                self._undo_stack.clear()
        self.tracker = self._redo_stack.pop()
        self._populate_persons()
        self._autosave_session("Rehacer")
        self._update_action_state()

    def _clear_history(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_action_state()

    def _update_action_state(self):
        has_tracker = self.tracker is not None
        is_analysis = bool(self._analysis_thread and self._analysis_thread.isRunning())
        is_render = bool(self._render_thread and self._render_thread.isRunning())
        is_regroup = bool(self._regroup_thread and self._regroup_thread.isRunning())
        if getattr(self, "_act_undo", None):
            self._act_undo.setEnabled(bool(self._undo_stack))
            self._act_undo.setToolTip(self._history_disabled_reason)
        if getattr(self, "_act_redo", None):
            self._act_redo.setEnabled(bool(self._redo_stack))
            self._act_redo.setToolTip(self._history_disabled_reason)
        if getattr(self, "_act_export_cfg", None):
            self._act_export_cfg.setEnabled(has_tracker)
        if getattr(self, "_act_import_cfg", None):
            self._act_import_cfg.setEnabled(has_tracker)
        can_regroup = has_tracker and not is_analysis and not is_render and not is_regroup
        can_merge = can_regroup and len(self.tracker.persons) >= 2 if self.tracker else False
        if getattr(self, "_act_regroup", None):
            self._act_regroup.setEnabled(can_regroup)
        if getattr(self, "btn_merge", None):
            self.btn_merge.setEnabled(can_merge)

    def _question_es(self, title: str, text: str, yes: str = "Sí", no: str = "No") -> bool:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle(title)
        msg.setText(text)
        apply_dark_title_bar(msg)
        btn_yes = msg.addButton(yes, QMessageBox.YesRole)
        msg.addButton(no, QMessageBox.NoRole)
        msg.setDefaultButton(btn_yes)
        msg.exec_()
        return msg.clickedButton() == btn_yes

    def _information_question_es(self, title: str, text: str, yes: str = "Sí", no: str = "No") -> bool:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(text)
        apply_dark_title_bar(msg)
        btn_yes = msg.addButton(yes, QMessageBox.YesRole)
        msg.addButton(no, QMessageBox.NoRole)
        msg.setDefaultButton(btn_yes)
        msg.exec_()
        return msg.clickedButton() == btn_yes

    def _toggle_detection_overlay(self, checked: bool):
        self._show_detections = checked
        if self._playing and self._last_raw_frame is not None and self._timeline:
            # Durante playback: recomponer el último frame sin tocar el player
            fi = self._timeline.current_frame()
            self.preview.show_frame(self._compose_preview_frame(self._last_raw_frame, fi))
        elif self._timeline:
            self._on_scrub(self._timeline.current_frame())

    def _toggle_play_from_shortcut(self):
        focused = self.focusWidget()
        if isinstance(focused, (QPushButton, QCheckBox, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox)):
            return
        self._toggle_play()

    def _show_manual(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Manual - GhostFrame Studio")
        apply_dark_title_bar(dlg)
        dlg.resize(760, 620)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet(
            "QTextBrowser { background: #050505; color: #f2f2f2; border: 1px solid #222; "
            "selection-background-color: #3a6aee; selection-color: #ffffff; }"
        )
        browser.setHtml("""
        <style>
          body {
            background: #050505;
            color: #f2f2f2;
            font-family: Segoe UI, Arial, sans-serif;
            font-size: 13px;
            line-height: 1.45;
          }
          h2 {
            color: #ffffff;
            margin: 0 0 14px 0;
            font-size: 22px;
          }
          h3 {
            color: #ffffff;
            margin-top: 18px;
            margin-bottom: 8px;
            font-size: 16px;
          }
          li {
            color: #e6e6e6;
            margin: 3px 0;
          }
          b {
            color: #ffffff;
          }
          a {
            color: #78a6ff;
          }
        </style>
        <h2>GhostFrame Studio - manual rapido</h2>
        <h3>Flujo basico</h3>
        <ol>
          <li><b>Abrir video</b>: carga el archivo que quieres censurar.</li>
          <li><b>Analizar caras</b>: detecta personas y crea las tarjetas de la derecha.</li>
          <li><b>Marca cada persona</b>: activa "Censurar esta persona" y elige efecto, intensidad y margen.</li>
          <li><b>Revisar</b>: usa la linea de tiempo, Play o Mostrar detecciones.</li>
          <li><b>Renderizar</b>: crea el video final conservando el audio.</li>
        </ol>
        <h3>Atajos</h3>
        <ul>
          <li><b>Espacio</b>: reproducir o pausar el preview.</li>
          <li><b>Ctrl+O</b>: abrir video.</li>
          <li><b>Ctrl+S</b>: guardar el frame actual como imagen.</li>
          <li><b>Ctrl+Z</b>: deshacer fusion/division de personas.</li>
          <li><b>Ctrl+Y</b> o <b>Ctrl+Shift+Z</b>: rehacer.</li>
          <li><b>F1</b>: abrir este manual.</li>
        </ul>
        <h3>Botones principales</h3>
        <ul>
          <li><b>Analizar caras</b>: busca caras con InsightFace. Usa GPU si el proveedor ONNX lo permite.</li>
          <li><b>Fusionar</b>: abre el diálogo para fusionar dos personas detectadas como si fueran la misma.</li>
          <li><b>Re-agrupar</b> (menú Herramientas): vuelve a juntar personas con los ajustes actuales sin analizar de nuevo el vídeo.</li>
          <li><b>Mostrar detecciones</b>: dibuja cajas de colores sobre las caras detectadas.</li>
          <li><b>Renderizar</b>: aplica la censura y guarda el video final. El encoder puede usar GPU.</li>
        </ul>
        <h3>Ajustes importantes</h3>
        <ul>
          <li><b>Analizar cada</b>: menos frames es mas preciso; mas frames es mas rapido.</li>
          <li><b>Umbral de similitud</b>: sube si mezcla personas; baja si divide a la misma persona.</li>
          <li><b>Confianza minima</b>: sube si salen falsos positivos; baja si faltan caras reales.</li>
          <li><b>Proveedor ONNX</b>: CUDA para NVIDIA, DirectML para otras GPU, CPU como fallback.</li>
          <li><b>Tamano detector</b>: 320 rapido para entrevistas; 640 mejor para caras pequenas.</li>
          <li><b>Encoder hardware</b>: usa GPU para guardar el video si NVENC/QSV/AMF funciona.</li>
          <li><b>CRF</b>: calidad del render por CPU. 18 es alta calidad; mas alto pesa menos.</li>
        </ul>
        <h3>Consejos</h3>
        <ul>
          <li>Si tienes NVIDIA, usa CUDA para deteccion y encoder hardware para render.</li>
          <li>Si una persona aparece duplicada, prueba Re-agrupar o fusiona manualmente.</li>
          <li>Si una persona se mezcla con otra, sube el umbral de similitud y reagrupa.</li>
          <li>El audio se reproduce en el preview si <b>libmpv-2.dll</b> está junto a main.py (ver README). El render siempre conserva el audio original.</li>
        </ul>
        """)
        layout.addWidget(browser)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignRight)
        dlg.exec_()

    def _show_diagnostics(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Diagnóstico - GhostFrame Studio")
        apply_dark_title_bar(dlg)
        dlg.resize(820, 620)
        layout = QVBoxLayout(dlg)

        browser = QTextBrowser()
        browser.setStyleSheet(
            "QTextBrowser { background: #101015; color: #f2f2f2; border: 1px solid #333; "
            "font-family: Consolas, 'Segoe UI', monospace; font-size: 12px; }"
        )
        browser.setPlainText("Calculando diagnóstico...")
        layout.addWidget(browser)

        row = QHBoxLayout()
        btn_refresh = QPushButton("Actualizar")
        row.addWidget(btn_refresh)
        row.addStretch()
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(dlg.accept)
        row.addWidget(btn_close)
        layout.addLayout(row)

        def safe_set_text(text: str):
            try:
                browser.setPlainText(text)
            except RuntimeError:
                pass

        def safe_enable_refresh(enabled: bool):
            try:
                btn_refresh.setEnabled(enabled)
            except RuntimeError:
                pass

        def start_refresh():
            if self._diagnostics_thread and self._diagnostics_thread.isRunning():
                return
            safe_enable_refresh(False)
            safe_set_text("Calculando diagnóstico...")
            worker = DiagnosticsWorker(self._build_diagnostics_text)
            thread = QThread(self)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(safe_set_text)
            worker.error.connect(lambda msg: safe_set_text(f"Diagnóstico falló:\n{msg}"))
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(lambda: safe_enable_refresh(True))
            thread.finished.connect(self._on_diagnostics_finished)
            self._diagnostics_worker = worker
            self._diagnostics_thread = thread
            thread.start()

        btn_refresh.clicked.connect(start_refresh)
        start_refresh()
        dlg.exec_()

    def _on_diagnostics_finished(self):
        self._diagnostics_worker = None
        self._diagnostics_thread = None

    def _build_diagnostics_text(self) -> str:
        import platform
        import site
        import subprocess

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.environ.get("GHOSTFRAME_LOG_PATH") or os.path.join(root, "logs", "ghostframe.log")
        lines = [
            "GhostFrame Studio - Diagnostico",
            "=" * 72,
            f"Python: {platform.python_version()}",
            f"Executable: {sys.executable}",
            f"Proyecto: {root}",
            f"Log: {log_path}",
            "",
            "Configuracion",
            "-" * 72,
            f"Proveedor configurado: {cfg.get('execution_provider')}",
            f"Modelo configurado: {cfg.get('model_name')}",
            f"Detector: {cfg.get('det_size')}",
            f"Analizar cada: {cfg.get('analysis_step')} frames",
            f"Encoder hardware: {cfg.get('use_hw_encode')}",
            f"FFmpeg path configurado: {cfg.get('ffmpeg_path') or '(auto)'}",
            "",
        ]

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except Exception as exc:
            providers = [f"ERROR: {exc}"]
        lines.extend([
            "ONNX Runtime",
            "-" * 72,
            "Providers disponibles:",
            *[f"  - {p}" for p in providers],
            f"Proveedor activo app: {active_provider() or '(no inicializado)'}",
            f"Modelo activo app: {active_model() or '(no inicializado)'}",
            f"Fallback GPU a CPU: {'SI' if gpu_fallback_occurred() else 'NO'}",
            "",
        ])

        try:
            from core.ffmpeg_utils import find_ffprobe, get_ffmpeg_version, best_encoder
            ffmpeg = find_ffmpeg()
            ffprobe = find_ffprobe(ffmpeg)
            lines.extend([
                "FFmpeg",
                "-" * 72,
                f"ffmpeg: {ffmpeg or 'NO ENCONTRADO'}",
                f"ffprobe: {ffprobe or 'NO ENCONTRADO'}",
                f"version: {get_ffmpeg_version(ffmpeg) if ffmpeg else 'unknown'}",
            ])
            if ffmpeg:
                hw = detect_hardware_encoders(ffmpeg)
                lines.append("Encoders listados:")
                for key, value in sorted(hw.items()):
                    lines.append(f"  - {key}: {'SI' if value else 'NO'}")
                try:
                    codec, args = best_encoder(ffmpeg)
                    lines.append(f"Mejor encoder verificado: {codec} {args}")
                except Exception as exc:
                    lines.append(f"Mejor encoder verificado: ERROR {exc}")
            lines.append("")
        except Exception as exc:
            lines.extend(["FFmpeg", "-" * 72, f"ERROR: {exc}", ""])

        lines.extend(["GPU Windows", "-" * 72])
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object Name,AdapterRAM,DriverVersion | Format-List",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            txt = (result.stdout or result.stderr or "").strip()
            lines.append(txt if txt else "Sin datos GPU.")
        except Exception as exc:
            lines.append(f"ERROR GPU: {exc}")
        lines.append("")

        lines.extend(["CUDA DLLs pip", "-" * 72])
        nvidia_dirs = [
            ("cuda_runtime", os.path.join("nvidia", "cuda_runtime", "bin")),
            ("cublas", os.path.join("nvidia", "cublas", "bin")),
            ("cudnn", os.path.join("nvidia", "cudnn", "bin")),
            ("cuda_nvrtc", os.path.join("nvidia", "cuda_nvrtc", "bin")),
            ("cusolver", os.path.join("nvidia", "cusolver", "bin")),
            ("cufft", os.path.join("nvidia", "cufft", "bin")),
            ("curand", os.path.join("nvidia", "curand", "bin")),
            ("cusparse", os.path.join("nvidia", "cusparse", "bin")),
            ("nvjitlink", os.path.join("nvidia", "nvjitlink", "bin")),
        ]
        bases = []
        try:
            bases.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            bases.append(site.getusersitepackages())
        except Exception:
            pass
        for name, rel in nvidia_dirs:
            found = next((os.path.join(base, rel) for base in bases if os.path.isdir(os.path.join(base, rel))), None)
            lines.append(f"{name}: {'OK - ' + found if found else 'FALTA'}")

        lines.extend([
            "",
            "Audio preview / mpv",
            "-" * 72,
            f"libmpv-2.dll junto a main.py: {'OK' if os.path.isfile(os.path.join(root, 'libmpv-2.dll')) else 'FALTA'}",
            "Si falta, el preview usa reloj interno sin audio; el render final conserva audio.",
            "",
            "Comandos utiles",
            "-" * 72,
            "pip install -r requirements.txt",
            "pip install onnxruntime-gpu==1.19.2",
            "pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12",
        ])
        return "\n".join(lines)

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_topbar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        # ── Panel izquierdo: preview + timeline ───────────────────────────────
        self._left = QWidget()
        self._left.setObjectName("leftPanel")
        self._left.setStyleSheet("QWidget#leftPanel { background: #1f1f23; border: none; }")
        ll = QVBoxLayout(self._left)
        ll.setContentsMargins(10, 8, 10, 6)
        ll.setSpacing(6)

        self.lbl_info = QLabel("Arrastra un vídeo aquí  o  usa 'Abrir vídeo'")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("color: #ffffff; font-size: 13px;")
        ll.addWidget(self.lbl_info)

        self.preview = PreviewWidget()
        ll.addWidget(self.preview, stretch=1)

        # Controles de reproducción
        play_bar = QHBoxLayout()
        play_bar.setContentsMargins(0, 0, 0, 0)

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setText("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.setFixedWidth(90)
        self.btn_play.setToolTip("Reproduce o pausa el preview. Atajo: Espacio.")
        self.btn_play.clicked.connect(self._toggle_play)
        play_bar.addWidget(self.btn_play)

        play_bar.addStretch()

        btn_export_frame = QPushButton("Guardar frame")
        btn_export_frame.setFixedWidth(130)
        btn_export_frame.clicked.connect(self._export_frame)
        play_bar.addWidget(btn_export_frame)

        ll.addLayout(play_bar)

        # Placeholder del timeline (se reemplaza al cargar vídeo)
        self._tl_placeholder = QLabel("─── Timeline ───")
        self._tl_placeholder.setAlignment(Qt.AlignCenter)
        self._tl_placeholder.setStyleSheet("color: #ffffff; font-size: 11px;")
        self._tl_placeholder.setFixedHeight(44)
        ll.addWidget(self._tl_placeholder)

        splitter.addWidget(self._left)

        # ── Panel derecho: personas ────────────────────────────────────────────
        right = QWidget()
        right.setObjectName("rightPanel")
        right.setMinimumWidth(284)
        right.setMaximumWidth(320)
        right.setStyleSheet("QWidget#rightPanel { background: #25252a; border: none; }")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 10, 8, 6)
        rl.setSpacing(4)

        hdr = QLabel("Personas detectadas")
        hdr.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff; background: transparent; border: none;")
        rl.addWidget(hdr)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setViewportMargins(0, 0, 3, 0)
        self.persons_container = QWidget()
        self.persons_layout = QVBoxLayout(self.persons_container)
        self.persons_layout.setAlignment(Qt.AlignTop)
        self.persons_layout.setSpacing(8)
        self.persons_layout.setContentsMargins(2, 2, 14, 2)
        self.scroll.setWidget(self.persons_container)
        rl.addWidget(self.scroll)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([820, 300])
        root.addWidget(splitter, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(15)
        root.addWidget(self.progress_bar)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Iniciando…")

    def _make_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setMinimumHeight(86)
        bar.setStyleSheet(
            "background: #222228; border-top: 1px solid #4a4a52; "
            "border-bottom: 2px solid #4a4a52;"
        )
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 10, 14, 10)
        bl.setSpacing(8)

        title = QLabel("GhostFrame")
        title.setFixedSize(220, 64)
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets",
            "ghostframe-logo-topbar.png",
        )
        logo = QPixmap(logo_path)
        if not logo.isNull():
            title.setPixmap(
                logo.scaled(220, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        bl.addWidget(title)
        bl.addStretch()

        self.btn_open = QPushButton("Abrir vídeo")
        self.btn_open.setObjectName("topbarButton")
        self.btn_open.setMinimumHeight(30)
        self.btn_open.setToolTip("Elige el video que quieres censurar.")
        self.btn_open.clicked.connect(self._open_video_dialog)
        bl.addWidget(self.btn_open)

        self.btn_analyze = QPushButton("Analizar caras")
        self.btn_analyze.setObjectName("topbarButton")
        self.btn_analyze.setMinimumHeight(30)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setToolTip("Busca caras en el video y crea una tarjeta por persona.")
        self.btn_analyze.clicked.connect(self._start_analysis)
        bl.addWidget(self.btn_analyze)

        self.btn_merge = QPushButton("Fusionar")
        self.btn_merge.setObjectName("topbarButton")
        self.btn_merge.setMinimumHeight(30)
        self.btn_merge.setEnabled(False)
        self.btn_merge.setToolTip("Une manualmente dos tarjetas que son la misma persona.")
        self.btn_merge.clicked.connect(self._merge_persons)
        bl.addWidget(self.btn_merge)

        self.chk_overlay = QCheckBox("Mostrar detecciones")
        self.chk_overlay.setObjectName("topbarCheck")
        self.chk_overlay.setToolTip("Muestra cajas de colores sobre las caras detectadas en el preview.")
        self.chk_overlay.toggled.connect(self._toggle_detection_overlay)
        bl.addWidget(self.chk_overlay)

        self.btn_render = QPushButton("Renderizar")
        self.btn_render.setObjectName("renderButton")
        self.btn_render.setMinimumHeight(30)
        self.btn_render.setEnabled(False)
        self.btn_render.setToolTip("Guarda el video final con la censura aplicada y el audio conservado.")
        self.btn_render.setStyleSheet(
            "QPushButton { background: #123018; border: 1px solid #4fff6a; "
            "color: #ffffff; border-radius: 4px; padding: 5px 14px; }"
            "QPushButton:hover { background: #184020; border-color: #ffffff; }"
            "QPushButton:pressed { background: #0e2413; color: #ffffff; }"
            "QPushButton:disabled { background: #25252a; border-color: #3a3a40; color: #8a8a8f; }"
        )
        self.btn_render.clicked.connect(self._start_render)
        bl.addWidget(self.btn_render)

        self.btn_cancel_op = QPushButton("Cancelar")
        self.btn_cancel_op.setObjectName("topbarButton")
        self.btn_cancel_op.setMinimumHeight(30)
        self.btn_cancel_op.setVisible(False)
        self.btn_cancel_op.setToolTip("Cancela el análisis o render en curso.")
        self.btn_cancel_op.setStyleSheet(
            "QPushButton { background: #301212; border: 1px solid #ff6a4f; "
            "color: #ffffff; border-radius: 4px; padding: 5px 14px; }"
            "QPushButton:hover { background: #401818; border-color: #ffffff; }"
            "QPushButton:pressed { background: #240e0e; color: #ffffff; }"
        )
        self.btn_cancel_op.clicked.connect(self._cancel_current_operation)
        bl.addWidget(self.btn_cancel_op)

        return bar

    # ── Drag & drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.splitext(p)[1].lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp", ".ts", ".flv"}:
                self._load_video(p)
                break

    # ── Carga de vídeo ────────────────────────────────────────────────────────

    def _open_video_dialog(self):
        if MainWindow._last_video_dir and os.path.isdir(MainWindow._last_video_dir):
            start_dir = MainWindow._last_video_dir
        else:
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            start_dir = downloads_dir if os.path.isdir(downloads_dir) else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar vídeo", start_dir,
            "Vídeos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.3gp *.ts *.flv);;Todos (*)"
        )
        if path:
            MainWindow._last_video_dir = os.path.dirname(path)
            self._load_video(path)

    def _load_video(self, path: str):
        if self._load_video_thread and self._load_video_thread.isRunning():
            QMessageBox.information(
                self, "Carga en curso",
                "Espera a que termine de abrirse el vídeo actual."
            )
            return
        if self._analysis_thread and self._analysis_thread.isRunning():
            QMessageBox.warning(
                self, "Análisis en curso",
                "Espera a que termine el análisis actual antes de cargar otro vídeo."
            )
            return
        if self._render_thread and self._render_thread.isRunning():
            QMessageBox.warning(
                self, "Render en curso",
                "Espera a que termine el render actual antes de cargar otro vídeo."
            )
            return
        # Parar el player AHORA para que el audio anterior no siga sonando
        # mientras carga el nuevo vídeo (no esperar a _on_video_loaded).
        if not self._close_player():
            QMessageBox.warning(
                self, "Reproducción activa",
                "No se pudo detener la reproducción a tiempo. Inténtalo de nuevo en unos segundos."
            )
            return
        self.status_bar.showMessage(f"Abriendo vídeo: {os.path.basename(path)}...")
        self.btn_analyze.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_render.setEnabled(False)
        worker = LoadVideoWorker(path, cfg.get("ffmpeg_path") or None)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_video_loaded)
        worker.error.connect(self._on_video_load_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_video_load_finished)
        self._load_video_worker = worker
        self._load_video_thread = thread
        thread.start()

    def _on_video_loaded(self, path: str, info: dict, reader: VideoReader):
        self.video_path = path
        self.video_info = info
        self._clear_history()
        if self._reader:
            self._reader.release()
        self._reader = reader
        if not self._close_player():
            self.status_bar.showMessage("Vídeo cargado, pero la reproducción anterior aún se está cerrando.")

        # Info detallada incluyendo flags iPhone
        dur   = info["duration"]
        h, rm = divmod(dur, 3600)
        m, s  = divmod(rm, 60)
        tags  = []
        if info.get("is_vfr"):
            tags.append("VFR→CFR")
        if info.get("rotation", 0) in (90, 270):
            tags.append(f"rotación {info['rotation']}°")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        orient = "▲" if info["orientation"] == "vertical" else "◀▶"

        self.lbl_info.setText(
            f"{os.path.basename(path)}{tag_str}   ·   "
            f"{info['width']}×{info['height']} {orient}   ·   "
            f"{info['fps']:.3f} fps   ·   "
            f"{int(h):02d}:{int(m):02d}:{s:04.1f}   ·   "
            f"{info['codec_video']} / {info['codec_audio']}   ·   "
            f"{info.get('file_size_mb', 0)} MB"
        )
        self.lbl_info.setStyleSheet("color: #aaa; font-size: 10px;")

        self.tracker = None
        self.btn_analyze.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_render.setEnabled(False)
        self._update_action_state()
        self._clear_persons()
        self._install_timeline(info)

        frame = self._reader.read_frame(0)
        if frame is not None:
            self.preview.show_frame(frame)

        self.status_bar.showMessage(f"Cargado: {os.path.basename(path)}")
        self.scrub_open_requested.emit(path, info.get("rotation", 0))
        self._try_load_session(path)

    def _on_video_load_error(self, path: str, msg: str):
        if self.video_path:
            self.status_bar.showMessage(
                f"No se pudo abrir {os.path.basename(path)}. "
                f"Se mantiene el vídeo anterior (pulsa Play para reanudar)."
            )
        else:
            self.status_bar.showMessage("No se pudo abrir el vídeo.")
        self.btn_analyze.setEnabled(self.tracker is not None)
        self.btn_play.setEnabled(self.video_path is not None)
        self.btn_render.setEnabled(self.tracker is not None)
        QMessageBox.critical(self, "Error", f"No se pudo abrir el vídeo:\n{msg}")

    def _on_video_load_finished(self):
        self._load_video_worker = None
        self._load_video_thread = None
        self._update_action_state()

    def _install_timeline(self, info: dict):
        ll = self._left.layout()
        if self._timeline:
            ll.removeWidget(self._timeline)
            self._timeline.deleteLater()
        self._tl_placeholder.hide()
        self._timeline = TimelineWidget(info["frame_count"], info["fps"])
        self._timeline.scrub_started.connect(self._on_scrub_started)
        self._timeline.frame_selected.connect(self._on_scrub)
        self._timeline.scrub_released.connect(self._on_scrub_released)
        ll.addWidget(self._timeline)

    def _try_load_session(self, path: str):
        try:
            tracker, info, _ = session_mod.load_session(path)
        except FileNotFoundError:
            return
        except ValueError as e:
            msg = str(e)
            self.status_bar.showMessage(f"Sesión ignorada — {msg}")
            QMessageBox.information(
                self, "Sesión no compatible",
                f"Se encontró un análisis previo pero no se puede cargar:\n\n{msg}\n\n"
                "Vuelve a analizar el vídeo para generar una sesión nueva."
            )
            return
        except Exception as e:
            self.status_bar.showMessage(f"Sesión ignorada — {e}")
            return
        if self._question_es(
            "Sesión guardada",
            "Se encontró un análisis previo para este vídeo.\n"
            "¿Cargar en lugar de re-analizar?"
        ):
            self.tracker = tracker
            # Sync video_info with corrected values stored in the session.
            if isinstance(info, dict):
                self.video_info.update(info)
            if self._timeline is not None:
                corrected_fc = self.video_info.get("frame_count", 0)
                if corrected_fc > 0:
                    self._timeline.update_total_frames(corrected_fc)
            self._populate_persons()
            self.btn_render.setEnabled(True)
            self._update_action_state()
            self.status_bar.showMessage(
                f"Sesión cargada · {len(tracker.persons)} persona(s)"
            )

    # ── Análisis ──────────────────────────────────────────────────────────────

    def _start_analysis(self):
        if not self.video_path:
            return
        if self._analysis_thread and self._analysis_thread.isRunning():
            return
        if self._render_thread and self._render_thread.isRunning():
            QMessageBox.warning(
                self, "Render en curso",
                "No se puede analizar mientras hay un render activo."
            )
            return
        if not self._close_player():
            QMessageBox.warning(
                self, "Reproducción activa",
                "No se pudo detener la reproducción a tiempo. Inténtalo de nuevo en unos segundos."
            )
            return
        self.btn_analyze.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.btn_cancel_op.setVisible(True)
        self.btn_cancel_op.setText("Cancelar análisis")
        self.tracker = None
        self._clear_history()
        self._clear_persons()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Analizando…")

        worker = AnalysisWorker(
            self.video_path,
            cfg.get("analysis_step"),
            cfg.get("min_det_score"),
            cfg.get("checkpoint_every"),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_analysis_progress)
        worker.eta.connect(self.status_bar.showMessage)
        worker.checkpoint.connect(lambda fi: self.status_bar.showMessage(
            f"Checkpoint guardado · frame {fi}"
        ))
        worker.finished.connect(self._on_analysis_done)
        worker.error.connect(self._on_analysis_error)
        worker.cancelled.connect(self._on_analysis_cancelled)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._analysis_thread = thread
        self._analysis_worker = worker
        thread.start()

    def _on_analysis_progress(self, cur: int, tot: int):
        pct = int(100 * cur / tot) if tot else 0
        pct = max(0, min(100, pct))
        self.progress_bar.setValue(pct)

    def _on_analysis_done(self, tracker: FaceTracker, info: dict):
        print(f"[Análisis] Completado: {len(tracker.persons)} persona(s)")
        self.tracker = tracker
        self.video_info.update(info)
        self.progress_bar.setVisible(False)
        self.btn_cancel_op.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_render.setEnabled(True)
        self._clear_history()
        self._populate_persons()
        n = len(tracker.persons)
        self.status_bar.showMessage(f"Análisis completo — {n} persona(s) detectada(s)")
        # Corregir timeline con el frame_count verificado por OpenCV durante el análisis.
        # probe_video() puede sobreestimar si usa duration×fps; el worker llama a probe_video
        # de nuevo y el cross-check de OpenCV lo refina — actualizamos el slider aquí por si
        # el valor inicial (cargado al abrir el vídeo) era incorrecto.
        if self._timeline:
            corrected_fc = info.get("frame_count", 0)
            if corrected_fc > 0:
                self._timeline.update_total_frames(corrected_fc)
        self._analysis_worker = None
        self._analysis_thread = None
        self._update_action_state()
        if self._close_after_analysis_cancel:
            self._close_after_analysis_cancel = False
            self.close()

    def _on_analysis_error(self, msg: str):
        print(f"[Análisis] ERROR: {msg}", file=sys.stderr)
        self.progress_bar.setVisible(False)
        self.btn_cancel_op.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_render.setEnabled(self.tracker is not None)
        QMessageBox.critical(self, "Error de análisis", msg)
        self._analysis_worker = None
        self._analysis_thread = None
        self._update_action_state()
        if self._close_after_analysis_cancel:
            self._close_after_analysis_cancel = False
            self.close()

    def _on_analysis_cancelled(self):
        self.progress_bar.setVisible(False)
        self.btn_cancel_op.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_render.setEnabled(self.tracker is not None)
        self.status_bar.showMessage("Análisis cancelado.")
        self._analysis_worker = None
        self._analysis_thread = None
        self._update_action_state()
        if self._close_after_analysis_cancel:
            self._close_after_analysis_cancel = False
            self.close()

    def _start_regroup(self):
        if not self.tracker or not self.video_path:
            return
        if self._analysis_thread and self._analysis_thread.isRunning():
            return
        if self._render_thread and self._render_thread.isRunning():
            return
        if self._regroup_thread and self._regroup_thread.isRunning():
            return
        self.status_bar.showMessage("Re-agrupando personas...")
        if getattr(self, "btn_regroup", None):
            self.btn_regroup.setEnabled(False)
        estimated_mb = self._estimate_tracker_mb(self.tracker)
        if estimated_mb > 180:
            if getattr(self, "btn_regroup", None):
                self.btn_regroup.setEnabled(True)
            self.status_bar.showMessage(
                f"Re-agrupar omitido: el análisis ocupa ~{estimated_mb:.0f} MB y podría saturar la RAM."
            )
            return
        # Deepcopy en hilo de fondo para no congelar la UI ni pasar referencia viva
        try:
            tracker_snapshot = self._deepcopy_async(self.tracker)
        except MemoryError:
            if getattr(self, "btn_regroup", None):
                self.btn_regroup.setEnabled(True)
            self.status_bar.showMessage("No hay RAM suficiente para re-agrupar este análisis.")
            return
        worker = RegroupWorker(
            tracker_snapshot,
            self.video_path,
            self.video_info,
            cfg.get("analysis_step"),
            cfg.get("similarity_threshold"),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_regroup_done)
        worker.error.connect(self._on_regroup_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_regroup_thread_finished)
        thread.start()
        self._regroup_worker = worker
        self._regroup_thread = thread
        self._update_action_state()

    def _on_regroup_done(self, tracker: FaceTracker, info: dict, merges: int, removed: int):
        if self.tracker is not None:
            try:
                snapshot = self._deepcopy_async(self.tracker)
            except (MemoryError, Exception):
                snapshot = None   # si falla deepcopy, saltamos esta entrada de historial
            if snapshot is not None:
                self._undo_stack.append(snapshot)
                if len(self._undo_stack) > 15:
                    self._undo_stack = self._undo_stack[-15:]
            self._redo_stack.clear()
        self.tracker = tracker
        self.video_info.update(info)
        self._populate_persons()
        self.status_bar.showMessage(
            f"Re-agrupado: {len(tracker.persons)} personas ({merges} fusiones, {removed} eliminadas)"
        )

    def _on_regroup_error(self, msg: str):
        QMessageBox.critical(self, "Error al re-agrupar", msg)

    def _on_regroup_thread_finished(self):
        self._regroup_worker = None
        self._regroup_thread = None
        self._update_action_state()
        if self._close_after_regroup:
            self._close_after_regroup = False
            QTimer.singleShot(0, self.close)

    # ── Panel de personas ─────────────────────────────────────────────────────

    def _clear_persons(self):
        self.person_cards.clear()
        while self.persons_layout.count():
            item = self.persons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _populate_persons(self):
        self._clear_persons()
        if not self.tracker:
            return
        fps   = self.video_info.get("fps", 25.0)
        total = self.video_info.get("frame_count", 1)
        ordered = sorted(self.tracker.persons, key=lambda x: (-x.frame_count, x.person_id))
        for display_idx, p in enumerate(ordered, start=1):
            card = PersonCard(p, fps, total, display_index=display_idx)
            card.config_changed.connect(self._on_config_changed)
            card.person_renamed.connect(lambda: self._autosave_session("Persona renombrada"))
            self.persons_layout.addWidget(card)
            self.person_cards.append(card)
        self._update_action_state()

    def _get_frame_data(self) -> dict:
        if not self.tracker:
            return {}
        return {p.person_id: p.frame_data for p in self.tracker.persons}

    # ── Preview & play ────────────────────────────────────────────────────────

    def _close_player(self, timeout_ms: int = 1500) -> bool:
        """
        Para el worker de preview y espera a que el hilo termine.
        Procesa eventos Qt en chunks de 100 ms para que la UI no se congele.
        """
        worker = self._playback_worker
        thread = self._playback_thread
        if worker:
            worker.stop()
        if thread:
            thread.quit()
            elapsed = 0
            chunk = 100
            while elapsed < timeout_ms and thread.isRunning():
                thread.wait(chunk)
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents | QEventLoop.ExcludeSocketNotifiers)
                elapsed += chunk
            if thread.isRunning():
                # Forzar terminación — preferible a un deadlock o a que la UI
                # quede bloqueada esperando un hilo que no para solo.
                thread.terminate()
                thread.wait(500)
                if thread.isRunning():
                    self.status_bar.showMessage("Advertencia: hilo de reproducción no pudo detenerse.")
                    return False   # el hilo sigue vivo tras terminate(); los callers deben abortar
            if self._playback_worker is worker:
                self._playback_worker = None
            if self._playback_thread is thread:
                self._playback_thread = None
        self._playing = False
        self._user_scrubbing = False
        self._pending_playback_seek_fi = None
        self._pending_playback_seek_token += 1
        self._scrub_preview_pending = False
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setText("Play")
        return True

    def _on_scrub_started(self):
        self._user_scrubbing = True
        self._pending_playback_seek_fi = None
        self._pending_playback_seek_token += 1

    def _on_scrub(self, fi: int):
        if not self._reader:
            return
        if self._playing:
            self._scrub_target_fi = fi
            # Mientras el usuario arrastra durante playback, no enviamos seeks
            # continuos al audio. Solo mostramos preview del punto bajo el cursor.
            self._scrub_debounce.start(40)
            return
        # La lectura OpenCV se hace con debounce (40 ms) para no congelar la UI
        # al arrastrar rápido sobre vídeos HEVC (cada read_frame puede tardar 100-500 ms)
        self._scrub_target_fi = fi
        self._scrub_debounce.start(40)

    def _on_scrub_released(self, fi: int):
        """
        Al soltar el slider: seek definitivo al punto exacto.
        Cancela el debounce para no duplicar el seek (el debounce y el release
        podrían llamar seek_to_frame dos veces en <40 ms).

        IMPORTANTE: cuando no se está reproduciendo, el debounce (40 ms) puede
        no haber disparado todavía si el usuario hizo clic rápido o soltó el
        slider antes de los 40 ms. En ese caso el frame nunca se mostraría.
        Por eso se emite scrub_frame_requested directamente aquí para garantizar
        que el preview siempre refleja la posición exacta al soltar.
        """
        self._was_playing = False
        self._scrub_debounce.stop()   # cancelar seek pendiente del debounce
        if self._playing and self._playback_worker:
            self._pending_playback_seek_fi = max(0, int(fi))
            self._pending_playback_seek_token += 1
            token = self._pending_playback_seek_token
            self._playback_worker.seek_to_frame(fi)
            QTimer.singleShot(8000, lambda t=token: self._clear_stale_playback_seek_guard(t))
        else:
            # Sin reproducción activa: garantizar que el frame en la posición
            # exacta de release se muestra siempre, incluso si el debounce no
            # alcanzó a disparar (clic rápido < 40 ms o drag muy corto).
            self._scrub_target_fi = fi
            self._scrub_preview_pending = False   # limpiar cualquier pending anterior
            self.scrub_frame_requested.emit(fi)
        self._user_scrubbing = False

    def _clear_stale_playback_seek_guard(self, token: int | None = None):
        if token is not None and token != self._pending_playback_seek_token:
            return
        self._pending_playback_seek_fi = None

    def _do_scrub_frame_read(self):
        """Fired by debounce timer: delegates frame read to ScrubWorker (worker thread)."""
        if not self._reader:
            return
        if self._playing and self._user_scrubbing:
            if self._scrub_preview_pending:
                return
            self._scrub_preview_pending = True
            self.scrub_frame_requested.emit(self._scrub_target_fi)
            QTimer.singleShot(1500, self._clear_scrub_preview_pending)
            return
        if self._playing and self._playback_worker:
            self._playback_worker.seek_to_frame(self._scrub_target_fi)
            return
        # read_frame is a slot on a moved-to-thread worker; Qt queues the call automatically.
        self.scrub_frame_requested.emit(self._scrub_target_fi)

    def _on_scrub_frame_received(self, frame, fi: int):
        """Called in UI thread when ScrubWorker delivers the decoded frame."""
        self._scrub_preview_pending = False
        # Only apply if this is still the frame we care about (user may have moved on).
        if fi == self._scrub_target_fi:
            self.preview.show_frame(self._compose_preview_frame(frame, fi))

    def _clear_scrub_preview_pending(self):
        self._scrub_preview_pending = False

    def _on_config_changed(self):
        if self._playing:
            # Durante playback: recomponer el último frame crudo con la nueva censura,
            # sin tocar el player (no cerrar, no hacer seek).
            if self._last_raw_frame is not None and self._timeline:
                fi = self._timeline.current_frame()
                self.preview.show_frame(self._compose_preview_frame(self._last_raw_frame, fi))
        elif self._timeline:
            self._on_scrub(self._timeline.current_frame())

    def _apply_censure(self, frame, fi: int):
        if not self.tracker:
            return frame
        total = self.video_info.get("frame_count", 1)
        fd    = self._get_frame_data()
        pending = []
        for card in self.person_cards:
            c = card.get_config()
            if not c["enabled"]:
                continue
            pid     = c["person_id"]
            fi_data = fd.get(pid, {}).get(fi)
            if not fi_data:
                continue
            end = c["end_frame"] if c["end_frame"] != -1 else total
            if not (c["start_frame"] <= fi <= end):
                continue
            pending.append((fi_data["bbox"], c["effect"], c["intensity"], c["padding_pct"]))
        if not pending:
            return frame
        out = frame.copy()
        for bbox, effect, intensity, padding_pct in pending:
            censure_roi_inplace(out, bbox, effect, intensity, padding_pct)
        return out

    @staticmethod
    def _draw_dashed_rect(img, x1, y1, x2, y2, color, dash=7, gap=5):
        """Rectángulo discontinuo para detecciones interpoladas."""
        for x in range(x1, x2, dash + gap):
            xe = min(x + dash, x2)
            cv2.line(img, (x, y1), (xe, y1), color, 1, cv2.LINE_AA)
            cv2.line(img, (x, y2), (xe, y2), color, 1, cv2.LINE_AA)
        for y in range(y1, y2, dash + gap):
            ye = min(y + dash, y2)
            cv2.line(img, (x1, y), (x1, ye), color, 1, cv2.LINE_AA)
            cv2.line(img, (x2, y), (x2, ye), color, 1, cv2.LINE_AA)

    def _apply_overlay(self, frame, fi: int):
        if not self.tracker:
            return frame
        out = frame.copy()
        ordered = sorted(self.tracker.persons, key=lambda x: (-x.frame_count, x.person_id))

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = 0.34   # tamaño del texto — discreto, legible

        for i, person in enumerate(ordered):
            fi_data = person.frame_data.get(fi)
            if fi_data is None:
                continue
            qcolor = PERSON_COLORS[i % len(PERSON_COLORS)]
            color  = (qcolor.blue(), qcolor.green(), qcolor.red())  # BGR
            h, w   = out.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in fi_data["bbox"]]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            is_interp = fi_data.get("interpolated", False)

            # ── 1. Borde fino alrededor de la cara ───────────────────────────
            # Interpolado → discontinuo (guiones); real → sólido 1 px
            if is_interp:
                self._draw_dashed_rect(out, x1, y1, x2, y2, color)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            # ── 2. Tab de etiqueta sobre el borde superior ───────────────────
            label = str(i + 1)
            (tw, th), bl = cv2.getTextSize(label, font, fs, 1)
            pad_x, pad_y = 7, 4
            tab_w = tw + pad_x * 2
            tab_h = th + bl + pad_y * 2

            # Posición del tab: por encima del borde superior del cuadro.
            # Si no hay espacio arriba, lo mete dentro del cuadro pegado al borde.
            tab_x1 = x1
            tab_x2 = min(w, tab_x1 + tab_w)
            tab_y2 = y1                          # base del tab = borde superior
            tab_y1 = tab_y2 - tab_h
            if tab_y1 < 0:                       # sin espacio arriba → dentro
                tab_y1 = y1
                tab_y2 = y1 + tab_h

            tab_x1 = max(0, tab_x1)
            tab_y1 = max(0, tab_y1)
            tab_x2 = min(w, tab_x2)
            tab_y2 = min(h, tab_y2)

            if tab_x2 > tab_x1 and tab_y2 > tab_y1:
                # Fondo plano del color de la persona, sin sombras ni gradientes
                cv2.rectangle(out, (tab_x1, tab_y1), (tab_x2, tab_y2), color, -1)
                # Texto blanco centrado en el tab
                tx = tab_x1 + pad_x
                ty = tab_y2 - pad_y - bl
                cv2.putText(out, label, (tx, ty), font, fs,
                            (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _compose_preview_frame(self, frame, fi: int):
        out = self._apply_censure(frame, fi)
        if self._show_detections:
            out = self._apply_overlay(out, fi)
        return out

    def _toggle_play(self):
        if not self._reader or not self._timeline or not self.video_path:
            return

        fi  = self._timeline.current_frame()
        fps = self.video_info.get("fps", 25.0)

        if self._playing:
            if not self._close_player():
                self.status_bar.showMessage("Pausando... espera a que termine de detenerse la reproducción.")
        else:
            worker = PlaybackWorker(
                self.video_path,
                fi,
                fps,
                cfg.get("play_speed"),
                self.video_info.get("rotation", 0),
                self.video_info.get("frame_count", 0),
                # None = "desconocido" (ffprobe no disponible) → intentar mpv igualmente
                # False = "sin audio confirmado" → usar InternalClock directamente
                self.video_info.get("has_audio") is not False,
            )
            thread = QThread(self)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.frame_ready.connect(self._on_frame_ready)
            worker.finished.connect(self._on_playback_finished)
            worker.finished.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._on_playback_thread_finished)
            thread.start()
            self._playback_worker = worker
            self._playback_thread = thread
            self._playing = True
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.btn_play.setText("Pausa")

    def _on_frame_ready(self, frame_bgr, fi: int):
        """Slot llamado desde el hilo del worker vía señal Qt (QueuedConnection)."""
        total = max(1, self.video_info.get("frame_count", 1))   # guard contra 0
        fi    = min(fi, total - 1)

        # Si el usuario está arrastrando la línea de tiempo, no dejar que los
        # frames de playback muevan el slider hacia atrás/adelante bajo su mano.
        if self._user_scrubbing:
            return

        # Tras soltar durante playback, pueden llegar frames antiguos que estaban
        # en cola antes del seek. Ignorarlos hasta recibir uno cercano al destino.
        if self._pending_playback_seek_fi is not None:
            fps = max(1.0, float(self.video_info.get("fps", 25.0) or 25.0))
            tolerance = max(2, int(fps * 0.25))
            if abs(fi - self._pending_playback_seek_fi) > tolerance:
                return
            self._pending_playback_seek_fi = None

        self._last_raw_frame = frame_bgr   # guardar para recomponer en config_changed
        if self._timeline:
            self._timeline.set_frame(fi)
        self.preview.show_frame(self._compose_preview_frame(frame_bgr, fi))

    def _on_playback_finished(self):
        self._playing = False
        self._user_scrubbing = False
        self._pending_playback_seek_fi = None
        self._pending_playback_seek_token += 1
        self._scrub_preview_pending = False
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setText("Play")

    def _on_playback_thread_finished(self):
        self._playback_worker = None
        self._playback_thread = None
        if self._close_after_playback_stop:
            self._close_after_playback_stop = False
            self.close()

    # ── Renderizado ───────────────────────────────────────────────────────────

    def _start_render(self):
        if not self.video_path or not self.tracker:
            return
        if self._analysis_thread and self._analysis_thread.isRunning():
            QMessageBox.warning(
                self, "Análisis en curso",
                "Espera a que termine el análisis antes de renderizar."
            )
            return
        if self._render_thread and self._render_thread.isRunning():
            return
        if not self._close_player():
            QMessageBox.warning(
                self, "Reproducción activa",
                "No se pudo detener la reproducción a tiempo. Inténtalo de nuevo en unos segundos."
            )
            return
        try:
            assert_ffmpeg(cfg.get("ffmpeg_path") or None)
        except RuntimeError as e:
            QMessageBox.critical(self, "FFmpeg no encontrado", str(e))
            return

        configs = [c.get_config() for c in self.person_cards]
        if not any(c["enabled"] for c in configs):
            QMessageBox.information(
                self, "Sin selección",
                "Activa 'Censurar' en al menos una persona."
            )
            return

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        renders_dir = os.path.join(project_root, "renders")
        os.makedirs(renders_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        default_output = os.path.join(renders_dir, f"{base_name}_censored.mp4")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar vídeo renderizado",
            default_output,
            "Vídeo MP4 (*.mp4);;Todos (*)"
        )
        if not out_path:
            return
        ext = os.path.splitext(out_path)[1].lower()
        if ext != ".mp4":
            out_path = os.path.splitext(out_path)[0] + ".mp4"
            QMessageBox.information(
                self, "Formato ajustado",
                f"El render siempre se guarda en MP4.\n"
                f"Archivo de salida: {os.path.basename(out_path)}"
            )
        if os.path.normcase(os.path.abspath(out_path)) == os.path.normcase(os.path.abspath(self.video_path)):
            QMessageBox.warning(
                self, "Ruta no permitida",
                "El vídeo de salida no puede ser el mismo archivo que el vídeo original."
            )
            return

        self._set_render_busy(True)

        worker = RenderWorker(
            self.video_path, out_path, configs,
            self._get_frame_data(), self.video_info
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.encoder_selected.connect(self._on_encoder_selected)
        worker.progress.connect(self._on_render_progress)
        worker.warning.connect(self._on_render_warning)
        worker.finished.connect(self._on_render_done)
        worker.error.connect(self._on_render_error)
        worker.cancelled.connect(self._on_render_cancelled)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_render_thread_finished)
        thread.start()
        self._render_thread = thread
        self._render_worker = worker

    def _on_encoder_selected(self, label: str):
        self._active_encoder = label
        self.status_bar.showMessage(f"Renderizando con {label}…")

    def _on_render_progress(self, cur: int, tot: int):
        pct = int(100 * cur / tot) if tot else 0
        pct = max(0, min(100, pct))
        self.progress_bar.setValue(pct)
        enc = f"  [{self._active_encoder}]" if self._active_encoder else ""
        self.status_bar.showMessage(f"Renderizando… {pct}%  ({cur}/{tot} frames){enc}")

    def _on_render_warning(self, msg: str):
        self.status_bar.showMessage(msg, 8000)

    def _set_render_busy(self, busy: bool) -> None:
        """Activa/desactiva el modo 'renderizando': botones y menús protegidos."""
        self.btn_render.setEnabled(not busy)
        self.btn_analyze.setEnabled(not busy)
        if getattr(self, "btn_merge", None):
            self.btn_merge.setEnabled((not busy) and self.tracker is not None and len(self.tracker.persons) >= 2)
        if getattr(self, "btn_cancel_op", None):
            self.btn_cancel_op.setVisible(busy)
            if busy:
                self.btn_cancel_op.setText("Cancelar render")
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setValue(0)
            self.status_bar.showMessage(
                "Renderizando… Los cambios en las tarjetas aplican al siguiente render."
            )
        # Bloquear tarjetas de persona durante el render para evitar confusión:
        # el render usa el snapshot capturado al inicio, no los valores actuales.
        _card_tip = (
            "Controles bloqueados durante el render.\n"
            "El render usa la configuración capturada al pulsar Renderizar.\n"
            "Los cambios que hagas aquí se aplicarán al siguiente render."
        ) if busy else ""
        for card in getattr(self, "person_cards", []):
            card.setEnabled(not busy)
            card.setToolTip(_card_tip)
        # Deshabilita los menús que abren diálogos modales (evita bloqueos de Qt)
        if self._act_settings:
            self._act_settings.setEnabled(not busy)
        if self._act_batch:
            self._act_batch.setEnabled(not busy)
        self._update_action_state()

    def _on_render_done(self, out_path: str):
        print(f"[Render] Completado: {out_path}")
        self.progress_bar.setValue(100)   # garantizar 100% visual aunque total_frames esté mal
        self._set_render_busy(False)
        enc = f"  ·  {self._active_encoder}" if self._active_encoder else ""
        self.status_bar.showMessage(f"Renderizado completo{enc}  →  {os.path.basename(out_path)}")
        self._active_encoder = ""
        self._close_after_render_cancel = False
        folder = os.path.dirname(out_path)
        if cfg.get("open_folder_after_render"):
            QMessageBox.information(
                self, "¡Listo!",
                f"Vídeo guardado en:\n{out_path}\n\n"
                f"Audio conservado en el render.",
            )
            self._open_folder(folder)
        else:
            if self._information_question_es(
                "¡Listo!",
                f"Vídeo guardado en:\n{out_path}\n\n"
                f"Audio conservado en el render.\n¿Abrir carpeta?"
            ):
                self._open_folder(folder)

    def _on_render_error(self, msg: str):
        print(f"[Render] Error: {msg}", file=sys.stderr)
        self._set_render_busy(False)
        self._active_encoder = ""
        self._close_after_render_cancel = False
        QMessageBox.critical(self, "Error de renderizado", msg)

    def _on_render_cancelled(self):
        print("[Render] Cancelado por el usuario.")
        self._set_render_busy(False)
        self._active_encoder = ""
        self.status_bar.showMessage("Render cancelado. Archivo parcial eliminado.")

    def _cancel_current_operation(self):
        """Called by the single Cancel button in the topbar."""
        if self._analysis_worker:
            self._analysis_worker.cancel()
            self.btn_cancel_op.setEnabled(False)
        elif self._render_worker:
            self._render_worker.cancel()
            self.btn_cancel_op.setEnabled(False)

    def _on_render_thread_finished(self):
        if getattr(self, "btn_cancel_op", None):
            self.btn_cancel_op.setEnabled(True)
        close_after_cancel = self._close_after_render_cancel
        self._render_worker = None
        self._render_thread = None
        if self._close_after_render_cancel:
            self._close_after_render_cancel = False
        if close_after_cancel:
            QTimer.singleShot(0, self.close)

    # ── Herramientas ──────────────────────────────────────────────────────────

    def _export_frame(self):
        if not self._reader or not self._timeline:
            QMessageBox.information(self, "Sin vídeo", "Carga un vídeo primero.")
            return
        fi    = self._timeline.current_frame()
        frame = self._reader.read_frame(fi)
        if frame is None:
            return
        frame = self._apply_censure(frame, fi)
        default_dir = os.path.dirname(self.video_path) if getattr(self, "video_path", None) else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar frame",
            os.path.join(default_dir, f"frame_{fi:06d}.jpg"),
            "JPEG (*.jpg);;PNG (*.png);;Todos (*)"
        )
        if path:
            ok = cv2.imwrite(path, frame)
            if ok:
                self.status_bar.showMessage(f"Frame guardado: {path}")
            else:
                QMessageBox.critical(
                    self, "Error al guardar",
                    f"cv2.imwrite() no pudo escribir el frame en:\n{path}\n\n"
                    "Comprueba que la ruta es válida y tienes permisos de escritura."
                )

    def _export_log(self):
        if not self.tracker:
            QMessageBox.information(self, "Sin datos", "Analiza un vídeo primero.")
            return
        default_dir = os.path.dirname(self.video_path) if getattr(self, "video_path", None) else ""
        path, _ = QFileDialog.getSaveFileName(self, "Exportar log", default_dir, "JSON (*.json)")
        if path:
            try:
                session_mod.export_detection_log(self.tracker, self.video_info, path)
                QMessageBox.information(self, "Log exportado", f"Guardado en:\n{path}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error al exportar",
                    f"No se pudo guardar el log de detección:\n{e}"
                )

    def _export_censure_config(self):
        if not self.tracker or not self.person_cards:
            QMessageBox.information(self, "Sin datos", "Analiza o carga una sesión primero.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar configuración de censura", "", "GhostFrame Config (*.gfscfg);;JSON (*.json)"
        )
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in {".gfscfg", ".json"}:
            path += ".gfscfg"
        try:
            session_mod.export_censure_config(path, self.person_cards)
            QMessageBox.information(self, "Configuración exportada", f"Guardado en:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo exportar:\n{e}")

    def _import_censure_config(self):
        if not self.tracker or not self.person_cards:
            QMessageBox.information(self, "Sin datos", "Analiza o carga una sesión primero.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Importar configuración de censura", "", "GhostFrame Config (*.gfscfg *.json);;Todos (*)"
        )
        if not path:
            return
        try:
            applied = session_mod.import_censure_config(path, self.person_cards)
            self._on_config_changed()
            self.status_bar.showMessage(f"Configuración importada: {applied} persona(s) actualizada(s)")
            if applied < len(self.person_cards):
                QMessageBox.information(
                    self,
                    "Importación parcial",
                    f"Se aplicó a {applied} de {len(self.person_cards)} persona(s).\n\n"
                    "Las personas sin coincidencia mantienen su configuración anterior.\n"
                    "Esto puede ocurrir si cambiaron las condiciones de luz o el ángulo.\n\n"
                    "Para ajustarlas manualmente, configura efecto e intensidad\n"
                    "en cada tarjeta de persona sin match."
                )
            self._update_action_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo importar:\n{e}")
            self._update_action_state()

    def _delete_cache(self):
        if not self.video_path:
            QMessageBox.information(self, "Sin vídeo", "Carga un vídeo primero.")
            return
        session_paths = [
            p for p in session_mod.session_path_candidates(self.video_path)
            if os.path.exists(p)
        ]
        if not session_paths:
            QMessageBox.information(self, "Sin caché", "No hay sesión guardada.")
            return
        session_list = "\n".join(session_paths)
        if self._question_es("Borrar caché", f"¿Borrar?\n{session_list}"):
            session_mod.delete_session(self.video_path)
            self.status_bar.showMessage("Caché borrada.")

    def _merge_persons(self):
        if not self.tracker or len(self.tracker.persons) < 2:
            QMessageBox.information(self, "Insuficiente", "Se necesitan al menos 2 personas.")
            return
        dlg = MergePersonsDialog(self.tracker, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        pid_a, pid_b, force_merge = dlg.result_pids()
        if pid_a is None or pid_b is None or pid_a == pid_b:
            return
        self._push_undo()
        try:
            self.tracker.merge_persons(pid_a, pid_b, force=force_merge)
        except ValueError as exc:
            # Fusión fallida — descartar la entrada undo recién creada (estado no cambió)
            if self._undo_stack:
                self._undo_stack.pop()
            self._update_action_state()
            QMessageBox.warning(self, "No se puede fusionar", str(exc))
            return
        self._populate_persons()
        self._autosave_session("Personas fusionadas")
        self._update_action_state()

    def _split_person(self):
        if not self.tracker:
            return
        persons = sorted(self.tracker.persons, key=lambda p: (-p.frame_count, p.person_id))
        ids = [
            f"{getattr(p, '_custom_label', None) or f'Persona {i}'}  ({p.frame_count} frames)  [#{p.person_id}]"
            for i, p in enumerate(persons, start=1)
        ]
        id_str, ok = QInputDialog.getItem(self, "Dividir", "¿Qué persona dividir?", ids, 0, False)
        if not ok:
            return
        person = persons[ids.index(id_str)]
        frames = sorted(person.frame_data.keys())
        if not frames:
            return
        fps = self.video_info.get("fps", 25.0)
        start_f, ok2 = QInputDialog.getInt(
            self, "Dividir — inicio",
            f"Frame de INICIO (rango detectado: {frames[0]}–{frames[-1]}):",
            frames[0], frames[0], frames[-1]
        )
        if not ok2:
            return
        end_f, ok3 = QInputDialog.getInt(
            self, "Dividir — fin", "Frame de FIN del segmento:",
            min(start_f + int(fps * 5), frames[-1]), start_f, frames[-1]
        )
        if not ok3:
            return
        to_split = [f for f in frames if start_f <= f <= end_f]
        if not to_split:
            QMessageBox.warning(self, "Sin frames", "No hay frames en ese rango.")
            return
        self._push_undo()
        new_pid = self.tracker.split_person(person.person_id, to_split)
        if new_pid < 0:
            self._undo_stack.pop()   # descarta el undo push: nada cambió
            QMessageBox.warning(self, "Error", "No se pudo dividir.")
            self._update_action_state()
            return
        self._populate_persons()
        self._autosave_session("Persona dividida")
        self._update_action_state()

    def _autosave_session(self, action_label: str):
        """Guarda la sesión tras una operación manual (merge, split, rename)."""
        if not self.video_path or not self.tracker:
            self.status_bar.showMessage(action_label + ".")
            return
        try:
            session_mod.save_session(
                self.video_path, self.tracker,
                self.video_info, cfg.get("analysis_step"),
            )
            self.status_bar.showMessage(f"{action_label}. Sesion guardada.")
        except Exception as _save_err:
            self.status_bar.showMessage(f"{action_label}.")
            QMessageBox.warning(
                self, "Error al guardar sesión",
                f"No se pudo guardar la sesión automáticamente:\n{_save_err}\n\n"
                "Los cambios de esta operación no se han escrito en disco."
            )

    def _open_settings(self):
        if self._analysis_thread and self._analysis_thread.isRunning():
            QMessageBox.warning(self, "Análisis en curso", "Espera a que termine el análisis actual.")
            return
        if self._render_thread and self._render_thread.isRunning():
            QMessageBox.warning(self, "Render en curso", "Espera a que termine el render actual.")
            return
        if self._playing and not self._close_player():
            QMessageBox.warning(self, "Reproducción activa", "No se pudo pausar a tiempo. Inténtalo de nuevo.")
            return
        from ui.settings_dialog import SettingsDialog
        SettingsDialog(self).exec_()

    def _open_batch(self):
        if self._analysis_thread and self._analysis_thread.isRunning():
            QMessageBox.warning(self, "Análisis en curso", "Espera a que termine el análisis actual.")
            return
        if self._render_thread and self._render_thread.isRunning():
            QMessageBox.warning(self, "Render en curso", "Espera a que termine el render actual.")
            return
        if self._playing and not self._close_player():
            QMessageBox.warning(self, "Reproducción activa", "No se pudo pausar a tiempo. Inténtalo de nuevo.")
            return
        if not self.video_path:
            QMessageBox.information(
                self, "Sin vídeo",
                "Carga un vídeo en la sesión principal antes de usar el modo batch.\n"
                "El efecto de censura se tomará de la configuración activa."
            )
            return
        configs = [c.get_config() for c in self.person_cards]
        from ui.batch_dialog import BatchDialog
        BatchDialog(configs, cfg.get("analysis_step"), self).exec_()

    @staticmethod
    def _open_folder(folder: str):
        import subprocess
        if not os.path.isdir(folder):
            return
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            subprocess.Popen(["xdg-open", folder])

    def closeEvent(self, event):
        if self._render_thread and self._render_thread.isRunning():
            reply = self._question_es(
                "Render en curso",
                "Hay un render activo.\n¿Cancelar el render y cerrar cuando termine la cancelación?",
            )
            if reply and self._render_worker:
                self._close_after_render_cancel = True
                self._render_worker.cancel()
                self.status_bar.showMessage("Cancelando render…")
            event.ignore()
            return
        if self._regroup_thread and self._regroup_thread.isRunning():
            self._close_after_regroup = True
            self.status_bar.showMessage("Re-agrupando… se cerrará al terminar.")
            event.ignore()
            return
        if self._analysis_thread and self._analysis_thread.isRunning():
            if self._analysis_worker:
                self._close_after_analysis_cancel = True
                self._analysis_worker.cancel()
            self.status_bar.showMessage("Cancelando análisis…")
            event.ignore()
            return
        if self._load_video_thread and self._load_video_thread.isRunning():
            self.status_bar.showMessage("Abriendo vídeo... espera unos segundos y vuelve a cerrar.")
            event.ignore()
            return
        if self._diagnostics_thread and self._diagnostics_thread.isRunning():
            self.status_bar.showMessage("Diagnóstico en curso... espera unos segundos y vuelve a cerrar.")
            event.ignore()
            return
        if self._warmup_thread and self._warmup_thread.isRunning():
            self.status_bar.showMessage("Inicializando el detector... espera unos segundos y vuelve a cerrar.")
            event.ignore()
            return
        if not self._close_player():
            self._close_after_playback_stop = True
            event.ignore()
            return
        if self._reader:
            self._reader.release()
        # Liberar VideoCapture del ScrubWorker ANTES de quit() para que
        # cap.read() en curso retorne y el hilo no quede bloqueado.
        self._scrub_worker.stop()
        self._scrub_thread.quit()
        if not self._scrub_thread.wait(3000):
            self.status_bar.showMessage("Esperando a que se cierre el lector de preview...")
            event.ignore()
            return
        event.accept()
