"""
Frame reading (OpenCV) and censure effect application.
The three censure effects (blur, pixelate, blackbox) all use feathered edges
so the boundary doesn't look like a hard rectangle.
"""
from __future__ import annotations

import functools
import cv2
import numpy as np


# ─── Censure effects ──────────────────────────────────────────────────────────

def _expand_bbox(
    bbox: list[int],
    padding_pct: float,
    frame_w: int,
    frame_h: int,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    pw = int((x2 - x1) * padding_pct)
    ph = int((y2 - y1) * padding_pct)
    return [
        max(0, x1 - pw),
        max(0, y1 - ph),
        min(frame_w, x2 + pw),
        min(frame_h, y2 + ph),
    ]


@functools.lru_cache(maxsize=256)
def _feather_mask(h: int, w: int, feather: int) -> np.ndarray:
    """
    Float32 mask [0..1] with smooth edges using distance-to-border.
    Avoids the hard rectangular outline that looks unnatural.
    Para ROIs muy grandes la máscara se calcula a escala reducida y se reescala
    para evitar OOM, en lugar de desactivar el feather por completo.

    Cacheada con lru_cache: los argumentos son int (hashables) y el array
    resultante NO se modifica nunca in-place (se usa solo en multiplicaciones),
    por lo que compartir la instancia cacheada entre llamadas es seguro.
    """
    if feather < 1:
        return np.ones((h, w), dtype=np.float32)

    def _compute(ch: int, cw: int, cf: int) -> np.ndarray:
        xs = np.arange(cw, dtype=np.float32)
        ys = np.arange(ch, dtype=np.float32)
        dx = np.minimum(xs, (cw - 1) - xs)
        dy = np.minimum(ys, (ch - 1) - ys)
        dist_x = np.minimum(dx, cf) / cf
        dist_y = np.minimum(dy, cf) / cf
        return (np.outer(dist_y, np.ones(cw)) * np.outer(np.ones(ch), dist_x)).astype(np.float32)

    max_pixels = 2_000_000
    if h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        sh = max(2, int(h * scale))
        sw = max(2, int(w * scale))
        # Garantizar que sh*sw ≤ max_pixels pese a truncado entero y aspect extremo
        if sh * sw > max_pixels:
            sw = max(2, max_pixels // sh)
        scaled_feather = max(1, int(feather * scale))
        small = _compute(sh, sw, scaled_feather)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return _compute(h, w, feather)


def censure_roi_inplace(
    frame: np.ndarray,
    bbox: list[int],
    effect: str,
    intensity: int,
    padding_pct: float,
) -> None:
    """Apply censure directly into `frame` (no copy). Internal use only."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(bbox, padding_pct, w, h)
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 2 or rw < 2:
        return

    if effect == "blackbox":
        censured = np.zeros_like(roi)

    elif effect == "pixelate":
        block = max(4, intensity * 5)
        small_w = max(1, rw // block)
        small_h = max(1, rh // block)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        censured = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

    else:  # blur (default); efecto desconocido cae aquí
        if effect not in ("blur", "pixelate", "blackbox"):
            print(f"[video_processor] Efecto desconocido '{effect}' — usando blur por defecto")
        k = max(3, intensity * 10 - 1)
        if k % 2 == 0:
            k += 1
        # Un kernel ≈ √2·k equivale a dos pasadas con kernel k, pero en una sola
        # operación (~2× más rápido). Se fuerza impar con | 1.
        k2 = int(k * 1.4) | 1
        censured = cv2.GaussianBlur(roi, (k2, k2), 0)

    feather_px = max(0, min(rh // 5, rw // 5, 15))
    if feather_px > 1:
        mask = _feather_mask(rh, rw, feather_px)
        mask3 = np.stack([mask] * 3, axis=-1)
        blended = (
            censured.astype(np.float32) * mask3
            + roi.astype(np.float32) * (1.0 - mask3)
        ).clip(0, 255).astype(np.uint8)
    else:
        blended = censured

    frame[y1:y2, x1:x2] = blended


def apply_censure(
    frame: np.ndarray,
    bbox: list[int],
    effect: str,
    intensity: int,
    padding_pct: float = 0.15,
) -> np.ndarray:
    """
    Apply censure on a copy of `frame` within `bbox` (x1,y1,x2,y2).

    effect    : 'blur' | 'pixelate' | 'blackbox'
    intensity : 1–10  (controls blur kernel / mosaic block size)
    Returns the modified frame (always a new array, original untouched).
    """
    out = frame.copy()
    censure_roi_inplace(out, bbox, effect, intensity, padding_pct)
    return out


# ─── Video reader ─────────────────────────────────────────────────────────────

class VideoReader:
    """
    Thin OpenCV wrapper for random-access and sequential frame reading.
    Use iter_frames() for analysis, read_frame() for preview/seeking.

    rotation: grados de rotación del vídeo (0, 90, 180, 270).
    OpenCV ignora los metadatos de rotación; hay que aplicarla manualmente para
    que los bboxes del análisis coincidan con los frames que FFmpeg produce
    durante el render (donde -autorotate sí los aplica por defecto).
    """

    def __init__(self, video_path: str, rotation: int = 0,
                 frame_count_hint: int | None = None):
        """
        frame_count_hint: si se pasa, se usa directamente como frame_count y se
        omite el seek-and-read de corrección de frames fantasma (ya fue ejecutado
        por probe_video()). Pasar siempre info["frame_count"] desde probe_video()
        para evitar la doble corrección que ralentiza la apertura.
        """
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"No se puede abrir el vídeo: {video_path}")
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

        if frame_count_hint is not None and frame_count_hint > 0:
            # Valor ya corregido por probe_video() — no repetir el seek-and-read.
            self.frame_count = frame_count_hint
        else:
            # Sin hint: corregir frames fantasma del final (H.264/H.265 con B-frames).
            # CAP_PROP_FRAME_COUNT lee el header del contenedor, que puede incluir
            # entradas de relleno de B-frame delay al final del stream.
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _tmp = None
            try:
                if self.frame_count > 0:
                    _lookback = 60
                    _start = max(0, self.frame_count - _lookback)
                    _tmp = cv2.VideoCapture(self.path)
                    _tmp.set(cv2.CAP_PROP_POS_FRAMES, _start)
                    _fi = _start
                    _last_good = -1
                    while True:
                        _ok, _ = _tmp.read()
                        if not _ok:
                            break
                        _last_good = _fi
                        _fi += 1
                    if _last_good >= 0:
                        self.frame_count = _last_good + 1
            except Exception:
                pass
            finally:
                if _tmp is not None:
                    _tmp.release()
        raw_w: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._rotation = rotation % 360
        # Tras rotar 90°/270° el ancho y el alto se intercambian
        if self._rotation in (90, 270):
            self.width, self.height = raw_h, raw_w
        else:
            self.width, self.height = raw_w, raw_h

    @staticmethod
    def _rotate(frame: np.ndarray, rotation: int) -> np.ndarray:
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        """Seek to frame_idx and return the BGR frame (rotated if needed), or None."""
        if frame_idx < 0 or self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return self._rotate(frame, self._rotation)

    def iter_frames(self, step: int = 1):
        """
        Yield (frame_idx, frame_bgr) for every `step` frames.
        Uses sequential reads for speed (avoids repeated seek overhead).
        Frames are returned in display orientation (rotation applied).
        """
        step = max(1, int(step or 1))
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield idx, self._rotate(frame, self._rotation)
            idx += 1

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        try:
            if getattr(self, "cap", None) is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass
