"""
FFmpeg detection, video probing, and pipe-based rendering.

Problemas específicos de iPhone que resuelve este módulo
=========================================================
1. VFR (Variable Frame Rate): los vídeos de iPhone usan VFR para ahorrar espacio.
   Al hacer pipe de frames raw, VFR desincroniza el audio.
   Fix: -fps_mode cfr en el decoder fuerza CFR al ritmo de avg_frame_rate.

2. Rotación por metadatos: los vídeos grabados en vertical tienen los píxeles
   en horizontal con un metadato "rotate=90". OpenCV ignora ese metadato.
   ffmpeg lo aplica automáticamente con -autorotate (activo por defecto), pero
   probe_video() lee el stream antes de la rotación y devuelve dimensiones
   incorrectas. Detectamos la rotación y hacemos swap de width/height.

3. Dimensiones impares: H.264/H.265 exigen ancho y alto pares.
   Si el vídeo es 1081px, libx264 falla. Fix: filtro scale=trunc(iw/2)*2:trunc(ih/2)*2.

4. HEVC (H.265): todos los iPhones modernos graban en HEVC por defecto.
   ffmpeg lo decodifica correctamente sin flags extra gracias a libx265 software.
   En Windows con GPU NVIDIA se intenta hevc_nvenc/h264_nvenc para acelerar el encode.

Soporte de formatos
===================
Compatible con cualquier formato que ffmpeg decodifique: MP4, MOV, MKV, AVI,
WEBM, M4V, 3GP, TS, FLV y más. El vídeo de entrada puede ser H.264, HEVC,
VP9, AV1, MPEG-4, etc.
"""
from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import threading
import time as _time
from collections import deque
from typing import Callable

import numpy as np


class RenderCancelled(RuntimeError):
    """Raised when the user cancels an in-progress render."""


class NoVideoStream(ValueError):
    """Raised when a media file contains no usable video stream."""


# ─── Localización de ffmpeg / ffprobe ────────────────────────────────────────

def find_ffmpeg() -> str | None:
    # 1. Variable de entorno explícita (máxima prioridad)
    env = os.environ.get("FFMPEG_PATH")
    if env and os.path.isfile(env):
        return env

    # 2. PATH del sistema
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 3. Carpeta del proyecto — busca ffmpeg.exe en la raíz y en cualquier
    #    subcarpeta ffmpeg*/bin/ (útil si la carpeta está incluida en el repo)
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _candidates_relative = []
    try:
        for entry in os.listdir(_project_root):
            full = os.path.join(_project_root, entry)
            if os.path.isdir(full) and "ffmpeg" in entry.lower():
                _candidates_relative.append(os.path.join(full, "bin", "ffmpeg.exe"))
                _candidates_relative.append(os.path.join(full, "ffmpeg.exe"))
    except Exception:
        pass
    _candidates_relative += [
        os.path.join(_project_root, "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(_project_root, "ffmpeg", "ffmpeg.exe"),
        os.path.join(_project_root, "ffmpeg.exe"),
    ]
    for c in _candidates_relative:
        if os.path.isfile(c):
            return c

    # 4. Rutas típicas de instalación en Windows
    if platform.system() == "Windows":
        for c in [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\tools\ffmpeg\bin\ffmpeg.exe",
            os.path.join(os.path.expanduser("~"), r"ffmpeg\bin\ffmpeg.exe"),
        ]:
            if os.path.isfile(c):
                return c

    return None


def find_ffprobe(ffmpeg_path: str | None = None) -> str | None:
    ffmpeg = ffmpeg_path if ffmpeg_path and os.path.isfile(ffmpeg_path) else find_ffmpeg()
    if ffmpeg:
        probe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe" if platform.system() == "Windows" else "ffprobe")
        if os.path.isfile(probe):
            return probe
    return shutil.which("ffprobe")


def get_ffmpeg_version(ffmpeg_path: str) -> str:
    try:
        r = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, timeout=5)
        parts = r.stdout.split()
        return parts[2] if len(parts) > 2 else "unknown"
    except Exception:
        return "unknown"


def is_ffmpeg_executable(path: str) -> bool:
    """Return True only when path points to an ffmpeg binary, not just any file."""
    try:
        r = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and r.stdout.lower().startswith("ffmpeg version")
    except Exception:
        return False


def assert_ffmpeg(ffmpeg_path: str | None = None) -> str:
    if ffmpeg_path:
        if not os.path.isfile(ffmpeg_path):
            raise RuntimeError(f"FFmpeg configurado no existe:\n{ffmpeg_path}")
        if not is_ffmpeg_executable(ffmpeg_path):
            raise RuntimeError(f"La ruta configurada no es un ejecutable FFmpeg válido:\n{ffmpeg_path}")
    path = ffmpeg_path or find_ffmpeg()
    if path is None:
        raise RuntimeError(
            "FFmpeg no encontrado.\n\n"
            "Descárgalo en https://ffmpeg.org/download.html y añade\n"
            "la carpeta bin\\ a la variable PATH del sistema.\n\n"
            "O configura la ruta directamente en Herramientas → Configuración → FFmpeg."
        )
    return path


def _audio_args_for_output(output_path: str, audio_codec: str | None) -> list[str]:
    """
    Prefer streamcopy, but transcode unusual audio when muxing to MP4/M4V.
    FFmpeg streamcopy is lossless/fast, but can fail if the output container
    cannot mux the original audio codec.
    """
    ext = os.path.splitext(output_path)[1].lower()
    codec = (audio_codec or "unknown").lower()
    if ext in {".mp4", ".m4v"}:
        mp4_safe = {"aac", "mp3", "alac", "ac3", "eac3", "mp4a"}
        if codec == "unknown":
            return ["-c:a", "aac", "-b:a", "192k"]
        if codec not in mp4_safe and codec != "none":
            return ["-c:a", "aac", "-b:a", "192k"]
    return ["-c:a", "copy"]


def _subtitle_args_for_output(output_path: str, subtitle_codecs: list[str] | None) -> tuple[list[str], list[str]]:
    """
    Preserve subtitle streams only when the output container can mux them safely.
    MP4 accepts mov_text directly and can transcode common text subtitles to
    mov_text, but bitmap subtitles need OCR and would make FFmpeg fail.
    """
    codecs = [(c or "unknown").lower() for c in (subtitle_codecs or [])]
    if not codecs:
        return [], ["-sn"]

    ext = os.path.splitext(output_path)[1].lower()
    if ext in {".mp4", ".m4v"}:
        text_subs = {"mov_text", "subrip", "srt", "webvtt", "ass", "ssa", "text"}
        if all(c in text_subs for c in codecs):
            return ["-map", "1:s?"], ["-c:s", "mov_text"]
        return [], ["-sn"]

    return ["-map", "1:s?"], ["-c:s", "copy"]


# ─── Detección de codecs hardware disponibles ────────────────────────────────

_hw_cache: dict[str, dict] = {}
_HW_CACHE_TTL = 3600   # segundos
_encoder_verify_cache: dict[str, dict[str, bool]] = {}


def verify_encoder(ffmpeg_bin: str, codec: str) -> bool:
    """
    Intenta codificar 1 frame negro para comprobar que el encoder hardware
    inicializa correctamente. Sin esto, ffmpeg puede listar 'h264_nvenc' en
    -encoders pero fallar al renderizar (driver no instalado, GPU no soportada).

    Éxitos cacheados indefinidamente (para la sesión). Fallos cacheados 5 minutos
    para evitar el test de 15 s en cada render batch sin bloquear encoders que
    fallan de forma transitoria (carga alta, driver ocupado).
    """
    import time
    global _encoder_verify_cache
    path_cache = _encoder_verify_cache.setdefault(ffmpeg_bin, {})
    if codec in path_cache:
        cached = path_cache[codec]
        if cached is True:
            return True
        # Failure entry: monotonic timestamp until which the failure is still valid
        if isinstance(cached, float) and time.monotonic() < cached:
            return False
        # TTL expired — fall through and retry

    # NVENC rechaza tamaños muy pequeños (por ejemplo 32x32), aunque el encoder
    # funcione correctamente. Usar 256x256 evita falsos negativos.
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "lavfi", "-i", "color=black:s=256x256:d=0.04:r=25",
        "-frames:v", "1",
        "-c:v", codec,
        "-f", "null", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        ok = r.returncode == 0
    except Exception:
        ok = False

    if ok:
        path_cache[codec] = True                         # éxito: cachear indefinidamente
    else:
        path_cache[codec] = time.monotonic() + 300.0    # fallo: cachear 5 minutos
        print(f"[Encoder] {codec} listado en ffmpeg pero falla al inicializar - omitido")
    return ok


def detect_hardware_encoders(ffmpeg_bin: str) -> dict:
    """
    Detecta qué encoders hardware están disponibles en este sistema Windows.
    Devuelve un dict con claves booleanas.
    Resultado cacheado por ruta de ffmpeg — si el usuario cambia ffmpeg_path
    en configuración, la detección se repite con el nuevo binario.
    """
    global _hw_cache
    entry = _hw_cache.get(ffmpeg_bin)
    if entry and (_time.monotonic() - entry.get("ts", 0.0)) < _HW_CACHE_TTL:
        return entry.get("data", {})

    try:
        r = subprocess.run(
            [ffmpeg_bin, "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        txt = r.stdout + r.stderr
        result = {
            "h264_nvenc":  "h264_nvenc"  in txt,   # NVIDIA GPU H.264
            "hevc_nvenc":  "hevc_nvenc"  in txt,   # NVIDIA GPU H.265
            "h264_qsv":    "h264_qsv"    in txt,   # Intel Quick Sync H.264
            "hevc_qsv":    "hevc_qsv"    in txt,   # Intel Quick Sync H.265
            "h264_amf":    "h264_amf"    in txt,   # AMD VCE H.264
            "hevc_amf":    "hevc_amf"    in txt,   # AMD VCE H.265
        }
    except Exception:
        result = {}
    _hw_cache[ffmpeg_bin] = {"data": result, "ts": _time.monotonic()}
    return result


def best_encoder(ffmpeg_bin: str, prefer_hevc: bool = False) -> tuple[str, str]:
    """
    Elige el mejor encoder VERIFICADO disponible en este sistema.
    Devuelve (codec_name, extra_args_string).
    Prioridad: NVIDIA NVENC > Intel QSV > AMD AMF > libx264 software.

    Cada candidato se prueba con verify_encoder() antes de seleccionarlo.
    Si está listado pero falla (driver no instalado, GPU no compatible),
    se descarta y se prueba el siguiente. Resultado cacheado.

    Args NVENC (RTX 3060/3050 y todos los NVIDIA desde GTX 900 con driver ≥ 471):
      -rc:v constqp  → modo calidad constante (equivale al CRF de x264)
      -qp:v 20       → calidad visual ~CRF 20 (0=lossless, 51=peor)
      -preset p4     → velocidad media (p1=fastest … p7=slowest/best)
    """
    hw = detect_hardware_encoders(ffmpeg_bin)

    if prefer_hevc:
        candidates = [
            ("hevc_nvenc", "-rc:v constqp -qp:v 22 -preset p4"),
            ("hevc_qsv",   "-global_quality 22 -preset medium"),
            ("hevc_amf",   "-quality balanced"),
        ]
        fallback = ("libx265", "-crf 22 -preset fast")
    else:
        candidates = [
            ("h264_nvenc", "-rc:v constqp -qp:v 20 -preset p4"),
            ("h264_qsv",   "-global_quality 20 -preset medium"),
            ("h264_amf",   "-quality balanced"),
        ]
        fallback = ("libx264", "-crf 18 -preset fast")

    for codec, args in candidates:
        if hw.get(codec):
            if verify_encoder(ffmpeg_bin, codec):
                return codec, args
            # Si falla, seguimos con el siguiente candidato

    return fallback


# ─── Probe de vídeo (VFR + rotación + metadatos iPhone) ──────────────────────

def _get_rotation_from_stream(stream: dict) -> int:
    """
    Extrae la rotación (0, 90, 180, 270) de los metadatos del stream.
    Los iPhones almacenan esto en side_data_list[].rotation o en tags.rotate.
    """
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            return int(sd["rotation"]) % 360
    tags = stream.get("tags", {})
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360
        except (ValueError, TypeError):
            pass
    return 0


def _parse_fps(fps_str: str) -> float:
    """Parsea "30000/1001" o "30" → float."""
    try:
        if "/" in fps_str:
            n, d = fps_str.split("/")
            return float(n) / float(d) if float(d) != 0 else 25.0
        return float(fps_str)
    except Exception:
        return 25.0


def probe_video(video_path: str, ffmpeg_path: str | None = None) -> dict:
    """
    Extrae metadatos completos del vídeo usando ffprobe.
    Maneja correctamente:
    - VFR (usa avg_frame_rate para frame_count real)
    - Rotación iPhone (detecta y hace swap width/height si es 90/270)
    - HEVC, AV1, VP9 y cualquier otro codec
    - Streams sin audio (entrevistas grabadas con micrófono externo)
    Fallback a OpenCV si ffprobe no está disponible.
    """
    ffprobe = find_ffprobe(ffmpeg_path)
    if ffprobe:
        try:
            cmd = [
                ffprobe, "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                "-show_entries",
                "stream=index,codec_type,codec_name,width,height,color_space,color_transfer,color_primaries,color_range,"
                "r_frame_rate,avg_frame_rate,nb_frames,duration,side_data_list,tags:"
                "format=duration,size,bit_rate",
                video_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                raise RuntimeError(
                    f"ffprobe returned {r.returncode}: {r.stderr.strip()[:300]}"
                )
            data = json.loads(r.stdout)

            streams = data.get("streams", [])
            fmt = data.get("format", {})

            vstream = next((s for s in streams if s.get("codec_type") == "video"), {})
            astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
            subtitle_codecs = [
                s.get("codec_name", "unknown")
                for s in streams
                if s.get("codec_type") == "subtitle"
            ]
            if not vstream:
                raise NoVideoStream(f"El archivo no contiene stream de vídeo: {video_path}")

            # Usar avg_frame_rate para VFR (más preciso que r_frame_rate para iPhone)
            avg_fps = _parse_fps(vstream.get("avg_frame_rate", "25/1"))
            r_fps   = _parse_fps(vstream.get("r_frame_rate",   "25/1"))
            fps = avg_fps if avg_fps > 0 else (r_fps if r_fps > 0 else 25.0)

            # Duración: formato > stream
            duration = float(fmt.get("duration") or vstream.get("duration") or 0)

            # Frame count: nb_frames > duración × fps
            nb = vstream.get("nb_frames")
            try:
                frame_count = int(nb) if nb and int(nb) > 0 else 0
            except (ValueError, TypeError):
                frame_count = 0
            if frame_count == 0:
                if duration > 0 and fps > 0:
                    frame_count = int(duration * fps)

            # ── Corrección de frames fantasma (B-frame tail, MP4/MOV) ───────────
            # nb_frames de ffprobe y CAP_PROP_FRAME_COUNT de OpenCV leen el mismo
            # header del contenedor, que en H.264/H.265 incluye frames de relleno
            # del decoder delay al final (normalmente 4-8 frames extra).
            # La única forma fiable de saber cuántos frames son realmente decodificables
            # es buscar cerca del final y leer hasta que cap.read() falle.
            # Se leen como máximo 60 frames (≈ 2 s a 30 fps): impacto mínimo en el
            # tiempo de carga incluso para vídeos de horas (seek directo por índice MP4).
            if frame_count > 0:
                try:
                    import cv2 as _cv2_cross
                    _lookback = 60
                    _start = max(0, frame_count - _lookback)
                    _cap = _cv2_cross.VideoCapture(video_path)
                    _cap.set(_cv2_cross.CAP_PROP_POS_FRAMES, _start)
                    _fi = _start
                    _last_good = -1
                    while True:
                        _ret, _ = _cap.read()
                        if not _ret:
                            break
                        _last_good = _fi
                        _fi += 1
                    _cap.release()
                    # Solo corregimos si al menos un frame fue legible desde _start
                    # (descarta el caso de vídeo corrupto donde ningún seek funciona)
                    if _last_good >= 0:
                        frame_count = _last_good + 1
                except Exception:
                    pass

            try:
                raw_w = int(vstream.get("width",  0) or 0)
                raw_h = int(vstream.get("height", 0) or 0)
            except (ValueError, TypeError):
                raw_w, raw_h = 0, 0

            # Rotación — iPhones graban horizontal con rotate=90/270
            rotation = _get_rotation_from_stream(vstream)
            rotated = rotation in (90, 270)
            # ffmpeg aplica autorotate automáticamente al decodificar,
            # así que las dimensiones reales de los frames serán las invertidas
            width  = raw_h if rotated else raw_w
            height = raw_w if rotated else raw_h

            is_vfr = abs(avg_fps - r_fps) > 0.5  # diferencia significativa → es VFR

            return {
                "path":        video_path,
                "fps":         fps,
                "avg_fps":     avg_fps,
                "r_fps":       r_fps,
                "is_vfr":      is_vfr,
                "frame_count": frame_count,
                "width":       width,
                "height":      height,
                "raw_width":   raw_w,
                "raw_height":  raw_h,
                "rotation":    rotation,
                "duration":    duration,
                "has_audio":   astream is not None,
                "has_subtitles": bool(subtitle_codecs),
                "subtitle_codecs": subtitle_codecs,
                "codec_video": vstream.get("codec_name", "unknown"),
                "codec_audio": astream.get("codec_name", "none") if astream else "none",
                "color_space": vstream.get("color_space") or "",
                "color_primaries": vstream.get("color_primaries") or "",
                "color_transfer": vstream.get("color_transfer") or "",
                "color_range": vstream.get("color_range") or "",
                "orientation": "vertical" if height > width else "horizontal",
                "file_size_mb": round(int(fmt.get("size", 0)) / 1_048_576, 1),
            }
        except NoVideoStream:
            raise
        except Exception as e:
            print(f"[ffprobe] Error: {e} - usando fallback OpenCV")

    # ── Fallback a OpenCV ─────────────────────────────────────────────────────
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se puede abrir el vídeo: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fc > 0:
        try:
            _lookback = 60
            _start = max(0, fc - _lookback)
            _cap = cv2.VideoCapture(video_path)
            _cap.set(cv2.CAP_PROP_POS_FRAMES, _start)
            _fi = _start
            _last_good = -1
            while True:
                _ret, _ = _cap.read()
                if not _ret:
                    break
                _last_good = _fi
                _fi += 1
            _cap.release()
            if _last_good >= 0:
                fc = _last_good + 1
        except Exception:
            pass
    if w <= 0 or h <= 0 or fc <= 0:
        raise NoVideoStream(f"El archivo no contiene stream de vídeo usable: {video_path}")
    return {
        "path":        video_path,
        "fps":         fps, "avg_fps": fps, "r_fps": fps,
        "is_vfr":      False,
        "frame_count": fc,
        "width":       w, "height": h, "raw_width": w, "raw_height": h,
        "rotation":    0,
        "duration":    fc / fps if fps else 0,
        "has_audio":   None,    # OpenCV no puede detectar presencia de audio — desconocido
        "has_subtitles": False,
        "subtitle_codecs": [],
        "codec_video": "unknown", "codec_audio": "unknown",
        "color_space": "", "color_primaries": "", "color_transfer": "", "color_range": "",
        "orientation": "vertical" if h > w else "horizontal",
        "file_size_mb": 0,
    }


# ─── Renderizado pipe (decode → Python → encode) ─────────────────────────────

def _build_decode_cmd(ffmpeg_bin: str, input_path: str, fps: float) -> list[str]:
    """
    Construye el comando ffmpeg decoder que vuelca frames BGR24 a stdout.

    -fps_mode cfr convierte VFR a CFR (crítico para iPhone / contenido VFR).
    -r fija el FPS exacto para que el encoder reciba el timing correcto.
    -map 0:v:0 selecciona explícitamente el primer stream de vídeo.
    -sws_flags lanczos+accurate_rnd: conversión de color sin artefactos.
    """
    return [
        ffmpeg_bin,
        "-sws_flags", "lanczos+accurate_rnd",
        "-i", input_path,
        "-fps_mode", "cfr",
        "-r", f"{fps:.6f}",
        "-map", "0:v:0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "pipe:1",
    ]


def _build_encode_cmd(
    ffmpeg_bin: str,
    output_path: str,
    input_path: str,
    width: int,
    height: int,
    fps: float,
    codec: str,
    hw_args_list: list[str],
    audio_args: list[str],
    subtitle_map_args: list[str],
    subtitle_codec_args: list[str],
    out_color_space: str,
    out_color_primaries: str,
    out_color_transfer: str,
    out_color_range: str,
) -> list[str]:
    """
    Construye el comando ffmpeg encoder que lee frames BGR24 desde stdin
    y muxea con el audio del archivo original.

    scale=trunc(iw/2)*2:trunc(ih/2)*2  garantiza dimensiones pares para H.264/H.265.
    format=yuv420p convierte antes del encoder hardware (NVENC con bgr24 directo
    produce vídeo verde por un bug del driver).
    """
    return [
        ffmpeg_bin, "-y",
        # Input 0: frames censados desde stdin
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        # Input 1: archivo original (para audio y subtítulos)
        "-i", input_path,
        # Mapeo
        "-map", "0:v:0",
        "-map", "1:a?",
        *subtitle_map_args,
        # Vídeo encode
        "-c:v", codec,
        *hw_args_list,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        # Metadatos de color
        "-colorspace",      out_color_space,
        "-color_primaries",  out_color_primaries,
        "-color_trc",       out_color_transfer,
        "-color_range",     out_color_range,
        # Audio
        *audio_args,
        # Subtítulos
        *subtitle_codec_args,
        "-movflags", "+faststart",
        output_path,
    ]


def render_via_pipe(
    ffmpeg_bin: str,
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    fps: float,
    process_fn: Callable[[np.ndarray, int], np.ndarray],
    progress_cb: Callable[[int, int], None] | None = None,
    total_frames: int = 0,
    crf: int = 18,
    preset: str = "fast",
    is_vfr: bool = False,
    use_hw_encode: bool = True,
    cancel_cb: Callable[[], bool] | None = None,
    audio_codec: str | None = None,
    subtitle_codecs: list[str] | None = None,
    color_space: str | None = None,
    color_primaries: str | None = None,
    color_transfer: str | None = None,
    color_range: str | None = None,
    warn_cb: Callable[[str], None] | None = None,
    _hdr_warned: bool = False,
):
    """
    Pipeline sin archivos temporales y sin doble compresión:

        ffmpeg (decode) ──stdout[BGR24]──► Python (censura) ──stdin[BGR24]──► ffmpeg (encode+mux)

    Maneja:
    - VFR iPhone: -fps_mode cfr convierte a CFR en el decoder
    - Rotación: ffmpeg aplica autorotate automáticamente (por defecto)
    - Dimensiones impares: filtro scale=trunc(iw/2)*2:trunc(ih/2)*2 en el encoder
    - Hardware encode: intenta NVENC/QSV/AMF, fallback a libx264 CRF 18
    - Audio: streamcopy si es compatible; AAC si MP4 lo requiere
    - Cualquier codec de entrada: ffmpeg decodifica software cualquier formato
    """
    frame_bytes = width * height * 3  # BGR24: 3 bytes por pixel

    # ── Advertencia HDR ───────────────────────────────────────────────────────
    # OpenCV y el pipe bgr24 no aplican tone-mapping HDR: el vídeo de salida
    # tendrá los píxeles con los valores originales pero etiquetado como SDR
    # (BT.709). En HDR moderado (bt2020-10/bt2020-12) el resultado es aceptable;
    # en PQ (smpte2084) o HLG (arib-std-b67) puede quedar muy oscuro/saturado.
    _HDR_TRANSFERS  = {"smpte2084", "arib-std-b67"}
    _HDR_PRIMARIES  = {"bt2020"}
    _src_transfer   = (color_transfer  or "").lower()
    _src_primaries  = (color_primaries or "").lower()
    if not _hdr_warned and (_src_transfer in _HDR_TRANSFERS or _src_primaries in _HDR_PRIMARIES):
        _hdr_note = (
            "El vídeo de entrada es HDR "
            f"(primarios: {_src_primaries or '?'}, transferencia: {_src_transfer or '?'}). "
            "El render convierte a SDR/BT.709 sin tone-mapping. "
            "El resultado puede quedar más oscuro o con colores menos saturados que el original."
        )
        if warn_cb:
            warn_cb(_hdr_note)
        else:
            print(f"[Renderer] Aviso HDR: {_hdr_note}")

    # ── Elegir encoder ────────────────────────────────────────────────────────
    if use_hw_encode:
        codec, hw_args = best_encoder(ffmpeg_bin, prefer_hevc=False)
        # Si la cascada HW no encontró nada y cayó a libx264, usar los ajustes del usuario
        if codec == "libx264":
            hw_args = f"-crf {crf} -preset {preset}"
    else:
        codec, hw_args = "libx264", f"-crf {crf} -preset {preset}"

    hw_args_list = shlex.split(hw_args) if hw_args else []

    # ── Resolver metadatos de color ───────────────────────────────────────────
    audio_args = _audio_args_for_output(output_path, audio_codec)
    subtitle_map_args, subtitle_codec_args = _subtitle_args_for_output(output_path, subtitle_codecs)
    out_color_space     = (color_space    or "bt709").lower()
    out_color_primaries = (color_primaries or "bt709").lower()
    out_color_transfer  = (color_transfer  or "bt709").lower()
    valid_color_spaces     = {"bt709", "bt470bg", "smpte170m", "smpte240m", "bt2020nc", "bt2020c"}
    valid_color_primaries  = {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "bt2020"}
    valid_color_transfers  = {"bt709", "smpte170m", "smpte240m", "bt2020-10", "bt2020-12", "iec61966-2-1"}
    if out_color_space not in valid_color_spaces:
        out_color_space = "bt709"
    if out_color_primaries not in valid_color_primaries:
        out_color_primaries = "bt709"
    if out_color_transfer not in valid_color_transfers:
        # HDR transfer coerced → coerce también space+primaries para tags consistentes
        # (bt2020+bt709 combinados producen tinte verde en Telegram y algunos players).
        out_color_transfer  = "bt709"
        out_color_space     = "bt709"
        out_color_primaries = "bt709"
    out_color_range = "pc" if (color_range or "").lower() in {"pc", "jpeg", "full"} else "tv"

    # ── Construir comandos con helpers extraídos ──────────────────────────────
    cmd_decode = _build_decode_cmd(ffmpeg_bin, input_path, fps)
    cmd_encode = _build_encode_cmd(
        ffmpeg_bin, output_path, input_path, width, height, fps,
        codec, hw_args_list,
        audio_args, subtitle_map_args, subtitle_codec_args,
        out_color_space, out_color_primaries, out_color_transfer, out_color_range,
    )

    def _terminate_process(proc: subprocess.Popen | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        except Exception:
            pass

    proc_decode = None
    proc_encode = None
    try:
        proc_decode = subprocess.Popen(
            cmd_decode,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # drenado en hilo — necesario para diagnóstico de errores
            bufsize=frame_bytes * 4,
        )
        proc_encode = subprocess.Popen(
            cmd_encode,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=frame_bytes * 4,
        )
    except Exception:
        _terminate_process(proc_decode)
        raise

    frame_idx = 0
    cancelled = False
    decode_error_lines: deque[bytes] = deque(maxlen=40)
    encode_error_lines: deque[bytes] = deque(maxlen=80)

    def _drain_decode_stderr():
        if proc_decode.stderr is None:
            return
        for line in iter(proc_decode.stderr.readline, b""):
            decode_error_lines.append(line)

    def _drain_encode_stderr():
        if proc_encode.stderr is None:
            return
        for line in iter(proc_encode.stderr.readline, b""):
            encode_error_lines.append(line)

    decode_stderr_thread = threading.Thread(target=_drain_decode_stderr, daemon=True)
    decode_stderr_thread.start()
    encode_stderr_thread = threading.Thread(target=_drain_encode_stderr, daemon=True)
    encode_stderr_thread.start()

    if progress_cb:
        progress_cb(0, total_frames or 1)

    try:
        while True:
            if cancel_cb and cancel_cb():
                cancelled = True
                break

            raw = proc_decode.stdout.read(frame_bytes)
            if len(raw) == 0:
                break  # EOF o error
            if len(raw) != frame_bytes:
                raise RuntimeError(
                    f"FFmpeg entregó un frame incompleto: {len(raw)} bytes de {frame_bytes}."
                )

            if cancel_cb and cancel_cb():
                cancelled = True
                break

            # Crear array sin copia extra (tobytes() al escribir sí copia)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            processed = process_fn(frame, frame_idx)

            if cancel_cb and cancel_cb():
                cancelled = True
                break

            proc_encode.stdin.write(processed.tobytes())
            frame_idx += 1

            if progress_cb and (frame_idx == 1 or frame_idx % 5 == 0):
                progress_cb(frame_idx, total_frames or frame_idx + 1)

    except BrokenPipeError:
        # El encoder cerró su stdin (error en ffmpeg) — recogemos el stderr abajo
        pass
    finally:
        for stream in (proc_decode.stdout, proc_encode.stdin):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass

        def _wait_or_stop(proc: subprocess.Popen):
            try:
                if cancelled and proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                proc.wait()

        _wait_or_stop(proc_decode)
        _wait_or_stop(proc_encode)
        decode_stderr_thread.join(timeout=1)
        encode_stderr_thread.join(timeout=1)

    if cancelled:
        raise RenderCancelled("Render cancelado por el usuario.")

    # Comprobar decoder ANTES de cualquier reintento: si el decode falló
    # (vídeo corrupto, codec no soportado, ruta incorrecta) no tiene sentido
    # reintentar con otros ajustes de encoder o subtítulos.
    if proc_decode.returncode not in (0, None):
        decode_error = b"".join(decode_error_lines).decode(errors="replace")[-1000:]
        raise RuntimeError(
            f"FFmpeg decoder falló (código {proc_decode.returncode}) — "
            f"vídeo corrupto, codec no soportado, ruta incorrecta o lectura interrumpida.\n"
            f"Frames procesados antes del fallo: {frame_idx}\n"
            f"Archivo: {input_path}\n"
            + (f"Detalle: {decode_error}" if decode_error.strip() else "")
        )

    if proc_encode.returncode != 0 and subtitle_map_args:
        print("[Renderer] Subtitulos incompatibles con el contenedor - reintentando sin subtitulos (el progreso se reinicia)")
        if warn_cb:
            warn_cb("Subtítulos incompatibles con el contenedor: render reintentado sin subtítulos.")
        if progress_cb:
            progress_cb(0, total_frames or 1)   # avisa a la UI que el progreso se reinicia
        return render_via_pipe(
            ffmpeg_bin, input_path, output_path, width, height, fps,
            process_fn, progress_cb, total_frames, crf, preset,
            is_vfr=is_vfr, use_hw_encode=use_hw_encode, cancel_cb=cancel_cb,
            audio_codec=audio_codec,
            subtitle_codecs=[],
            color_space=color_space,
            color_primaries=color_primaries,
            color_transfer=color_transfer,
            color_range=color_range,
            warn_cb=warn_cb,
            _hdr_warned=True,
        )

    # Si el encoder falló con hw, reintentar con software
    if proc_encode.returncode != 0 and use_hw_encode and codec != "libx264":
        print(f"[Renderer] Hardware encoder '{codec}' fallo - reintentando con libx264 (el progreso se reinicia)")
        if warn_cb:
            warn_cb(f"Encoder hardware '{codec}' falló: reintentando con libx264 por CPU.")
        if progress_cb:
            progress_cb(0, total_frames or 1)
        return render_via_pipe(
            ffmpeg_bin, input_path, output_path, width, height, fps,
            process_fn, progress_cb, total_frames, crf, preset,
            is_vfr=is_vfr, use_hw_encode=False, cancel_cb=cancel_cb,
            audio_codec=audio_codec,
            subtitle_codecs=subtitle_codecs,
            color_space=color_space,
            color_primaries=color_primaries,
            color_transfer=color_transfer,
            color_range=color_range,
            warn_cb=warn_cb,
            _hdr_warned=True,
        )

    if proc_encode.returncode != 0:
        encode_error = b"".join(encode_error_lines)
        raise RuntimeError(
            f"FFmpeg encode falló (código {proc_encode.returncode}) con codec '{codec}':\n"
            + encode_error.decode(errors="replace")[-2000:]
        )

    if progress_cb:
        progress_cb(frame_idx, total_frames or max(frame_idx, 1))

    return frame_idx
