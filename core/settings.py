"""
Configuración centralizada y persistente de GhostFrame Studio.
Se guarda como JSON en el directorio home del usuario.
"""
from __future__ import annotations

import json
import os
import threading as _threading_settings
import uuid

_PATH = os.path.join(os.path.expanduser("~"), ".ghostframe_studio_settings.json")

DEFAULTS: dict = {
    # Análisis
    "analysis_step":         5,      # analizar 1 de cada N frames
    "similarity_threshold":  0.65,   # similitud coseno ArcFace mínima (literatura ArcFace: ≥0.6–0.65 = misma persona)
    "iou_threshold":         0.40,   # IoU para tracking in-scene
    "max_interp_gap":        30,     # máximo gap a interpolar (frames)
    "min_det_score":         0.50,   # score mínimo para aceptar una detección
    "checkpoint_every":      500,    # guardar .gfs parcial cada N frames

    # Modelo / hardware
    "execution_provider":    "cuda", # "auto" | "cpu" | "cuda" | "directml"
    "model_name":            "auto", # "auto" | "buffalo_l" | "buffalo_s"
    "det_size":              320,    # tamaño de entrada del detector (320 o 640). 320 = 4x más rápido, suficiente para entrevistas
    "use_hw_encode":         True,   # intentar hardware encoder (NVENC/QSV/AMF)

    # Renderizado
    "crf":                   18,
    "encode_preset":         "fast",
    "ffmpeg_path":           "",

    # UX
    "default_effect":        "pixelate",
    "default_intensity":     6,
    "default_padding_pct":   15,
    "open_folder_after_render": True,
    "play_speed":            1.0,    # multiplicador de velocidad del preview
}

_cache: dict | None = None
_cache_lock = _threading_settings.RLock()

# ── Reglas de saneamiento ──────────────────────────────────────────────────────
# Cada entrada: clave → función de coerción que devuelve el valor válido.
# Si la función lanza TypeError/ValueError se usa el DEFAULTS correspondiente.
_VALID_PRESETS  = {"ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"}
_VALID_EFFECTS  = {"blur", "pixelate", "blackbox"}
_VALID_PROVIDERS = {"auto", "cpu", "cuda", "directml"}
_VALID_MODELS   = {"auto", "buffalo_l", "buffalo_s"}

_COERCE: dict = {
    "analysis_step":           lambda v: max(1,   min(100,  int(v))),
    "similarity_threshold":    lambda v: max(0.0, min(1.0,  float(v))),
    "iou_threshold":           lambda v: max(0.0, min(1.0,  float(v))),
    "max_interp_gap":          lambda v: max(1,   min(500,  int(v))),
    "min_det_score":           lambda v: max(0.0, min(1.0,  float(v))),
    "checkpoint_every":        lambda v: max(10,  min(10000, int(v))),
    "crf":                     lambda v: max(0,   min(51,   int(v))),
    "det_size":                lambda v: 640 if int(v) == 640 else 320,
    "default_intensity":       lambda v: max(1,   min(10,   int(v))),
    "default_padding_pct":     lambda v: max(0,   min(100,  int(v))),
    "play_speed":              lambda v: max(0.1, min(4.0,  float(v))),
    "use_hw_encode":           lambda v: bool(v),
    "open_folder_after_render":lambda v: bool(v),
    "ffmpeg_path":             lambda v: str(v) if v else "",
    "execution_provider":      lambda v: str(v) if str(v) in _VALID_PROVIDERS  else DEFAULTS["execution_provider"],
    "model_name":              lambda v: str(v) if str(v) in _VALID_MODELS     else DEFAULTS["model_name"],
    "encode_preset":           lambda v: str(v) if str(v) in _VALID_PRESETS    else DEFAULTS["encode_preset"],
    "default_effect":          lambda v: str(v) if str(v) in _VALID_EFFECTS    else DEFAULTS["default_effect"],
}


def _sanitize(s: dict) -> dict:
    """
    Devuelve una copia de `s` con todos los valores saneados.
    Valores fuera de rango, tipo incorrecto o texto inválido se reemplazan
    silenciosamente por el default correspondiente.
    """
    out = dict(s)
    for key, coerce in _COERCE.items():
        if key in out:
            try:
                out[key] = coerce(out[key])
            except (TypeError, ValueError, KeyError):
                out[key] = DEFAULTS.get(key, out[key])
    return out


def load() -> dict:
    global _cache
    with _cache_lock:
        if _cache is not None:
            return dict(_cache)   # devolver copia — los callers no deben mutar el caché directamente
        try:
            if os.path.exists(_PATH):
                with open(_PATH, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                clean = _sanitize({**DEFAULTS, **saved})
            else:
                clean = dict(DEFAULTS)
        except Exception:
            clean = dict(DEFAULTS)
        _cache = clean
        return dict(_cache)


def save(settings: dict):
    global _cache
    clean = _sanitize(settings)   # sanear antes de escribir → JSON siempre válido
    tmp = _PATH + f".tmp_{os.getpid()}_{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        os.replace(tmp, _PATH)
        with _cache_lock:
            _cache = clean   # actualizar cache solo tras escritura exitosa → disco y memoria siempre consistentes
    except Exception as e:
        print(f"[Configuración] No se pudo guardar: {e}")
        try:
            os.remove(tmp)
        except Exception:
            pass


def get(key: str):
    # load() ya devuelve valores saneados; solo necesitamos el fallback a DEFAULTS
    with _cache_lock:
        if _cache is not None:
            return _cache.get(key, DEFAULTS.get(key))
    return load().get(key, DEFAULTS.get(key))


def set(key: str, value):
    s = load()      # ya devuelve copia — mutar `s` no afecta `_cache` hasta save()
    s[key] = value
    save(s)
