"""
Save / load .gfs (GhostFrame session) files in the project cache directory.
Sessions are versioned, HMAC-signed, and tied to a fast content fingerprint of
the video, so stale or tampered sessions are rejected automatically.
"""
from __future__ import annotations

import base64
import os
import json
import hashlib
import hmac
import io
import threading
import uuid
import numpy as np

SESSION_VERSION = 6  # bump this to force re-analysis on format changes
_MAGIC = b"GFS6\n"
_KEY_PATH = os.path.join(os.path.expanduser("~"), ".ghostframe_studio_session.key")
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SESSION_CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache", "sessions")
_FINGERPRINT_CACHE: dict[tuple[str, int, int], str] = {}
_FINGERPRINT_LOCK = threading.Lock()   # protects concurrent BatchWorker + MainWindow access
_FNAME_CACHE: dict[tuple[str, int], str] = {}
_FNAME_LOCK = threading.Lock()


def _key_path_candidates() -> list[str]:
    paths = [_KEY_PATH]
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        paths.append(os.path.join(local_appdata, "GhostFrame Studio", "ghostframe_session.key"))
    paths.append(os.path.join(os.path.dirname(__file__), "..", "logs", "ghostframe_session.key"))
    return [os.path.abspath(p) for p in paths]


def _session_key() -> bytes:
    for path in _key_path_candidates():
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    key = f.read().strip()
                    if len(key) >= 32:
                        return key
        except Exception:
            continue
    key = os.urandom(32).hex().encode("ascii")
    last_error = None
    for path in _key_path_candidates():
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(key)
            return key
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"No se pudo crear la clave de sesión: {last_error}")


def _sign(data: bytes) -> bytes:
    return hmac.new(_session_key(), data, hashlib.sha256).hexdigest().encode("ascii")


def _video_fingerprint(video_path: str) -> str:
    """
    Fast stale-video fingerprint.

    Full hashing multi-GB videos on every checkpoint makes analysis look frozen.
    Small/medium videos are hashed completely. Large videos use a dense sampled
    hash across the whole file to avoid multi-GB reads on every checkpoint.
    """
    stat = os.stat(video_path)
    size = int(stat.st_size)
    cache_key = (os.path.normcase(os.path.abspath(video_path)), size, int(getattr(stat, "st_mtime_ns", 0)))
    with _FINGERPRINT_LOCK:
        cached = _FINGERPRINT_CACHE.get(cache_key)
    if cached:
        return cached

    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    # mtime_ns eliminado intencionalmente: copiar/mover el mismo vídeo
    # cambia el mtime pero no el contenido → sesión válida rechazada.
    # La identidad del vídeo se establece solo por tamaño + muestras de contenido.

    full_hash_limit = 64 * 1024 * 1024   # >64 MB → hash muestral para no congelar en checkpoints
    chunk_size = 2 * 1024 * 1024
    if size <= full_hash_limit:
        offsets = range(0, size, chunk_size)
    else:
        # 32 puntos interiores uniformes + inicio + fin → probabilidad de colisión
        # entre dos vídeos distintos del mismo tamaño es astronómicamente baja.
        offsets = {0, max(0, size - chunk_size)}
        for numerator in range(1, 32):
            offsets.add(max(0, (size * numerator // 32) - chunk_size // 2))

    with open(video_path, "rb") as f:
        for offset in sorted(offsets):
            f.seek(offset)
            h.update(offset.to_bytes(8, "little", signed=False))
            h.update(f.read(min(chunk_size, max(0, size - offset))))

    digest = h.hexdigest()
    with _FINGERPRINT_LOCK:
        if len(_FINGERPRINT_CACHE) >= 64:   # cap size — evict only when full
            _FINGERPRINT_CACHE.clear()
        _FINGERPRINT_CACHE[cache_key] = digest
    return digest


def _array_to_payload(arr: np.ndarray) -> dict:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return {
        "npy_b64": base64.b64encode(bio.getvalue()).decode("ascii"),
    }


def _array_from_payload(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["npy_b64"].encode("ascii"))
    return np.load(io.BytesIO(raw), allow_pickle=False)


def _person_to_payload(person) -> dict:
    return {
        "person_id": int(person.person_id),
        "custom_label": getattr(person, "_custom_label", None),
        "is_manual_split": bool(getattr(person, "_is_manual_split", False)),
        "thumbnail": _array_to_payload(person.thumbnail),
        "embeddings": [
            _array_to_payload(np.asarray(e, dtype=np.float32))
            for e in person.embeddings
        ],
        "frame_data": {
            str(int(fi)): {
                "bbox": [int(v) for v in data.get("bbox", [])],
                "det_score": float(data.get("det_score", 0.0)),
                **({"interpolated": True} if data.get("interpolated", False) else {}),
            }
            for fi, data in person.frame_data.items()
        },
    }


def _person_from_payload(payload: dict):
    from core.face_tracker import Person

    person = Person(int(payload["person_id"]), _array_from_payload(payload["thumbnail"]))
    person.embeddings = [
        _array_from_payload(e).astype(np.float32)
        for e in payload.get("embeddings", [])
    ]
    person.frame_data = {
        int(fi): {
            "bbox": [int(v) for v in data.get("bbox", [])],
            "det_score": float(data.get("det_score", 0.0)),
            **({"interpolated": True} if data.get("interpolated", False) else {}),
        }
        for fi, data in payload.get("frame_data", {}).items()
    }
    if payload.get("custom_label"):
        person._custom_label = payload["custom_label"]
    person._is_manual_split = bool(payload.get("is_manual_split", False))
    return person


def _legacy_session_path_for(video_path: str) -> str:
    base, _ = os.path.splitext(video_path)
    return base + ".gfs"


def _session_cache_filename(video_path: str) -> str:
    """
    Stable filename for any source video path.

    Uses size plus small sampled content chunks, not the absolute folder path.
    Moving the same video to another folder can still find the same project cache.
    Stale detection still comes from the signed video fingerprint in the payload.
    """
    stat = os.stat(video_path)
    size = int(stat.st_size)
    cache_key = (os.path.normcase(os.path.abspath(video_path)), size)
    with _FNAME_LOCK:
        cached = _FNAME_CACHE.get(cache_key)
        if cached:
            return cached

    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    chunk_size = 256 * 1024
    offsets = {0}
    if size > chunk_size:
        offsets.add(max(0, size // 2 - chunk_size // 2))
        offsets.add(max(0, size - chunk_size))
    with open(video_path, "rb") as f:
        for offset in sorted(offsets):
            f.seek(offset)
            h.update(offset.to_bytes(8, "little", signed=False))
            h.update(f.read(min(chunk_size, max(0, size - offset))))
    digest = h.hexdigest()[:20]
    # Prefijo legible (basename) solo para facilitar depuración manual;
    # la identidad real es el digest de contenido (renombrar el vídeo cambia
    # el prefijo pero el digest es el mismo → aún se encuentra la caché).
    # El prefijo no forma parte de la búsqueda en session_path_candidates().
    base = os.path.splitext(os.path.basename(video_path))[0].strip() or "video"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)[:40]
    result = f"{digest}_{safe}.gfs"
    with _FNAME_LOCK:
        if len(_FNAME_CACHE) >= 128:
            _FNAME_CACHE.clear()
        _FNAME_CACHE[cache_key] = result
    return result


def _path_based_session_cache_filename(video_path: str) -> str:
    """Old cache naming scheme kept only to load/delete existing caches."""
    abs_path = os.path.normcase(os.path.abspath(video_path))
    digest = hashlib.sha256(abs_path.encode("utf-8", errors="surrogatepass")).hexdigest()[:20]
    base = os.path.splitext(os.path.basename(video_path))[0].strip() or "video"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)[:60]
    return f"{safe}_{digest}.gfs"


def session_cache_dir() -> str:
    """Directory where GhostFrame stores analysis caches."""
    return _SESSION_CACHE_DIR


def session_path_for(video_path: str) -> str:
    """Return the project-local cache path for a video's analysis session."""
    return os.path.join(_SESSION_CACHE_DIR, _session_cache_filename(video_path))


def session_path_candidates(video_path: str) -> list[str]:
    """
    Candidatos de sesión en orden de preferencia:
    1. Caché por contenido (nuevo esquema: {hash}_{basename}.gfs)
    2. Búsqueda por hash en el directorio de caché (cubre vídeos renombrados)
    3. Caché antigua por ruta absoluta (compatibilidad)
    4. Sidecar legacy junto al vídeo
    """
    preferred    = session_path_for(video_path)
    old_preferred = os.path.join(_SESSION_CACHE_DIR, _path_based_session_cache_filename(video_path))
    legacy       = _legacy_session_path_for(video_path)

    # Buscar en el directorio de caché cualquier .gfs que empiece con el hash de contenido
    # Esto permite encontrar la sesión aunque el vídeo haya sido renombrado.
    extra: list[str] = []
    content_filename = _session_cache_filename(video_path)
    content_hash     = content_filename.split("_")[0]   # primeros 20 chars del hexdigest
    cache_dir        = _SESSION_CACHE_DIR
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith(content_hash) and fname.endswith(".gfs"):
                full = os.path.join(cache_dir, fname)
                if full != preferred:
                    extra.append(full)

    paths: list[str] = []
    seen: set[str] = set()
    for candidate in [preferred] + extra + [old_preferred, legacy]:
        norm = os.path.normcase(os.path.abspath(candidate))
        if norm not in seen:
            seen.add(norm)
            paths.append(candidate)
    return paths


def save_session(
    video_path: str,
    tracker,
    video_info: dict,
    step: int,
) -> str:
    """Persist tracker state. Returns the .gfs file path."""
    gfs_path = session_path_for(video_path)
    os.makedirs(os.path.dirname(gfs_path), exist_ok=True)
    payload = {
        "version": SESSION_VERSION,
        "video_hash": _video_fingerprint(video_path),
        "video_info": video_info,
        "step": step,
        "persons": [_person_to_payload(p) for p in tracker.persons],
        "similarity_threshold": tracker.similarity_threshold,
        "iou_threshold": tracker.iou_threshold,
        "max_interp_gap": tracker.max_interp_gap,
        # scene_cuts: frame indices detected as scene cuts (used to skip
        # interpolation across cuts). Optional — old sessions default to empty set.
        "scene_cuts": sorted(tracker.scene_cuts),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    # Nombre único para el temp — evita colisiones si dos procesos guardan el mismo vídeo
    tmp_path = gfs_path + f".tmp_{os.getpid()}_{uuid.uuid4().hex}"
    with open(tmp_path, "wb") as f:
        f.write(_MAGIC)
        f.write(_sign(raw))
        f.write(b"\n")
        f.write(raw)
    os.replace(tmp_path, gfs_path)
    return gfs_path


def load_session(video_path: str):
    """
    Load a saved session.
    Returns (tracker, video_info, step) or raises:
        FileNotFoundError  — no .gfs file exists for this video
        ValueError         — todos los candidatos fallaron (versión/firma/vídeo)

    Prueba los candidatos en orden y salta los que fallen con ValueError
    (versión incompatible, firma incorrecta, vídeo distinto). Solo lanza
    ValueError si todos los candidatos existentes han fallado.
    """
    from core.face_tracker import FaceTracker

    candidates = [p for p in session_path_candidates(video_path) if os.path.exists(p)]
    if not candidates:
        raise FileNotFoundError(session_path_for(video_path))

    last_error: Exception | None = None
    for gfs_path in candidates:
        try:
            result = _load_session_from_file(gfs_path, video_path)
            return result
        except Exception as exc:
            last_error = exc
            continue   # intentar el siguiente candidato

    raise last_error or ValueError("No se pudo cargar ninguna sesión válida.")


def _load_session_from_file(gfs_path: str, video_path: str):
    """Carga y valida un único archivo .gfs. Lanza ValueError si no es válido."""
    from core.face_tracker import FaceTracker

    with open(gfs_path, "rb") as f:
        blob = f.read()
    if not blob.startswith(_MAGIC):
        raise ValueError("Formato de sesión antiguo — re-analiza para actualizar.")
    try:
        sig, raw = blob[len(_MAGIC):].split(b"\n", 1)
    except ValueError as exc:
        raise ValueError("Formato de sesión inválido — re-analiza para actualizar.") from exc
    if not hmac.compare_digest(sig, _sign(raw)):
        raise ValueError("Firma de sesión incorrecta — el archivo fue modificado externamente.")

    payload = json.loads(raw.decode("utf-8"))

    if payload.get("version") != SESSION_VERSION:
        raise ValueError(
            f"Versión de sesión {payload.get('version')} incompatible con {SESSION_VERSION} — re-analiza para actualizar."
        )

    if payload.get("video_hash") != _video_fingerprint(video_path):
        raise ValueError("El vídeo ha cambiado desde el último análisis — re-analiza para actualizar.")

    tracker = FaceTracker(
        similarity_threshold=payload.get("similarity_threshold", 0.65),
        iou_threshold=payload.get("iou_threshold", 0.40),
        max_interp_gap=payload.get("max_interp_gap", 30),
    )
    tracker.persons = [_person_from_payload(p) for p in payload.get("persons", [])]
    # Recalibrate monotonic ID counter so new persons don't collide with loaded ones.
    if tracker.persons:
        tracker._next_id = max(p.person_id for p in tracker.persons) + 1
    # scene_cuts: backward-compatible — old sessions without this key default to empty set.
    tracker.scene_cuts = set(payload.get("scene_cuts", []))

    return tracker, payload["video_info"], payload["step"]


def delete_session(video_path: str):
    for gfs in session_path_candidates(video_path):
        if os.path.exists(gfs):
            os.remove(gfs)


def _thumbnail_hash(thumbnail: np.ndarray) -> str:
    arr = np.ascontiguousarray(thumbnail)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _thumbnail_ahash(thumbnail: np.ndarray) -> str:
    arr = np.asarray(thumbnail)
    if arr.size == 0:
        return "0" * 16
    h, w = arr.shape[:2]
    ys = np.linspace(0, max(0, h - 1), 8).astype(int)
    xs = np.linspace(0, max(0, w - 1), 8).astype(int)
    sample = arr[np.ix_(ys, xs)]
    if sample.ndim == 3 and sample.shape[2] >= 3:
        gray = sample[..., 2] * 0.299 + sample[..., 1] * 0.587 + sample[..., 0] * 0.114
    else:
        gray = sample.astype(np.float32)
    bits = gray >= float(gray.mean())
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def _hamming_hex(a: str, b: str) -> int:
    try:
        xor = int(a, 16) ^ int(b, 16)
        if hasattr(int, "bit_count"):
            return xor.bit_count()
        return bin(xor).count("1")
    except Exception:
        return 64


def export_censure_config(path: str, person_cards: list) -> str:
    entries = []
    for card in person_cards:
        config = card.get_config()
        entries.append({
            "thumbnail_hash": _thumbnail_hash(card.person.thumbnail),
            "thumbnail_ahash": _thumbnail_ahash(card.person.thumbnail),
            "enabled": bool(config.get("enabled", False)),
            "effect": config.get("effect", "blur"),
            "intensity": int(config.get("intensity", 5)),
            "padding_pct": float(config.get("padding_pct", 0.15)),
            "start_frame": int(config.get("start_frame", 0)),
            "end_frame": int(config.get("end_frame", -1)),
        })
    payload = {"version": 1, "entries": entries}
    tmp_path = path + f".tmp_{os.getpid()}_{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)
    return path


def import_censure_config(path: str, person_cards: list) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"El archivo de configuración está corrupto o no es JSON válido: {exc}"
        ) from exc
    if payload.get("version") != 1:
        raise ValueError("Versión de configuración de censura no compatible.")
    entries = {
        entry.get("thumbnail_hash"): entry
        for entry in payload.get("entries", [])
        if entry.get("thumbnail_hash")
    }
    fuzzy_entries = [
        entry for entry in payload.get("entries", [])
        if entry.get("thumbnail_ahash")
    ]
    applied = 0
    used_fuzzy: set[int] = set()   # índices de fuzzy_entries ya asignados (1 entrada → 1 persona)
    for card in person_cards:
        entry = entries.get(_thumbnail_hash(card.person.thumbnail))
        if not entry:
            current = _thumbnail_ahash(card.person.thumbnail)
            # Puntuar solo entradas fuzzy aún no asignadas
            scored = [
                (dist, idx)
                for idx, candidate in enumerate(fuzzy_entries)
                if idx not in used_fuzzy
                for dist in [_hamming_hex(current, candidate["thumbnail_ahash"])]
            ]
            scored.sort(key=lambda item: item[0])
            if scored and scored[0][0] <= 5:
                best_idx = scored[0][1]
                entry    = fuzzy_entries[best_idx]
                used_fuzzy.add(best_idx)   # marcar como usado para no reutilizarlo
        if not entry:
            continue
        card.apply_config(entry)
        applied += 1
    return applied


def export_detection_log(tracker, video_info: dict, output_path: str):
    """
    Export a JSON file listing every person and all their timestamps.
    Useful for legal review or audit.
    """
    fps = video_info.get("fps", 25.0) or 25.0
    log = {
        "video": {k: v for k, v in video_info.items() if k != "path"},
        "video_path": video_info.get("path", ""),
        "persons": [],
    }

    for person in sorted(tracker.persons, key=lambda p: p.person_id):
        frames = sorted(person.frame_data.keys())
        timestamps = [round(fi / fps, 3) for fi in frames]
        # Build contiguous segments
        segments = []
        if frames:
            seg_start = frames[0]
            seg_prev = frames[0]
            for fi in frames[1:]:
                if fi - seg_prev > 5:  # gap > 5 frames = new segment
                    segments.append({
                        "start_frame": seg_start,
                        "end_frame": seg_prev,
                        "start_sec": round(seg_start / fps, 3),
                        "end_sec": round(seg_prev / fps, 3),
                    })
                    seg_start = fi
                seg_prev = fi
            segments.append({
                "start_frame": seg_start,
                "end_frame": seg_prev,
                "start_sec": round(seg_start / fps, 3),
                "end_sec": round(seg_prev / fps, 3),
            })

        log["persons"].append({
            "person_id": person.person_id,
            "label": getattr(person, "_custom_label", None) or f"Persona {person.person_id + 1}",
            "total_frames": person.frame_count,
            "segments": segments,
            "all_timestamps_sec": timestamps,
        })

    tmp_path = output_path + f".tmp_{os.getpid()}_{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)
