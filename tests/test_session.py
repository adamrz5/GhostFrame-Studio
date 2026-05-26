from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import session as session_mod
from core.face_tracker import FaceTracker, Person


def _tracker() -> FaceTracker:
    tracker = FaceTracker()
    person = Person(0, np.zeros((8, 8, 3), dtype=np.uint8))
    person.add_observation(
        frame_idx=25,
        bbox=[1, 2, 6, 7],
        det_score=0.9,
        embedding=np.ones(512, dtype=np.float32),
    )
    tracker.persons = [person]
    return tracker


def _signed_session(payload: dict) -> bytes:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return session_mod._MAGIC + session_mod._sign(raw) + b"\n" + raw


def test_save_session_load_session_roundtrip(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(session_mod, "_video_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(session_mod, "_SESSION_CACHE_DIR", str(tmp_path / "cache"))

    info = {"path": str(video), "fps": 25.0, "frame_count": 50}
    saved_path = session_mod.save_session(str(video), _tracker(), info, step=2)

    loaded, loaded_info, step = session_mod.load_session(str(video))

    assert saved_path.startswith(str(tmp_path / "cache"))
    assert not (tmp_path / "input.gfs").exists()
    assert step == 2
    assert loaded_info == info
    assert len(loaded.persons) == 1
    assert loaded.persons[0].frame_data[25]["bbox"] == [1, 2, 6, 7]


def test_load_session_wrong_version_raises(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(session_mod, "_video_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(session_mod, "_SESSION_CACHE_DIR", str(tmp_path / "cache"))

    payload = {
        "version": session_mod.SESSION_VERSION + 1,
        "video_hash": "fingerprint",
        "video_info": {},
        "step": 1,
        "persons": [],
    }
    gfs = session_mod.session_path_for(str(video))
    (tmp_path / "cache").mkdir()
    Path(gfs).write_bytes(_signed_session(payload))

    with pytest.raises(ValueError):
        session_mod.load_session(str(video))


def test_load_session_tampered_content_raises(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(session_mod, "_video_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(session_mod, "_SESSION_CACHE_DIR", str(tmp_path / "cache"))
    session_mod.save_session(str(video), _tracker(), {"path": str(video)}, step=1)

    gfs = Path(session_mod.session_path_for(str(video)))
    blob = bytearray(gfs.read_bytes())
    blob[-1] = ord("}")
    gfs.write_bytes(bytes(blob) + b" ")

    with pytest.raises(ValueError):
        session_mod.load_session(str(video))


def test_load_session_legacy_sidecar_still_supported(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(session_mod, "_video_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(session_mod, "_SESSION_CACHE_DIR", str(tmp_path / "cache"))

    payload = {
        "version": session_mod.SESSION_VERSION,
        "video_hash": "fingerprint",
        "video_info": {"path": str(video)},
        "step": 3,
        "persons": [],
    }
    (tmp_path / "input.gfs").write_bytes(_signed_session(payload))

    tracker, info, step = session_mod.load_session(str(video))

    assert tracker.persons == []
    assert info["path"] == str(video)
    assert step == 3


def test_video_fingerprint_consistent(tmp_path):
    """El mismo archivo siempre produce el mismo hash (contenido, no nombre/mtime)."""
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake video content " * 500)
    h1 = session_mod._video_fingerprint(str(video))
    h2 = session_mod._video_fingerprint(str(video))
    assert h1 == h2
    assert len(h1) == 64   # SHA-256 hex


def test_video_fingerprint_different_content(tmp_path):
    """Archivos con distinto contenido producen hashes distintos."""
    v1 = tmp_path / "a.mp4"
    v2 = tmp_path / "b.mp4"
    v1.write_bytes(b"content A " * 500)
    v2.write_bytes(b"content B " * 500)
    assert session_mod._video_fingerprint(str(v1)) != session_mod._video_fingerprint(str(v2))


def test_video_fingerprint_stable_on_rename(tmp_path):
    """Renombrar el archivo no cambia el hash (identidad por contenido, no por nombre)."""
    video = tmp_path / "original.mp4"
    video.write_bytes(b"fake video content " * 500)
    h_original = session_mod._video_fingerprint(str(video))
    renamed = tmp_path / "renamed_copy.mp4"
    video.rename(renamed)
    h_renamed = session_mod._video_fingerprint(str(renamed))
    assert h_original == h_renamed


def test_export_detection_log_valid_json(tmp_path):
    out = tmp_path / "detections.json"
    tracker = _tracker()

    session_mod.export_detection_log(
        tracker,
        {"path": "video.mp4", "fps": 25.0, "frame_count": 50},
        str(out),
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["video_path"] == "video.mp4"
    assert data["persons"][0]["person_id"] == 0
    assert data["persons"][0]["all_timestamps_sec"] == [1.0]


# ── Nuevos tests añadidos ──────────────────────────────────────────────────────

def test_next_id_recalibration_after_load(tmp_path, monkeypatch):
    """
    Tras cargar una sesión, tracker._next_id debe ser max(person_id)+1
    para que las nuevas personas no colisionen con las cargadas.
    """
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video data" * 100)
    monkeypatch.setattr(session_mod, "_video_fingerprint", lambda _path: "fp")
    monkeypatch.setattr(session_mod, "_SESSION_CACHE_DIR", str(tmp_path / "cache"))

    tracker = FaceTracker()
    for pid in (3, 7, 12):
        p = Person(pid, np.zeros((8, 8, 3), dtype=np.uint8))
        p.add_observation(pid * 10, [0, 0, 10, 10], 0.9, np.ones(512, dtype=np.float32))
        tracker.persons.append(p)
    tracker._next_id = 13

    session_mod.save_session(str(video), tracker, {}, step=1)
    loaded, _, _ = session_mod.load_session(str(video))

    assert loaded._next_id == 13   # max(3,7,12) + 1


def test_scene_cuts_roundtrip(tmp_path, monkeypatch):
    """scene_cuts se serializa y se recupera correctamente."""
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video data" * 100)
    monkeypatch.setattr(session_mod, "_video_fingerprint", lambda _path: "fp2")
    monkeypatch.setattr(session_mod, "_SESSION_CACHE_DIR", str(tmp_path / "cache2"))

    tracker = _tracker()
    tracker.scene_cuts = {150, 300, 750}

    session_mod.save_session(str(video), tracker, {}, step=5)
    loaded, _, _ = session_mod.load_session(str(video))

    assert loaded.scene_cuts == {150, 300, 750}


def test_scene_cuts_missing_in_old_session_defaults_empty(tmp_path, monkeypatch):
    """Sesiones antiguas sin 'scene_cuts' deben cargarse sin error (set vacío)."""
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video data" * 100)
    monkeypatch.setattr(session_mod, "_video_fingerprint", lambda _path: "fp3")
    monkeypatch.setattr(session_mod, "_SESSION_CACHE_DIR", str(tmp_path / "cache3"))

    # Construir payload sin 'scene_cuts' (como una sesión antigua)
    payload = {
        "version": session_mod.SESSION_VERSION,
        "video_hash": "fp3",
        "video_info": {},
        "step": 1,
        "persons": [],
        "similarity_threshold": 0.65,
        "iou_threshold": 0.40,
        "max_interp_gap": 30,
        # "scene_cuts" ausente — simula sesión v6 pre-optimización
    }
    (tmp_path / "cache3").mkdir(parents=True, exist_ok=True)
    gfs = Path(session_mod.session_path_for(str(video)))
    gfs.write_bytes(_signed_session(payload))

    loaded, _, _ = session_mod.load_session(str(video))
    assert loaded.scene_cuts == set()


def test_fingerprint_cache_thread_safe(tmp_path):
    """
    Acceso concurrente a _FINGERPRINT_CACHE desde varios hilos no lanza
    excepciones ni produce datos corruptos.
    """
    import threading as _threading

    video = tmp_path / "video.mp4"
    video.write_bytes(b"content " * 1000)

    errors = []

    def _work():
        try:
            for _ in range(10):
                session_mod._video_fingerprint(str(video))
        except Exception as exc:
            errors.append(exc)

    threads = [_threading.Thread(target=_work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errores en acceso concurrente: {errors}"


def test_fingerprint_cache_eviction(tmp_path, monkeypatch):
    """
    El cache no crece ilimitadamente: al superar 64 entradas se vacía
    y la siguiente llamada recalcula el hash sin fallo.
    """
    # Limpiar cache antes del test para que el conteo sea predecible
    with session_mod._FINGERPRINT_LOCK:
        session_mod._FINGERPRINT_CACHE.clear()

    videos = []
    for i in range(70):
        v = tmp_path / f"v{i}.mp4"
        v.write_bytes(f"unique content {i} " .encode() * 100)
        videos.append(v)

    for v in videos:
        session_mod._video_fingerprint(str(v))

    # Después de 70 inserciones el cache ha purgado y tiene como máximo 6 entradas
    with session_mod._FINGERPRINT_LOCK:
        assert len(session_mod._FINGERPRINT_CACHE) <= 64

    # El hash sigue siendo correcto tras la evicción
    h1 = session_mod._video_fingerprint(str(videos[0]))
    h2 = session_mod._video_fingerprint(str(videos[0]))
    assert h1 == h2
