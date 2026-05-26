# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

GhostFrame Studio is a Windows PyQt5 desktop app for censoring faces in interview videos. It detects, tracks, and re-identifies people across the entire video (even after cuts or re-entries), lets the user selectively censor each person, and renders the output without quality loss.

UI direction: high-contrast black theme, white text, per-person accent colors. Do not remove per-person colors — they are used for cards, timelines, and detection overlays.

---

## Running the App

```powershell
cd C:\Users\adamr\Desktop\Proyectos\Edicion\faceblur_studio
.\.venv\Scripts\activate
python main.py
# Or open a video directly:
python main.py C:\path\to\video.mp4
```

`exe.bat` / `run_silent.bat` — shortcuts for daily use (no console window with `pythonw`).

Runtime output is tee'd to `logs/ghostframe.log`. Fallback paths: `%LOCALAPPDATA%\GhostFrame Studio\logs\` then system temp.

`Ayuda → Diagnóstico` builds a live report: Python, FFmpeg/ffprobe, ONNX providers, GPU, CUDA pip DLLs.

## Running Tests

```powershell
.\.venv\Scripts\activate
pytest tests/
# Single test file:
pytest tests/test_face_tracker.py -v
```

`tests/conftest.py` adds the project root to `sys.path` so `core.*` imports work from tests.

---

## Key Dependency Constraints

- **`numpy < 2.0`** — insightface 0.7.3 uses `np.int` (removed in NumPy 2.0); faiss-cpu 1.8.x compiled against NumPy 1.x ABI.
- **`onnxruntime` pinned at `1.19.2`** — insightface is incompatible with ≥1.20. Install exactly ONE variant: `onnxruntime`, `onnxruntime-gpu`, or `onnxruntime-directml`. They conflict.
- **`opencv-python-headless`** (not `opencv-python`) — both bundle Qt; the non-headless version causes `ImportError` with PyQt5 on Windows.
- **`onnx==1.16.1`** — exact pin; 1.17+ breaks ops used by buffalo_l/buffalo_s ONNX models.
- **Python 3.12** confirmed working. Python 3.14 is not yet compatible with any AI packages.
- **Print statements must use ASCII** (`OK`/`FALLO`, not `✓`/`✗`) — Windows cp1252 console crashes on those codepoints.

---

## GPU Acceleration

### CUDA (current dev setup — fastest)

`main.py::_preload_nvidia_dlls()` registers all `nvidia/<pkg>/bin/` dirs into `os.environ["PATH"]` and `os.add_dll_directory()` **before** any `import onnxruntime`. Required packages:

```
nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, nvidia-cudnn-cu12,
nvidia-cuda-nvrtc-cu12, nvidia-cufft-cu12, nvidia-curand-cu12,
nvidia-cusparse-cu12, nvidia-cusolver-cu12, nvidia-nvjitlink-cu12
```

If the app logs `Applied providers: ['CPUExecutionProvider']` when CUDA was requested, a DLL from that list is missing or PATH was not set before onnxruntime loaded.

`onnxruntime.get_available_providers()` lists compiled-in providers, not ones that work at runtime. `_cuda_runtime_loadable()` in `face_detector.py` does an actual ctypes DLL probe to verify.

### DirectML (any GPU, no CUDA Toolkit)

```powershell
pip uninstall onnxruntime -y
pip install onnxruntime-directml==1.19.2
```

Set provider to `auto` or `directml` in Settings. ~2.8× slower than CUDA on NVIDIA but works on AMD/Intel.

---

## Architecture

### Data Flow

```
Video file
  │
  ▼ probe_video() [ffprobe + OpenCV fallback]
  Video metadata: fps, dimensions, VFR flag, rotation, color tags
  │
  ▼ VideoReader.iter_frames(step=N) [OpenCV — rotation applied manually]
  Raw BGR frames (every Nth frame)
  │
  ▼ detect_faces() [InsightFace buffalo_l via ONNX Runtime]
  List of {bbox, embedding(512-d ArcFace), det_score, kps} per frame
  │
  ▼ FaceTracker.process_frame()  →  interpolate_bboxes()
     → consolidate_persons()  →  prune_persons()
  Finalized tracker with Person objects
  │
  ▼ session.save_session() → signed .gfs in cache/sessions/
  │
  ▼ render_video() → render_via_pipe() [two ffmpeg processes piped through Python]
  Censored output video (no temp files, no double compression)
```

### Two-Stage Identity Tracking (`core/face_tracker.py`)

**Stage 1 — IoU (in-scene):** Face in frame N overlaps a recent frame at IoU ≥ threshold (default 0.40) → same person, no embedding needed. Fast for continuous motion.

**Stage 2 — ArcFace re-id (cross-scene):** No IoU match → compare 512-d ArcFace embedding against rolling-mean embeddings of all known persons via FAISS `IndexFlatIP` (inner product = cosine similarity after L2 normalization). Falls back to scipy cosine if faiss unavailable.

**Key implementation details:**
- `_invalidate_index()` is called **once per frame** (after the full detections loop), not once per detection. This is intentional — rebuilding FAISS is O(n) and detections share `used_person_ids`, so a stale index within the same frame is safe.
- `_next_id` is a monotonic O(1) counter. `session.py` recalibrates it after loading persons (`tracker._next_id = max(ids) + 1`) so loaded IDs never collide.
- `Person.__init__` takes only `(person_id, thumbnail)`. Embeddings are added exclusively via `add_observation()`. `mean_embedding` returns a zero vector if no observations exist.
- Post-analysis passes: `consolidate_persons()` (O(n²) pairwise merge) then `prune_persons()` (drop persons with <5 real frames).

### Pipe-Based Rendering (`core/ffmpeg_utils.py` + `core/renderer.py`)

No temporary files, no double compression:

```
ffmpeg (decode) → stdout [raw BGR24] → Python (censure) → stdin [raw BGR24] → ffmpeg (encode+mux)
```

`render_video()` in `renderer.py` writes to a UUID temp file and atomically renames on success (`os.replace()`). Partial renders are deleted on error.

`verify_encoder()` caches **successes indefinitely** and **failures for 5 minutes** (TTL-based). This prevents the 15-second test-encode from re-running on every batch render video.

**Color metadata handling:** When the source video has an HDR transfer function (smpte2084/arib-std-b67), all three color tags (`colorspace`, `color_primaries`, `color_trc`) are forced to `bt709`. Forcing only the transfer while leaving `bt2020nc/bt2020` on the other two produces invalid mixed SDR tags that cause green tints on Telegram and some players.

**iPhone-specific:** VFR → CFR via `-fps_mode cfr`; rotation metadata handled by ffmpeg autorotate (width/height swapped in `probe_video` for 90°/270°); odd dimensions fixed with `scale=trunc(iw/2)*2:trunc(ih/2)*2`.

### Threading Model (`ui/main_window.py`)

All heavy work runs in `QObject` workers moved to `QThread`. Workers communicate back via Qt signals only. Workers:

| Worker | Purpose |
|---|---|
| `WarmupWorker` | Initializes InsightFace at startup |
| `AnalysisWorker` | Face detection + tracking, emits periodic checkpoints |
| `RenderWorker` | Orchestrates the ffmpeg pipe render |
| `RegroupWorker` | Re-runs interpolation/consolidation/pruning without re-calling InsightFace |
| `LoadVideoWorker` | Probes video metadata in background |
| `ScrubWorker` | Decodes a single frame for preview on timeline scrub |
| `PlaybackWorker` | Sequential frame decode for real-time preview |
| `DiagnosticsWorker` | Builds the Ayuda → Diagnóstico report |

**Cancel button:** A single `btn_cancel_op` topbar button (hidden by default) appears labeled "Cancelar análisis" or "Cancelar render" when a worker starts. Calls `_cancel_current_operation()` which delegates to the active worker's `.cancel()`.

`_set_render_busy(busy)` disables Settings and Batch menu actions during rendering to prevent modal dialog conflicts.

**Preview playback:** Video-only. `PlaybackWorker` reads frames with OpenCV. Audio preview uses `ui/playback_manager.py` (`AudioPlaybackManager`): mpv is the master clock; if `libmpv-2.dll` or python-mpv is missing, falls back silently to `InternalClock` (monotonic). The rendered output always keeps audio.

### Detection Overlay (`_apply_overlay` in `ui/main_window.py`)

Professional-style overlay (Adobe Premiere-inspired):
- **Thin 1px colored rectangle** around the face (person's accent color). Dashed for interpolated frames, solid for real detections.
- **Flat color tab** above the top-left corner of the box with white person number. If there is no space above the box (face at top of frame), the tab is placed inside the box at the top.
- No filled shapes inside the box, no shadows, no gradients.

### Session Files (`core/session.py`)

`.gfs` files in `cache/sessions/`. Filename derived from a fast sampled content fingerprint of the video (not path). HMAC-SHA256 signed, `SESSION_VERSION = 6`. Bump this version when changing `Person` or `FaceTracker` structure.

Fingerprint strategy: reads 2 MB chunks at the start, end, and 30 evenly-spaced interior offsets. Not a full hash — doesn't detect edits outside those regions. `_FINGERPRINT_CACHE` is a process-level dict capped at 64 entries (evicts on overflow, never clears prematurely).

`load_session()` catches `Exception` (not just `ValueError`) across all candidates so a corrupted HMAC key (`RuntimeError`) doesn't escape the retry loop.

`.gfscfg` censure profiles: export/import censure settings by thumbnail hash (exact) with perceptual ahash fallback (Hamming ≤ 5). Does not change `.gfs` format, no `SESSION_VERSION` bump needed.

### Settings (`core/settings.py`)

Persisted as JSON at `~/.ghostframe_studio_settings.json`. Temp file uses PID+UUID suffix to avoid collision between simultaneous instances.

API: `cfg.get(key)` / `cfg.set(key, value)`. All values sanitized through `_COERCE` dict on load and save.

Default `execution_provider = "cuda"`. Default `det_size = 320` (4× faster than 640, sufficient for interview footage).

### FFmpeg Discovery (`find_ffmpeg` in `core/ffmpeg_utils.py`)

Priority:
1. `FFMPEG_PATH` env var
2. System PATH (`shutil.which`)
3. Any `ffmpeg*/bin/ffmpeg.exe` inside the project root (auto-detected by name glob)
4. Common Windows install paths

The bundled `ffmpeg-2026-05-21-git-0857141823-essentials_build/bin/ffmpeg.exe` is auto-detected via rule 3.

### Face Detector Error Handling (`core/face_detector.py`)

Single-frame errors (GPU hiccup, OOM spike) return `[]` and log the failure — they do **not** abort analysis. A `_consecutive_failures` counter aborts only if 10 frames fail in a row. This prevents a transient GPU error from killing a long analysis run.

---

## Important Gotchas

- **`VideoReader(path, rotation=N)`** — OpenCV ignores video rotation metadata. Always pass `rotation=info["rotation"]` from `probe_video()` so frames are in display orientation, matching FFmpeg's autorotate output. Mismatched rotation causes bbox misalignment on iPhone vertical videos.
- **`apply_censure()` always works on a copy** — never mutates the input array. `censure_roi_inplace()` mutates in place; used by `renderer.py` for efficiency.
- **FAISS index invalidated once per frame** — do not assume the index is current between `process_frame()` calls. It's rebuilt lazily on the next `_embedding_match()`.
- **`_next_id` counter** — after calling `tracker.persons = [...]` from a loaded session, always recalibrate: `tracker._next_id = max(p.person_id for p in tracker.persons) + 1`.
- **Batch mode censors ALL persons** — `BatchWorker` builds its own `persons_config` from each video's local tracker. Effect/intensity inherited from the first enabled person in the main session, or from global defaults.
- **Undo/redo stores deep copies of `FaceTracker` only** — for merge/split/regroup operations. Not for arbitrary card UI edits.
- **`_autosave_session()` shows a `QMessageBox.warning`** on failure — do not silently swallow session save errors.
- **`_open_video_dialog()`** remembers `MainWindow._last_video_dir` across instances within a session.
- **Settings dialog `_timeout_recommendation()`** — if the hardware scan hangs, a `QTimer.singleShot(8000, ...)` forcibly terminates the thread and re-enables the dialog so the user is never permanently locked out.
- **`_FORCE_CPU_PROVIDER`** — was a hardcoded debug flag that forced CPU regardless of settings. It was removed. Never re-add a hardcoded provider override.
- **Person display order** — cards and labels are sorted by `frame_count` descending (most-seen person = Persona 1). Internal `person_id` values are never renumbered to avoid breaking saved sessions.

---

## Current Environment

```
OS:           Windows 10/11
Python:       3.12.x (in .venv/)
GPU:          NVIDIA RTX (driver compatible with CUDA 12.9)
CUDA Runtime: 12.9.79 (via pip nvidia-cuda-runtime-cu12)
cuDNN:        9.22.0.52 (via pip nvidia-cudnn-cu12)
onnxruntime:  gpu 1.19.2
InsightFace:  0.7.3 (buffalo_l model, ~500 MB at ~/.insightface/models/buffalo_l/)
FFmpeg:       essentials build 2026-05-21, inside project root (auto-detected)
```

## Default Settings (`~/.ghostframe_studio_settings.json`)

```json
{
  "analysis_step": 5,
  "similarity_threshold": 0.65,
  "iou_threshold": 0.40,
  "max_interp_gap": 30,
  "min_det_score": 0.50,
  "checkpoint_every": 500,
  "execution_provider": "cuda",
  "model_name": "auto",
  "det_size": 320,
  "use_hw_encode": true,
  "crf": 18,
  "encode_preset": "fast",
  "ffmpeg_path": "",
  "default_effect": "pixelate",
  "default_intensity": 6,
  "default_padding_pct": 15,
  "open_folder_after_render": true,
  "play_speed": 1.0
}
```
