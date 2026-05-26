"""
Detección facial y extracción de embeddings ArcFace con InsightFace.

Proveedores ONNX Runtime — orden de prioridad en Windows (modo "auto"):
  1. CUDAExecutionProvider   — GPU NVIDIA via CUDA
                               Requiere: pip install onnxruntime-gpu==1.19.2
  2. DmlExecutionProvider    — DirectML: GPU universal (NVIDIA/AMD/Intel)
                               Requiere: pip install onnxruntime-directml==1.19.2
  3. CPUExecutionProvider    — CPU puro, siempre disponible (fallback automático)

Notas de rendimiento:
- La inferencia ONNX (detection + embedding) es el paso más costoso en CPU.
  Moverla a GPU reduce el uso de CPU drásticamente.
- Si GPU init falla, el código vuelve automáticamente a CPU y avisa en consola.
- En modo CPU se limitan los hilos OMP/MKL a cpu_count//2 para no bloquear la UI.
"""
from __future__ import annotations

import os
import traceback
import numpy as np


_app = None
_active_provider: str = "cpu"
_active_model: str = "buffalo_l"
_last_detection_exception_str: str | None = None
_detector_init_error: Exception | None = None
_gpu_fallback: bool = False      # True si se pidió GPU pero se cayó a CPU
_consecutive_failures: int = 0   # frames fallidos seguidos; abortar si supera umbral
_MAX_CONSECUTIVE_FAILURES = 10


# ── Resolución de modelo ───────────────────────────────────────────────────────

def _resolve_model(setting: str) -> str:
    """buffalo_l en Windows (alta precisión); buffalo_s solo si el usuario lo pide."""
    valid = {"auto", "buffalo_l", "buffalo_s"}
    if setting not in valid:
        print(f"[FaceDetector] Modelo desconocido '{setting}', usando buffalo_l")
        return "buffalo_l"
    if setting != "auto":
        return setting
    return "buffalo_l"


# ── Construcción de lista de proveedores ──────────────────────────────────────

def _cuda_runtime_loadable() -> bool:
    """
    Comprueba si las DLLs de CUDA Runtime están realmente disponibles en el sistema.

    onnxruntime-gpu lista 'CUDAExecutionProvider' en get_available_providers() incluso
    cuando faltan las DLLs CUDA de runtime. Intentar usarlo sin las DLLs produce el
    error 126 (LoadLibrary failed) y ONNX cae silenciosamente a CPU.

    Esta función carga 'nvcuda.dll' (driver de NVIDIA, siempre presente si hay GPU NVIDIA)
    y uno de los cudart DLLs (paquetes NVIDIA vía pip o Toolkit). Si solo está nvcuda pero no cudart, CUDA
    no funcionará para inferencia ONNX.
    """
    import ctypes
    # nvcuda.dll viene con los drivers de NVIDIA — necesario pero no suficiente
    nvcuda_ok = False
    try:
        ctypes.CDLL("nvcuda.dll")
        nvcuda_ok = True
    except OSError:
        return False  # Sin drivers NVIDIA no tiene sentido seguir

    # cudart64_*.dll — viene del CUDA Toolkit del sistema O de nvidia-cuda-runtime-cu12 (pip)
    # cudart64_12.dll  → paquete pip nvidia-cuda-runtime-cu12 (sin número menor)
    # cudart64_120.dll → CUDA 12.0 sistema / cudart64_121.dll → CUDA 12.1, etc.
    for dll in (
        "cudart64_12.dll",                                           # pip nvidia-cuda-runtime-cu12
        "cudart64_110.dll", "cudart64_111.dll", "cudart64_112.dll", # CUDA 11.x sistema
        "cudart64_120.dll", "cudart64_121.dll", "cudart64_122.dll", # CUDA 12.x sistema
        "cudart64_125.dll", "cudart64_128.dll", "cudart64_129.dll",
    ):
        try:
            ctypes.CDLL(dll)
            return True
        except OSError:
            continue

    # nvcuda existe (drivers OK) pero no cudart (paquetes CUDA no disponibles)
    if nvcuda_ok:
        print("[FaceDetector] NVIDIA GPU detectada pero faltan DLLs CUDA. "
              "Instala onnxruntime-gpu y los paquetes nvidia-*-cu12 vía pip, o usa DirectML.")
    return False


def _build_providers(setting: str) -> list:
    """
    Devuelve la lista de proveedores ONNX Runtime con opciones óptimas.
    Cada entrada puede ser un str simple o una tupla (nombre, opciones_dict).

    CUDA options reference: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
    DML  options reference: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
    """
    # ── CUDA (NVIDIA, máximo rendimiento con onnxruntime-gpu) ─────────────────
    cuda_opts = {
        "device_id": 0,
        # Crecimiento de arena de memoria: evita re-alloc en cada frame
        "arena_extend_strategy": "kNextPowerOfTwo",
        # Límite de VRAM. RTX 3060 tiene 12 GB; 4 GB es suficiente para buffalo_l
        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
        # Búsqueda exhaustiva de algoritmos cuDNN → más rápido después del warmup
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        # Copia en el stream por defecto → evita race conditions con PyTorch/OpenCV
        "do_copy_in_default_stream": True,
    }

    # ── DirectML (NVIDIA/AMD/Intel con DirectX 12, onnxruntime-directml) ──────
    # DirectML no expone muchas opciones; device_id=0 selecciona la GPU primaria
    dml_opts = {"device_id": 0}

    if setting == "cuda":
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]

    if setting == "directml":
        return [("DmlExecutionProvider", dml_opts), "CPUExecutionProvider"]

    if setting == "cpu":
        return ["CPUExecutionProvider"]

    # ── "auto" — detectar el mejor proveedor REALMENTE disponible ────────────
    # IMPORTANTE: ort.get_available_providers() lista lo que el paquete compila,
    # no lo que funciona en runtime. CUDAExecutionProvider puede aparecer en la
        # lista aunque falten las DLLs CUDA de runtime (error 126 al cargar).
    # Por eso comprobamos si las DLLs de CUDA Runtime son accesibles antes de
    # incluirlo — evitamos el fallback silencioso a CPU sin notificarlo al usuario.
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
        providers: list = []
        if "CUDAExecutionProvider" in available and _cuda_runtime_loadable():
            providers.append(("CUDAExecutionProvider", cuda_opts))
        if "DmlExecutionProvider" in available:
            providers.append(("DmlExecutionProvider", dml_opts))
        providers.append("CPUExecutionProvider")
        return providers
    except Exception:
        return ["CPUExecutionProvider"]


def _provider_label(providers: list) -> str:
    """Nombre corto del proveedor principal de la lista."""
    first = providers[0] if providers else "CPUExecutionProvider"
    name = first[0] if isinstance(first, tuple) else first
    if "CUDA" in name:
        return "cuda"
    if "Dml" in name:
        return "directml"
    return "cpu"


# ── Optimización de hilos CPU ──────────────────────────────────────────────────

def _limit_cpu_threads() -> None:
    """
    Limita los hilos de cálculo paralelo (OpenMP, MKL, OpenBLAS) a cpu_count//2.
    Objetivo: dejar núcleos libres para la UI de Qt y no saturar el 100% de CPU.

    Debe llamarse ANTES de que onnxruntime cree las sesiones ONNX, porque los
    pools de hilos se inicializan en el primer InferenceSession, no en el import.
    """
    n = max(1, (os.cpu_count() or 4) // 2)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ORT_NUM_THREADS",           # ONNX Runtime propio
    ):
        os.environ.setdefault(var, str(n))
    try:
        import cv2
        cv2.setNumThreads(n)
    except Exception:
        pass
    print(f"[FaceDetector] Hilos CPU limitados a {n} (de {os.cpu_count()} disponibles)")


# ── Inicialización del detector ────────────────────────────────────────────────

def init_detector(model_name: str = "auto", provider: str = "auto") -> str:
    """
    Inicializa InsightFace buffalo_l con el proveedor de inferencia indicado.
    Llamar desde un hilo de fondo al arrancar la app (WarmupWorker en main_window).

    Estrategia:
    1. Intenta GPU (CUDA o DirectML según configuración/auto-detect).
    2. Si falla (package no instalado, driver incompatible, VRAM insuficiente),
       hace fallback automático a CPU con logging claro.
    3. En modo CPU limita los hilos para no saturar el procesador.

    Devuelve: "cuda" | "directml" | "cpu"
    """
    global _app, _active_provider, _active_model, _detector_init_error, _gpu_fallback

    from insightface.app import FaceAnalysis

    model     = _resolve_model(model_name)
    providers = _build_providers(provider)
    label     = _provider_label(providers)

    _active_model = model

    # Limitar hilos OMP/MKL SIEMPRE — incluso con GPU, el preprocesado de frames
    # (resize, normalización numpy/OpenCV) corre en CPU y se beneficia del límite.
    # Sin esto, OpenCV puede lanzar tantos hilos como núcleos haya y saturar la CPU
    # incluso cuando la inferencia ONNX ya está en la GPU.
    _limit_cpu_threads()

    # ── Intento principal (puede ser GPU) ─────────────────────────────────────
    from core import settings as cfg
    _det = cfg.get("det_size") or 320
    det_size = (_det, _det)

    try:
        _app = FaceAnalysis(name=model, providers=providers)
        _app.prepare(ctx_id=0, det_size=det_size)
        _active_provider = label
        _gpu_fallback = False
        _detector_init_error = None
        print(f"[FaceDetector] OK Proveedor activo: {label.upper()}  ({model})")
        return _active_provider

    except Exception as gpu_err:
        # Muestra el error real del intento GPU para diagnóstico
        print(f"[FaceDetector] FALLO con {label.upper()}: {gpu_err}")

        # ── Fallback a CPU si el intento no era ya CPU ─────────────────────
        if label == "cpu":
            _app = None
            _detector_init_error = gpu_err
            raise

        # Limita hilos para no saturar CPU con el fallback
        _limit_cpu_threads()

        try:
            print("[FaceDetector] Reintentando con CPUExecutionProvider...")
            _app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"])
            _app.prepare(ctx_id=0, det_size=det_size)
            _active_provider = "cpu"
            _gpu_fallback = True
            _detector_init_error = None
            print("[FaceDetector] OK Fallback a CPU exitoso. "
                  "Para GPU: instala onnxruntime-gpu o onnxruntime-directml "
                  "y configura el proveedor en Herramientas > Configuracion.")
            return _active_provider

        except Exception as cpu_err:
            _app = None
            _detector_init_error = cpu_err
            raise


# ── Lazy-init (si detect_faces se llama antes de init_detector) ───────────────

def _get_app():
    global _app
    if _detector_init_error is not None:
        raise RuntimeError(
            "InsightFace no se inicializó correctamente."
        ) from _detector_init_error
    if _app is None:
        from core import settings as cfg
        init_detector(
            model_name=cfg.get("model_name"),
            provider=cfg.get("execution_provider"),
        )
    return _app


# ── Detección por frame ────────────────────────────────────────────────────────

def detect_faces(frame_bgr: np.ndarray, min_det_score: float = 0.50) -> list[dict]:
    """
    Detecta caras en un frame BGR y devuelve lista de dicts:
        {
            'bbox':      [x1, y1, x2, y2],
            'embedding': np.ndarray (512-d, ArcFace) | None,
            'det_score': float,
            'kps':       np.ndarray (5 keypoints) | None,
        }

    min_det_score: descarta detecciones poco fiables (reflejos, fondos borrosos).
    Devuelve [] si no hay caras. Si el detector falla, lanza RuntimeError para
    evitar renderizar un vídeo sin censura por un fallo silencioso de GPU/modelo.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    global _last_detection_exception_str, _consecutive_failures
    try:
        app   = _get_app()
        faces = app.get(frame_bgr)
        results = []
        for face in faces:
            score = float(getattr(face, "det_score", 1.0))
            if score < min_det_score:
                continue
            bbox = [int(v) for v in face.bbox.tolist()]
            embedding = (
                face.embedding.copy()
                if hasattr(face, "embedding") and face.embedding is not None
                else None
            )
            kps = (
                face.kps.copy()
                if hasattr(face, "kps") and face.kps is not None
                else None
            )
            results.append({
                "bbox":      bbox,
                "embedding": embedding,
                "det_score": score,
                "kps":       kps,
            })
        _consecutive_failures = 0   # reset on success
        return results

    except Exception as e:
        _consecutive_failures += 1
        exc_str = traceback.format_exc()
        if exc_str != _last_detection_exception_str:
            _last_detection_exception_str = exc_str
            print("[FaceDetector] Nuevo error al detectar caras (traza completa):")
            print(exc_str)
        else:
            print(
                f"[FaceDetector] Frame omitido por error "
                f"({_consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES}): "
                f"{type(e).__name__}: {e}"
            )
        if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                f"El detector facial falló en {_MAX_CONSECUTIVE_FAILURES} frames "
                "consecutivos. Se detiene el proceso para evitar generar un vídeo "
                "sin censurar."
            ) from e
        return []


# ── Consultas de estado ────────────────────────────────────────────────────────

def active_provider() -> str:
    """Devuelve el proveedor activo: 'cpu', 'cuda' o 'directml'."""
    return _active_provider


def active_model() -> str:
    """Devuelve el nombre del modelo cargado."""
    return _active_model


def gpu_fallback_occurred() -> bool:
    """True si se solicitó GPU pero se tuvo que caer a CPU por error de init."""
    return _gpu_fallback
