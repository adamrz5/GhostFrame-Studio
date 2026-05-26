# Contexto acumulado — GhostFrame Studio

Registro de decisiones técnicas, problemas resueltos y estado actual del proyecto.  
Última actualización: 2026-05-25.

---

## Estado actual del proyecto

El programa arranca y funciona correctamente con:
- **InsightFace buffalo_l** como modelo de detección + embeddings
- **CUDA GPU** como proveedor ONNX (RTX de la máquina del usuario)
- **FFmpeg** detectado automáticamente desde la carpeta del proyecto
- **det_size = 320** (4× más rápido que 640)
- **Python 3.12** en `.venv/`
- **Interfaz negro/blanco** de alto contraste, manteniendo colores por persona/cara
- **Preview sin audio** por diseño; el render final conserva audio y copia subtítulos cuando existen
- **Manual integrado** en `Ayuda → Manual y atajos` (`F1`)
- **Recomendación automática de configuración** en `Herramientas → Configuración`
- **Botón "Cancelar"** en topbar (oculto por defecto): aparece como "Cancelar análisis" o "Cancelar render" cuando arranca un worker
- **Overlay de detección estilo Adobe Premiere**: rectángulo fino de 1 px en color de persona + etiqueta plana encima de la esquina superior izquierda

---

## Problema principal resuelto: CPU al 80-83% con GPU al 50%

### Causa raíz
`face_detector.py` tenía un flag hardcodeado `_FORCE_CPU_PROVIDER = True` en la línea 27 que forzaba CPU independientemente de la configuración. Era un flag de debug que nunca se revirtió.

### Solución completa (varias fases)
1. **Eliminar `_FORCE_CPU_PROVIDER`** — causa raíz.
2. **Cambiar proveedor por defecto** de `"auto"` a `"cuda"` en `core/settings.py`.
3. **DirectML descartado**: arquitecturalmente 2.8× más lento que CUDA en NVIDIA — usa sincronizaciones CPU en cada operación GPU. CPU seguía al 80% incluso con GPU al 50%.
4. **Instalar `onnxruntime-gpu==1.19.2`** en lugar de `onnxruntime-directml`.
5. **Resolver `LoadLibrary error 126`** (ver sección CUDA abajo).
6. **`_limit_cpu_threads()`** llamado para todos los proveedores (no solo CPU) — reduce hilos de OpenMP/MKL a `cpu_count//2` para no saturar con preprocesado.
7. **`det_size` reducido a 320** (configurable, antes hardcodeado a 640).

---

## Problema CUDA: LoadLibrary error 126 → resuelto

### Historia completa

onnxruntime-gpu lista `CUDAExecutionProvider` en `get_available_providers()` aunque falten las DLLs de CUDA. Al intentar usarlo, fallaba silenciosamente a CPU con:
```
LoadLibrary failed with error 126 when trying to load onnxruntime_providers_cuda.dll
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*
```

### Por qué fallaban los primeros enfoques
- **`os.add_dll_directory()`**: Solo funciona para DLLs cargadas directamente por Python, NO para cargas transitivas que hace `onnxruntime_providers_cuda.dll` internamente con su propio `LoadLibrary`.
- **`os.environ["PATH"]`** con solo 3-4 paquetes: faltaban los paquetes `cufft`, `curand`, `cusparse`, `cusolver`, `nvjitlink`.
- **Copiar DLLs a `onnxruntime/capi/`**: Windows busca el directorio de la DLL cargadora para dependencias *estáticas* (import table), pero onnxruntime carga cuDNN *dinámicamente* con `LoadLibrary("cudnn64_9.dll")` — esto no busca el directorio de la DLL sino el PATH del proceso.

### Solución que funcionó
Añadir **todos** los directorios `bin/` de los paquetes NVIDIA a PATH del proceso antes de que onnxruntime cargue la sesión:

```python
nvidia_pkgs = ['cuda_runtime', 'cublas', 'cudnn', 'cuda_nvrtc', 
               'cufft', 'curand', 'cusparse', 'cusolver', 'nvjitlink']
for base in site.getsitepackages():
    for pkg in nvidia_pkgs:
        full = os.path.join(base, 'nvidia', pkg, 'bin')
        if os.path.isdir(full):
            os.environ['PATH'] = full + os.pathsep + os.environ['PATH']
            os.add_dll_directory(full)
```

### Paquetes pip instalados para CUDA (sin CUDA Toolkit del sistema)
```
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cublas-cu12==12.9.2.10
nvidia-cudnn-cu12==9.22.0.52
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cufft-cu12==11.4.1.4
nvidia-curand-cu12==10.3.10.19
nvidia-cusparse-cu12==12.5.10.65
nvidia-cusolver-cu12==11.7.5.82
nvidia-nvjitlink-cu12==12.9.86
onnxruntime-gpu==1.19.2
```

Los DLLs también están copiados físicamente en `.venv/Lib/site-packages/onnxruntime/capi/` (por si acaso, aunque lo que realmente resolvió fue el PATH completo).

### Verificación
```powershell
.\.venv\Scripts\python.exe -c "
import onnxruntime as ort
import glob, os

models = glob.glob(os.path.expanduser('~/.insightface/models/buffalo_l/*.onnx'))
sess = ort.InferenceSession(models[0], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(sess.get_providers())  # debe mostrar ['CUDAExecutionProvider', 'CPUExecutionProvider']
"
```

---

## Overlay de detección — diseño profesional (2026-05-25)

Rediseñado al estilo Adobe Premiere. Implementado en `_apply_overlay()` y `_draw_dashed_rect()` de `ui/main_window.py`.

### Especificación visual
- **Rectángulo fino 1 px** en color de acento de la persona:
  - **Sólido** para detecciones reales (en frames analizados)
  - **Discontinuo** (dashes 7 px, huecos 5 px) para frames interpolados
- **Etiqueta plana** arriba-izquierda del rectángulo:
  - Fondo: color de acento, sin transparencia, sin bordes redondeados
  - Texto: número de persona en blanco, fuente `FONT_HERSHEY_SIMPLEX` escala 0.42
  - Si la cara está en el borde superior del frame, la etiqueta se coloca dentro de la caja
- Sin círculos, sin sombras, sin rellenos, sin degradados

### Implementación clave
```python
@staticmethod
def _draw_dashed_rect(img, x1, y1, x2, y2, color, dash=7, gap=5):
    for x in range(x1, x2, dash + gap):
        xe = min(x + dash, x2)
        cv2.line(img, (x, y1), (xe, y1), color, 1, cv2.LINE_AA)
        cv2.line(img, (x, y2), (xe, y2), color, 1, cv2.LINE_AA)
    for y in range(y1, y2, dash + gap):
        ye = min(y + dash, y2)
        cv2.line(img, (x1, y), (x1, ye), color, 1, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, ye), color, 1, cv2.LINE_AA)
```

---

## Cambios acumulados en cada fichero

### `main.py`
- `_preload_nvidia_dlls()`: añade todos los directorios `nvidia/*/bin/` de los paquetes pip a PATH y `os.add_dll_directory()`. Lista completa: `cuda_runtime`, `cublas`, `cudnn`, `cuda_nvrtc`, `cusolver`, `cufft`, `curand`, `cusparse`, `nvjitlink`.
- Llamada antes de cualquier import de onnxruntime o insightface.
- Import de `onnx.onnx_cpp2py_export` antes de PyQt5 (fix para setup específico de Windows).

### `core/face_detector.py`
- Eliminado `_FORCE_CPU_PROVIDER = True`.
- `_cuda_runtime_loadable()`: prueba `ctypes.CDLL("cudart64_12.dll")` (pip) y versiones de sistema CUDA 11.x/12.x.
- `_build_providers()`: en modo "auto" verifica CUDA realmente cargable antes de incluirlo.
- `_limit_cpu_threads()`: llamado para todos los proveedores (no solo CPU).
- `det_size` leído de `cfg.get("det_size")` en lugar de hardcodeado.
- `_gpu_fallback: bool` y `gpu_fallback_occurred()` para informar a la UI.
- Módulo-level counters `_consecutive_failures` y `_MAX_CONSECUTIVE_FAILURES = 10` para abortar solo tras 10 fallos consecutivos.
- Prints en ASCII.

### `core/face_tracker.py`
- `_next_id: int = 0` como counter monotónico O(1) (antes `max(ids)+1` por cada persona nueva).
- `_invalidate_index()` movido fuera del bucle de detecciones — se llama una vez por frame.
- `last_bbox_before(window: int = 30)` — default corregido de 10 a 30.

### `core/settings.py`
- `"execution_provider": "cuda"` (era `"auto"`).
- `"det_size": 320` (nueva clave, antes hardcodeado 640 en el detector).
- Fichero temp con sufijo `f".tmp_{os.getpid()}_{uuid.uuid4().hex}"` para evitar colisiones.

### `core/ffmpeg_utils.py`
- `find_ffmpeg()`: busca subcarpetas `ffmpeg*/bin/ffmpeg.exe` dentro del directorio raíz del proyecto.
- Render con subtítulos opcionales: `-map 1:s?` y `-c:s copy`.
- HDR coerción completa: cuando `color_trc` es `smpte2084` o `arib-std-b67`, se fuerzan los tres tags (`colorspace`, `color_primaries`, `color_trc`) a `bt709`.
- `verify_encoder()` con TTL-based failure caching: éxitos cacheados indefinidamente, fallos 5 minutos (`time.monotonic() + 300.0`).

### `core/session.py`
- `.gfs`: sesión firmada, versionada y ligada al vídeo mediante fingerprint rápido.
- `.gfscfg`: export/import de configuración de censura por hash de thumbnail.
- `_FINGERPRINT_CACHE` capeado a 64 entradas — evicción al superar el límite en lugar de `clear()`.
- `except Exception` (ampliado de `except ValueError`) para capturar `RuntimeError` de HMAC corrupto.
- `tracker._next_id = max(p.person_id for p in tracker.persons) + 1` después de cargar personas.

### `ui/main_window.py`
- `_last_video_dir: str = ""` — clase variable, recuerda la última carpeta de vídeo abierta.
- `btn_cancel_op` en topbar (rojo, oculto por defecto): aparece como "Cancelar análisis" o "Cancelar render" cuando arranca un worker.
- `_cancel_current_operation()` — delega a `_current_worker.cancel()`.
- `_set_render_busy(busy)` — deshabilita menús y muestra/oculta botón cancelar.
- `_autosave_session()` — muestra `QMessageBox.warning` si falla el guardado.
- Split/merge labels con sufijo `[#id]` para evitar ambigüedad en `ids.index()`.
- Export frame guarda en la carpeta del vídeo por defecto.
- `_apply_overlay()` + `_draw_dashed_rect()` — overlay estilo Adobe Premiere.
- Import `gpu_fallback_occurred` de `core.face_detector`.
- `_on_warmup_done()`: labels diferenciadas `"CUDA ✓ GPU"` / `"DirectML ✓ GPU"` / `"CPU ⚠"`.
- Undo/redo para merge/split/regroup (`Ctrl+Z`, `Ctrl+Y`, `Ctrl+Shift+Z`).

### `ui/settings_dialog.py`
- `self.cmb_det_size`: ComboBox 320/640.
- Bloque `Recomendación para este PC`.
- `QTimer.singleShot(8000, self._timeout_recommendation)` — timeout de seguridad 8 s para el hilo de recomendación.
- `_timeout_recommendation()` — fuerza cierre del hilo y rehabilita el diálogo.

### `ui/preview_widget.py`
- Eliminada asignación duplicada `h2, w2 = rgb.shape[:2]`.
- `QImage` usa `w, h` correctos (las variables muertas `h2, w2` ya no existen).
- Zoom visual con rueda del ratón. Doble clic resetea a 1.0.

### `ui/person_card.py`
- `_ft()` con fallback `fps = 25.0` cuando `self.fps <= 0`.
- Tarjetas negras con texto blanco.
- Cada persona conserva color propio en borde, nombre y barra de apariciones.

### `ui/batch_dialog.py`
- Título corregido a `"Modo Batch — GhostFrame"`.

---

## Pendiente / próximos pasos

- **M-2** (interpolación en cortes de escena detectados): pospuesto. Requiere detección de scene cuts con superficie de cambio alta. Todos los demás bugs están resueltos.

---

## Entorno actual

```
OS:           Windows 10/11
Python:       3.12.x (en .venv/)
GPU:          NVIDIA (RTX, driver compatible con CUDA 12.9)
CUDA Runtime: 12.9.79 (vía pip nvidia-cuda-runtime-cu12)
cuDNN:        9.22.0.52 (vía pip nvidia-cudnn-cu12)
onnxruntime:  gpu 1.19.2
InsightFace:  0.7.3 (modelo buffalo_l, ~500 MB en ~/.insightface/models/buffalo_l/)
FFmpeg:       essentials build 2026-05-21, dentro del proyecto (auto-detectado)
```

## Configuración actual guardada en `~/.ghostframe_studio_settings.json`

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
