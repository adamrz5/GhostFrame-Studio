# Guía de instalación — GhostFrame Studio
### Windows · Python 3.12 (recomendado) o 3.11

---

## Compatibilidad de versiones Python

| Versión | Estado |
|---|---|
| Python 3.14 | No compatible — ningún paquete de IA tiene wheels todavía. |
| **Python 3.12** | **Compatible y probado.** Funciona con esta guía. |
| **Python 3.11** | **Compatible.** Opción más segura si tienes problemas con 3.12. |
| Python 3.10 | Compatible también. |
| Python 3.9 o menor | No soportado. |

---

## Instalación automática recomendada

En Windows, la forma más segura es ejecutar:

```bat
INSTALACION.bat
```

El script hace preguntas y configura la instalación:

- Detecta/usa Python 3.12.
- Crea `.venv`.
- Instala `requirements.txt`.
- Pregunta si tu GPU es **NVIDIA**, **AMD/Intel/DirectML** o si quieres **CPU**.
- Desinstala proveedores ONNX conflictivos antes de instalar el correcto.
- Para NVIDIA instala `onnxruntime-gpu` y paquetes `nvidia-*-cu12` vía pip.
- Para AMD/Intel instala `onnxruntime-directml`.
- Para CPU instala `onnxruntime`.
- Comprueba FFmpeg.
- Comprueba `libmpv-2.dll` para audio en preview.
- Da enlaces directos para FFmpeg, mpv/libmpv y Visual C++ Redistributable.

`libmpv-2.dll` no viene dentro de `python-mpv`. Si quieres audio en el preview,
descarga una build de mpv y copia `libmpv-2.dll` junto a `main.py`.
Si falta, GhostFrame funciona igual, pero el preview no tendrá audio.

---

## PASO 1 — Verificar Python

Abre **PowerShell** y ejecuta:

```powershell
python --version
```

Si muestra `Python 3.12.x` o `Python 3.11.x`, continúa al paso 2.  
Si no tienes Python instalado: https://www.python.org/downloads/

---

## PASO 2 — FFmpeg

FFmpeg es el motor de vídeo para leer y renderizar.

### Opción A — Dentro del proyecto (recomendado, sin tocar el PATH)

1. Descarga la build más reciente:  
   https://github.com/BtbN/FFmpeg-Builds/releases  
   Archivo: `ffmpeg-master-latest-win64-gpl.zip`

2. Descomprime el ZIP **dentro de la carpeta `ghostframe_studio/`**:
   ```
   ghostframe_studio/
   └── ffmpeg-master-latest-win64-gpl/
       └── bin/
           ├── ffmpeg.exe
           └── ffprobe.exe
   ```

3. El programa detecta automáticamente cualquier carpeta `ffmpeg*/bin/` dentro del proyecto. No necesitas configurar nada más.

### Opción B — Instalación global

1. Descomprime en `C:\ffmpeg\`
2. Añade `C:\ffmpeg\bin` al PATH del sistema:
   - `Win + R` → `sysdm.cpl` → Opciones avanzadas → Variables de entorno
   - Variables del sistema → `Path` → Editar → Nuevo → escribe `C:\ffmpeg\bin`
   - Acepta todo y abre una terminal nueva para verificar: `ffmpeg -version`

---

## PASO 3 — Crear entorno virtual

Abre **PowerShell** en la carpeta del proyecto:

```powershell
cd C:\ruta\a\ghostframe_studio
python -m venv .venv
.\.venv\Scripts\activate
```

El prompt cambiará a `(.venv)`. Verifica:

```powershell
python --version
```

---

## PASO 4 — Instalar dependencias base

Con el entorno virtual activado:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Tiempo estimado: 3-8 minutos, ~200 MB.

---

## PASO 5 — Activar GPU para el análisis de caras

> El análisis de caras es la fase más pesada. Con CPU puede consumir el 60-80% del procesador.  
> Con GPU baja al 20-40% y termina mucho más rápido.

Hay dos rutas según tu GPU. Elige una:

---

### RUTA A — DirectML (cualquier GPU, sin instalar nada extra)

Funciona con cualquier GPU con DirectX 12: NVIDIA, AMD e Intel desde 2015.  
**No necesitas instalar CUDA Toolkit ni ningún driver adicional.**

```powershell
pip uninstall onnxruntime -y
pip install onnxruntime-directml==1.19.2
```

En el programa: **Herramientas → Configuración → Proveedor ONNX → `auto` o `directml`**  
Reinicia el programa. La barra inferior mostrará `[buffalo_l / DirectML GPU]`.

---

### RUTA B — CUDA (solo NVIDIA, máximo rendimiento)

**Generalmente más rápido que DirectML** en GPUs NVIDIA (la diferencia varía según la GPU, el modelo InsightFace elegido y el driver; en pruebas con RTX y `buffalo_l` se observó un 30–50 % de mejora, pero puede ser menor en GPUs de gama baja).  
No requiere instalar CUDA Toolkit del sistema: las DLLs necesarias se instalan vía pip.

#### B.1 — Instalar onnxruntime-gpu y DLLs CUDA vía pip

> Solo necesitas tener el driver NVIDIA actualizado. No hace falta `nvcc` ni CUDA Toolkit.

```powershell
pip uninstall onnxruntime-directml -y
pip uninstall onnxruntime -y
pip install onnxruntime-gpu==1.19.2
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12
```

#### B.2 — Configurar el proveedor

En el programa: **Herramientas → Configuración → Proveedor ONNX → `cuda`**  
Guarda y reinicia. La barra inferior mostrará `[buffalo_l / CUDA GPU]`.

#### B.3 — Verificar que CUDA funciona realmente

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Debe mostrar `CUDAExecutionProvider` en la lista.

> **Si ves errores `LoadLibrary failed with error 126`** al arrancar el programa:  
> Falta uno de los paquetes `nvidia-*-cu12` o no está en el `PATH` del proceso.  
> Reinstala el bloque pip anterior dentro de la `.venv` y reinicia el programa.

---

### Verificar qué proveedor está activo

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

| Resultado | Estado |
|---|---|
| `['DmlExecutionProvider', 'CPUExecutionProvider']` | DirectML listo (Ruta A) |
| `['CUDAExecutionProvider', ...]` | CUDA listo (Ruta B) |
| `['CPUExecutionProvider']` | Solo CPU — instala un paquete GPU |

---

## PASO 6 — Primera ejecución

```powershell
python main.py
```

La primera vez descarga el modelo `buffalo_l` (~500 MB). Necesitas internet.  
Se guarda en `C:\Users\TU_USUARIO\.insightface\models\buffalo_l\` y no se vuelve a descargar.

Si falta algún componente, GhostFrame muestra un aviso al arrancar con:
- paquetes Python ausentes,
- comando `pip install` recomendado,
- enlace a Visual C++ Redistributable,
- enlace a FFmpeg,
- comando extra para CUDA/NVIDIA si corresponde.

---

## PASO 7 — Arranque rápido (uso diario)

Crea un archivo `run.bat` en la carpeta del proyecto:

```bat
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python main.py
pause
```

Doble clic en `run.bat` para arrancar sin abrir PowerShell.

La versión actual incluye además `run_silent.bat`, que usa `pythonw main.py`
para abrir la app sin consola:

```bat
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
pythonw main.py
```

Si usas `run_silent.bat`, revisa errores en `logs/ghostframe.log` o abre
**Ayuda → Diagnóstico** dentro de la app. Si la carpeta del proyecto no permite
escribir logs, GhostFrame usa `%LOCALAPPDATA%\GhostFrame Studio\logs\` como
fallback.

---

## Ajustes de rendimiento recomendados

Una vez el programa funciona, optimiza estos valores en **Herramientas → Configuración**:

La versión actual incluye un bloque **Recomendación para este PC**. Pulsa
**Aplicar configuración recomendada** para que el programa detecte CPU, RAM, GPU,
proveedores ONNX Runtime y encoders FFmpeg, y rellene valores equilibrados.

| Ajuste | Valor recomendado | Motivo |
|---|---|---|
| Analizar cada | 5-8 frames | Más alto = más rápido, mínima pérdida |
| Tamaño detector | 320 | 4x más rápido que 640, suficiente para entrevistas |
| Proveedor ONNX | directml o cuda | GPU siempre mejor que cpu |
| Umbral similitud | 0.65 | Evita fragmentar la misma persona en múltiples IDs |

Con NVIDIA y `onnxruntime-gpu`, normalmente el recomendado será `cuda`. El render
puede usar NVENC para codificar, mientras la censura visual se aplica en CPU con
OpenCV/NumPy.

---

## Uso rápido y atajos

| Acción | Atajo / ubicación |
|---|---|
| Abrir vídeo | `Ctrl+O` o botón **Abrir vídeo** |
| Reproducir / pausar preview | `Espacio` |
| Guardar frame actual | `Ctrl+S` |
| Deshacer fusión/división | `Ctrl+Z` |
| Rehacer fusión/división | `Ctrl+Y` o `Ctrl+Shift+Z` |
| Manual integrado | `F1` o **Ayuda → Manual y atajos** |
| Mostrar cajas de detección | Checkbox **Mostrar detecciones** |
| Zoom del preview | Rueda del ratón; doble clic para reset |
| Re-agrupar personas | **Re-agrupar**, sin volver a analizar con InsightFace |
| Cancelar análisis / render | Botón **Cancelar** en la barra superior (visible solo durante operaciones largas) |

La interfaz usa fondo negro, texto blanco y colores distintos por persona/cara.
Las detecciones se muestran con un rectángulo fino en el color de cada persona
(sólido para frames analizados, discontinuo para frames interpolados) más una
etiqueta plana con el número de persona.

---

## Solución de problemas frecuentes

### `pip install` falla con error en `insightface`
```
error: legacy-install-failure
```
Actualiza las herramientas de compilación:
```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
Si sigue fallando, prueba Python 3.11 en lugar de 3.12.

---

### `No module named 'cv2'`
El entorno virtual no está activado:
```powershell
.\.venv\Scripts\activate
```

---

### `ImportError: DLL load failed` al importar onnxruntime
Instala Visual C++ Redistributable:  
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

### FFmpeg no encontrado

**Comprueba que el programa lo ve:**
```powershell
python -c "from core.ffmpeg_utils import find_ffmpeg; print(find_ffmpeg())"
```
Si muestra `None`, coloca la carpeta de FFmpeg dentro de `ghostframe_studio/` (Paso 2 Opción A)  
o configura la ruta en Herramientas → Configuración → FFmpeg.

---

### GPU detectada pero el análisis sigue en CPU

**Comprueba el proveedor instalado:**
```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

- Solo aparece `CPUExecutionProvider` → instala `onnxruntime-directml` o `onnxruntime-gpu`
- Aparece `CUDAExecutionProvider` pero hay errores `LoadLibrary failed 126` → falta algún paquete `nvidia-*-cu12` en la `.venv`
- Aparece `DmlExecutionProvider` → DirectML activo, configura proveedor en el programa y reinicia

---

### El modelo InsightFace no descarga
Descárgalo manualmente:  
https://github.com/deepinsight/insightface/releases  
Descomprime `buffalo_l.zip` en `C:\Users\TU_USUARIO\.insightface\models\buffalo_l\`

---

### Ajuste si venías de una versión anterior

La versión anterior usaba `similarity_threshold = 0.55` y `det_size = 640`.  
Los nuevos valores por defecto son `0.65` y `320`.

Para aplicarlos borra la configuración guardada:
```powershell
del %USERPROFILE%\.ghostframe_studio_settings.json
```

---

### `LoadLibrary error 126` al iniciar con CUDA

Falta al menos un paquete `nvidia-*-cu12`. Reinstala el bloque completo:
```powershell
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 ^
            nvidia-cuda-nvrtc-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 ^
            nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12
```
Deben estar todos en la misma `.venv`. El programa los registra en el `PATH` del proceso
automáticamente antes de cargar ONNX Runtime.

---

## Estructura del proyecto

```
ghostframe_studio/
├── main.py                  <- Punto de entrada
├── requirements.txt         <- Dependencias pip
├── INSTALACION.md           <- Esta guía
├── run.bat                  <- Arranque rapido (crear manualmente)
├── .venv/                   <- Entorno virtual Python
├── ffmpeg-*/                <- FFmpeg (auto-detectado si esta aqui dentro)
|   └── bin/
|       ├── ffmpeg.exe
|       └── ffprobe.exe
├── core/
│   ├── face_detector.py     <- InsightFace / ArcFace + seleccion GPU
│   ├── face_tracker.py      <- Tracking IoU + FAISS + consolidacion
│   ├── ffmpeg_utils.py      <- Pipe ffmpeg + auto-deteccion de ruta
│   ├── renderer.py          <- Orquestador de renderizado
│   ├── session.py           <- Guardado/carga de analisis (.gfs en cache/sessions)
│   │                           y perfiles de censura (.gfscfg)
│   ├── settings.py          <- Configuracion persistente JSON
│   └── video_processor.py   <- Efectos blur/pixelate/blackbox
└── ui/
    ├── main_window.py       <- Ventana principal
    ├── person_card.py       <- Tarjeta por persona detectada
    ├── preview_widget.py    <- Previsualizacion de frames + zoom
    ├── timeline_widget.py   <- Barra de tiempo / scrubber
    ├── settings_dialog.py   <- Dialogo de configuracion + recomendacion hardware
    └── batch_dialog.py      <- Procesado en lote
```

---

## Requisitos del sistema

| Componente | Minimo | Recomendado |
|---|---|---|
| SO | Windows 10 64-bit | Windows 11 |
| Python | 3.11.x o 3.12.x | 3.12.10 |
| RAM | 8 GB | 16 GB |
| GPU | Cualquiera con DirectX 12 | NVIDIA RTX con 4+ GB VRAM |
| Disco | 3 GB libres | 10 GB (modelos + videos) |
| FFmpeg | 4.x+ | 7.x (BtbN build) |
| CUDA Toolkit | No necesario | No necesario; CUDA va vía paquetes pip NVIDIA |
