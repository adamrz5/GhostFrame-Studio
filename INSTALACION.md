# Guía de instalación — FaceBlur Studio
### Windows · Python 3.11 o 3.12

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

2. Descomprime el ZIP **dentro de la carpeta `faceblur_studio/`**:
   ```
   faceblur_studio/
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
cd C:\ruta\a\faceblur_studio
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

**30-50% más rápido que DirectML** en GPUs NVIDIA.  
Requiere instalar el CUDA Toolkit de NVIDIA antes del paso de pip.

#### B.1 — Instalar CUDA Toolkit 12.6

> Solo necesitas hacer esto una vez. El instalador no toca tus drivers de juegos.

1. Descarga el instalador desde:  
   **https://developer.nvidia.com/cuda-12-6-0-download-archive**

   Selecciona:
   - Operating System: `Windows`
   - Architecture: `x86_64`
   - Version: `11` (Windows 11) o `10` (Windows 10)
   - Installer Type: `exe (local)` — descarga todo el instalador (~3 GB)

2. Ejecuta el instalador con permisos de administrador.

3. En el paso "Installation Options" elige **Custom (Advanced)**:
   - Marca: `CUDA → Runtime` (obligatorio)
   - Marca: `CUDA → Development` (opcional, para desarrolladores)
   - Desmarca: `CUDA → Visual Studio Integration` (no lo necesitas)
   - Desmarca: `Documentation` y `Samples` (ahorras espacio)
   - El instalador detecta y actualiza los drivers NVIDIA si hace falta

4. Completa la instalación (puede tardar 5-10 minutos).

5. **Verifica** abriendo una terminal nueva:
   ```powershell
   nvcc --version
   ```
   Debe mostrar `release 12.6`. Si el comando no existe, reinicia el PC y vuelve a probar.

#### B.2 — Instalar onnxruntime-gpu

```powershell
pip uninstall onnxruntime-directml -y
pip uninstall onnxruntime -y
pip install onnxruntime-gpu==1.19.2
```

#### B.3 — Configurar el proveedor

En el programa: **Herramientas → Configuración → Proveedor ONNX → `cuda`**  
Guarda y reinicia. La barra inferior mostrará `[buffalo_l / CUDA GPU]`.

#### B.4 — Verificar que CUDA funciona realmente

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Debe mostrar `CUDAExecutionProvider` en la lista.

> **Si ves errores `LoadLibrary failed with error 126`** al arrancar el programa:  
> El CUDA Toolkit no se instaló correctamente o la variable PATH no se actualizó.  
> Reinicia el PC e inténtalo de nuevo. Si persiste, usa la Ruta A (DirectML).

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

---

## Ajustes de rendimiento recomendados

Una vez el programa funciona, optimiza estos valores en **Herramientas → Configuración**:

| Ajuste | Valor recomendado | Motivo |
|---|---|---|
| Analizar cada | 5-8 frames | Más alto = más rápido, mínima pérdida |
| Tamaño detector | 320 | 4x más rápido que 640, suficiente para entrevistas |
| Proveedor ONNX | directml o cuda | GPU siempre mejor que cpu |
| Umbral similitud | 0.65 | Evita fragmentar la misma persona en múltiples IDs |

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
Si muestra `None`, coloca la carpeta de FFmpeg dentro de `faceblur_studio/` (Paso 2 Opción A)  
o configura la ruta en Herramientas → Configuración → FFmpeg.

---

### GPU detectada pero el análisis sigue en CPU

**Comprueba el proveedor instalado:**
```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

- Solo aparece `CPUExecutionProvider` → instala `onnxruntime-directml` o `onnxruntime-gpu`
- Aparece `CUDAExecutionProvider` pero hay errores `LoadLibrary failed 126` → CUDA Toolkit no instalado, usa DirectML
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
del %USERPROFILE%\.faceblur_studio_settings.json
```

---

## Estructura del proyecto

```
faceblur_studio/
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
│   ├── session.py           <- Guardado/carga de analisis (.fbs)
│   ├── settings.py          <- Configuracion persistente JSON
│   └── video_processor.py   <- Efectos blur/pixelate/blackbox
└── ui/
    ├── main_window.py       <- Ventana principal
    ├── person_card.py       <- Tarjeta por persona detectada
    ├── preview_widget.py    <- Previsualizacion de frames
    ├── timeline_widget.py   <- Barra de tiempo / scrubber
    ├── settings_dialog.py   <- Dialogo de configuracion
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
| CUDA Toolkit | No necesario con DirectML | 12.6 si usas onnxruntime-gpu |
