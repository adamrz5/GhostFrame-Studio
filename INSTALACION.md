# Guía de instalación — GhostFrame Studio
### Windows · Python 3.12 recomendado

---

## Compatibilidad de versiones Python

| Versión | Estado |
|---|---|
| Python 3.14 | No compatible — ningún paquete de IA tiene wheels todavía. |
| **Python 3.12** | **Compatible y probado. Recomendado.** |
| **Python 3.11** | Compatible. |
| Python 3.10 | Compatible también. |
| Python 3.9 o menor | No soportado. |

---

## PASO 0 — Obtener el código

### Opción A — Clonar desde GitHub (recomendado)

Necesitas tener [Git para Windows](https://git-scm.com/download/win) instalado.

```powershell
git clone https://github.com/adamrz5/GhostFrame-Studio.git
cd GhostFrame-Studio
```

### Opción B — Descargar ZIP

1. Ve a https://github.com/adamrz5/GhostFrame-Studio
2. Haz clic en **Code → Download ZIP**
3. Descomprime en la carpeta que quieras y entra en ella

---

## PASO 1 — Verificar Python

Abre **PowerShell** y ejecuta:

```powershell
python --version
```

Si muestra `Python 3.12.x` o `Python 3.11.x`, continúa al paso 2.  
Si no tienes Python instalado: https://www.python.org/downloads/  
Durante la instalación, **marca la opción "Add Python to PATH"**.

---

## PASO 2 — FFmpeg

FFmpeg es el motor de vídeo para leer y renderizar. La build incluida en el repositorio ya está dentro de la carpeta del proyecto y se detecta automáticamente. Si no está o la quieres actualizar:

### Opción A — Dentro del proyecto (recomendado, sin tocar el PATH)

1. Descarga la build más reciente:  
   https://github.com/BtbN/FFmpeg-Builds/releases  
   Archivo: `ffmpeg-master-latest-win64-gpl.zip`

2. Descomprime el ZIP **dentro de la carpeta `GhostFrame-Studio/`**:
   ```
   GhostFrame-Studio/
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
cd C:\ruta\a\GhostFrame-Studio
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

### RUTA A — DirectML (cualquier GPU, opción más fácil)

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
**No necesitas instalar el CUDA Toolkit del sistema** — las DLLs necesarias se instalan directamente vía pip.

#### B.1 — Instalar onnxruntime-gpu y DLLs CUDA

```powershell
pip uninstall onnxruntime onnxruntime-directml -y
pip install onnxruntime-gpu==1.19.2
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12
```

> **¿Por qué tantos paquetes `nvidia-*-cu12`?**  
> En lugar de instalar el CUDA Toolkit completo del sistema (~3 GB), estos paquetes pip
> contienen exactamente las DLLs que necesita ONNX Runtime. El programa las registra
> automáticamente en el PATH del proceso al arrancar (`_preload_nvidia_dlls`).
> No tienes que tocar variables de entorno ni instalar nada más.

#### B.2 — Configurar el proveedor

En el programa: **Herramientas → Configuración → Proveedor ONNX → `cuda`**  
Guarda y reinicia. La barra inferior mostrará `[buffalo_l / CUDA GPU]`.

#### B.3 — Verificar que CUDA funciona

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Debe mostrar `CUDAExecutionProvider` en la lista.

> **Si ves errores `LoadLibrary failed with error 126`** al arrancar el programa:  
> Falta alguno de los paquetes `nvidia-*-cu12`. Ejecuta de nuevo el comando `pip install nvidia-*-cu12`
> del punto B.1 completo, con el entorno virtual activado. Si persiste, usa la Ruta A (DirectML).

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

## PASO 6 — Audio en preview (opcional)

El render final **siempre conserva el audio** aunque no hagas este paso.  
Este paso solo añade audio al **preview en tiempo real** dentro de la app.

GhostFrame usa `libmpv-2.dll` para reproducir audio en el preview. Esta DLL **no está incluida en el repositorio** de GitHub (supera el límite de 100 MB de GitHub).

### Cómo conseguirla

1. Ve a https://github.com/shinchiro/mpv-winbuild-cmake/releases
2. Descarga el archivo `mpv-x86_64-<fecha>-git-<hash>.7z` más reciente
3. Abre el `.7z` con [7-Zip](https://www.7-zip.org/) o WinRAR
4. Extrae únicamente el archivo `libmpv-2.dll`
5. Cópialo junto a `main.py` dentro de la carpeta `GhostFrame-Studio/`

Si `libmpv-2.dll` no está disponible, GhostFrame abre igual pero el preview funciona sin audio.

---

## PASO 7 — Primera ejecución

Con el entorno virtual activado:

```powershell
python main.py
```

La primera vez descarga el modelo `buffalo_l` (~500 MB). Necesitas conexión a internet.  
Se guarda en `C:\Users\TU_USUARIO\.insightface\models\buffalo_l\` y no se vuelve a descargar.

---

## PASO 8 — Arranque rápido (uso diario)

El proyecto incluye `run_silent.bat` para arrancar sin consola visible con doble clic.  
También puedes crear tu propio `run.bat`:

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

El menú **Herramientas → Configuración → Aplicar configuración recomendada** detecta tu hardware y sugiere valores automáticamente.

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
Si muestra `None`, coloca la carpeta de FFmpeg dentro de `GhostFrame-Studio/` (Paso 2 Opción A)  
o configura la ruta en Herramientas → Configuración → FFmpeg.

---

### GPU detectada pero el análisis sigue en CPU

**Comprueba el proveedor instalado:**
```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

- Solo aparece `CPUExecutionProvider` → instala `onnxruntime-directml` o `onnxruntime-gpu`
- Aparece `CUDAExecutionProvider` pero hay errores `LoadLibrary failed 126` → falta algún paquete `nvidia-*-cu12`, repite el Paso 5 Ruta B.1 completo
- Aparece `DmlExecutionProvider` → DirectML activo, configura proveedor en el programa y reinicia

---

### El modelo InsightFace no descarga
Descárgalo manualmente:  
https://github.com/deepinsight/insightface/releases  
Descomprime `buffalo_l.zip` en `C:\Users\TU_USUARIO\.insightface\models\buffalo_l\`

---

### Restablecer configuración predeterminada

Si quieres borrar la configuración guardada y volver a los valores por defecto:
```powershell
del %USERPROFILE%\.ghostframe_studio_settings.json
```

---

### Los análisis guardados no se cargan tras mover el proyecto

Las sesiones `.gfs` se guardan en `cache/sessions/` dentro del proyecto y se identifican por el contenido del vídeo, no por la ruta. Si mueves la carpeta del proyecto, los análisis siguen funcionando mientras el vídeo original esté accesible.

---

## Estructura del proyecto

```
GhostFrame-Studio/
├── main.py                  <- Punto de entrada
├── requirements.txt         <- Dependencias pip
├── INSTALACION.md           <- Esta guía
├── run_silent.bat           <- Arranque sin consola
├── .venv/                   <- Entorno virtual Python (no en GitHub)
├── libmpv-2.dll             <- Audio en preview (descargar por separado, no en GitHub)
├── ffmpeg-*/                <- FFmpeg (auto-detectado si está aquí)
│   └── bin/
│       ├── ffmpeg.exe
│       └── ffprobe.exe
├── core/
│   ├── face_detector.py     <- InsightFace / ArcFace + selección GPU
│   ├── face_tracker.py      <- Tracking IoU + FAISS + consolidación
│   ├── ffmpeg_utils.py      <- Pipe ffmpeg + auto-detección de ruta
│   ├── renderer.py          <- Orquestador de renderizado
│   ├── session.py           <- Guardado/carga de análisis (.gfs)
│   ├── settings.py          <- Configuración persistente JSON
│   └── video_processor.py   <- Efectos blur/pixelate/blackbox
└── ui/
    ├── main_window.py       <- Ventana principal
    ├── person_card.py       <- Tarjeta por persona detectada
    ├── preview_widget.py    <- Previsualización de frames
    ├── timeline_widget.py   <- Barra de tiempo / scrubber
    ├── settings_dialog.py   <- Diálogo de configuración
    └── batch_dialog.py      <- Procesado en lote
```

---

## Requisitos del sistema

| Componente | Mínimo | Recomendado |
|---|---|---|
| SO | Windows 10 64-bit | Windows 11 |
| Python | 3.11.x o 3.12.x | 3.12.x |
| RAM | 8 GB | 16 GB |
| GPU | Cualquiera con DirectX 12 | NVIDIA RTX con 4+ GB VRAM |
| Disco | 3 GB libres | 10 GB (modelos + vídeos) |
| FFmpeg | 4.x+ | 7.x (BtbN build) |
| CUDA Toolkit | No necesario | No necesario (DLLs vía pip) |
