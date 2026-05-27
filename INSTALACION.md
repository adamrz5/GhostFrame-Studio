# Instalar GhostFrame Studio
### Windows — guía paso a paso

---

## Antes de empezar — instala esto primero

Necesitas tener estas tres cosas en tu ordenador antes de continuar:

| Qué | Para qué sirve | Descarga |
|-----|---------------|----------|
| **Python 3.12** | El lenguaje en el que está hecho el programa | [python.org/downloads](https://www.python.org/downloads/) |
| **FFmpeg** | Lee y exporta los vídeos | [Descargar FFmpeg](https://github.com/BtbN/FFmpeg-Builds/releases) |
| **Visual C++ Redistributable** | Necesario para que funcionen las librerías internas | [Descargar VC++ x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) |

> Si ya tienes algo de esto instalado, puedes saltarte ese paso.

---

## PASO 1 — Descargar el programa

Ve a [github.com/adamrz5/GhostFrame-Studio](https://github.com/adamrz5/GhostFrame-Studio) y haz clic en **Code → Download ZIP**.

Descomprime la carpeta donde quieras, por ejemplo en `C:\GhostFrame-Studio\`.

---

## PASO 2 — Instalar Python 3.12

1. Descarga **Python 3.12** desde [python.org/downloads](https://www.python.org/downloads/)
2. Ejecuta el instalador
3. **MUY IMPORTANTE:** antes de pulsar Instalar, marca la casilla **"Add Python to PATH"**

   [![Add to PATH](https://i.imgur.com/placeholder.png)](https://www.python.org)

4. Verifica que funciona: abre PowerShell y escribe:
   ```
   python --version
   ```
   Debe mostrar `Python 3.12.x`. Si ves un error, el PATH no se configuró bien — vuelve a instalar con esa casilla marcada.

---

## PASO 3 — Instalar Visual C++ Redistributable

Esto es necesario para que las librerías internas del programa funcionen en Windows.

1. Descarga desde: [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Ejecuta el instalador y sigue los pasos
3. Reinicia el ordenador si te lo pide

---

## PASO 4 — Instalar FFmpeg

FFmpeg es el programa que GhostFrame usa por dentro para leer y exportar vídeos.

1. Ve a [github.com/BtbN/FFmpeg-Builds/releases](https://github.com/BtbN/FFmpeg-Builds/releases)
2. Descarga `ffmpeg-master-latest-win64-gpl.zip`
3. Descomprime esa carpeta **dentro de la carpeta de GhostFrame**:
   ```
   GhostFrame-Studio/
   └── ffmpeg-master-latest-win64-gpl/
       └── bin/
           ├── ffmpeg.exe
           └── ffprobe.exe
   ```

GhostFrame lo detecta automáticamente — no tienes que configurar nada más.

---

## PASO 5 — Abrir PowerShell en la carpeta del programa

1. Abre la carpeta `GhostFrame-Studio` en el Explorador de archivos
2. Haz clic en la **barra de direcciones** (donde aparece la ruta), escribe `powershell` y pulsa **Enter**

PowerShell se abre ya dentro de la carpeta correcta.

---

## PASO 6 — Crear el entorno e instalar dependencias

Copia y pega estos tres comandos uno a uno. Espera a que cada uno termine antes del siguiente:

```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\activate
```
```powershell
pip install -r requirements.txt
```

> Esto tarda entre 3 y 8 minutos. Es completamente normal — está descargando e instalando todo lo necesario.

---

## PASO 7 — Instalar el soporte de GPU

Elige **solo una** opción según tu tarjeta gráfica:

### Tengo GPU NVIDIA
```powershell
pip install onnxruntime-gpu==1.19.2
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12
```

### Tengo GPU AMD o Intel
```powershell
pip install onnxruntime-directml==1.19.2
```

### No tengo GPU / no estoy seguro
```powershell
pip install onnxruntime==1.19.2
```

> No necesitas instalar CUDA Toolkit del sistema. El programa carga las librerías necesarias automáticamente desde los paquetes de pip.

---

## PASO 8 — Instalar el audio del preview (libmpv)

Para escuchar el audio mientras previsualizas el vídeo dentro de la app necesitas instalar `libmpv-2.dll`. Sin este archivo el preview se ve pero no se escucha.

> El vídeo que exportes **siempre tendrá audio** — esto solo afecta a lo que escuchas dentro del programa mientras trabajas.

1. Ve a [github.com/shinchiro/mpv-winbuild-cmake/releases](https://github.com/shinchiro/mpv-winbuild-cmake/releases)
2. Descarga el archivo `.7z` más reciente — el que se llama `mpv-x86_64-...`
3. Ábrelo con [7-Zip](https://www.7-zip.org/) o WinRAR
4. Extrae **únicamente** el archivo `libmpv-2.dll`
5. Cópialo dentro de la carpeta `GhostFrame-Studio/`, junto a `main.py`

A partir de ahí el audio se escucha automáticamente en el preview.

---

## PASO 9 — Ejecutar el programa

```powershell
python main.py
```

> **La primera vez** descarga el modelo de detección de caras (~500 MB). Solo ocurre una vez. Ten paciencia.

---

## PASO 10 — Arranque rápido para el día a día

Para no tener que abrir PowerShell cada vez que uses el programa, haz doble clic en **`run_silent.bat`** — ya viene incluido en la carpeta. Abre GhostFrame directamente.

---

## ¿Algo no funciona?

### El programa no arranca / error de DLL
Asegúrate de haber instalado el **Visual C++ Redistributable** del Paso 3.  
Descarga: [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### FFmpeg no encontrado
Comprueba que la carpeta `ffmpeg*/bin/` está **dentro** de `GhostFrame-Studio/` exactamente como se muestra en el Paso 4.

### CUDA no funciona / el análisis va muy lento
Ve a **Herramientas → Configuración → Proveedor ONNX** y cambia a `directml` o `cpu`.

Si usas NVIDIA y sigue fallando, reinstala los paquetes `nvidia-*-cu12` con el entorno `.venv` activado:
```powershell
.\.venv\Scripts\activate
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12
```

### El modelo de caras no se descarga
Descárgalo manualmente desde [insightface releases](https://github.com/deepinsight/insightface/releases).  
Descomprime `buffalo_l.zip` en:
```
C:\Users\TU_USUARIO\.insightface\models\buffalo_l\
```

### No se escucha audio en el preview
Falta `libmpv-2.dll` en la carpeta del programa. Sigue el **Paso 8**.

### Error `numpy 2.x incompatible`
```powershell
pip install "numpy<2"
```
