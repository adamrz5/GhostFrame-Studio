# Instalar GhostFrame Studio
### Windows — guía paso a paso

---

## Lo que necesitas antes de empezar

- Windows 10 o 11 (64 bits)
- Conexión a internet
- ~5 GB de espacio libre en disco

---

## PASO 1 — Descargar el programa

Ve a https://github.com/adamrz5/GhostFrame-Studio y haz clic en **Code → Download ZIP**.  
Descomprime la carpeta donde quieras, por ejemplo en `C:\GhostFrame-Studio\`.

---

## PASO 2 — Instalar Python

1. Ve a https://www.python.org/downloads/ y descarga **Python 3.12**
2. Ejecuta el instalador
3. **IMPORTANTE:** marca la casilla **"Add Python to PATH"** antes de darle a instalar
4. Verifica que se instaló bien abriendo PowerShell y escribiendo:
   ```
   python --version
   ```
   Debe mostrar `Python 3.12.x`

---

## PASO 3 — Instalar FFmpeg

FFmpeg es el programa que usa GhostFrame para leer y exportar vídeos.

1. Ve a https://github.com/BtbN/FFmpeg-Builds/releases
2. Descarga `ffmpeg-master-latest-win64-gpl.zip`
3. Descomprime esa carpeta **dentro de la carpeta de GhostFrame**:
   ```
   GhostFrame-Studio/
   └── ffmpeg-master-latest-win64-gpl/
       └── bin/
           ├── ffmpeg.exe
           └── ffprobe.exe
   ```
GhostFrame lo detecta automáticamente, no tienes que configurar nada más.

---

## PASO 4 — Abrir PowerShell en la carpeta del programa

1. Abre la carpeta `GhostFrame-Studio` en el explorador de archivos
2. Haz clic en la barra de direcciones, escribe `powershell` y pulsa Enter

---

## PASO 5 — Crear el entorno virtual e instalar dependencias

Copia y pega estos comandos uno a uno:

```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\activate
```
```powershell
pip install -r requirements.txt
```

Esto tarda entre 3 y 8 minutos. Es normal.

---

## PASO 6 — Instalar el soporte de GPU

Elige según tu tarjeta gráfica:

### Tengo GPU NVIDIA
```powershell
pip install onnxruntime-gpu==1.19.2
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12
```

### Tengo GPU AMD o Intel
```powershell
pip install onnxruntime-directml==1.19.2
```

### No tengo GPU / no sé
```powershell
pip install onnxruntime==1.19.2
```

---

## PASO 7 — Ejecutar el programa

```powershell
python main.py
```

La primera vez descarga el modelo de detección de caras (~500 MB). Solo ocurre una vez.

---

## PASO 8 — Arranque rápido (para el día a día)

Para no tener que abrir PowerShell cada vez, haz doble clic en **`run_silent.bat`** que ya viene incluido en la carpeta.

---

## ¿Algo no funciona?

### El programa no arranca / error de DLL
Instala esto: https://aka.ms/vs/17/release/vc_redist.x64.exe  
Luego vuelve a intentarlo.

### FFmpeg no encontrado
Asegúrate de que la carpeta `ffmpeg*/bin/` está dentro de `GhostFrame-Studio/` como se indica en el Paso 3.

### CUDA no funciona y el análisis va lento
Ve a **Herramientas → Configuración → Proveedor ONNX** y cambia a `directml` o `cpu`.

### El modelo de caras no descarga
Descárgalo manualmente desde https://github.com/deepinsight/insightface/releases  
Descomprime `buffalo_l.zip` en `C:\Users\TU_USUARIO\.insightface\models\buffalo_l\`

---

## PASO 9 — Instalar audio para el preview

Para escuchar el audio mientras previsualizas el vídeo dentro de la app necesitas instalar `libmpv-2.dll`. El vídeo exportado siempre tiene audio, pero el preview dentro del programa lo necesita para reproducirlo.

1. Ve a https://github.com/shinchiro/mpv-winbuild-cmake/releases
2. Descarga el archivo `.7z` más reciente (el que se llama `mpv-x86_64-...`)
3. Ábrelo con [7-Zip](https://www.7-zip.org/) o WinRAR
4. Extrae únicamente el archivo `libmpv-2.dll`
5. Cópialo dentro de la carpeta `GhostFrame-Studio/`, junto a `main.py`
6. Reinicia el programa

A partir de ahí el audio se escucha automáticamente en el preview.
