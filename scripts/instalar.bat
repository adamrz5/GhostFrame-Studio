@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Instalacion - GhostFrame Studio

cd /d "%~dp0"

echo ============================================================
echo GhostFrame Studio - instalador interactivo Windows
echo ============================================================
echo.
echo Este instalador:
echo   - crea/usa .venv
echo   - instala dependencias base
echo   - pregunta tu GPU e instala SOLO un proveedor ONNX
echo   - comprueba FFmpeg y libmpv-2.dll
echo   - configura GhostFrame segun tu eleccion
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo [FALLO] No se encontro Python Launcher "py".
    echo Instala Python 3.12 desde:
    echo https://www.python.org/downloads/
    start "" "https://www.python.org/downloads/"
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('py -3.12 -c "import sys; print(sys.version.split()[0])" 2^>nul') do set PYVER=%%v
if not defined PYVER (
    echo [FALLO] No se encontro Python 3.12.
    echo GhostFrame esta probado con Python 3.12. Instala Python 3.12 x64:
    echo https://www.python.org/downloads/release/python-31210/
    start "" "https://www.python.org/downloads/release/python-31210/"
    pause
    exit /b 1
)
echo [OK] Python 3.12 detectado: %PYVER%

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Creando entorno virtual .venv...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo [FALLO] No se pudo crear .venv.
        pause
        exit /b 1
    )
) else (
    echo [OK] Entorno virtual existente: .venv
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [FALLO] No se pudo activar .venv.
    pause
    exit /b 1
)

echo.
echo Actualizando pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [FALLO] pip no pudo actualizarse.
    pause
    exit /b 1
)

echo.
echo Instalando dependencias base...
pip install -r requirements.txt
if errorlevel 1 (
    echo [FALLO] Error instalando requirements.txt.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Seleccion del proveedor de IA / GPU
echo ============================================================
echo.
echo Deteccion automatica de GPU por Windows:
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | Format-Table -AutoSize" 2>nul
echo.
echo Elige UNA opcion:
echo   1 - NVIDIA CUDA  (mas rapido en NVIDIA, recomendado si tienes RTX/GTX)
echo   2 - AMD / Intel / NVIDIA DirectML  (GPU universal Windows)
echo   3 - CPU solamente  (mas compatible, mas lento)
echo.
set /p GPUCHOICE="Opcion [1/2/3]: "
if "%GPUCHOICE%"=="" set GPUCHOICE=3

echo.
echo Limpiando proveedores ONNX para evitar conflictos...
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml >nul 2>nul

set PROVIDER=cpu
if "%GPUCHOICE%"=="1" (
    set PROVIDER=cuda
    echo.
    echo Instalando ONNX Runtime GPU CUDA y DLLs NVIDIA via pip...
    pip install onnxruntime-gpu==1.19.2
    if errorlevel 1 goto :onnx_fail
    pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12
    if errorlevel 1 goto :onnx_fail
) else if "%GPUCHOICE%"=="2" (
    set PROVIDER=directml
    echo.
    echo Instalando ONNX Runtime DirectML...
    pip install onnxruntime-directml==1.19.2
    if errorlevel 1 goto :onnx_fail
) else (
    set PROVIDER=cpu
    echo.
    echo Instalando ONNX Runtime CPU...
    pip install onnxruntime==1.19.2
    if errorlevel 1 goto :onnx_fail
)

echo.
echo Configurando proveedor por defecto en GhostFrame: %PROVIDER%
python -c "from core import settings as cfg; s=cfg.load(); s['execution_provider']='%PROVIDER%'; cfg.save(s); print('Proveedor guardado:', s['execution_provider'])"

echo.
echo ============================================================
echo Comprobaciones externas
echo ============================================================

echo.
echo [FFmpeg]
python -c "from core.ffmpeg_utils import find_ffmpeg; p=find_ffmpeg(); print(p or 'NO ENCONTRADO'); raise SystemExit(0 if p else 2)"
if errorlevel 2 (
    echo [AVISO] FFmpeg no encontrado.
    echo Descarga recomendado:
    echo https://github.com/BtbN/FFmpeg-Builds/releases
    echo Extrae ffmpeg-master-latest-win64-gpl.zip dentro del proyecto
    echo o configura ffmpeg.exe en Herramientas ^> Configuracion.
    set /p OPENFF="Abrir pagina de FFmpeg ahora? [S/N]: "
    if /I "!OPENFF!"=="S" start "" "https://github.com/BtbN/FFmpeg-Builds/releases"
) else (
    echo [OK] FFmpeg detectado.
)

echo.
echo [Audio preview / mpv]
if exist "libmpv-2.dll" (
    echo [OK] libmpv-2.dll encontrada en la carpeta del proyecto.
) else (
    echo [AVISO] Falta libmpv-2.dll.
    echo GhostFrame abrira igual, pero el preview no tendra audio.
    echo Descarga:
    echo https://github.com/shinchiro/mpv-winbuild-cmake/releases
    echo Archivo recomendado: mpv-x86_64-*.7z
    echo Extrae libmpv-2.dll junto a main.py.
    set /p OPENMPV="Abrir pagina de mpv ahora? [S/N]: "
    if /I "!OPENMPV!"=="S" start "" "https://github.com/shinchiro/mpv-winbuild-cmake/releases"
)

echo.
echo [Visual C++ Redistributable]
echo Si al abrir falla alguna DLL de Windows/MSVC, instala VC++ x64:
echo https://aka.ms/vs/17/release/vc_redist.x64.exe
set /p OPENVC="Abrir pagina VC++ ahora? [S/N]: "
if /I "!OPENVC!"=="S" start "" "https://aka.ms/vs/17/release/vc_redist.x64.exe"

echo.
echo ============================================================
echo Verificacion final
echo ============================================================
python -c "import numpy, cv2, insightface, onnx, onnxruntime, PyQt5; print('Imports base OK'); print('ONNX providers:', onnxruntime.get_available_providers())"
if errorlevel 1 (
    echo [FALLO] Algo no importa correctamente. Revisa el error de arriba.
    pause
    exit /b 1
)

python -c "import importlib.util; print('python-mpv:', 'OK' if importlib.util.find_spec('mpv') else 'FALTA')"

echo.
echo [OK] Instalacion completada.
echo Ejecuta:
echo   exe.bat
echo o:
echo   run_silent.bat
echo.
pause
exit /b 0

:onnx_fail
echo.
echo [FALLO] Error instalando el proveedor ONNX elegido.
echo Prueba de nuevo y elige CPU si quieres el modo mas compatible.
pause
exit /b 1
