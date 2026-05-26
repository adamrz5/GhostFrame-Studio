"""
GhostFrame Studio — entry point.

Usage:
    python main.py
    python main.py /path/to/video.mp4    # optional: open video directly
"""
import sys
import os
import traceback
import importlib.util
import datetime
import atexit
import tempfile

# Ensure project root is on sys.path regardless of CWD
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_ICON_PATH = os.path.join(_ROOT, "assets", "ghostframe-app-icon.ico")
_LOG_DIR = os.path.join(_ROOT, "logs")
_LOG_PATH = os.path.join(_LOG_DIR, "ghostframe.log")
_LOG_FILE_HANDLE = None


class _Tee:
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, text):
        if not text:
            return
        try:
            self.log_file.write(text)
            self.log_file.flush()
        except Exception:
            pass
        if self.original:
            try:
                self.original.write(text)
                self.original.flush()
            except Exception:
                pass

    def flush(self):
        try:
            self.log_file.flush()
        except Exception:
            pass
        if self.original:
            try:
                self.original.flush()
            except Exception:
                pass


def _setup_logging() -> None:
    global _LOG_PATH, _LOG_FILE_HANDLE
    candidates = [
        os.path.join(_ROOT, "logs", "ghostframe.log"),
        os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()), "GhostFrame Studio", "logs", "ghostframe.log"),
        os.path.join(tempfile.gettempdir(), "ghostframe.log"),
    ]
    log_file = None
    last_error = None
    for candidate in candidates:
        try:
            os.makedirs(os.path.dirname(candidate), exist_ok=True)
            log_file = open(candidate, "a", encoding="utf-8", buffering=1)
            _LOG_PATH = candidate
            os.environ["GHOSTFRAME_LOG_PATH"] = candidate
            break
        except Exception as exc:
            last_error = exc
    if log_file is None:
        print(f"[Logging] No se pudo abrir archivo de log: {last_error}")
        return
    _LOG_FILE_HANDLE = log_file
    banner = (
        "\n"
        + "=" * 72
        + f"\nGhostFrame start {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        + f"Python: {sys.version}\n"
        + f"Executable: {sys.executable}\n"
        + f"CWD: {os.getcwd()}\n"
        + "=" * 72
        + "\n"
    )
    log_file.write(banner)
    log_file.flush()
    sys.stdout = _Tee(getattr(sys, "__stdout__", None), log_file)
    sys.stderr = _Tee(getattr(sys, "__stderr__", None), log_file)


def _close_logging() -> None:
    try:
        if _LOG_FILE_HANDLE:
            _LOG_FILE_HANDLE.flush()
            _LOG_FILE_HANDLE.close()
    except Exception:
        pass


# Keeps os.add_dll_directory() handles alive for the lifetime of the process.
# Letting them be GC'd removes the directory from the DLL search path.
_dll_directory_handles: list = []


def _native_alert(title: str, message: str) -> None:
    """Show a Windows message box without requiring PyQt5."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
            return
        except Exception:
            pass
    print(title)
    print(message)


def _module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def _preflight_dependencies() -> None:
    """
    Detect missing runtime pieces before importing the full app.
    PyQt may be missing too, so this uses a native Windows dialog as fallback.
    """
    required_modules = [
        ("PyQt5", "PyQt5", "pip install \"PyQt5>=5.15.9,<6.0.0\""),
        ("numpy", "numpy", "pip install \"numpy>=1.22.0,<2.0.0\""),
        ("cv2", "opencv-python-headless", "pip install \"opencv-python-headless>=4.8.0,<4.11.0\""),
        ("insightface", "insightface", "pip install insightface==0.7.3"),
        ("onnx", "onnx", "pip install onnx==1.16.1"),
        ("onnxruntime", "onnxruntime / onnxruntime-gpu / onnxruntime-directml", "pip install onnxruntime-gpu==1.19.2"),
        ("scipy", "scipy", "pip install \"scipy>=1.11.0,<2.0.0\""),
        ("PIL", "Pillow", "pip install \"Pillow>=10.0.0,<11.0.0\""),
        ("tqdm", "tqdm", "pip install tqdm>=4.65.0"),
    ]

    missing = [
        (package, command)
        for module, package, command in required_modules
        if _module_missing(module)
    ]

    ffmpeg_missing = False
    try:
        from core.ffmpeg_utils import find_ffmpeg
        ffmpeg_missing = find_ffmpeg() is None
    except Exception:
        ffmpeg_missing = True

    cuda_warning = False
    try:
        from core import settings as cfg
        provider = cfg.get("execution_provider")
        if provider == "cuda":
            import site
            bases = []
            try:
                bases.extend(site.getsitepackages())
            except Exception:
                pass
            try:
                bases.append(site.getusersitepackages())
            except Exception:
                pass
            required_cuda_dirs = [
                os.path.join("nvidia", "cuda_runtime", "bin"),
                os.path.join("nvidia", "cublas", "bin"),
                os.path.join("nvidia", "cudnn", "bin"),
                os.path.join("nvidia", "cuda_nvrtc", "bin"),
                os.path.join("nvidia", "cusolver", "bin"),
                os.path.join("nvidia", "cufft", "bin"),
                os.path.join("nvidia", "curand", "bin"),
                os.path.join("nvidia", "cusparse", "bin"),
                os.path.join("nvidia", "nvjitlink", "bin"),
            ]
            for rel in required_cuda_dirs:
                if not any(os.path.isdir(os.path.join(base, rel)) for base in bases):
                    cuda_warning = True
                    break
    except Exception:
        cuda_warning = False

    if not missing and not ffmpeg_missing and not cuda_warning:
        return

    lines = [
        "GhostFrame Studio necesita algunos componentes antes de arrancar correctamente.",
        "",
    ]
    if missing:
        lines.append("Faltan paquetes Python:")
        for package, command in missing:
            lines.append(f"  - {package}")
        lines.extend([
            "",
            "Comandos recomendados:",
            "  python -m pip install --upgrade pip setuptools wheel",
            "  pip install -r requirements.txt",
            "",
            "Si usas NVIDIA/CUDA:",
            "  pip uninstall onnxruntime onnxruntime-directml -y",
            "  pip install onnxruntime-gpu==1.19.2",
            "  pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12",
            "",
        ])
    if ffmpeg_missing:
        lines.extend([
            "FFmpeg no se ha encontrado.",
            "Descarga recomendada:",
            "  https://github.com/BtbN/FFmpeg-Builds/releases",
            "Coloca la carpeta ffmpeg*/bin dentro del proyecto o configura la ruta en ajustes.",
            "",
        ])
    if cuda_warning:
        lines.extend([
            "CUDA esta seleccionado, pero faltan carpetas DLL de NVIDIA instaladas por pip.",
            "Comando:",
            "  pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12",
            "",
        ])
    lines.extend([
        "Enlaces utiles:",
        "  Python: https://www.python.org/downloads/",
        "  Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe",
        "  FFmpeg: https://ffmpeg.org/download.html",
    ])

    _native_alert("GhostFrame Studio - faltan componentes", "\n".join(lines))
    if missing:
        sys.exit(1)


def _preload_nvidia_dlls() -> None:
    """
    Registra los directorios de DLLs de los paquetes pip de NVIDIA en el proceso
    de Windows ANTES de que onnxruntime intente cargarlas.

    Esto permite usar CUDAExecutionProvider sin instalar el CUDA Toolkit del sistema.
    Solo requiere haber instalado via pip:
        nvidia-cuda-runtime-cu12  →  cudart64_12.dll
        nvidia-cublas-cu12        →  cublas64_12.dll, cublasLt64_12.dll
        nvidia-cudnn-cu12         →  cudnn64_9.dll + companions
        nvidia-cufft-cu12         →  cufft64_11.dll
        nvidia-curand-cu12        →  curand64_10.dll
        nvidia-cusolver-cu12      →  cusolver64_11.dll
        nvidia-cusparse-cu12      →  cusparse64_12.dll
        nvidia-nvjitlink-cu12     →  nvJitLink_120_0.dll

    Sin esto, onnxruntime-gpu falla con "LoadLibrary error 126" porque Windows
    no encuentra las DLLs de CUDA en el PATH del sistema.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    import site
    bases: list[str] = []
    try:
        bases += site.getsitepackages()
    except Exception:
        pass
    try:
        bases.append(site.getusersitepackages())
    except Exception:
        pass

    # Subcarpetas donde los paquetes nvidia-*-cu12 instalan sus DLLs en Windows
    nvidia_bin_subdirs = [
        os.path.join("nvidia", "cuda_runtime", "bin"),
        os.path.join("nvidia", "cublas", "bin"),
        os.path.join("nvidia", "cudnn", "bin"),
        os.path.join("nvidia", "cuda_nvrtc", "bin"),
        os.path.join("nvidia", "cusolver", "bin"),
        os.path.join("nvidia", "cufft", "bin"),
        os.path.join("nvidia", "curand", "bin"),
        os.path.join("nvidia", "cusparse", "bin"),
        os.path.join("nvidia", "nvjitlink", "bin"),
    ]

    added = []
    current_path_entries = {
        os.path.normcase(os.path.abspath(p))
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p
    }
    for base in bases:
        for subdir in nvidia_bin_subdirs:
            full = os.path.join(base, subdir)
            full_norm = os.path.normcase(os.path.abspath(full))
            if os.path.isdir(full) and full_norm not in current_path_entries:
                # Añadir al PATH del proceso — método que usa Windows LoadLibrary
                # internamente al cargar DLLs transitivas (add_dll_directory no es suficiente
                # para DLLs cargadas por otras DLLs, solo para cargas directas de Python).
                os.environ["PATH"] = full + os.pathsep + os.environ.get("PATH", "")
                current_path_entries.add(full_norm)
                try:
                    # Store handle — if discarded, Windows removes the dir from the search path.
                    _dll_directory_handles.append(os.add_dll_directory(full))
                except Exception:
                    pass
                added.append(full)

    if added:
        print(f"[CUDA] {len(added)} carpeta(s) DLL de NVIDIA añadidas al PATH.")


def _preload_mpv_dll() -> None:
    """
    Añade la carpeta raíz del proyecto al PATH para que python-mpv encuentre
    libmpv-2.dll si está junto a main.py.
    También registra el directorio con os.add_dll_directory() para que Windows
    lo incluya en la búsqueda de DLLs transitivas.
    """
    dll_path = os.path.join(_ROOT, "libmpv-2.dll")
    if not os.path.isfile(dll_path):
        return  # La DLL no está aquí; python-mpv buscará en PATH del sistema.
    root_norm = os.path.normcase(_ROOT)
    path_entries = {os.path.normcase(p) for p in os.environ.get("PATH", "").split(os.pathsep)}
    if root_norm not in path_entries:
        os.environ["PATH"] = _ROOT + os.pathsep + os.environ.get("PATH", "")
    try:
        _dll_directory_handles.append(os.add_dll_directory(_ROOT))
    except Exception:
        pass
    print("[mpv] libmpv-2.dll encontrada en la carpeta del proyecto. Audio en preview activo.")


def _prepare_runtime() -> None:
    _setup_logging()
    atexit.register(_close_logging)
    # Precargar libmpv ANTES de cualquier import de python-mpv
    _preload_mpv_dll()
    # Precargar DLLs de NVIDIA ANTES de cualquier import de onnxruntime o insightface
    _preload_nvidia_dlls()
    _preflight_dependencies()
    # ONNX must be loaded before PyQt5 on this Windows setup. If Qt loads first,
    # onnx_cpp2py_export can fail later while InsightFace initializes.
    try:
        from onnx import onnx_cpp2py_export  # noqa: F401
    except Exception as exc:
        print(f"[Startup] Aviso: no se pudo precargar onnx_cpp2py_export: {exc}")


def _set_windows_app_id() -> None:
    """Make Windows taskbar group/icon use GhostFrame instead of python.exe."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "GhostFrameStudio.GhostFrame"
        )
    except Exception:
        pass


def _global_exception_hook(exc_type, exc_value, exc_tb):
    """Show unhandled exceptions in a dialog instead of a silent crash."""
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(tb, file=sys.stderr)
    try:
        from PyQt5.QtWidgets import QMessageBox
        from ui.window_chrome import apply_dark_title_bar
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error inesperado - GhostFrame Studio")
        apply_dark_title_bar(msg)
        msg.setText(f"<b>{exc_type.__name__}</b>: {exc_value}")
        msg.setDetailedText(tb)
        msg.exec_()
    except Exception:
        pass


def main():
    _prepare_runtime()
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QIcon

    # Install global crash handler before creating the app
    sys.excepthook = _global_exception_hook
    _set_windows_app_id()

    # HiDPI support (must be set before QApplication)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("GhostFrame")
    app.setOrganizationName("GhostFrameStudio")
    if os.path.exists(_APP_ICON_PATH):
        app.setWindowIcon(QIcon(_APP_ICON_PATH))

    from ui.main_window import MainWindow
    from ui.window_chrome import apply_dark_title_bar
    window = MainWindow()
    apply_dark_title_bar(window)
    window.show()

    # Optional: open a video passed as CLI argument
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window._load_video(sys.argv[1])

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
