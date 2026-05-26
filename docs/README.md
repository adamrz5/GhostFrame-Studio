# GhostFrame Studio

> **Repositorio:** https://github.com/adamrz5/GhostFrame-Studio

Editor de censura facial para vídeos de entrevistas. Detecta, rastrea y re-identifica
personas a lo largo de todo el vídeo (aunque salgan y vuelvan a aparecer), y permite
censurarlas de forma selectiva sin pérdida de calidad de audio ni de imagen.

La interfaz usa un estilo oscuro de alto contraste: fondo negro, texto blanco y
colores distintos por persona para que cada cara detectada sea fácil de reconocer.

---

## Instalación rápida

```powershell
git clone https://github.com/adamrz5/GhostFrame-Studio.git
cd GhostFrame-Studio
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# Elige UN proveedor ONNX según tu GPU (ver abajo)
python main.py
```

Para la guía de instalación completa paso a paso (GPU, libmpv, FFmpeg, solución de problemas):  
→ **[INSTALACION.md](../INSTALACION.md)**

---

## Requisitos del sistema

| Requisito | Detalle |
|-----------|---------|
| Python | **3.12** recomendado (3.11 compatible; 3.14 no soportado) |
| FFmpeg | En el PATH, configurado en Ajustes, o carpeta `ffmpeg*/bin/` dentro del proyecto ([descargar](https://github.com/BtbN/FFmpeg-Builds/releases)) |
| Visual C++ Redist. | Requerido en Windows para ONNX Runtime ([descargar](https://aka.ms/vs/17/release/vc_redist.x64.exe)) |
| numpy | **<2.0** — insightface 0.7.3 no es compatible con numpy 2.x |
| onnxruntime | **1.19.2 exacto** — instalar solo UNA variante: `onnxruntime`, `onnxruntime-gpu` o `onnxruntime-directml` |
| libmpv-2.dll | Opcional — solo para audio en preview. **No incluida en el repositorio** (>100 MB). Descargar de [mpv-winbuild-cmake](https://github.com/shinchiro/mpv-winbuild-cmake/releases) |

---

## Proveedor ONNX — elige uno

```powershell
# GPU NVIDIA (más rápido, no requiere CUDA Toolkit del sistema):
pip install onnxruntime-gpu==1.19.2
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12

# GPU universal Windows — NVIDIA / AMD / Intel (DirectX 12):
pip install onnxruntime-directml==1.19.2

# CPU puro (siempre funciona):
pip install onnxruntime==1.19.2
```

No hace falta instalar el CUDA Toolkit del sistema para la opción NVIDIA: el programa añade
al `PATH` del proceso las DLL de los paquetes `nvidia-*-cu12` instalados por pip antes de
cargar ONNX Runtime.

---

## Ejecución

```bash
python main.py
# O con un vídeo directamente:
python main.py C:\ruta\al\video.mp4
```

Al arrancar, GhostFrame comprueba paquetes Python, FFmpeg y DLLs CUDA opcionales.
Si falta algo, muestra un aviso con comandos `pip` y enlaces directos de instalación.

Para arrancar sin consola en Windows:

```bat
run_silent.bat
```

Los logs se guardan en `logs/ghostframe.log`. También puedes abrir
**Ayuda → Diagnóstico** para ver Python, GPU, FFmpeg, ONNX Runtime, CUDA y rutas.

---

## Flujo de trabajo

1. **Abrir vídeo** — arrastra el archivo o usa el botón. Soporta MP4, MOV, MKV, AVI.
2. **Analizar caras** — el programa procesa el vídeo (cada N frames, configurable) y
   detecta todas las personas. Los resultados se guardan como `.gfs` dentro de
   `cache/sessions/` en el proyecto, no junto al vídeo original.
3. **Configurar censura** — panel derecho con una tarjeta por persona:
   - Las tarjetas se ordenan por cantidad de frames detectados: la persona que más aparece se muestra como **Persona 1**.
   - Toggle on/off
   - Efecto: Pixelado / Gaussian Blur / Caja negra
   - Intensidad (1-10)
   - Margen alrededor de la cara (0-50%)
   - Rango de tiempo: completo o personalizado con sliders
   - **Barra visual de apariciones**: muestra en qué momentos del vídeo está presente cada persona
4. **Preview en tiempo real** — arrastra el timeline para ver la censura aplicada al instante.
   - `Espacio` reproduce/pausa el preview.
   - Rueda del ratón sobre el preview: zoom visual. Doble clic: reset de zoom.
   - **Mostrar detecciones** dibuja el overlay por persona: rectángulo fino (sólido = frame analizado, discontinuo = frame interpolado) con etiqueta plana en el color de acento de la persona. Estilo Adobe Premiere.
5. **Re-agrupar** — vuelve a fusionar personas con los ajustes actuales sin ejecutar de nuevo InsightFace.
   - **Cancelar** — botón visible en la barra superior durante análisis o render. Detiene la operación al terminar el frame/vídeo actual.
6. **Renderizar** — genera un MP4 final. Por defecto propone la carpeta `renders/`
   dentro del proyecto, aunque puedes elegir otra ruta. El audio se copia bit a bit cuando MP4 lo soporta;
   si el codec original no es compatible con MP4, se convierte a AAC.
   Los subtítulos se copian sin modificarlos cuando existen y el contenedor lo permite.

Funciones útiles:
- **Ayuda → Manual y atajos** (`F1`): manual integrado dentro de la app.
- **Archivo → Exportar/Importar configuración**: guarda/carga qué personas se censuran y con qué efecto en `.gfscfg`.
- **Editar → Deshacer/Rehacer**: `Ctrl+Z`, `Ctrl+Y` o `Ctrl+Shift+Z` para fusionar/dividir personas.
- **Herramientas → Configuración → Aplicar configuración recomendada**: detecta CPU, RAM, GPU y proveedores ONNX para proponer valores equilibrados.

---

## Arquitectura técnica

### Tracking de identidad en dos etapas

A diferencia de proyectos similares (deface, blurfaces) que solo detectan caras frame a frame
sin identidad persistente, GhostFrame Studio usa un tracker de dos etapas:

**Etapa 1 — IoU in-scene** (rápida, para movimiento continuo):
Si la cara de un frame se superpone con la de un frame reciente (IoU ≥ 0.40), se asigna
automáticamente a la misma persona sin necesidad de comparar embeddings.

**Etapa 2 — ArcFace re-id** (para re-entradas tras cortes o ausencias):
Si la etapa 1 no encuentra match, se compara el embedding ArcFace (512 dimensiones) contra
todos los perfiles conocidos usando FAISS `IndexFlatIP` (similitud coseno tras L2-normalización).
Si la similitud supera el umbral configurado (por defecto 0.65), es la misma persona.

**Robustez del detector:** errores de GPU en un frame individual no abortan el análisis —
se devuelve `[]` y se cuenta. Solo se aborta si 10 frames consecutivos fallan.
El índice FAISS se reconstruye una vez por frame (no una vez por detección), lo que
mantiene el coste O(n) total en lugar de O(n²).

### Renderizado sin degradación

```
ffmpeg (decode) → stdout [raw BGR24] → Python (censura) → stdin [raw BGR24] → ffmpeg (encode)
```

- **Sin archivo temporal**: frames van de la decodificación a la codificación en memoria.
- **Sin doble compresión**: los frames viajan sin comprimir entre los dos procesos.
- **Audio**: copia bit a bit cuando el codec original es compatible con el contenedor MP4; si no lo es (o si ffprobe no pudo detectarlo y devuelve `unknown`), se transcodifica a AAC para garantizar reproducibilidad.
- **Subtítulos**: si el origen trae subtítulos de texto compatibles con MP4, se convierten a `mov_text`; si FFmpeg los rechaza o son bitmap/incompatibles, el render reintenta sin subtítulos para no perder el vídeo.
- **Vídeo**: intenta codificación hardware (`h264_nvenc`, `h264_qsv`, `h264_amf`) y cae a
  `libx264 -crf 18` si no hay encoder hardware usable.
- **Nota CPU/GPU**: la inferencia facial puede usar CUDA/DirectML y el encode puede usar GPU,
  pero la censura visual (`blur`, `pixelate`, `blackbox`) se aplica en CPU con OpenCV/NumPy.

### Búsqueda de embeddings con FAISS

Si `faiss-cpu` está instalado, la búsqueda del embedding más similar usa `IndexFlatIP`
(producto interno = similitud coseno tras L2-normalización), con complejidad O(1) en la
práctica para el número de personas típico en una entrevista.

---

## Configuración avanzada

**Menú Herramientas → Configuración** (o `Ctrl+,`):

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| Analizar cada N frames | 5 | Mayor = más rápido, puede perder apariciones breves |
| Umbral de similitud | 0.65 | Sube si une personas distintas; baja si parte a la misma |
| Umbral IoU | 0.40 | Para tracking continuo en escena |
| Máx. interpolación | 30 frames | Gaps más largos no se interpolan |
| Proveedor ONNX | cuda | CUDA en NVIDIA, DirectML como alternativa Windows, CPU como fallback |
| Tamaño detector | 320 | 320 es más rápido; 640 detecta mejor caras pequeñas |
| CRF de vídeo | 18 | 0 = lossless, 18 = visualmente idéntico, 28 = comprimido |
| Preset ffmpeg | fast | Más lento = menor tamaño de archivo |

El diálogo de configuración muestra una recomendación automática basada en el hardware detectado:
CPU, RAM, GPU, proveedores disponibles de ONNX Runtime y encoders FFmpeg.

---

## Estructura de archivos

```
GhostFrame-Studio/
├── main.py                    # Punto de entrada + crash handler global
├── requirements.txt
├── core/
│   ├── face_detector.py       # InsightFace: detección + embeddings ArcFace
│   ├── face_tracker.py        # Tracking IoU + re-id FAISS/scipy
│   ├── video_processor.py     # VideoReader + efectos de censura con feathering
│   ├── renderer.py            # Orquestación del renderizado
│   ├── ffmpeg_utils.py        # Detección de ffmpeg + pipe de renderizado
│   ├── session.py             # Cache .gfs en cache/sessions + export JSON/.gfscfg
│   └── settings.py            # Configuración persistente (~/.ghostframe_studio_settings.json)
└── ui/
    ├── main_window.py         # Ventana principal con threading seguro
    ├── person_card.py         # Tarjeta por persona + barra de apariciones visual
    ├── timeline_widget.py     # Scrubber de frames con tiempo formateado
    ├── preview_widget.py      # Vista previa con zoom visual
    ├── settings_dialog.py     # Diálogo de configuración + recomendación hardware
    └── batch_dialog.py        # Modo batch: procesar carpeta de vídeos
```

---

## Problemas conocidos y soluciones

| Problema | Solución |
|----------|----------|
| `DLL load failed` en Windows | Instala [Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `numpy 2.x incompatible` | `pip install "numpy<2"` |
| FFmpeg no encontrado | Coloca una carpeta `ffmpeg*/bin/` dentro del proyecto, añade `bin/` al PATH o configura la ruta en Configuración |
| CUDA cae a CPU / `LoadLibrary error 126` | Reinstala todos los paquetes `nvidia-*-cu12` dentro de la `.venv` (ver INSTALACION.md Paso 5 Ruta B) |
| El tracker une dos personas distintas | Aumenta el umbral de similitud (>0.65) y/o usa Fusionar/Dividir |
| El tracker separa a la misma persona | Reduce el umbral de similitud o usa Fusionar/Dividir |
| Vídeo sin audio en el output | El vídeo original no tiene audio, o FFmpeg no pudo leer/convertir el stream de audio |
| Vídeo con tinte verde/rosa en Telegram después del render | Fuente HDR (smpte2084/HLG) — la versión actual fuerza los tres tags de color a bt709 automáticamente |
| El diálogo de Configuración se quedó congelado | Espera 8 segundos; hay un timeout automático que lo rehabilita |
| No hay audio en el preview | `libmpv-2.dll` no está en la carpeta del proyecto — ver INSTALACION.md Paso 6 |

---

## Referencias oficiales

- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html): CUDA EP acelera inferencia en GPU NVIDIA; ONNX Runtime 1.19.x usa CUDA 12.x/cuDNN 9.x en PyPI.
- [ONNX Runtime DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html): alternativa Windows para GPUs DirectX 12, con soporte amplio de hardware.
- [FFmpeg Download](https://ffmpeg.org/download.html): página oficial de FFmpeg con enlaces a builds Windows compiladas.
