<div align="center">

# GhostFrame Studio

**Censura facial inteligente para vídeos — detecta, rastrea y re-identifica personas automáticamente**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D4?logo=windows&logoColor=white)](https://github.com/adamrz5/GhostFrame-Studio)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-required-green?logo=ffmpeg&logoColor=white)](https://github.com/BtbN/FFmpeg-Builds/releases)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.19.2-orange)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/adamrz5/GhostFrame-Studio?label=version)](https://github.com/adamrz5/GhostFrame-Studio/releases)

[📥 Descargar instalador](https://github.com/adamrz5/GhostFrame-Studio/releases) · [📖 Guía de instalación](INSTALACION.md) · [🐛 Reportar un problema](https://github.com/adamrz5/GhostFrame-Studio/issues)

</div>

---

## Demo

https://github.com/user-attachments/assets/fe1f2fd0-ec49-4e6f-a274-bda56f3768ec

---

GhostFrame Studio es una herramienta de escritorio para **censurar caras en vídeos de entrevistas y reportajes**. A diferencia de otras soluciones, recuerda quién es cada persona a lo largo de todo el vídeo — aunque salga del plano y vuelva — y te permite decidir con un toggle a quién censurar y con qué efecto, sin pérdida de calidad.

---

## ⚠️ Aviso legal

GhostFrame Studio está diseñado exclusivamente para **proteger la identidad** de personas en vídeos: periodismo, documentales, investigación y cualquier contexto donde la privacidad deba preservarse.

El uso de esta herramienta para manipular vídeos sin consentimiento o con fines ilegales es responsabilidad exclusiva del usuario. El autor no asume ninguna responsabilidad por un uso indebido del software.

---

## Características

- 🎯 **Detección y tracking por identidad** — reconoce a cada persona individualmente aunque desaparezca y reaparezca
- 🎛️ **Control total por persona** — pixelado, blur gaussiano o caja negra, intensidad y margen ajustables
- ⚡ **Aceleración GPU** — compatible con NVIDIA (CUDA), AMD e Intel (DirectML)
- 🔇 **Sin pérdida de calidad** — el audio se copia bit a bit; el vídeo se renderiza sin doble compresión
- 👁️ **Preview en tiempo real** — ve la censura aplicada antes de exportar
- 📁 **Proceso por lotes** — censura una carpeta entera de vídeos de una vez
- 💾 **Sesiones guardadas** — el análisis se cachea; no tienes que repetirlo si vuelves al mismo vídeo
- 🔊 **Audio en el preview** — reproducción con audio incluida (requiere libmpv-2.dll)

---

## Requisitos del sistema

| Requisito | Versión / Detalle |
|-----------|-------------------|
| Sistema operativo | Windows 10 o 11 (64 bits) |
| Python | **3.12** recomendado · 3.11 compatible · 3.14 no soportado |
| FFmpeg | Carpeta `ffmpeg*/bin/` dentro del proyecto o en el PATH · [Descargar](https://github.com/BtbN/FFmpeg-Builds/releases) |
| Visual C++ Redist. | Requerido en Windows · [Descargar](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| numpy | `< 2.0` — insightface 0.7.3 no es compatible con numpy 2.x |
| onnxruntime | **1.19.2 exacto** — instalar solo una variante (ver abajo) |
| libmpv-2.dll | Para audio en el preview · [Descargar de mpv-winbuild-cmake](https://github.com/shinchiro/mpv-winbuild-cmake/releases) |

---

## Instalación rápida

```powershell
git clone https://github.com/adamrz5/GhostFrame-Studio.git
cd GhostFrame-Studio
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Elige **una sola opción** según tu GPU:

```powershell
# NVIDIA (más rápido)
pip install onnxruntime-gpu==1.19.2
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12

# AMD / Intel — DirectX 12
pip install onnxruntime-directml==1.19.2

# Sin GPU / no estoy seguro
pip install onnxruntime==1.19.2
```

```powershell
python main.py
```

> **En el primer arranque** GhostFrame descarga automáticamente el modelo de detección de caras (~450 MB) en `C:\Users\TU_USUARIO\.insightface\models\buffalo_l\`. Solo ocurre una vez.

> ⚠️ Para la guía completa con GPU, audio y solución de problemas → **[INSTALACION.md](INSTALACION.md)**

---

## Cómo se usa

### 1 · Abrir el vídeo
Arrastra el archivo o usa el botón **Abrir**. Formatos: MP4, MOV, MKV, AVI.

### 2 · Analizar las caras
Pulsa **Analizar**. El programa detecta todas las personas del vídeo y guarda los resultados en `cache/sessions/` para no repetir el análisis.

### 3 · Configurar la censura
En el panel derecho aparece una tarjeta por persona. Las tarjetas se ordenan por protagonismo (Persona 1 = la que más aparece).

Cada tarjeta permite:
- Activar o desactivar la censura con un toggle
- Elegir el efecto: **Pixelado**, **Gaussian Blur** o **Caja negra**
- Ajustar intensidad (1–10) y margen alrededor de la cara (0–50%)
- Limitar la censura a un tramo concreto del vídeo
- Ver una barra visual con los momentos exactos en que aparece esa persona

### 4 · Previsualizar
Arrastra el timeline para ver la censura aplicada al instante.

| Atajo | Acción |
|-------|--------|
| `Espacio` | Reproducir / Pausar |
| Rueda del ratón | Zoom en el preview |
| Doble clic | Resetear zoom |
| `F1` | Manual y atajos de teclado |

### 5 · Renderizar
Pulsa **Renderizar**. El vídeo se guarda en `renders/`. El audio y los subtítulos se conservan sin recomprimir.

### Funciones adicionales

| Función | Acceso |
|---------|--------|
| Manual y atajos de teclado | `F1` |
| Exportar / Importar configuración | Archivo → Exportar/Importar (`.gfscfg`) |
| Diagnóstico del sistema | Ayuda → Diagnóstico |
| Configuración recomendada automática | Herramientas → Configuración → Aplicar recomendada |
| Proceso por lotes | Herramientas → Proceso por lotes |
| Deshacer / Rehacer | `Ctrl+Z` / `Ctrl+Y` |

---

## Configuración avanzada

**Herramientas → Configuración** o `Ctrl+,`

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| Analizar cada N frames | 5 | Más alto = más rápido, puede perder apariciones breves |
| Umbral de similitud | 0.65 | Sube si une personas distintas; baja si divide a la misma |
| Umbral IoU | 0.40 | Sensibilidad del tracking continuo en escena |
| Máx. interpolación | 30 frames | Huecos más largos no se interpolan |
| Proveedor ONNX | cuda | CUDA para NVIDIA, DirectML para AMD/Intel, CPU como fallback |
| Tamaño detector | 320 | 640 detecta mejor caras pequeñas, más lento |
| CRF de vídeo | 18 | 0 = sin pérdidas · 18 = visualmente idéntico · 28 = comprimido |
| Preset ffmpeg | fast | Más lento = archivo más pequeño |

---

## Updates

- **v1.0.0** — Mayo 2026 · Lanzamiento público inicial · Instalador Windows con Python embebido · Detección automática de GPU

Ver historial completo en [CHANGELOG.md](CHANGELOG.md).

---

## Solución de problemas

| Problema | Solución |
|----------|----------|
| `DLL load failed` al arrancar | Instala [Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `numpy 2.x incompatible` | `pip install "numpy<2"` |
| FFmpeg no encontrado | Pon la carpeta `ffmpeg*/bin/` dentro de `GhostFrame-Studio/` como indica la instalación |
| CUDA cae a CPU / `LoadLibrary error 126` | Reinstala los paquetes `nvidia-*-cu12` dentro de la `.venv` |
| El tracker une a dos personas distintas | Sube el umbral de similitud (> 0.65) y/o usa Fusionar/Dividir |
| El tracker separa a la misma persona en dos | Baja el umbral de similitud o usa Fusionar/Dividir |
| Sin audio en el vídeo exportado | El vídeo original no tenía audio, o FFmpeg no pudo convertirlo |
| Tinte verde/rosa en Telegram | Fuente HDR — el programa corrige los tags de color a bt709 automáticamente |
| Sin audio en el preview | `libmpv-2.dll` no está en la carpeta del proyecto — ver [INSTALACION.md](INSTALACION.md) |
| El diálogo de Configuración se congeló | Espera 8 s; hay un timeout automático que lo rehabilita |

---

## Arquitectura técnica

### Tracking de identidad en dos etapas

A diferencia de herramientas que solo detectan caras frame a frame sin memoria de identidad, GhostFrame usa un tracker de dos etapas:

**Etapa 1 — IoU in-scene** *(rápida, para movimiento continuo)*
Si la cara de un frame se superpone con la de un frame reciente (IoU ≥ 0.40), se asigna automáticamente a la misma persona sin comparar embeddings.

**Etapa 2 — ArcFace re-id** *(para reentradas tras cortes o ausencias)*
Si la etapa 1 no encuentra coincidencia, se compara el embedding ArcFace (512 dimensiones) contra todos los perfiles conocidos. Si supera el umbral (0.65 por defecto), es la misma persona.

### Pipeline de renderizado sin degradación

```
ffmpeg (decode) → stdout [raw BGR24] → Python (censura) → stdin [raw BGR24] → ffmpeg (encode)
```

- Sin archivo temporal — los frames viajan en memoria
- Sin doble compresión entre los dos procesos
- Audio copiado bit a bit si el codec es compatible; si no, transcodifica a AAC
- Vídeo: intenta `h264_nvenc` → `h264_qsv` → `h264_amf` → `libx264 -crf 18`

### Estructura del proyecto

```
GhostFrame-Studio/
├── main.py                    # Punto de entrada + crash handler global
├── requirements.txt
├── INSTALACION.md
├── CHANGELOG.md
├── core/
│   ├── face_detector.py       # InsightFace: detección + embeddings ArcFace
│   ├── face_tracker.py        # Tracking IoU + re-id FAISS/scipy
│   ├── video_processor.py     # VideoReader + efectos de censura con feathering
│   ├── renderer.py            # Orquestación del renderizado
│   ├── ffmpeg_utils.py        # Detección de ffmpeg + pipe de renderizado
│   ├── session.py             # Cache .gfs en cache/sessions + export .gfscfg
│   └── settings.py            # Configuración persistente
└── ui/
    ├── main_window.py         # Ventana principal con threading seguro
    ├── person_card.py         # Tarjeta por persona + barra de apariciones
    ├── timeline_widget.py     # Scrubber de frames
    ├── preview_widget.py      # Vista previa con zoom visual
    ├── settings_dialog.py     # Configuración + recomendación hardware automática
    └── batch_dialog.py        # Proceso por lotes
```

---

## Referencias

- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [FFmpeg Builds para Windows](https://github.com/BtbN/FFmpeg-Builds/releases)
- [InsightFace — modelos de detección facial](https://github.com/deepinsight/insightface)

---

## Licencia

Distribuido bajo la licencia **MIT**. Ver [LICENSE](LICENSE) para más detalles.
