# Changelog — GhostFrame Studio

Todos los cambios notables de este proyecto se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [1.0.0] — 2026-05-27

### Añadido
- Lanzamiento público inicial
- Detección y tracking de caras con InsightFace (buffalo_l / buffalo_s)
- Efectos de censura: blur gaussiano, pixelado y caja negra
- Tracking de identidad en dos etapas: IoU in-scene + re-identificación ArcFace
- Preview en tiempo real con scrubbing frame a frame y zoom visual
- Audio en el preview mediante libmpv (opcional)
- Renderizado con FFmpeg: soporte NVENC, QSV, AMF y libx264 como fallback
- Sesiones guardables y reanudables (`.gfs`)
- Exportación e importación de configuración de censura (`.gfscfg`)
- Proceso por lotes para carpetas completas de vídeos
- Aceleración GPU: NVIDIA CUDA, AMD/Intel DirectML, CPU
- Instalador Windows con Python embebido y detección automática de GPU
- Configuración persistente en `~/.ghostframe_studio_settings.json`
- Manual integrado con atajos de teclado (`F1`)
- Recomendación automática de configuración según el hardware detectado
- Botón Cancelar para análisis y renderizado en curso
- Undo/Redo para operaciones de fusión y separación de personas (`Ctrl+Z` / `Ctrl+Y`)
- Corrección automática de vídeos HDR (bt709) para evitar tinte verde en Telegram y reproductores

### Técnico
- Pipeline de renderizado en memoria (sin archivo temporal intermedio)
- Preload de DLLs NVIDIA desde paquetes pip (sin CUDA Toolkit del sistema)
- Rutas adaptativas para instalaciones en Archivos de Programa (sin permisos de escritura)
- Checkpoint automático del análisis cada 500 frames
