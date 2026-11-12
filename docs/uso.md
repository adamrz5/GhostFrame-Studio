# Guía de uso

## Editar y exportar un vídeo

### 1 · Abrir el vídeo
Arrastra el archivo o usa el botón **Abrir**. Formatos: MP4, MOV, MKV, AVI.

### 2 · Analizar las caras
Pulsa **Analizar**. El programa detecta todas las personas del vídeo y guarda los resultados en `cache/sessions/` para no repetir el análisis.

### 3 · Configurar la censura
En el panel derecho aparece una tarjeta por persona. Las tarjetas se ordenan por protagonismo (Persona 1 = la que más aparece).

Desde cada tarjeta puedes hacer estos ajustes.
- Activar o desactivar la censura
- Elegir entre **pixelado**, **desenfoque gaussiano** y **caja negra**
- Ajustar intensidad (1–10) y margen alrededor de la cara (0–50%)
- Limitar la censura a un tramo concreto del vídeo
- Ver una barra visual con los momentos exactos en que aparece esa persona

### 4 · Previsualizar
Arrastra el cursor de la línea de tiempo para ver la censura aplicada al instante.

| Atajo | Acción |
|-------|--------|
| `Espacio` | Reproducir / Pausar |
| Rueda del ratón | Zoom en el preview |
| Doble clic | Resetear zoom |
| `F1` | Manual y atajos de teclado |

### 5 · Renderizar
Pulsa **Renderizar**. El vídeo se guarda en `renders/`. El audio y los subtítulos se conservan cuando el formato de salida lo permite.

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
| CRF de vídeo | 18 | 0 = sin pérdidas · 18 = alta calidad · 28 = comprimido |
| Preset ffmpeg | fast | Más lento = archivo más pequeño |

## Solución de problemas

| Problema | Solución |
|----------|----------|
| El tracker une a dos personas distintas | Sube el umbral de similitud (> 0.65) y/o usa Fusionar/Dividir |
| El tracker separa a la misma persona en dos | Baja el umbral de similitud o usa Fusionar/Dividir |
| Sin audio en el vídeo exportado | El vídeo original no tenía audio, o FFmpeg no pudo convertirlo |
| Tinte verde/rosa en Telegram | Fuente HDR — el programa corrige los tags de color a bt709 automáticamente |
| El diálogo de Configuración se congeló | Espera 8 segundos a que se reactive automáticamente |

## Aviso legal

GhostFrame Studio está diseñado para proteger la identidad de personas en vídeos. El usuario es responsable de cualquier uso ilegal o manipulación sin consentimiento. El autor no asume responsabilidad por el uso indebido del software.
