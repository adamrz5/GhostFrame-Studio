# Arquitectura

| Ruta | Contenido |
| --- | --- |
| [main.py](../main.py) | Arranque de la aplicación. |
| [ghostframe/core/](../ghostframe/core/) | Detección, seguimiento, efectos, renderizado, sesiones y configuración. |
| [ghostframe/ui/](../ghostframe/ui/) | Ventanas, tarjetas por persona, vista previa y línea de tiempo. |
| [ghostframe/assets/](../ghostframe/assets/) | Icono y logotipo. |
| [tests/](../tests/) | Pruebas automatizadas. |

## Seguimiento facial

InsightFace detecta los rostros y genera embeddings ArcFace. El seguimiento intenta asociar primero cada detección con una reciente mediante el solapamiento de sus cajas (IoU). Si no encuentra una coincidencia, compara los embeddings con los perfiles conocidos para reconocer a quien reaparece tras un corte o una ausencia.

## Exportación

FFmpeg decodifica los fotogramas, Python aplica los efectos y FFmpeg codifica el resultado. Los fotogramas pasan entre procesos en memoria. El renderizador intenta usar NVENC, QSV o AMF y puede recurrir a libx264. Según el contenedor y los códecs de origen, puede copiar o convertir el audio y los subtítulos compatibles.

## Referencias

- [InsightFace](https://github.com/deepinsight/insightface)
- [ONNX Runtime con CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime con DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [FFmpeg para Windows](https://github.com/BtbN/FFmpeg-Builds/releases)
