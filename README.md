<div align="center">
  <img src="ghostframe/assets/ghostframe-app-icon.png" alt="" width="96">
  <h1>GhostFrame Studio</h1>
  <p>Censura caras en vídeo desde Windows.</p>
</div>

**Español** · [English](README.en.md)

[![Última versión](https://img.shields.io/github/v/release/adamrz5/GhostFrame-Studio?label=versi%C3%B3n)](https://github.com/adamrz5/GhostFrame-Studio/releases/latest)
[![Windows 10 y 11 de 64 bits](https://img.shields.io/badge/Windows-10%20%2F%2011%20x64-0078D4)](#instalación)
[![Licencia MIT](https://img.shields.io/badge/licencia-MIT-blue)](LICENSE)

**[Download for Windows](https://github.com/adamrz5/GhostFrame-Studio/releases/latest)** · [Watch the demo](#demo) · [Install from source](docs/instalacion.md)

GhostFrame Studio detecta las caras de un vídeo en tu equipo. Tú eliges cuáles ocultar y con qué efecto. Si alguien sale de plano y vuelve a aparecer, el programa intenta seguir a esa persona con los mismos ajustes.

## Demo

https://github.com/user-attachments/assets/fe1f2fd0-ec49-4e6f-a274-bda56f3768ec

## Funciones

- Aplica pixelado, desenfoque o un recuadro negro a cada cara. Puedes ajustar la intensidad, ampliar el área que cubre el efecto o limitarlo a un tramo.
- Localiza en la línea de tiempo cuándo aparece cada persona. Si alguien sale en dos tarjetas, puedes fusionarlas.
- Previsualiza los cambios antes de exportar a MP4 con FFmpeg.
- Recupera un análisis guardado o procesa una carpeta de vídeos por lotes.

## Instalación

1. Descarga `GhostFrame-Setup-*.exe` desde [Releases](https://github.com/adamrz5/GhostFrame-Studio/releases/latest) y ejecútalo.
2. Abre GhostFrame Studio con conexión a internet para que descargue el modelo de detección facial. La descarga solo se hace la primera vez.

Funciona en **Windows 10 y 11 de 64 bits**. El análisis puede usar CUDA en tarjetas NVIDIA, DirectML en tarjetas compatibles con DirectX 12 o el procesador. La [guía de instalación](docs/instalacion.md) explica cómo arrancar desde el código con Python 3.12 y configurar FFmpeg.

## Tu primer vídeo

1. Arrastra un MP4, MOV, MKV o AVI a la ventana y pulsa **Analizar**.
2. Activa la censura en la tarjeta de cada persona que quieras ocultar.
3. Recorre la línea de tiempo y comprueba que el efecto cubre bien las caras, sobre todo cuando vuelven a entrar en plano.
4. Pulsa **Renderizar** y elige dónde guardar el MP4.

## Documentación y contribuciones

- [Guía de uso](docs/uso.md) · Atajos, ajustes y problemas frecuentes.
- [Arquitectura](docs/arquitectura.md) · Seguimiento facial, exportación y estructura del código.
- [Historial de cambios](CHANGELOG.md) · Novedades de cada versión.

Si algo falla, indica qué versión usas, cómo reproducir el fallo y qué mensaje aparece al [abrir un issue](https://github.com/adamrz5/GhostFrame-Studio/issues/new). También puedes enviar cambios mediante un [pull request](https://github.com/adamrz5/GhostFrame-Studio/pulls).

[Licencia MIT](LICENSE)
