"""Small Windows title-bar helpers for Qt windows."""
from __future__ import annotations

import sys


def _apply_dark_title_bar_now(widget) -> None:
    if sys.platform != "win32" or widget is None:
        return
    try:
        import ctypes

        hwnd = int(widget.winId())
        dwmapi = ctypes.windll.dwmapi

        enabled = ctypes.c_int(1)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE, legacy fallback.
            try:
                dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    attr,
                    ctypes.byref(enabled),
                    ctypes.sizeof(enabled),
                )
            except Exception:
                pass

        # COLORREF is 0x00BBGGRR.
        caption = ctypes.c_int(0x00231F1F)  # rgb(31, 31, 35)
        text = ctypes.c_int(0x00FFFFFF)
        border = ctypes.c_int(0x00423A3A)
        for attr, value in ((35, caption), (36, text), (34, border)):
            try:
                dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    attr,
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                )
            except Exception:
                pass
    except Exception:
        pass


def apply_dark_title_bar(widget) -> None:
    """Ask Windows DWM to use GhostFrame's dark title bar on this widget."""
    _apply_dark_title_bar_now(widget)
    try:
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda w=widget: _apply_dark_title_bar_now(w))
    except Exception:
        pass
