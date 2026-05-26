"""
AudioPlaybackManager — reloj maestro de audio para la preview de GhostFrame Studio.

Usa python-mpv con video=False para reproducir solo el audio del vídeo. El tiempo de
reproducción del audio (``time_pos``) dicta qué frame muestra OpenCV: nunca al revés.

Degradación elegante
────────────────────
Si python-mpv no está instalado, falta libmpv-2.dll, o el archivo no tiene pista de
audio, recae en ``InternalClock`` (``time.monotonic``). La imagen se muestra sin
audio, igual que antes.

Flujo en PlaybackWorker
────────────────────────
1. Crear AudioPlaybackManager(path, fps, speed, prefer_mpv=has_audio)
2. Llamar .play()  → mpv arranca PAUSADO (sin flash de audio) o reloj interno arranca
3. Esperar primer time_pos non-None → si tarda > MPV_START_TIMEOUT → fallback_to_clock()
4. Si start_pos > 0: seek() → confirmar (margen ≤ 0.15 s) → resume()
   Si start_pos ≈ 0: resume() directamente
5. Bucle: t = manager.time_pos → fi = int(t * fps) → leer frame → emitir
6. EOF con mpv: time_pos vuelve a None y se mantiene > 200 ms → break
7. Al terminar: manager.terminate()

Sincronización de frames
────────────────────────
- Gap ≤ SEQUENTIAL_THRESHOLD frames desde el último leído → grab() sin decodificar
  los intermedios (más rápido que seek() para saltos pequeños).
- Gap > SEQUENTIAL_THRESHOLD → cap.set(CAP_PROP_POS_FRAMES, fi) (seek OpenCV).
- Gap ≤ 0 → el audio aún no avanzó suficiente; sleep medio frame.
- Gap > MAX_LAG_FRAMES → el proceso va muy retrasado; salta frames para ponerse al día.
"""
from __future__ import annotations

import time
from typing import Optional


# ─── Reloj de sustitución (sin audio) ────────────────────────────────────────

class InternalClock:
    """
    Reloj monotónico para reproducción sin audio.
    Respeta velocidad de reproducción, pausa y reanudación.
    """

    def __init__(self, fps: float, speed: float = 1.0) -> None:
        self._fps        = max(0.1, fps)
        self._speed      = max(0.05, speed)
        self._start_mono: Optional[float] = None
        self._start_pos:  float           = 0.0
        self._paused_pos: float           = 0.0
        self._is_paused:  bool            = False

    # ── Control ───────────────────────────────────────────────────────────────

    def play(self, pos_sec: float = 0.0) -> None:
        self._start_pos  = pos_sec
        self._start_mono = time.monotonic()
        self._is_paused  = False

    def pause(self) -> None:
        if not self._is_paused:
            self._paused_pos = self._current_pos() or 0.0
            self._is_paused  = True

    def resume(self) -> None:
        if self._is_paused:
            self._start_pos  = self._paused_pos
            self._start_mono = time.monotonic()
            self._is_paused  = False

    def seek(self, pos_sec: float) -> None:
        self._start_pos  = pos_sec
        self._start_mono = time.monotonic()
        # Mantener estado de pausa — seek no despausa.
        if self._is_paused:
            self._paused_pos = pos_sec

    def stop(self) -> None:
        self._start_mono = None
        self._is_paused  = False

    # ── Consulta ──────────────────────────────────────────────────────────────

    @property
    def time_pos(self) -> Optional[float]:
        if self._is_paused:
            return self._paused_pos
        return self._current_pos()

    def _current_pos(self) -> Optional[float]:
        if self._start_mono is None:
            return None
        elapsed = (time.monotonic() - self._start_mono) * self._speed
        return self._start_pos + elapsed

    # ── Velocidad ─────────────────────────────────────────────────────────────

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        cur = self.time_pos
        if cur is not None and not self._is_paused:
            self._start_pos  = cur
            self._start_mono = time.monotonic()
        self._speed = max(0.05, value)


# ─── Gestor de audio mpv ──────────────────────────────────────────────────────

class AudioPlaybackManager:
    """
    Encapsula python-mpv como reloj maestro de audio.

    Usa ``video=False`` para que mpv decodifique solo el audio (6× menos CPU
    que ``vo=null`` que igualmente decodifica el vídeo).

    Si python-mpv o libmpv no están disponibles, ``using_mpv`` es False y todas
    las operaciones delegan en ``InternalClock``.

    Flujo de inicio sin flash de audio
    ───────────────────────────────────
    ``play()`` arranca mpv PAUSADO. El worker espera el primer ``time_pos`` non-None,
    luego hace el seek de inicio, confirma y llama ``resume()``. Así el audio nunca
    suena antes del punto de inicio.
    """

    def __init__(
        self,
        video_path: str,
        fps: float,
        speed: float = 1.0,
        prefer_mpv: bool = True,
    ) -> None:
        self._video_path = video_path
        self._fps        = max(0.1, fps)
        self._speed      = max(0.05, speed)
        self._player     = None
        self._clock:     Optional[InternalClock] = None
        self._using_mpv  = False

        if not prefer_mpv:
            self._clock = InternalClock(fps, speed)
            return

        try:
            import mpv  # type: ignore
            player = mpv.MPV(
                video                  = False,
                terminal               = False,
                input_default_bindings = False,
                osc                    = False,
                ytdl                   = False,
            )
            player.speed   = self._speed
            self._player   = player
            self._using_mpv = True
        except Exception as exc:
            print(f"[mpv] No disponible ({type(exc).__name__}: {exc}). Usando reloj interno sin audio.")
            self._clock = InternalClock(fps, speed)

    # ── Propiedades públicas ──────────────────────────────────────────────────

    @property
    def using_mpv(self) -> bool:
        """True si mpv está activo; False si se usa InternalClock."""
        return self._using_mpv

    @property
    def time_pos(self) -> Optional[float]:
        """
        Posición actual en segundos.
        Con mpv: None antes de arrancar y después de EOF.
        Con InternalClock: None solo antes de llamar play().
        """
        if self._using_mpv and self._player is not None:
            try:
                return self._player.time_pos
            except Exception as exc:
                print(f"[mpv] time_pos fallo: {exc}")
                return None
        if self._clock is not None:
            return self._clock.time_pos
        return None

    @property
    def speed(self) -> float:
        if self._using_mpv and self._player is not None:
            return float(self._player.speed)
        if self._clock is not None:
            return self._clock.speed
        return 1.0

    @speed.setter
    def speed(self, value: float) -> None:
        value = max(0.05, value)
        self._speed = value
        if self._using_mpv and self._player is not None:
            self._player.speed = value
        elif self._clock is not None:
            self._clock.speed = value

    # ── Control de reproducción ───────────────────────────────────────────────

    def play(self, pos_sec: float = 0.0) -> None:
        """
        Inicia la carga del archivo. mpv arranca PAUSADO para evitar flash de
        audio antes del seek de inicio. El worker llamará resume() cuando esté listo.
        InternalClock arranca directamente desde pos_sec.
        """
        if self._using_mpv and self._player is not None:
            # Arrancar pausado → sin audio hasta que el worker haga seek + resume.
            self._player.pause = True
            self._player.play(self._video_path)
        elif self._clock is not None:
            self._clock.play(pos_sec)

    def pause(self) -> None:
        if self._using_mpv and self._player is not None:
            self._player.pause = True
        elif self._clock is not None:
            self._clock.pause()

    def resume(self) -> None:
        """Despausa. Con mpv: el audio empieza a sonar. Con clock: reanuda reloj."""
        if self._using_mpv and self._player is not None:
            self._player.pause = False
        elif self._clock is not None:
            self._clock.resume()

    def seek(self, pos_sec: float) -> bool:
        """
        Salto absoluto (en segundos).
        Devuelve True si el seek se inició con éxito, False si mpv lanzó una excepción.
        Con InternalClock siempre devuelve True.
        """
        if self._using_mpv and self._player is not None:
            try:
                self._player.seek(pos_sec, reference="absolute", precision="exact")
                return True
            except Exception as exc:
                print(f"[mpv] seek({pos_sec:.3f}s) fallo: {exc}")
                return False
        elif self._clock is not None:
            self._clock.seek(pos_sec)
            return True
        return False

    def stop(self) -> None:
        if self._using_mpv and self._player is not None:
            try:
                self._player.stop()
            except Exception:
                pass
        elif self._clock is not None:
            self._clock.stop()

    def fallback_to_clock(self, pos_sec: float) -> None:
        """
        Abandona mpv y activa InternalClock desde pos_sec.
        Llamar cuando mpv arrancó pero nunca devolvió time_pos (sin pista de audio,
        formato no soportado, etc.). El video sigue reproduciéndose en silencio.
        """
        if self._player is not None:
            try:
                self._player.stop()
                self._player.terminate()
            except Exception:
                pass
            self._player = None
        self._using_mpv = False
        self._clock = InternalClock(self._fps, self._speed)
        self._clock.play(pos_sec)
        print("[mpv] Sin pista de audio o mpv no respondio. Reproduccion en silencio.")

    def terminate(self) -> None:
        """Libera todos los recursos. Llamar al finalizar PlaybackWorker."""
        if self._using_mpv and self._player is not None:
            try:
                self._player.stop()
                self._player.terminate()
            except Exception:
                pass
            self._player = None
        elif self._clock is not None:
            self._clock.stop()
            self._clock = None
        self._using_mpv = False
