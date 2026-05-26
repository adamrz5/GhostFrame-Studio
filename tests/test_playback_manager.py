from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from ui.playback_manager import AudioPlaybackManager, InternalClock


def test_audio_playback_manager_can_skip_mpv_for_silent_preview():
    audio = AudioPlaybackManager("missing.mp4", fps=25.0, speed=1.0, prefer_mpv=False)

    audio.play(2.0)
    pos = audio.time_pos
    audio.terminate()

    assert audio.using_mpv is False
    assert pos is not None
    assert 2.0 <= pos < 2.2


def test_audio_playback_manager_internal_clock_pause_seek_resume():
    audio = AudioPlaybackManager("missing.mp4", fps=25.0, speed=1.0, prefer_mpv=False)

    audio.play(1.0)
    audio.pause()
    paused = audio.time_pos
    time.sleep(0.02)
    still_paused = audio.time_pos
    audio.seek(5.0)
    seeked = audio.time_pos
    audio.resume()
    time.sleep(0.02)
    resumed = audio.time_pos
    audio.terminate()

    assert paused is not None
    assert still_paused == paused
    assert seeked == 5.0
    assert resumed is not None
    assert resumed > 5.0


def test_seek_internal_clock_returns_true():
    """AudioPlaybackManager.seek() returns True when backed by InternalClock."""
    audio = AudioPlaybackManager("missing.mp4", fps=25.0, speed=1.0, prefer_mpv=False)
    audio.play(0.0)
    result = audio.seek(3.5)
    pos = audio.time_pos
    audio.terminate()

    assert result is True
    assert pos is not None
    assert abs(pos - 3.5) < 0.05   # position should reflect the seek immediately


def test_seek_returns_false_when_mpv_throws():
    """AudioPlaybackManager.seek() returns False when mpv raises, without crashing."""
    audio = AudioPlaybackManager("missing.mp4", fps=25.0, speed=1.0, prefer_mpv=False)
    audio.play(0.0)

    # Simulate mpv being active but its seek() throwing (e.g., format not loaded)
    mock_player = MagicMock()
    mock_player.seek.side_effect = RuntimeError("seek failed")
    audio._player = mock_player
    audio._using_mpv = True
    audio._clock = None

    result = audio.seek(5.0)
    audio._player = None
    audio._using_mpv = False
    audio.terminate()

    assert result is False


def test_seek_returns_false_when_no_player_or_clock():
    """AudioPlaybackManager.seek() returns False when neither mpv nor clock is active."""
    audio = AudioPlaybackManager("missing.mp4", fps=25.0, speed=1.0, prefer_mpv=False)
    audio.terminate()  # clears _clock
    audio._using_mpv = False
    audio._player = None
    audio._clock = None

    result = audio.seek(2.0)
    assert result is False


def test_fallback_to_clock_after_seek_failure():
    """After fallback_to_clock(), time_pos advances from the given position."""
    audio = AudioPlaybackManager("missing.mp4", fps=25.0, speed=1.0, prefer_mpv=False)
    audio._player = MagicMock()
    audio._player.stop = MagicMock()
    audio._player.terminate = MagicMock()
    audio._using_mpv = True
    audio._clock = None

    audio.fallback_to_clock(10.0)
    pos = audio.time_pos
    time.sleep(0.02)
    pos2 = audio.time_pos
    audio.terminate()

    assert audio.using_mpv is False
    assert pos is not None
    assert pos >= 10.0
    assert pos2 is not None and pos2 >= pos   # clock advances after fallback
