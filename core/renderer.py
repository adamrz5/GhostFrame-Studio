"""
Orquestador de renderizado final.
Delega el trabajo al pipeline de pipe ffmpeg en ffmpeg_utils.py.
"""
from __future__ import annotations

import os
import uuid
from typing import Callable

from core import settings as cfg
from core.ffmpeg_utils import assert_ffmpeg, render_via_pipe
from core.video_processor import censure_roi_inplace


def build_process_fn(
    persons_config: list[dict],
    frame_data: dict[int, dict[int, dict]],
    total_frames: int,
) -> Callable:
    def _process(frame, frame_idx: int):
        # Collect all active censures first, then apply with a single frame copy.
        pending = []
        for item in persons_config:
            if not item.get("enabled", False):
                continue
            pid   = item["person_id"]
            start = item.get("start_frame", 0)
            end   = item.get("end_frame", -1)
            person_frames = frame_data.get(pid, {})
            actual_last = max(person_frames.keys(), default=-1)
            if actual_last < 0:
                continue
            if end == -1 or end > actual_last + 1:
                end = actual_last + 1
            if not (start <= frame_idx <= end):
                continue
            fi_data = person_frames.get(frame_idx)
            if fi_data is None:
                continue
            pending.append((
                fi_data["bbox"],
                item.get("effect",      "blur"),
                item.get("intensity",   5),
                item.get("padding_pct", 0.15),
            ))
        if not pending:
            return frame
        out = frame.copy()
        for bbox, effect, intensity, padding_pct in pending:
            censure_roi_inplace(out, bbox, effect, intensity, padding_pct)
        return out
    return _process


def render_video(
    input_path: str,
    output_path: str,
    persons_config: list[dict],
    frame_data: dict[int, dict[int, dict]],
    video_info: dict,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> str:
    ffmpeg_bin = assert_ffmpeg(cfg.get("ffmpeg_path") or None)

    total   = video_info["frame_count"]
    process = build_process_fn(persons_config, frame_data, total)
    root, ext = os.path.splitext(output_path)
    # Include PID so the temp name is unique and never collides with a user file.
    tmp_output = f"{root}.rendering_{os.getpid()}_{uuid.uuid4().hex}{ext or '.mp4'}"
    if os.path.abspath(tmp_output) == os.path.abspath(output_path):
        tmp_output = output_path + f".rendering_{os.getpid()}.mp4"

    try:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        render_via_pipe(
            ffmpeg_bin   = ffmpeg_bin,
            input_path   = input_path,
            output_path  = tmp_output,
            width        = video_info["width"],
            height       = video_info["height"],
            fps          = video_info["fps"],
            process_fn   = process,
            progress_cb  = progress_callback,
            total_frames = total,
            crf          = cfg.get("crf"),
            preset       = cfg.get("encode_preset"),
            is_vfr       = video_info.get("is_vfr", False),
            use_hw_encode= cfg.get("use_hw_encode"),
            cancel_cb    = cancel_callback,
            audio_codec  = video_info.get("codec_audio"),
            subtitle_codecs = video_info.get("subtitle_codecs") or [],
            color_space = video_info.get("color_space"),
            color_primaries = video_info.get("color_primaries"),
            color_transfer = video_info.get("color_transfer"),
            color_range = video_info.get("color_range"),
            warn_cb = warning_callback,
        )
        os.replace(tmp_output, output_path)
    except Exception:
        try:
            if os.path.exists(tmp_output):
                os.remove(tmp_output)
        except Exception:
            pass
        raise
    return output_path
