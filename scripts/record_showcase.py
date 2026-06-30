#!/usr/bin/env python3
"""Render a terminal-style MP4/GIF of `engram showcase run` without a screen recorder."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "assets" / "showcase"

WIDTH = 1080
HEIGHT = 640
FPS = 24
PACE = {
    "intro_hold": 1.0,
    "typing_char": 0.07,
    "post_command": 1.4,
    "beat_header": 1.1,
    "beat_gap": 0.7,
    "highlight": 0.75,
    "body": 0.55,
    "blank": 0.1,
    "summary": 4.5,
}
FONT_SIZE = 15
LINE_HEIGHT = 22
PADDING_X = 28
PADDING_Y = 58
MAX_COLS = 96

BG = (12, 12, 20)
PANEL = (26, 26, 42)
BORDER = (45, 45, 74)
TEXT = (226, 232, 240)
DIM = (100, 116, 139)
CMD = (165, 180, 252)
PASS = (52, 211, 153)
FAIL = (248, 113, 113)
HIGHLIGHT = (148, 163, 184)
PROMPT = (129, 140, 248)

DOTS = [(239, 68, 68), (245, 158, 11), (34, 197, 94)]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _run_showcase() -> str:
    result = subprocess.run(
        ["uv", "run", "engram", "showcase", "run"],
        cwd=SERVER_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() + "\n"


def _wrap_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        if len(raw) <= MAX_COLS:
            lines.append(raw)
            continue
        wrapped = textwrap.wrap(
            raw,
            width=MAX_COLS,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def _line_color(line: str, *, typing_command: bool) -> tuple[int, int, int]:
    if typing_command:
        return CMD
    if line.startswith("$ "):
        return CMD
    if "[PASS]" in line:
        return PASS
    if "[FAIL]" in line:
        return FAIL
    if line.startswith("  User:") or line.startswith("  Action:") or line.startswith("  Story:"):
        return DIM
    if line.startswith("  Recall highlights:") or line.startswith("    - "):
        return HIGHLIGHT
    if line.startswith("  Matched:") or line.startswith("  Suggested reply:"):
        return TEXT
    if line.startswith("Summary:"):
        return TEXT
    if line.startswith("Beat "):
        return TEXT
    if line.startswith("Engram showcase"):
        return TEXT
    return TEXT


def _max_visible_lines() -> int:
    return max(8, (HEIGHT - PADDING_Y - 24) // LINE_HEIGHT)


def _window_lines(lines: list[str], *, scroll_to_end: bool) -> list[str]:
    max_lines = _max_visible_lines()
    if len(lines) <= max_lines:
        return lines
    if scroll_to_end:
        return lines[-max_lines:]
    return lines[-max_lines:]


def _draw_frame(
    *,
    command: str,
    visible_lines: list[str],
    cursor_on_command: bool,
    cursor_visible: bool,
    scroll_to_end: bool = False,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    font = _load_font(FONT_SIZE)

    draw.rounded_rectangle((12, 12, WIDTH - 12, HEIGHT - 12), radius=14, fill=PANEL, outline=BORDER)
    draw.rectangle((12, 12, WIDTH - 12, 46), fill=(22, 22, 36))
    for index, color in enumerate(DOTS):
        x = 30 + index * 18
        draw.ellipse((x, 24, x + 10, 34), fill=color)
    draw.text((78, 22), "server — zsh — engram showcase run", font=font, fill=DIM)

    y = PADDING_Y
    prompt = "$ "
    command_line = prompt + command
    draw.text((PADDING_X, y), prompt, font=font, fill=PROMPT)
    draw.text((PADDING_X + font.getlength(prompt), y), command, font=font, fill=CMD)
    if cursor_on_command and cursor_visible:
        cursor_x = PADDING_X + font.getlength(command_line)
        draw.rectangle((cursor_x, y, cursor_x + 8, y + LINE_HEIGHT - 4), fill=CMD)
    y += LINE_HEIGHT + 8

    for line in _window_lines(visible_lines, scroll_to_end=scroll_to_end):
        color = _line_color(line, typing_command=False)
        draw.text((PADDING_X, y), line, font=font, fill=color)
        y += LINE_HEIGHT

    return image


def _save_frames(frames: list[Image.Image], frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(frame_dir / f"frame_{index:04d}.png")


def _encode_video(frame_dir: Path, out_mp4: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode showcase-demo.mp4")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ],
        check=True,
        capture_output=True,
    )


def _encode_gif(frame_dir: Path, out_gif: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode showcase-demo.gif")

    palette = out_gif.with_suffix(".palette.png")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-vf",
            "palettegen=stats_mode=diff",
            str(palette),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse=dither=bayer:bayer_scale=3",
            str(out_gif),
        ],
        check=True,
        capture_output=True,
    )
    palette.unlink(missing_ok=True)


def _seconds_to_frames(seconds: float, *, fps: int) -> int:
    return max(1, round(seconds * fps))


def _line_hold_seconds(line: str, *, previous_line: str | None) -> float:
    if line.startswith("Beat "):
        gap = PACE["beat_gap"] if previous_line and previous_line.startswith("  Suggested reply:") else 0.0
        return PACE["beat_header"] + gap
    if line.startswith("Summary:"):
        return PACE["summary"]
    if line.startswith("    - "):
        return PACE["highlight"]
    if line.startswith("  Recall highlights:"):
        return PACE["body"] + 0.25
    if not line.strip():
        return PACE["blank"]
    if line.startswith("  "):
        return PACE["body"]
    return PACE["body"] + 0.2


def build_frames(output_text: str, *, fps: int = FPS) -> list[Image.Image]:
    command = "cd server && uv run engram showcase run"
    lines = _wrap_lines(output_text)
    frames: list[Image.Image] = []

    def append(frame: Image.Image, repeats: int) -> None:
        frames.extend([frame.copy() for _ in range(repeats)])

    typed = ""
    append(
        _draw_frame(command=typed, visible_lines=[], cursor_on_command=True, cursor_visible=False),
        _seconds_to_frames(PACE["intro_hold"], fps=fps),
    )

    for char in command:
        typed += char
        char_frames = _seconds_to_frames(PACE["typing_char"], fps=fps)
        for tick in range(char_frames):
            cursor_visible = tick % 2 == 0
            append(
                _draw_frame(
                    command=typed,
                    visible_lines=[],
                    cursor_on_command=True,
                    cursor_visible=cursor_visible,
                ),
                1,
            )

    append(
        _draw_frame(command=command, visible_lines=[], cursor_on_command=False, cursor_visible=False),
        _seconds_to_frames(PACE["post_command"], fps=fps),
    )

    visible: list[str] = []
    previous_line: str | None = None
    for line in lines:
        visible.append(line)
        scroll_to_end = line.startswith("Beat 3/") or line.startswith("Summary:")
        repeats = _seconds_to_frames(_line_hold_seconds(line, previous_line=previous_line), fps=fps)
        append(
            _draw_frame(
                command=command,
                visible_lines=visible.copy(),
                cursor_on_command=False,
                cursor_visible=False,
                scroll_to_end=scroll_to_end,
            ),
            repeats,
        )
        previous_line = line

    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--pace",
        choices=["slow", "normal", "fast"],
        default="slow",
        help="Playback pacing for line reveals (default: slow)",
    )
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    global PACE
    if args.pace == "normal":
        PACE = {key: value * 0.65 for key, value in PACE.items()}
    elif args.pace == "fast":
        PACE = {key: value * 0.4 for key, value in PACE.items()}

    output_text = _run_showcase()
    frames = build_frames(output_text, fps=args.fps)
    frame_dir = args.out_dir / ".frames"
    _save_frames(frames, frame_dir)

    mp4_path = args.out_dir / "showcase-demo.mp4"
    gif_path = args.out_dir / "showcase-demo.gif"
    _encode_video(frame_dir, mp4_path, args.fps)
    _encode_gif(frame_dir, gif_path, args.fps)

    if not args.keep_frames:
        shutil.rmtree(frame_dir)

    print(f"Wrote {mp4_path}")
    print(f"Wrote {gif_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())