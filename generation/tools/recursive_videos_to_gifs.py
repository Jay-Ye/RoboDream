import argparse
import concurrent.futures
import os
import shutil
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursively convert videos to GIFs while preserving folder structure."
    )
    parser.add_argument("--input-root", required=True, help="Root folder containing videos.")
    parser.add_argument("--output-root", required=True, help="Root folder to save GIFs.")
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate. Default: 12")
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Output GIF width in pixels; height keeps aspect ratio. Default: 480",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite GIF files if they already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel ffmpeg workers. Default: min(8, CPU cores)",
    )
    return parser.parse_args()


def find_videos(input_root: Path):
    for path in input_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def convert_video_to_gif(video_path: Path, gif_path: Path, fps: int, width: int):
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:-1:flags=lanczos",
        str(gif_path),
    ]
    subprocess.run(cmd, check=True)


def convert_task(video_path: Path, gif_path: Path, fps: int, width: int):
    try:
        convert_video_to_gif(video_path, gif_path, fps, width)
        return video_path, gif_path, True, ""
    except subprocess.CalledProcessError as exc:
        return video_path, gif_path, False, str(exc)


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_root.is_dir():
        raise ValueError(f"Input root does not exist or is not a directory: {input_root}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not found in PATH.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    output_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    converted = 0
    skipped = 0
    failed = 0

    for video_path in find_videos(input_root):
        relative = video_path.relative_to(input_root)
        gif_path = output_root / relative.with_suffix(".gif")

        if gif_path.exists() and not args.overwrite:
            print(f"[skip] {gif_path}")
            skipped += 1
            continue

        jobs.append((video_path, gif_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(convert_task, video_path, gif_path, args.fps, args.width): (video_path, gif_path)
            for video_path, gif_path in jobs
        }

        for future in concurrent.futures.as_completed(future_to_job):
            video_path, gif_path, ok, error = future.result()
            if ok:
                print(f"[done] {video_path} -> {gif_path}")
                converted += 1
            else:
                print(f"[fail] {video_path} -> {gif_path}\n       {error}")
                failed += 1

    print("\nDone.")
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":
    main()
