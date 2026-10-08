"""Command line interface for local batch processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import process_image


def main() -> None:
    parser = argparse.ArgumentParser(description="신분증 스캔 이미지를 로컬에서 보정합니다.")
    parser.add_argument("inputs", nargs="+", help="처리할 JPG/PNG 파일 또는 디렉터리")
    parser.add_argument("-o", "--output", default="output", help="결과 저장 폴더 (기본: output)")
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in sorted(path.iterdir())
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        else:
            files.append(path)
    if not files:
        parser.error("처리할 이미지가 없습니다.")

    for path in files:
        result = process_image(path, args.output)
        # Do not print OCR text or any content from the identity document.
        print(f"{path.name}: {result.status} — {result.message}")
        if result.output_path:
            print(f"  결과: {result.output_path}")
        if result.diagnostic_path:
            print(f"  진단: {result.diagnostic_path}")
        if result.quality_warning:
            print(f"  품질 안내: {result.quality_warning}")


if __name__ == "__main__":
    main()
