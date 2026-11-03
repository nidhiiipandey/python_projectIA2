"""Privacy-conscious classroom face counter and document scanner."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
LOG_FIELDS = ("date", "time", "source", "face_count")


def read_image(path: Path) -> np.ndarray:
    """Load a supported image or report a useful input error."""
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix or '(no extension)'}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    return detector


def detect_faces(image: np.ndarray, detector: cv2.CascadeClassifier) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return [tuple(map(int, face)) for face in faces]


def annotate_faces(
    image: np.ndarray,
    faces: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    annotated = image.copy()
    for x, y, width, height in faces:
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 200, 0), 2)
    cv2.putText(
        annotated,
        f"Faces: {len(faces)}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 200, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def log_attendance(log_path: Path, source: str, face_count: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    now = datetime.now()
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "source": source,
                "face_count": face_count,
            }
        )


def _output_path(value: str | None, default: Path) -> Path:
    return Path(value) if value else default


def _confirm_camera_use() -> bool:
    print(
        "Camera consent: the webcam will be used to count faces. "
        "Only timestamps, source, and counts are logged; face images are not stored "
        "unless you explicitly enable annotated output."
    )
    return input("Do you consent to webcam processing? [y/N] ").strip().lower() in {"y", "yes"}


def run_face_image(args: argparse.Namespace) -> int:
    source = Path(args.source)
    detector = load_face_detector()
    image = read_image(source)
    faces = detect_faces(image, detector)
    log_attendance(Path(args.log), str(source), len(faces))
    print(f"Detected {len(faces)} face(s) in {source}; count logged to {args.log}.")
    if args.store_annotated:
        output = _output_path(args.output, source.with_name(f"{source.stem}_annotated.jpg"))
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), annotate_faces(image, faces)):
            raise OSError(f"Could not save annotated image: {output}")
        print(f"Annotated image saved to {output}.")
    else:
        print("Annotated image not saved. Use --store-annotated to opt in.")
    return 0


def run_face_stream(args: argparse.Namespace, source: str | int) -> int:
    is_camera = isinstance(source, int)
    if is_camera and not _confirm_camera_use():
        print("Camera processing cancelled; no camera was opened.")
        return 0

    detector = load_face_detector()
    capture = cv2.VideoCapture(source)
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    last_log_time: float | None = None
    output = Path(args.output) if args.output else None
    try:
        if not capture.isOpened():
            description = f"camera {source}" if is_camera else str(source)
            raise RuntimeError(f"Could not open video source: {description}")

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            faces = detect_faces(frame, detector)
            annotated = annotate_faces(frame, faces)
            if is_camera:
                sample_time = time.monotonic()
            else:
                sample_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if sample_time < 0:
                    sample_time = time.monotonic()
            if last_log_time is None or sample_time - last_log_time >= args.interval:
                log_attendance(Path(args.log), str(source), len(faces))
                last_log_time = sample_time

            if args.store_annotated:
                if output is None:
                    suffix = ".mp4" if is_camera else Path(str(source)).suffix
                    output = Path(f"{Path(str(source)).stem or 'camera'}_annotated{suffix}")
                output.parent.mkdir(parents=True, exist_ok=True)
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0:
                        fps = 20.0
                    height, width = frame.shape[:2]
                    codec = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output), codec, fps, (width, height))
                    if not writer.isOpened():
                        raise OSError(f"Could not create annotated video: {output}")
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Smart Attendance - press q to stop", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if frame_count == 0:
            raise RuntimeError(f"No frames could be read from video source: {source}")
        print(f"Processed {frame_count} frame(s); attendance counts logged to {args.log}.")
        if output and args.store_annotated:
            print(f"Annotated video saved to {output}.")
        return 0
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(differences)]
    rect[3] = points[np.argmax(differences)]
    return rect


def _four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = _order_points(points)
    width = max(
        int(np.linalg.norm(bottom_right - bottom_left)),
        int(np.linalg.norm(top_right - top_left)),
    )
    height = max(
        int(np.linalg.norm(top_right - bottom_right)),
        int(np.linalg.norm(top_left - bottom_left)),
    )
    if width < 2 or height < 2:
        raise ValueError("Detected document contour is too small to scan.")
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"),
        destination,
    )
    return cv2.warpPerspective(image, transform, (width, height))


def scan_document(image: np.ndarray, find_contour: bool = True) -> np.ndarray:
    """Deskew a detected page when possible, then produce a thresholded scan."""
    working = image
    if find_contour:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 75, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * 0.1:
                break
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approximation) == 4:
                working = _four_point_transform(image, approximation.reshape(4, 2))
                break

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def run_document(args: argparse.Namespace) -> int:
    source = Path(args.source)
    image = read_image(source)
    result = scan_document(image, find_contour=not args.no_contour)
    output = _output_path(args.output, source.with_name(f"{source.stem}_scan.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result):
        raise OSError(f"Could not save scanned document: {output}")
    print(f"Scanned document saved to {output}.")
    return 0


def transform_image(
    image: np.ndarray,
    operation: str,
    width: int | None = None,
    height: int | None = None,
    crop: tuple[int, int, int, int] | None = None,
    angle: float = 0,
    flip: str = "horizontal",
) -> np.ndarray:
    if operation == "resize":
        if not width or not height or width < 1 or height < 1:
            raise ValueError("Resize requires positive --width and --height values.")
        return cv2.resize(image, (width, height))
    if operation == "crop":
        if crop is None:
            raise ValueError("Crop requires --crop X Y WIDTH HEIGHT.")
        x, y, crop_width, crop_height = crop
        image_height, image_width = image.shape[:2]
        if x < 0 or y < 0 or crop_width < 1 or crop_height < 1:
            raise ValueError("Crop coordinates must be non-negative with positive dimensions.")
        if x + crop_width > image_width or y + crop_height > image_height:
            raise ValueError("Crop rectangle extends outside the image.")
        return image[y : y + crop_height, x : x + crop_width].copy()
    if operation == "rotate":
        image_height, image_width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (image_width, image_height))
    if operation == "flip":
        flip_code = {"horizontal": 1, "vertical": 0, "both": -1}[flip]
        return cv2.flip(image, flip_code)
    if operation == "watermark":
        result = image.copy()
        date_text = datetime.now().strftime("%Y-%m-%d")
        cv2.putText(
            result,
            date_text,
            (12, max(28, result.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            date_text,
            (12, max(28, result.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return result
    raise ValueError(f"Unknown image operation: {operation}")


def run_utility(args: argparse.Namespace) -> int:
    image = read_image(Path(args.source))
    result = transform_image(
        image,
        args.operation,
        width=args.width,
        height=args.height,
        crop=tuple(args.crop) if args.crop else None,
        angle=args.angle,
        flip=args.flip,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result):
        raise OSError(f"Could not save transformed image: {output}")
    print(f"Saved {args.operation} image to {output}.")
    return 0


def run_batch(args: argparse.Namespace) -> int:
    source_dir = Path(args.input_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {source_dir}")
    images = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in {source_dir}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = random.Random(args.seed).sample(images, min(5, len(images)))
    print(f"Demo samples (seed {args.seed}): {', '.join(path.name for path in samples)}")

    processed = 0
    failed = 0
    for path in images:
        try:
            result = scan_document(read_image(path), find_contour=not args.no_contour)
            output = output_dir / f"{path.stem}_scan.png"
            if not cv2.imwrite(str(output), result):
                raise OSError(f"Could not save scanned image: {output}")
            processed += 1
        except (OSError, ValueError, cv2.error) as error:
            failed += 1
            print(f"Failed {path.name}: {error}", file=sys.stderr)
    print(f"Batch complete: {processed} processed, {failed} failed.")
    return 1 if failed else 0


def run_evaluation(args: argparse.Namespace) -> int:
    image_dir = Path(args.input_dir)
    labels_path = Path(args.labels)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Evaluation labels not found: {labels_path}")
    with labels_path.open(newline="", encoding="utf-8") as labels_file:
        rows = list(csv.DictReader(labels_file))
    if len(rows) != 10:
        raise ValueError(f"Evaluation requires exactly 10 labeled images; found {len(rows)}.")

    detector = load_face_detector()
    results: list[dict[str, str | int]] = []
    for row in rows:
        image_name = (row.get("image") or "").strip()
        try:
            actual = int(row["actual_faces"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid actual_faces value for {image_name or '(unnamed image)'}.") from error
        if not image_name or actual < 0:
            raise ValueError("Each evaluation row needs an image name and non-negative actual_faces.")
        faces = detect_faces(read_image(image_dir / image_name), detector)
        detected = len(faces)
        error_count = abs(actual - detected)
        results.append(
            {
                "image": image_name,
                "actual_faces": actual,
                "detected_faces": detected,
                "errors": error_count,
                "accuracy_percent": 100 if error_count == 0 else 0,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("image", "actual_faces", "detected_faces", "errors", "accuracy_percent")
    with output.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    exact_matches = sum(row["errors"] == 0 for row in results)
    accuracy = exact_matches / len(results) * 100
    print(f"Evaluation saved to {output}; exact-count accuracy: {accuracy:.1f}% ({exact_matches}/10).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count faces for attendance or turn document photos into clean scans."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    face = subparsers.add_parser("face", help="Detect faces in an image, video, or webcam.")
    face.add_argument("--source", required=True, help="Image/video path, 'webcam', or camera index.")
    face.add_argument("--log", default="attendance_log.csv")
    face.add_argument("--interval", type=float, default=5.0, help="Seconds between video attendance rows.")
    face.add_argument("--store-annotated", action="store_true", help="Opt in to saving imagery containing faces.")
    face.add_argument("--output", help="Annotated image/video path; inferred when omitted.")
    face.add_argument("--no-display", action="store_true", help="Do not open a preview window for video.")
    face.set_defaults(handler=run_face_image)

    document = subparsers.add_parser("document", help="Clean and threshold a document photo.")
    document.add_argument("--source", required=True)
    document.add_argument("--output")
    document.add_argument("--no-contour", action="store_true", help="Skip page-edge detection and perspective correction.")
    document.set_defaults(handler=run_document)

    utility = subparsers.add_parser("utility", help="Resize, crop, rotate, flip, or watermark an image.")
    utility.add_argument("--source", required=True)
    utility.add_argument("--output", required=True)
    utility.add_argument("--operation", required=True, choices=("resize", "crop", "rotate", "flip", "watermark"))
    utility.add_argument("--width", type=int)
    utility.add_argument("--height", type=int)
    utility.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    utility.add_argument("--angle", type=float, default=0.0)
    utility.add_argument("--flip", choices=("horizontal", "vertical", "both"), default="horizontal")
    utility.set_defaults(handler=run_utility)

    batch = subparsers.add_parser("batch", help="Scan every supported image in a folder.")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--output-dir", default="scanned_documents")
    batch.add_argument("--seed", type=int, default=42)
    batch.add_argument("--no-contour", action="store_true")
    batch.set_defaults(handler=run_batch)

    evaluation = subparsers.add_parser("evaluate", help="Evaluate face counts for exactly 10 labeled images.")
    evaluation.add_argument("--input-dir", required=True)
    evaluation.add_argument("--labels", required=True, help="CSV columns: image,actual_faces.")
    evaluation.add_argument("--output", default="evaluation_results.csv")
    evaluation.set_defaults(handler=run_evaluation)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "face":
        source = args.source.strip()
        if args.interval <= 0:
            parser.error("--interval must be greater than zero.")
        if source.lower() == "webcam":
            args.handler = run_face_stream
            return args.handler(args, 0)
        if source.isdigit():
            args.handler = run_face_stream
            return args.handler(args, int(source))
        path = Path(source)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return args.handler(args)
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path}")
            args.handler = run_face_stream
            return args.handler(args, str(path))
        raise ValueError(f"Unsupported face source or file format: {source}")
    return args.handler(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
