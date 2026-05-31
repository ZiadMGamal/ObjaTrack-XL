from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_URLS: dict[str, str] = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt",
    "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt",
    "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt",
}

MODEL_CHECKSUMS: dict[str, str] = {
    "yolov8n.pt": "",
    "yolov8s.pt": "",
    "yolov8m.pt": "",
    "yolov8l.pt": "",
    "yolov8x.pt": "",
}


class ModelDownloader:

    def __init__(self, model_dir: str = "models", timeout: float = 120.0) -> None:
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def get_model_path(self, model_name: str) -> Path:
        return self._model_dir / model_name

    def is_downloaded(self, model_name: str) -> bool:
        return self.get_model_path(model_name).exists()

    def download(self, model_name: str, force: bool = False) -> Path:
        target_path = self.get_model_path(model_name)

        if target_path.exists() and not force:
            logger.info("model_exists", model=model_name, path=str(target_path))
            return target_path

        url = MODEL_URLS.get(model_name)
        if not url:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_URLS.keys())}")

        logger.info("downloading_model", model=model_name, url=url)

        try:
            with httpx.stream("GET", url, timeout=self._timeout, follow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0

                temp_path = target_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            progress = (downloaded / total) * 100
                            if downloaded % (1024 * 1024) < 8192:
                                logger.info(
                                    "download_progress",
                                    model=model_name,
                                    progress=f"{progress:.1f}%",
                                    downloaded_mb=round(downloaded / (1024 * 1024), 1),
                                )

                shutil.move(str(temp_path), str(target_path))

        except httpx.HTTPError as e:
            logger.error("download_failed", model=model_name, error=str(e))
            raise

        if self._verify_checksum(model_name, target_path):
            logger.info("model_downloaded", model=model_name, path=str(target_path))
        else:
            logger.warning("checksum_skipped", model=model_name)

        return target_path

    def _verify_checksum(self, model_name: str, path: Path) -> bool:
        expected = MODEL_CHECKSUMS.get(model_name, "")
        if not expected:
            return False

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        actual = sha256.hexdigest()
        if actual != expected:
            logger.error(
                "checksum_mismatch",
                model=model_name,
                expected=expected[:16],
                actual=actual[:16],
            )
            path.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {model_name}")

        return True

    def ensure_model(self, model_name: str) -> Path:
        if self.is_downloaded(model_name):
            return self.get_model_path(model_name)
        return self.download(model_name)

    def list_models(self) -> list[dict[str, Any]]:
        models = []
        for path in self._model_dir.glob("*"):
            if path.is_file() and path.suffix in (".pt", ".onnx", ".engine", ".trt"):
                models.append({
                    "name": path.name,
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "format": path.suffix[1:],
                })
        return models

    def delete_model(self, model_name: str) -> bool:
        path = self.get_model_path(model_name)
        if path.exists():
            path.unlink()
            logger.info("model_deleted", model=model_name)
            return True
        return False
