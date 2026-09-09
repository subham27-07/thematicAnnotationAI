"""Run configuration shared by every backend."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path | None = None) -> None:
    """Read `.env` into the environment, without overriding a real shell value.

    Parsed by hand so the project has no hard dependency on python-dotenv, and
    so `export FOO=bar` works as written — that prefix is what people actually
    put in the file, and a bare `os.environ` read would silently miss the key.
    """
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_env_file()

DEFAULT_DATASET = PROJECT_ROOT / "Qual Analysis - accounts_tidy.csv"
DEFAULT_CODEBOOK = PROJECT_ROOT / "data" / "codebook.csv"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations-all-rounds.csv"
DEFAULT_ADJUDICATED = PROJECT_ROOT / "data" / "adjudicated-all-rounds.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Backend presets. `model` is overridable from the CLI / notebook.
BACKEND_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {"model": "gpt-5.1", "max_workers": 8},
    "ollama": {"model": "qwen3.5:latest", "max_workers": 2},
}


@dataclass
class RunConfig:
    """Everything that defines one annotation run.

    Two runs with the same `run_key` are interchangeable, so the on-disk cache
    is keyed by it and a re-run costs nothing.
    """

    backend: str = "ollama"
    model: str = ""
    prompt_variant: str = "codebook_only"

    # Generation controls. GPT-5.1 accepts reasoning effort "none"/"low"/
    # "medium"/"high"; Ollama models ignore it and use `think` instead.
    temperature: float = 0.0
    reasoning_effort: str = "low"
    think: bool = False
    max_output_tokens: int = 4096
    seed: int = 20260909

    # Few-shot settings (only used when prompt_variant == "few_shot").
    few_shot_k: int = 12
    few_shot_source: str = "adjudicated"  # adjudicated | union | intersection

    # Codebook rendering.
    include_codebook_examples: bool = True
    min_confidence: float = 0.0  # drop model codes below this confidence

    # Execution.
    max_workers: int = 0  # 0 -> backend default
    max_retries: int = 4
    retry_base_delay: float = 2.0
    request_timeout: float = 300.0
    use_cache: bool = True

    # Paths.
    dataset_path: Path = DEFAULT_DATASET
    codebook_path: Path = DEFAULT_CODEBOOK
    annotations_path: Path = DEFAULT_ANNOTATIONS
    adjudicated_path: Path = DEFAULT_ADJUDICATED
    output_dir: Path = DEFAULT_OUTPUT_DIR

    # Endpoint / auth.
    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", ""))

    # Free-text label that ends up in the output CSV, e.g. "test_split".
    run_label: str = ""

    def __post_init__(self) -> None:
        if self.backend not in BACKEND_DEFAULTS:
            raise ValueError(
                f"unknown backend {self.backend!r}; expected one of {sorted(BACKEND_DEFAULTS)}"
            )
        defaults = BACKEND_DEFAULTS[self.backend]
        if not self.model:
            self.model = defaults["model"]
        if not self.max_workers:
            self.max_workers = defaults["max_workers"]
        for attr in (
            "dataset_path",
            "codebook_path",
            "annotations_path",
            "adjudicated_path",
            "output_dir",
        ):
            setattr(self, attr, Path(getattr(self, attr)))

    @property
    def model_slug(self) -> str:
        """Filename-safe model identifier."""
        return self.model.replace(":", "-").replace("/", "-").replace(" ", "_")

    @property
    def run_key(self) -> str:
        """Stable identifier for cache files and output filenames."""
        parts = [self.backend, self.model_slug, self.prompt_variant]
        if self.prompt_variant == "few_shot":
            parts.append(f"k{self.few_shot_k}")
        if self.run_label:
            parts.append(self.run_label)
        return "__".join(parts)

    def prompt_fingerprint(self, system_prompt: str) -> str:
        """Hash of everything that changes the model's answer for a unit."""
        payload = "|".join(
            [
                self.backend,
                self.model,
                self.prompt_variant,
                f"{self.temperature}",
                self.reasoning_effort,
                f"{self.think}",
                f"{self.seed}",
                system_prompt,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("openai_api_key", None)  # never persist secrets
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}
