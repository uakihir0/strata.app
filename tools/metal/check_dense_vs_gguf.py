#!/usr/bin/env python3
"""Run the full dense.bin/GGUF byte comparison for the seven catalogue packs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from dense_vs_gguf import DenseGGUFError, verify_model_packs  # noqa: E402


def main() -> int:
    models_root = os.environ.get("STRATA_TEST_MODELS")
    if not models_root:
        print("dense_vs_gguf: STRATA_TEST_MODELS must point to the catalogue model root", file=sys.stderr)
        return 2
    try:
        results = verify_model_packs(models_root)
    except (DenseGGUFError, OSError) as exc:
        print(f"dense_vs_gguf: {exc}", file=sys.stderr)
        return 1
    total_tensors = 0
    total_bytes = 0
    total_conversions = 0
    total_conversion_bytes = 0
    for pack_dir, report in results:
        total_tensors += report.tensors_checked
        total_bytes += report.bytes_checked
        total_conversions += report.conversions_checked
        total_conversion_bytes += report.conversion_bytes_checked
        print(
            f"{pack_dir}: {report.tensors_checked} unlisted tensors ({report.bytes_checked} bytes), "
            f"{report.conversions_checked} converted tensors ({report.conversion_bytes_checked} dense.bin bytes)"
        )
    print(
        f"dense_vs_gguf: {len(results)} packs, {total_tensors} unlisted float tensors "
        f"({total_bytes} bytes), {total_conversions} converted tensors "
        f"({total_conversion_bytes} dense.bin bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
