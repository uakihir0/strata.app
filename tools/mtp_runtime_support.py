"""Validate and track the derived MTP runtime artifacts on macOS."""
from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
from mtp_fetch import PINNED_REVISION as MTP_PINNED_REVISION
from artifact_set_lock import pending_marker
from mtp_vocab import ArtifactSetLockError, install as install_mtp_vocabulary, validate_vocabulary


OLD_DRAFT_VOCABS = {
    "369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25"
}
DRAFT_VOCABS = {
    "cjk": "draft_vocab.bin",
    "en": "draft_vocab_en.bin",
    "cyrillic": "draft_vocab_cyrillic.bin",
}


MTP_RUNTIME_EXPERT_BYTES = 707_788_800
MTP_RUNTIME_DENSE_BYTES = 116_099_072
MTP_RUNTIME_DENSE_LAYOUT = {
    "fc_embedding.weight": ("q8_0", 2_560, 2_560),
    "fc_hidden.weight": ("q8_0", 2_560, 2_560),
    "self_attn.q_proj.weight": ("q8_0", 12_288, 2_560),
    "self_attn.k_proj.weight": ("q8_0", 512, 2_560),
    "self_attn.v_proj.weight": ("q8_0", 512, 2_560),
    "self_attn.o_proj.weight": ("q8_0", 2_560, 6_144),
    "self_attn.indexer.index_qk_proj.weight": ("q8_0", 640, 2_560),
    "mlp.shared_expert.gate_proj.weight": ("q8_0", 640, 2_560),
    "mlp.shared_expert.up_proj.weight": ("q8_0", 640, 2_560),
    "mlp.shared_expert.down_proj.weight": ("q8_0", 2_560, 640),
    "hyper_connection_mixer.hc_norm.weight": ("f32", 1, 10_240),
    "hyper_connection_mixer.input_mix_weight_down.weight": ("bf16", 320, 10_240),
    "hyper_connection_mixer.input_mix_weight_up.weight": ("bf16", 10_240, 320),
    "attn_hyper_connection.hc_norm.weight": ("f32", 1, 10_240),
    "attn_hyper_connection.input_mix_weight_down.weight": ("bf16", 320, 10_240),
    "attn_hyper_connection.input_mix_weight_up.weight": ("bf16", 10_240, 320),
    "attn_hyper_connection.block_inject_weight.weight": ("bf16", 4, 10_240),
    "mlp_hyper_connection.hc_norm.weight": ("f32", 1, 10_240),
    "mlp_hyper_connection.input_mix_weight_down.weight": ("bf16", 320, 10_240),
    "mlp_hyper_connection.input_mix_weight_up.weight": ("bf16", 10_240, 320),
    "mlp_hyper_connection.block_inject_weight.weight": ("bf16", 4, 10_240),
    "mlp.gate.weight": ("bf16", 512, 2_560),
    "mlp.shared_expert_gate.weight": ("bf16", 1, 2_560),
    "pre_fc_norm_embedding.weight": ("f32", 1, 2_560),
    "pre_fc_norm_hidden.weight": ("f32", 1, 10_240),
    "self_attn.q_norm.weight": ("f32", 1, 256),
    "self_attn.k_norm.weight": ("f32", 1, 256),
    "self_attn.indexer.q_layernorm.weight": ("f32", 1, 128),
    "self_attn.indexer.k_layernorm.weight": ("f32", 1, 128),
}
MTP_RUNTIME_DENSE_TENSORS = len(MTP_RUNTIME_DENSE_LAYOUT)
MTP_RUNTIME_SOURCE_FILE = "mtp-source.json"


def mtp_runtime_valid(runtime: Path) -> bool:
    """Reject truncated or malformed MTP output before a tool reuses it."""
    try:
        if (
            (runtime / "experts.bin").stat().st_size != MTP_RUNTIME_EXPERT_BYTES
            or (runtime / "dense.bin").stat().st_size != MTP_RUNTIME_DENSE_BYTES
        ):
            return False
        rows = (runtime / "dense.txt").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    if len(rows) != MTP_RUNTIME_DENSE_TENSORS:
        return False
    expected_offset = 0
    seen: set[str] = set()
    for line in rows:
        fields = line.split()
        if len(fields) != 6:
            return False
        name, kind = fields[:2]
        expected = MTP_RUNTIME_DENSE_LAYOUT.get(name)
        if name in seen or expected is None or kind != expected[0]:
            return False
        seen.add(name)
        try:
            row_count, columns, offset, byte_count = map(int, fields[2:])
        except ValueError:
            return False
        if (
            row_count <= 0
            or columns <= 0
            or offset != expected_offset
            or offset % 256
            or (kind, row_count, columns) != expected
        ):
            return False
        if kind == "q8_0":
            if columns % 32:
                return False
            expected_bytes = row_count * (columns // 32) * 34
        else:
            expected_bytes = row_count * columns * (4 if kind == "f32" else 2)
        if byte_count != expected_bytes or offset > MTP_RUNTIME_DENSE_BYTES - byte_count:
            return False
        expected_offset = (offset + byte_count + 255) // 256 * 256
    return seen == set(MTP_RUNTIME_DENSE_LAYOUT) and expected_offset == MTP_RUNTIME_DENSE_BYTES


def selected_mtp_revision() -> str:
    return os.environ.get("STRATA_MTP_REVISION") or MTP_PINNED_REVISION


def mtp_source_identity(mtp: Path) -> dict[str, str] | None:
    try:
        inventory = json.loads((mtp / "mtp-inventory.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(inventory, dict):
        return None
    requested_revision = inventory.get("requested_revision")
    repo = inventory.get("repo")
    resolved_revision = inventory.get("resolved_revision")
    prefix = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/"
    if (
        not isinstance(requested_revision, str)
        or not requested_revision
        or not isinstance(resolved_revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", resolved_revision)
        or repo != f"{prefix}{resolved_revision}/"
    ):
        return None
    return {
        "requested_revision": requested_revision,
        "repo": repo,
        "resolved_revision": resolved_revision,
    }


def mtp_runtime_source_matches(mtp: Path, runtime: Path) -> bool:
    identity = mtp_source_identity(mtp)
    if identity is None or identity["requested_revision"] != selected_mtp_revision():
        return False
    try:
        runtime_identity = json.loads(
            (runtime / MTP_RUNTIME_SOURCE_FILE).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError):
        return False
    return runtime_identity == identity


def write_mtp_runtime_source(mtp: Path, runtime: Path) -> None:
    identity = mtp_source_identity(mtp)
    if identity is None or identity["requested_revision"] != selected_mtp_revision():
        raise ValueError("MTP inventory does not match the selected checkpoint revision")
    runtime.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MTP_RUNTIME_SOURCE_FILE}.tmp.", dir=runtime
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(identity, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, runtime / MTP_RUNTIME_SOURCE_FILE)
        directory_fd = os.open(runtime, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def mtp_generation_needs_rebuild(mtp: Path, runtime: Path, corrupt: bool) -> bool:
    return (
        pending_marker(runtime).exists()
        or corrupt
        or not mtp_runtime_valid(runtime)
        or not mtp_runtime_source_matches(mtp, runtime)
    )


def saved_draft_vocab(config_path: Path) -> str | None:
    """Read a valid saved choice from a model configuration, if present."""
    try:
        choice = json.loads(config_path.read_text(encoding="utf-8-sig")).get("draft_vocab")
    except (OSError, UnicodeError, ValueError, AttributeError):
        return None
    return choice if choice in DRAFT_VOCABS else None


def refresh_draft_vocab(
    runtime: Path, choice: str = "cjk", inherited_locks: str | None = None
) -> bool:
    """Install a shipped vocabulary while preserving valid user-created choices."""
    filename = DRAFT_VOCABS.get(choice, DRAFT_VOCABS["cjk"])
    source = ROOT / "data" / filename
    destination = runtime / "draft_vocab.bin"
    if not source.is_file() or not runtime.is_dir():
        return False
    if inherited_locks is None and pending_marker(runtime).exists():
        raise ArtifactSetLockError(
            f"{pending_marker(runtime)}: an interrupted MTP runtime generation needs recovery"
        )

    new_hash = hashlib.sha256(validate_vocabulary(source)).hexdigest()
    if destination.exists():
        try:
            old_hash = hashlib.sha256(validate_vocabulary(destination)).hexdigest()
        except (OSError, ValueError):
            old_hash = None
        else:
            shipped = set(OLD_DRAFT_VOCABS)
            for shipped_name in DRAFT_VOCABS.values():
                shipped_path = ROOT / "data" / shipped_name
                if shipped_path.is_file():
                    shipped.add(hashlib.sha256(validate_vocabulary(shipped_path)).hexdigest())
            if old_hash not in shipped or old_hash == new_hash:
                return False

    install_mtp_vocabulary(source, runtime, inherited=inherited_locks)
    return True
