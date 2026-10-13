#!/usr/bin/env python3
"""Build the frozen prompt sets of docs/mac/09-testing-and-validation.md, section 2.

    python3 tools/metal/make_prompts.py --pack <pack folder> --upstream <Strata checkout>
        [--llama-cpp <llama.cpp checkout>] [--src tests/metal/prompts/src] [--out tests/metal/prompts] [--check]
    python3 tools/metal/make_prompts.py --self-test

Inputs: the pack's tokenizer/ folder (vocab.json, merges.txt, token_type.json, tokenizer.json,
chat_template.jinja), the project's own sources in tests/metal/prompts/src/, and two upstream sources that
are read with git from checkouts at pinned commits: the repository files of the ca-* sets (Niko1221/Strata at
1678de33) and the two documents of tf-doc (llama.cpp at 3cf03257). Upstream text is never copied into this
repository.

Outputs: <set>.ids, <set>.txt and MANIFEST.json. --check rebuilds every set into a temporary folder and fails
when a manifest entry, an ids hash recomputed from the .ids file, or a .txt file differs from the rebuild.
--self-test needs no pack or upstream: it builds a small set from a synthetic tokenizer, and checks that a changed
source byte and a changed ids byte are both detected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jinja2

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from serve.frontend import ChatTemplate  # noqa: E402
from strata_tokenizer import BYTE_TO_UNICODE, Tokenizer  # noqa: E402

BUILDER_VERSION = 1
MANIFEST_SCHEMA = 1
UPSTREAM_COMMIT = "1678de33d0e0711bc414ad992b640e1a37dd814"
LLAMA_CPP_COMMIT = "3cf03257"  # abbreviated; git resolves it and the full id must start with this
CA_ROOTS = ("src", "include", "serve", "tools")
CA_EXTENSIONS = (".cpp", ".hpp", ".h", ".cu", ".cuh", ".py")
TF_DOC_PATHS = ("docs/build.md", "docs/function-calling.md")
PLACEHOLDER = "<<<STRATA_FILES>>>"
CUT_TRIES = 16
TF_TOKENS = 1024
TF_CODE_TOKENS = 512
TF_CHAT_TURNS = 20
G1_IDS = tuple(f"g1-{n:02d}" for n in range(1, 11))
SRC_DIR = ROOT / "tests" / "metal" / "prompts" / "src"
OUT_DIR = ROOT / "tests" / "metal" / "prompts"
SRC_FILES = {"code": "code-agent.json", "tf_doc": "tf-doc.json", "tf_chat": "tf-chat.json", "g1": "g1.json"}
TOKENIZER_FILES = ("vocab.json", "merges.txt", "chat_template.jinja", "token_type.json", "tokenizer.json")
SET_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
SET_FILE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\.(?:ids|txt)\Z")
CANONICAL_INT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
ROLES = ("system", "user", "assistant")


class MakePromptsError(Exception):
    """A failure with a message for the user. The CLI prints it and exits with status 1."""


@dataclass(frozen=True)
class SetSpec:
    name: str
    kind: str       # ca | tf-code | tf-doc | tf-chat | g1
    tokens: int     # the target length; the whole prompt for g1 (0: no target)
    context: int


SETS: tuple[SetSpec, ...] = (
    SetSpec("ca-1k", "ca", 1024, 8192),
    SetSpec("ca-4k", "ca", 4096, 8192),
    SetSpec("ca-32k", "ca", 32256, 32768),
    SetSpec("ca-64k", "ca", 65024, 65536),
    SetSpec("ca-128k", "ca", 130560, 131072),
    SetSpec("ca-262k", "ca", 261632, 262144),
    SetSpec("tf-code", "tf-code", TF_CODE_TOKENS, 8192),
    SetSpec("tf-doc", "tf-doc", TF_TOKENS, 8192),
    SetSpec("tf-chat", "tf-chat", TF_TOKENS, 8192),
    *(SetSpec(name, "g1", 0, 8192) for name in G1_IDS),
)
SELF_TEST_NAMES = frozenset({"ca-4k", "tf-code", "tf-doc", "tf-chat", *G1_IDS})


@dataclass(frozen=True)
class Pack:
    tokenizer: Tokenizer
    template: ChatTemplate
    sha256: str       # vocab.json, merges.txt, chat_template.jinja, concatenated in that order
    aux_sha256: str   # token_type.json, tokenizer.json, concatenated in that order


@dataclass(frozen=True)
class Sources:
    upstream_commit: str
    ca_files: tuple[tuple[str, bytes], ...]
    llama_commit: str
    doc_files: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True)
class BuiltSet:
    name: str
    ids: list[int]
    context: int
    source: str


# ------------------------------------------------------------------ the pack's tokenizer/ folder
def load_pack(pack: Path) -> Pack:
    tdir = pack / "tokenizer"
    missing = [name for name in TOKENIZER_FILES if not (tdir / name).is_file()]
    if missing:
        raise MakePromptsError(f"make_prompts: {tdir} lacks {', '.join(missing)}; the sets need the pack's tokenizer/ folder")
    raw = {name: (tdir / name).read_bytes() for name in TOKENIZER_FILES}
    try:
        vocab = json.loads(raw["vocab.json"].decode("utf-8"))
        token_types = json.loads(raw["token_type.json"].decode("utf-8"))
        cfg = json.loads(raw["tokenizer.json"].decode("utf-8"))
        merges_text = raw["merges.txt"].decode("utf-8")
        template = ChatTemplate(tdir / "chat_template.jinja")
    except (UnicodeDecodeError, ValueError, OSError, jinja2.TemplateError) as exc:
        raise MakePromptsError(f"make_prompts: {tdir}: {exc}") from exc
    if not isinstance(vocab, dict) or not vocab:
        raise MakePromptsError(f"make_prompts: {tdir}/vocab.json must map token strings to ids")
    tokens: list[str | None] = [None] * len(vocab)
    for token, index in vocab.items():
        if type(index) is not int or not 0 <= index < len(vocab) or tokens[index] is not None:
            raise MakePromptsError(f"make_prompts: {tdir}/vocab.json: id {index!r} is not unique within 0..{len(vocab) - 1}")
        tokens[index] = token
    if not isinstance(token_types, list) or len(token_types) != len(tokens):
        raise MakePromptsError(f"make_prompts: {tdir}/token_type.json must hold one type per token ({len(tokens)})")
    pre = cfg.get("pre")
    if pre != "qwen35":
        raise MakePromptsError(f"make_prompts: pre-tokenizer {pre!r} is not transcribed; only qwen35 is")
    special = cfg.get("special_ids", {})
    if not isinstance(special, dict):
        raise MakePromptsError(f"make_prompts: {tdir}/tokenizer.json: special_ids must be an object")
    merges = merges_text.split("\n")
    if merges and merges[-1] == "":
        merges.pop()
    try:
        tokenizer = Tokenizer(tokens, merges, token_types, pre, special)
    except ValueError as exc:
        raise MakePromptsError(f"make_prompts: {tdir}: {exc}") from exc
    return Pack(
        tokenizer=tokenizer,
        template=template,
        sha256=_digest([raw["vocab.json"], raw["merges.txt"], raw["chat_template.jinja"]]),
        aux_sha256=_digest([raw["token_type.json"], raw["tokenizer.json"]]),
    )


def _digest(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


# ------------------------------------------------------------------ upstream sources, read with git
def _git(repo: Path, *args: str) -> bytes:
    try:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise MakePromptsError("make_prompts: git is needed to read the upstream checkouts") from exc
    if done.returncode:
        detail = done.stderr.decode("utf-8", "replace").strip()
        raise MakePromptsError(f"make_prompts: git {' '.join(args)} failed in {repo}: {detail}")
    return done.stdout


def _commit(repo: Path, rev: str, label: str) -> str:
    if not repo.is_dir():
        raise MakePromptsError(f"make_prompts: {label} checkout not found at {repo}; clone it and pass its path")
    try:
        return _git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").decode("ascii").strip()
    except (MakePromptsError, UnicodeDecodeError) as exc:
        raise MakePromptsError(f"make_prompts: the {label} checkout at {repo} has no commit {rev}") from exc


def read_sources(upstream: Path, llama_cpp: Path) -> Sources:
    ucommit = _commit(upstream, UPSTREAM_COMMIT, "upstream Strata")
    if ucommit != UPSTREAM_COMMIT:
        raise MakePromptsError(f"make_prompts: upstream resolved to {ucommit}, expected {UPSTREAM_COMMIT}")
    listing = _git(upstream, "ls-tree", "-r", "-z", ucommit, "--", *CA_ROOTS)
    entries: list[tuple[str, str]] = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        meta, sep, raw_path = record.partition(b"\t")
        if not sep:
            raise MakePromptsError("make_prompts: unexpected git ls-tree output")
        mode, kind, oid = meta.decode("ascii").split(" ")
        path = raw_path.decode("utf-8")
        if not path.endswith(CA_EXTENSIONS):
            continue
        if kind != "blob" or mode not in ("100644", "100755"):
            raise MakePromptsError(f"make_prompts: {path} at {ucommit} is {mode} {kind}, not a regular file")
        entries.append((path, oid))
    if not entries:
        raise MakePromptsError(f"make_prompts: no source files under {', '.join(CA_ROOTS)} at {ucommit}")
    entries.sort()
    ca_files = tuple((path, _git(upstream, "cat-file", "blob", oid)) for path, oid in entries)

    lcommit = _commit(llama_cpp, LLAMA_CPP_COMMIT, "llama.cpp")
    if not lcommit.startswith(LLAMA_CPP_COMMIT):
        raise MakePromptsError(f"make_prompts: llama.cpp resolved to {lcommit}, expected {LLAMA_CPP_COMMIT}")
    doc_files = tuple((path, _git(llama_cpp, "show", f"{lcommit}:{path}")) for path in TF_DOC_PATHS)
    return Sources(upstream_commit=ucommit, ca_files=ca_files, llama_commit=lcommit, doc_files=doc_files)


def files_text(files: tuple[tuple[str, bytes], ...]) -> str:
    """Each file under a `=== <path> ===` line, with a line end after its text."""
    blocks = []
    for path, data in files:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MakePromptsError(f"make_prompts: {path} is not UTF-8 text") from exc
        if not text.endswith("\n"):
            text += "\n"
        blocks.append(f"=== {path} ===\n{text}")
    return "".join(blocks)


# ------------------------------------------------------------------ the project's own sources
def load_src(src: Path) -> dict:
    loaded = {}
    for key, name in SRC_FILES.items():
        path = src / name
        if not path.is_file():
            raise MakePromptsError(f"make_prompts: missing source {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MakePromptsError(f"make_prompts: {path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema") != 1:
            raise MakePromptsError(f"make_prompts: {path} must be a JSON object with schema 1")
        loaded[key] = value
    _check_code(loaded["code"])
    _check_tf_doc(loaded["tf_doc"])
    _check_tf_chat(loaded["tf_chat"])
    _check_g1(loaded["g1"])
    return loaded


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise MakePromptsError(f"make_prompts: {message}")


def _check_message(message, where: str) -> None:
    _need(isinstance(message, dict) and message.get("role") in ROLES and isinstance(message.get("content"), str),
          f"{where}: each message needs a role (system, user or assistant) and string content")


def _check_code(code: dict) -> None:
    _need(isinstance(code.get("system"), str) and code["system"], "code-agent.json needs a system prompt")
    _need(isinstance(code.get("request"), str) and code["request"], "code-agent.json needs a request")
    _need(isinstance(code.get("tools"), list) and code["tools"] and all(isinstance(t, dict) for t in code["tools"]),
          "code-agent.json needs a non-empty list of tool definitions")


def _check_tf_doc(tf_doc: dict) -> None:
    _need(isinstance(tf_doc.get("system"), str) and tf_doc["system"], "tf-doc.json needs a system prompt")
    _need(isinstance(tf_doc.get("request"), str) and tf_doc["request"], "tf-doc.json needs a request")


def _check_tf_chat(tf_chat: dict) -> None:
    messages = tf_chat.get("messages")
    _need(isinstance(messages, list) and messages, "tf-chat.json needs a messages list")
    for message in messages:
        _check_message(message, "tf-chat.json")
    turns = [m["role"] for m in messages if m["role"] != "system"]
    _need(turns.count("user") == TF_CHAT_TURNS and turns[0] == "user" and turns[-1] == "assistant"
          and all(turns[i] != turns[i + 1] for i in range(len(turns) - 1)),
          f"tf-chat.json must alternate user and assistant messages over {TF_CHAT_TURNS} user turns")


def _check_g1(g1: dict) -> None:
    prompts = g1.get("prompts")
    _need(isinstance(prompts, list) and [p.get("id") for p in prompts] == list(G1_IDS),
          f"g1.json must list the prompts {G1_IDS[0]} .. {G1_IDS[-1]} in order")
    for prompt in prompts:
        where = prompt["id"]
        _need(isinstance(prompt.get("thinking"), bool), f"g1.json: {where} needs thinking true or false")
        _need(isinstance(prompt.get("expect"), str), f"g1.json: {where} needs an expect string")
        messages = prompt.get("messages")
        _need(isinstance(messages, list) and messages and messages[-1].get("role") == "user",
              f"g1.json: {where} needs messages ending with a user message")
        for message in messages:
            _check_message(message, f"g1.json {where}")
        _need(prompt.get("tools") is None or (isinstance(prompt["tools"], list) and prompt["tools"]),
              f"g1.json: {where} tools must be a non-empty list when present")


# ------------------------------------------------------------------ rendering and cutting
def render(pack: Pack, messages: list[dict], tools: list[dict] | None, thinking: bool) -> str:
    return pack.template.render(messages=messages, tools=tools or None, add_generation_prompt=True,
                                enable_thinking=thinking)


def _placeholder_halves(pack: Pack, system: str, request: str, tools, name: str) -> tuple[str, str]:
    """Render a chat whose user message is the file placeholder followed by the request; split at the placeholder."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": PLACEHOLDER + "\n" + request}]
    rendered = render(pack, messages, tools, thinking=False)
    count = rendered.count(PLACEHOLDER)
    if count != 1:
        raise MakePromptsError(f"make_prompts: {name}: the template placed the file text {count} times, not once")
    head, _, tail = rendered.partition(PLACEHOLDER)
    return head, tail


def decode_strict(tok: Tokenizer, ids: list[int]) -> str | None:
    try:
        return tok.decode(ids, errors="strict")
    except UnicodeDecodeError:
        return None


def round_trips(tok: Tokenizer, ids: list[int]) -> bool:
    text = decode_strict(tok, ids)
    return text is not None and tok.encode(text, parse_special=True) == ids


def cut_files(name: str, target: int, head: list[int], files: list[int], tail: list[int], round_trip) -> list[int]:
    """Keep `target - len(head) - len(tail)` file tokens, then step back one token at a time, at most CUT_TRIES
    times, until `round_trip(ids)` holds: the cut must not split a merge or a UTF-8 character."""
    keep = target - len(head) - len(tail)
    if keep < 0:
        raise MakePromptsError(f"make_prompts: {name}: the prompt around the files is longer than {target}")
    if keep > len(files):
        raise MakePromptsError(f"make_prompts: {name}: the sources hold {len(files)} file tokens, the set needs {keep}")
    for step in range(CUT_TRIES + 1):
        cut = keep - step
        if cut < 0:
            break
        ids = head + files[:cut] + tail
        if round_trip(ids):
            return ids
    raise MakePromptsError(f"make_prompts: {name}: no cut within {CUT_TRIES} tokens of {target} round-trips")


def canonical_ids(ids: list[int]) -> str:
    return ",".join(str(i) for i in ids)


def ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(canonical_ids(ids).encode("ascii")).hexdigest()


def _source_of(spec: SetSpec, sources: Sources, entry_id: str | None = None) -> str:
    if spec.kind == "ca":
        return f"tests/metal/prompts/src/code-agent.json; upstream {sources.upstream_commit} {'/'.join(CA_ROOTS)}"
    if spec.kind == "tf-code":
        return f"first {TF_CODE_TOKENS} ids of ca-4k"
    if spec.kind == "tf-doc":
        return f"tests/metal/prompts/src/tf-doc.json; llama.cpp {sources.llama_commit} {', '.join(TF_DOC_PATHS)}"
    if spec.kind == "tf-chat":
        return "tests/metal/prompts/src/tf-chat.json"
    return f"tests/metal/prompts/src/g1.json ({entry_id})"


def build_sets(pack: Pack, src: dict, sources: Sources, specs: tuple[SetSpec, ...]) -> list[BuiltSet]:
    tok = pack.tokenizer
    built: dict[str, list[int]] = {}
    out: list[BuiltSet] = []
    ca_stream: list[int] | None = None
    doc_stream: list[int] | None = None
    g1 = {prompt["id"]: prompt for prompt in src["g1"]["prompts"]}
    for spec in specs:
        if not SET_NAME.match(spec.name):
            raise MakePromptsError(f"make_prompts: bad set name {spec.name!r}")
        entry_id = None
        if spec.kind == "ca":
            if ca_stream is None:
                ca_stream = tok.encode(files_text(sources.ca_files), parse_special=True)
            head_text, tail_text = _placeholder_halves(pack, src["code"]["system"], src["code"]["request"],
                                                       src["code"]["tools"], spec.name)
            ids = cut_files(spec.name, spec.tokens, tok.encode(head_text, parse_special=True), ca_stream,
                            tok.encode(tail_text, parse_special=True), lambda c: round_trips(tok, c))
        elif spec.kind == "tf-code":
            if "ca-4k" not in built:
                raise MakePromptsError("make_prompts: tf-code is the first 512 ids of ca-4k; build ca-4k too")
            ids = built["ca-4k"][:TF_CODE_TOKENS]
            _need(len(ids) == TF_CODE_TOKENS, "ca-4k is shorter than 512 tokens")
        elif spec.kind == "tf-doc":
            if doc_stream is None:
                doc_stream = tok.encode(files_text(sources.doc_files), parse_special=True)
            head_text, tail_text = _placeholder_halves(pack, src["tf_doc"]["system"], src["tf_doc"]["request"],
                                                       None, spec.name)
            ids = (tok.encode(head_text, parse_special=True) + doc_stream
                   + tok.encode(tail_text, parse_special=True))[:spec.tokens]
            _need(len(ids) == spec.tokens, f"{spec.name}: the document is shorter than {spec.tokens} tokens")
        elif spec.kind == "tf-chat":
            ids = tok.encode(render(pack, src["tf_chat"]["messages"], None, thinking=False), parse_special=True)
            _need(len(ids) >= spec.tokens, f"{spec.name}: the transcript is shorter than {spec.tokens} tokens")
            ids = ids[:spec.tokens]
        else:
            entry_id = spec.name
            prompt = g1[spec.name]
            ids = tok.encode(render(pack, prompt["messages"], prompt.get("tools"), prompt["thinking"]),
                             parse_special=True)
            if not round_trips(tok, ids):
                raise MakePromptsError(f"make_prompts: {spec.name}: the prompt is not a tokenisation of its own text")
        _need(len(ids) <= spec.context, f"{spec.name}: {len(ids)} tokens do not fit a context of {spec.context}")
        built[spec.name] = ids
        out.append(BuiltSet(name=spec.name, ids=ids, context=spec.context,
                            source=_source_of(spec, sources, entry_id)))
    return out


# ------------------------------------------------------------------ manifest and files
def build_manifest(pack: Pack, sets: list[BuiltSet]) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "sets": [{
            "name": s.name,
            "tokens": len(s.ids),
            "ids_sha256": ids_hash(s.ids),
            "context": s.context,
            "source": s.source,
            "tokenizer_sha256": pack.sha256,
            "tokenizer_aux_sha256": pack.aux_sha256,
        } for s in sets],
    }


def render_files(pack: Pack, sets: list[BuiltSet], manifest: dict) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for s in sets:
        files[f"{s.name}.ids"] = (canonical_ids(s.ids) + "\n").encode("ascii")
        files[f"{s.name}.txt"] = pack.tokenizer.decode(s.ids, errors="replace").encode("utf-8")
    files["MANIFEST.json"] = (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode("ascii")
    return files


def write_files(out: Path, files: dict[str, bytes]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    for name, data in files.items():
        if name != "MANIFEST.json" and not SET_FILE.match(name):
            raise MakePromptsError(f"make_prompts: refusing to write unexpected name {name!r}")
        target = out / name
        if target.is_symlink():
            raise MakePromptsError(f"make_prompts: refusing to write through the symlink {target}")
        if target.resolve().parent != root:
            raise MakePromptsError(f"make_prompts: {target} is outside {out}")
        target.write_bytes(data)


def resolve_out(out: Path, sha256: str) -> Path:
    """The folder for this tokenizer: `out` itself, unless `out` already holds sets of another tokenizer, in which
    case `out/<hash[:12]>` (one folder per tokenizer)."""
    manifest = out / "MANIFEST.json"
    if not manifest.is_file() or manifest.is_symlink():
        return out
    existing = _read_json(manifest)
    seen = {entry.get("tokenizer_sha256") for entry in existing.get("sets", []) if isinstance(entry, dict)}
    if seen and seen != {sha256}:
        return out / sha256[:12]
    return out


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MakePromptsError(f"make_prompts: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MakePromptsError(f"make_prompts: {path} must hold a JSON object")
    return value


def parse_ids_file(data: bytes) -> list[int]:
    """One line of canonical decimal ids, LF or CRLF terminated. Line ends are not part of the identity."""
    try:
        text = data.replace(b"\r\n", b"\n").decode("ascii")
    except UnicodeDecodeError as exc:
        raise MakePromptsError("the ids file is not ASCII") from exc
    if not text.endswith("\n") or "\n" in text[:-1]:
        raise MakePromptsError("the ids file must be one LF-terminated line")
    fields = text[:-1].split(",")
    for field in fields:
        if not CANONICAL_INT.match(field):
            raise MakePromptsError(f"{field!r} is not a canonical token id")
    return [int(field) for field in fields]


def _lf(path: Path) -> bytes:
    """The file's bytes with CRLF line ends read as LF: a line-end conversion is not a change of content."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def compare_with_rebuild(out: Path, rebuilt: Path) -> list[str]:
    """Differences between the frozen files in `out` and the rebuild in `rebuilt` (empty when they agree)."""
    errors: list[str] = []
    rebuilt_manifest = _read_json(rebuilt / "MANIFEST.json")
    if not _regular(out / "MANIFEST.json"):
        errors.append(f"MANIFEST.json is missing from {out}")
    elif _lf(out / "MANIFEST.json") != _lf(rebuilt / "MANIFEST.json"):
        errors.append("MANIFEST.json differs from the rebuild")
    expected = {entry["name"]: entry["ids_sha256"] for entry in rebuilt_manifest["sets"]}
    for name in sorted(expected):
        ids_path, txt_path = out / f"{name}.ids", out / f"{name}.txt"
        if _regular(ids_path):
            try:
                ids = parse_ids_file(ids_path.read_bytes())
            except MakePromptsError as exc:
                errors.append(f"{name}.ids: {exc}")
            else:
                if ids_hash(ids) != expected[name]:
                    errors.append(f"{name}.ids: the ids hash differs from the manifest")
        else:
            errors.append(f"{name}.ids is missing or is a symlink")
        if _regular(txt_path):
            if _lf(txt_path) != _lf(rebuilt / f"{name}.txt"):
                errors.append(f"{name}.txt differs from the rebuild")
        else:
            errors.append(f"{name}.txt is missing or is a symlink")
    for path in sorted(out.glob("*.ids")) + sorted(out.glob("*.txt")):
        if path.name[:-4] not in expected:
            errors.append(f"{path.name} is not a set of the manifest")
    return errors


# ------------------------------------------------------------------ self-test support
def _synthetic_python(index: int) -> bytes:
    lines = []
    for j in range(40):
        lines.append(f"def function_{index}_{j}(value, scale={j}):")
        lines.append(f"    return value * scale + {index * 17 + j}  # synthetic sample {index}.{j}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _synthetic_prose(label: str) -> bytes:
    return "".join(f"The {label} section {j} explains option {j} and its default value.\n"
                   for j in range(60)).encode("utf-8")


def synthetic_sources() -> Sources:
    ca = tuple((f"src/synthetic_{i:02d}.py", _synthetic_python(i)) for i in range(24))
    docs = (("docs/build.md", _synthetic_prose("build")),
            ("docs/function-calling.md", _synthetic_prose("calls")))
    return Sources(upstream_commit=UPSTREAM_COMMIT, ca_files=ca, llama_commit=LLAMA_CPP_COMMIT, doc_files=docs)


_SPACE = BYTE_TO_UNICODE[ord(" ")]
_NEWLINE = BYTE_TO_UNICODE[ord("\n")]
SYNTHETIC_MERGES = (("t", "h"), ("th", "e"), (_SPACE, "the"), ("i", "n"), ("in", "g"), ("e", "r"),
                    ("a", "n"), (_SPACE, "a"), (_SPACE, "i"), ("l", "l"), ("ll", "o"), ("o", "n"),
                    ("e", "n"), (_SPACE, "t"), ("s", "t"), ("o", "r"), ("r", "e"), (_NEWLINE, "t"))
SYNTHETIC_SPECIALS = (("<|endoftext|>", 3), ("<|im_start|>", 3), ("<|im_end|>", 3), ("<think>", 4), ("</think>", 4))
SYNTHETIC_TEMPLATE = """{%- if tools %}<|im_start|>system
{% for tool in tools %}{{ tool | tojson }}
{% endfor %}<|im_end|>
{% endif %}{%- for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{%- if add_generation_prompt %}<|im_start|>assistant
{% if enable_thinking %}<think>
{% else %}<think>
</think>
{% endif %}{% endif %}"""


def write_synthetic_pack(pack: Path) -> Path:
    """A small byte-level BPE pack in the same files extract() writes. It is for the self-test only."""
    tdir = pack / "tokenizer"
    tdir.mkdir(parents=True)
    tokens = [BYTE_TO_UNICODE[b] for b in range(256)]
    merges = []
    for left, right in SYNTHETIC_MERGES:
        merges.append(f"{left} {right}")
        if left + right not in tokens:
            tokens.append(left + right)
    token_types = [1] * len(tokens)
    for name, kind in SYNTHETIC_SPECIALS:
        tokens.append(name)
        token_types.append(kind)
    vocab = {token: index for index, token in enumerate(tokens)}
    (tdir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (tdir / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (tdir / "token_type.json").write_text(json.dumps(token_types), encoding="utf-8")
    (tdir / "tokenizer.json").write_text(json.dumps({
        "model": "gpt2", "pre": "qwen35", "vocab_size": len(tokens), "n_merges": len(merges),
        "special_ids": {"eos_token_id": vocab["<|im_end|>"]}, "add_bos_token": False,
    }), encoding="utf-8")
    (tdir / "chat_template.jinja").write_text(SYNTHETIC_TEMPLATE, encoding="utf-8", newline="\n")
    return pack


def _rebuild(pack: Pack, src: dict, sources: Sources) -> tuple[list[BuiltSet], dict, dict[str, bytes]]:
    specs = tuple(s for s in SETS if s.name in SELF_TEST_NAMES)
    sets = build_sets(pack, src, sources, specs)
    manifest = build_manifest(pack, sets)
    return sets, manifest, render_files(pack, sets, manifest)


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="make_prompts-self-test-") as tmp:
        base = Path(tmp)
        pack = load_pack(write_synthetic_pack(base / "pack"))
        src = load_src(SRC_DIR)
        sources = synthetic_sources()
        _, _, files = _rebuild(pack, src, sources)
        out = base / "out"
        write_files(out, files)
        clean = _compare_against_fresh(out, files)
        if clean:
            print("\n".join(clean), file=sys.stderr)
            raise MakePromptsError("make_prompts: self-test: a fresh build does not match its own files")

        # A changed byte in a project source must change the frozen sets.
        mutated = base / "src"
        shutil.copytree(SRC_DIR, mutated)
        code = mutated / SRC_FILES["code"]
        data = code.read_bytes()
        if data.count(b"Find where") != 1:
            raise MakePromptsError("make_prompts: self-test: code-agent.json no longer has its request text")
        code.write_bytes(data.replace(b"Find where", b"Fand where", 1))
        _, _, changed = _rebuild(pack, load_src(mutated), sources)
        if not _compare_against_fresh(out, changed):
            raise MakePromptsError("make_prompts: self-test: a changed source byte was not detected")

        # A changed byte in a frozen ids file must be detected by the recomputed hash.
        ids_file = out / "ca-4k.ids"
        raw = bytearray(ids_file.read_bytes())
        raw[0] = ord("9") if raw[0] != ord("9") else ord("8")
        ids_file.write_bytes(bytes(raw))
        if not _compare_against_fresh(out, files):
            raise MakePromptsError("make_prompts: self-test: a changed ids byte was not detected")
    print(f"make_prompts: self-test passed ({len(SELF_TEST_NAMES)} sets, synthetic tokenizer)")
    return 0


def _compare_against_fresh(out: Path, files: dict[str, bytes]) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="make_prompts-rebuild-") as tmp:
        rebuilt = Path(tmp)
        write_files(rebuilt, files)
        return compare_with_rebuild(out, rebuilt)


# ------------------------------------------------------------------ command line
def build(args: argparse.Namespace) -> int:
    pack = load_pack(args.pack)
    src = load_src(args.src)
    sources = read_sources(args.upstream, args.llama_cpp)
    sets = build_sets(pack, src, sources, SETS)
    manifest = build_manifest(pack, sets)
    files = render_files(pack, sets, manifest)
    out = resolve_out(args.out, pack.sha256)
    if args.check:
        errors = _compare_against_fresh(out, files)
        if errors:
            for error in errors:
                print(f"make_prompts: {error}", file=sys.stderr)
            return 1
        print(f"make_prompts: {len(sets)} sets match {out}")
        return 0
    write_files(out, files)
    print(f"make_prompts: wrote {len(sets)} sets to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen benchmark prompt sets.")
    parser.add_argument("--pack", type=Path, help="model pack folder that holds tokenizer/")
    parser.add_argument("--upstream", type=Path, help="Niko1221/Strata checkout that has commit 1678de33")
    parser.add_argument("--llama-cpp", type=Path, default=ROOT / "third_party" / "llama.cpp",
                        help="llama.cpp checkout that has commit 3cf03257")
    parser.add_argument("--src", type=Path, default=SRC_DIR, help="project prompt sources")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output folder for the sets and MANIFEST.json")
    parser.add_argument("--check", action="store_true", help="fail on any difference from a rebuild")
    parser.add_argument("--self-test", action="store_true", help="check the builder on a synthetic tokenizer")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            if args.pack or args.upstream or args.check:
                parser.error("--self-test takes no other option")
            return self_test()
        if args.pack is None or args.upstream is None:
            parser.error("--pack and --upstream are required (or use --self-test)")
        return build(args)
    except (MakePromptsError, OSError) as exc:
        print(exc if isinstance(exc, MakePromptsError) else f"make_prompts: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
