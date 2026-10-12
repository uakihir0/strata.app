# 09. Testing and validation: how the Mac engine is shown to be right and fast

Status: design specification, updated 2026-10-10. The host and Metal CTest targets and the macOS Python test runner exist. Model-backed acceptance, the release gate, and end-to-end inference tests are not implemented.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Source audits: [audit/A-model-and-kernels.md](audit/A-model-and-kernels.md), sections E.0-E.8 (tests and oracles), and
[audit/K5-orchestration-vram-tests.md](audit/K5-orchestration-vram-tests.md), section 6 (determinism and the test
plan). Upstream's own evidence: `bench/results/2026-09-27-cache-parity/`, `2026-09-27-esp/`,
`2026-09-28-prefill-speed/` and `2026-09-29-speed-0126/`.
Every `file:line` reference is relative to the upstream repository root.

This document covers how the port is shown to work:
- the frozen inputs: the benchmark prompt sets, the test models, and the reference data from upstream's CUDA build and
  from llama.cpp;
- the correctness criteria C1-C4, the prompt-path criterion, long-context needles and the server end-to-end run;
- the performance criteria: the decode benchmark, PF-1 and PF-2, and the regression check;
- the gates G1-G3 of [00-feasibility-study.md](00-feasibility-study.md), section 7, as commands with exit codes;
- where and when each test runs: CI, the reference Mac, the family runs on M1-M4, the nightly and weekly runs, and
  the release acceptance runs (section 8);
- a map of every test that the other documents define, with its label and its schedule (section 9).

Other documents cover the rest:
- [04-metal-kernels.md](04-metal-kernels.md), section 10: the kernel twins, their oracles, input sets and tolerance
  classes. This document runs them; it does not restate them.
- [05-memory-and-residency.md](05-memory-and-residency.md), section 13, and [06-engine-host-port.md](06-engine-host-port.md),
  section 10.5: the tests of the planner and the exactness contract that C2-C4 check.
- [07-server-setup-port.md](07-server-setup-port.md), section 10: the Python tests.
- [08-build-and-distribution.md](08-build-and-distribution.md), sections 6 and 12: the test helper, the labels, the
  presets and the CI workflows that run what this document lists.
- [01-requirements.md](01-requirements.md): the numbers a release must reach. Each requirement there names the check
  here that verifies it.

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. Principles, layers and files

### 1.1 Principles

| # | Principle | What it means here |
| --- | --- | --- |
| P1 | Bitwise only where the arithmetic is the same | Mac against Mac is compared bit for bit: speculation against greedy, tiered against full, a resumed state against a re-read one, one run against the next. Mac against CUDA or llama.cpp is a tolerance comparison, because the order of floating-point additions differs ([04](04-metal-kernels.md), section 3.9). |
| P2 | Frozen reference results | Previously measured results from the upstream CUDA build and llama.cpp at `3cf03257f219afbe7334045ff7c6a06ac68c627d` inform the target comparisons. This project builds and supports the Metal product only; it does not build or run a CUDA product backend. |
| P3 | The tolerance comes from the references | A cross-backend threshold is set from how far upstream's own valid CUDA paths differ from each other, and how far CUDA differs from llama.cpp (measurements T-1 to T-3). The Mac must be no further from CUDA than that. No threshold is chosen to make a result pass. |
| P4 | Kernels against oracles, the model against references | A kernel is tested against its CPU oracle in [04](04-metal-kernels.md), section 10. This document tests the whole model. A model-level failure is bisected down to a kernel with the dumps of section 3.1. |
| P5 | Frozen inputs | Prompts, golden data and test models are named by SHA-256 in manifests in git. A run whose inputs do not match their manifest fails before it starts. |
| P6 | A criterion is a command | Every criterion is checked by a script that exits 0 on a pass and non-zero otherwise (section 3.2), and writes its numbers to a result file. Only the reviews named in section 7.6 need a person. |
| P7 | Keep reference formats stable | The upstream needle and conversation-cache tools remain the numerical reference. `tools/metal/tf_compare.py` reads the retained logits dump format directly and runs on macOS without the old workstation-specific launcher. |
| P8 | No retries | A test that fails once has failed. A test that passes and fails on the same inputs is a bug in the engine or the test, filed as one (section 10.2). |

### 1.2 Layers

| Layer | What runs | Label ([08](08-build-and-distribution.md), section 6.2) | Where |
| --- | --- | --- | --- |
| 1. Python | `tools/test_*.py` and `serve/test_*.py`, through `tools/metal/run_python_tests.py` ([07](07-server-setup-port.md), section 10) | none (not in CTest) | Hosted macOS CI and the reference Mac |
| 2. Host | Unit tests that need no GPU: parsers, planners, readers, oracles, fixtures | `host` | Every runner |
| 3. Kernel | The twins and properties of [04](04-metal-kernels.md), section 10, on synthetic, edge and fixture inputs | `metal` | Reference Mac, family runs; hosted Macs if X-8 allows (section 7.5) |
| 4. Model | CTest tests that load a real model from `STRATA_TEST_MODELS`: the `real` input sets of the twins, the model-level properties, the full-against-tiered tests | `model` | Reference Mac |
| 5. Acceptance | Scripts that run the engine binary or the server on the frozen inputs: C1-C4, G1-G3, the benchmarks, needles, the server run | not in CTest; run by `tools/metal/accept.py` | Reference Mac; part of them on the family Macs |
| 6. Measurement | One-off or periodic runs whose result sets a default (the M, R, L, H, X, B and T items) | none | As each item says; results in [appendix-a-measurements.md](appendix-a-measurements.md) |

Layers 2-4 are CTest tests registered with `strata_metal_test(<name> <label> <command>)`
([08](08-build-and-distribution.md), section 6.5). Layer 5 is outside CTest because its runs take minutes to hours,
need a model loading window of their own, and produce result files rather than a pass line.

### 1.3 Files this document adds

| Path | What it is | Section |
| --- | --- | --- |
| `tests/metal/prompts/` | The benchmark prompt sets: `<set>.ids`, `<set>.txt`, `MANIFEST.json`, and the sources in `src/` | 2.1 |
| `tools/metal/make_prompts.py` | Builds the prompt sets from their sources | 2.2 |
| `tests/metal/models.json` | The test models: tag, files, sizes and SHA-256 values | 2.3 |
| `tools/metal/test_models.py` | `fetch` and `check` for the models under `STRATA_TEST_MODELS` | 2.3 |
| `tests/metal/golden/MANIFEST.json` | The golden set: files, SHA-256 values, command lines, GPU, driver, upstream commit | 2.5 |
| `tools/metal/record_golden.py` | Records the golden set on the CUDA reference PC | 2.5 |
| `tools/metal/accept.py`, `tools/metal/accept/` | The acceptance runner and its runs (one module per run) | 3.2 |
| `tools/metal/tf_compare.py` | Compares two logits dumps: same top-1, KL, perplexity, margins | 3.3 |
| `tools/metal/ladder_compare.py` | Compares two `--dump-layers` files, or one with llama.cpp's ladder | 4.1 |
| `tools/metal/lc_dump/` | `lc_dump.cpp` and its `CMakeLists.txt`: a small program, built against llama.cpp at the pin, that writes every position's logits and the residual ladder in the engine's dump formats | 2.5 |
| `tools/metal/accept/engine.py` | The one parser of the engine's output lines | 3.1 |
| `tools/metal/accept/criteria.py` | Every criterion's ID, operator, threshold and gating flag, in one table | 3.2, 10.3 |
| `tools/metal/accept/fake_engine.py`, `tools/metal/test_accept.py` | The fake engine and the runner's own tests | 3.2 |
| `tools/metal/accept/engines.py` | Fetches and unpacks the previous release's engine | 5.4 |
| `tools/metal/accept/pressure_smoke.py` | The memory-pressure smoke step | 5.5, 9.2 |
| `tests/metal/images/` | `red.png` and `osaka.png`, the images of S.image | 4.7 |
| `tests/metal/upstream_speed.json` | Upstream's published speeds, with their source files | 5.3 |
| `tools/metal/balloon.py` | Holds memory so that a 64 GB Mac behaves as a 48 GB one | 6.2 |
| `docs/mac/gates/G1.md`, `G2.md`, `G3.md` | The record of each gate's decision | 6 |
| `tools/metal/family_run.sh` | One family run on a Mac of M1-M4 | 7.4 |
| `tests/metal/model/` | The `model`-label CTest sources this document adds (sections 4 and 5.5) | 9 |
| `tests/metal/test_map.json` | Every CTest test with its label and origin, in machine form | 9.4 |
| `tests/metal/requirements.json`, `tests/metal/reviews.json` | Each requirement of [01](01-requirements.md) with its checks; the records of the reviews | 6.3, 7.6, 9.4 |
| `tools/metal/test_map.py` | The host test that the CTest list, this document's map, the requirements and the reviews agree | 9.4 |
| `tools/metal/test_tf_compare.py`, `test_ladder_compare.py`, `test_make_prompts.py` | The unit tests of the comparison tools and the prompt builder | 9.3 |

Results are written outside the checkout, to `STRATA_RESULTS` (default `/Users/Shared/strata-results` on the
reference Mac), as section 3.2 describes.

---

## 2. Inputs: prompts, models and references

### 2.1 The benchmark prompt sets

Upstream's benchmarks read frozen prompts from `bench/prompts/<group>/<name>.ids`, for example
`bench/prompts/long/32k-01-code-agent.ids` (`bench/results/2026-09-28-prefill-speed/README.md`), and
`.gitattributes` marks `bench/prompts/** -text` because they are hashed byte for byte. The files are not in the
repository at the baseline: no commit in its history adds them. Upstream's numbers can therefore be compared with the
Mac's only as numbers, not on the same tokens. The project defines its own sets, in `tests/metal/prompts/`:

| Set | Tokens | Content | Used by |
| --- | ---: | --- | --- |
| `ca-1k` | 1,024 | Code agent: the system prompt and tool definitions of a coding agent, then a user message with repository files, then a request about them (section 2.2) | Decode benchmark; C2; R-2, R-3 ([03](03-model-reference.md)) |
| `ca-4k` | 4,096 | Same, more files | Decode benchmark; C2; G1 and G2 speed; R-2, R-3, M-9 ([05](05-memory-and-residency.md)) |
| `ca-32k` | 32,256 | Same | Decode benchmark; PF-1, PF-2; the prompt-path criterion (section 4.5); C2; needles' length class |
| `ca-64k` | 65,024 | Same | Decode benchmark (weekly) |
| `ca-128k` | 130,560 | Same | Decode benchmark (weekly); PF-1 |
| `ca-262k` | 261,632 | Same | Decode benchmark (weekly), on Macs whose plan holds a 262,144 context |
| `tf-code` | 512 | The first 512 tokens of `ca-4k` | C1 (teacher-forced), as upstream's "code agent (512)" |
| `tf-doc` | 1,024 | The first 1,024 tokens of a chat whose user message is a long technical document (section 2.2) | C1; R-1 |
| `tf-chat` | 1,024 | The first 1,024 tokens of a 20-turn chat transcript | C1 |
| `g1-01` .. `g1-10` | 30-3,200 each | Ten short chat prompts (section 2.2) | G1; C2; C3 repeats; the server run |

- Each long set is 512 tokens shorter than its context: `ca-32k` runs at a context of 32,768, `ca-64k` at 65,536,
  `ca-128k` at 131,072 and `ca-262k` at 262,144. `ca-1k`, `ca-4k`, `tf-*` and `g1-*` run at 8,192. So the runs of
  this document always use one of five contexts, the same as setup's menu offers ([07](07-server-setup-port.md),
  section 4.5).
- The token counts are targets. A set may be up to 16 tokens shorter (section 2.2); the manifest holds the exact
  count.
- Needle prompts are not frozen files: `tools/needle_bench.py` generates them deterministically
  (`random.Random(7)`, the repository's files in sorted order), so the same checkout gives the same prompts.

**File formats.**
- `<set>.ids`: one line of ASCII, the token ids separated by single commas, then LF. The engine's `--tokens-file`
  accepts commas and white space (`parse_i64_list`, `generate.cpp:578-596`).
- `<set>.txt`: the ids decoded by `tools/strata_tokenizer.py`, with LF line ends. It is for reading only; the ids
  are the input.
- `MANIFEST.json`: schema 1, with one entry per set: `name`, `tokens` (the count), `ids_sha256` (the SHA-256 of
  the canonical string `",".join(ids)` as ASCII, with no line end), `context`, `source` (the source file and the
  upstream or llama.cpp commit it reads), and `tokenizer_sha256` (below). The top level holds `builder_version`.
- Because the hash is over the canonical string and not the file bytes, a line-end conversion by git cannot change
  a set's identity, and no `.gitattributes` line is needed. `test_models.py check` and every acceptance run recompute
  `ids_sha256` from the file before using it.

**One set per tokeniser.** The ids depend on the tokeniser and the chat template. `tokenizer_sha256` is the
SHA-256 over `vocab.json`, `merges.txt` and `chat_template.jinja` of the pack's `tokenizer/` folder, concatenated in
that order. At the baseline all seven files are expected to share one tokeniser (248,320 tokens, `pre` = `qwen35`,
`tools/strata_tokenizer.py:3-4`). If `make_prompts.py` finds a second hash among the test models, it writes that
group's sets to `tests/metal/prompts/<tokenizer_sha256[:12]>/`, and each run picks the folder whose hash matches
its model's pack.

### 2.2 Building the sets: `tools/metal/make_prompts.py`

```
python3 tools/metal/make_prompts.py --pack <pack folder> [--out tests/metal/prompts] [--check]
```

The builder reads only sources with a known licence and a fixed commit:

| Source | Read from | Licence |
| --- | --- | --- |
| Repository files for the `ca-*` sets | `git -C ../upstream show 1678de333d0e0711bc414ad992b640e1a37dd814:<path>` from a temporary upstream checkout at that exact commit, for every path under `src/`, `include/`, `serve/` and `tools/` with the extensions `.cpp`, `.hpp`, `.h`, `.cu`, `.cuh`, `.py`, sorted by path. This source lives outside the project history. | MIT (upstream) |
| The document of `tf-doc` | `docs/build.md`, then `docs/function-calling.md`, from `third_party/llama.cpp` at `3cf03257` (the checkout setup makes) | MIT (llama.cpp) |
| System prompts, tool definitions, requests, the `tf-chat` transcript, the `g1-*` prompts | `tests/metal/prompts/src/*.json`, written for the project | The project's licence |

Steps, for each set:
1. Build the messages (and tools) from the source JSON. For a `ca-*` set, the user message is the repository files,
   each preceded by a line `=== <path> ===`, followed by the request in `src/code-agent.json`:
   "Find where the prompt cache decides how much of a conversation it can reuse, explain the rule, and propose a
   test for it." The files are a placeholder at this step.
2. Render with `serve.frontend.ChatTemplate(<pack>/tokenizer/chat_template.jinja).render(messages, tools,
   add_generation_prompt=True)`, with the placeholder `<<<STRATA_FILES>>>` where the files go. Split the rendered
   text at the placeholder into a prefix and a suffix.
3. Encode the prefix and suffix with `strata_tokenizer.Tokenizer(<pack>/tokenizer).encode(text,
   parse_special=True)`, and the file text with `parse_special=False`, so that a file that contains a special
   token's spelling cannot inject one.
4. Keep `k = target − len(prefix) − len(suffix)` file tokens: `ids = prefix + files[:k] + suffix`.
5. **Round trip.** `encode(decode(ids), parse_special=True)` must equal `ids`. If it does not, the cut split a
   merge: k decreases by one and step 5 repeats, at most 16 times; then the builder fails with
   `make_prompts: <set>: no cut within 16 tokens of <target> round-trips`. This makes every set a tokenisation of
   its own text, so a server request with `<set>.txt` gives the same ids as the file.
6. `tf-code` is the first 512 ids of `ca-4k`. `tf-doc` and `tf-chat` are the first 1,024 ids of their rendered
   chats, which are longer than 1,024 tokens by construction. A cut prefix needs no round trip: it is never sent as
   text.
7. Write `<set>.ids`, `<set>.txt` and the manifest entry.

The `g1-*` prompts, in `src/g1.json`:

| Set | Prompt | Template options | Expected in the answer (checked only by the server run, section 4.7) |
| --- | --- | --- | --- |
| `g1-01` | Write a Python function that merges overlapping intervals, with three unit tests | thinking off | `def ` |
| `g1-02` | A shop sells pencils at 3 for 40 yen. How much do 27 pencils cost? Answer with the number only. | thinking off | `360` |
| `g1-03` | The same kind of word problem, multi-step (trains meeting) | thinking on | the meeting time |
| `g1-04` | A Japanese question about the difference between a process and a thread, answered in Japanese | thinking off | a Japanese sentence |
| `g1-05` | A 600-character Chinese news text, summarise in three Chinese sentences | thinking off | Chinese text |
| `g1-06` | One tool `get_weather(city)`; "What is the weather in Osaka?" | tools | a tool call to `get_weather` with `city` Osaka |
| `g1-07` | A 3,000-token system prompt of coding rules, then a one-line question about rule 17 | thinking off | the text of rule 17 |
| `g1-08` | A 3-turn conversation, then a follow-up that needs turn 1 | thinking off | the fact from turn 1 |
| `g1-09` | A 250-line C++ file and a request for a unified diff that renames one function | thinking off | `@@` |
| `g1-10` | Return a JSON object with the keys `name`, `year` and `tags` for a given paragraph | thinking off | valid JSON with the three keys |

`--check` rebuilds every set into a temporary folder and compares the ids hashes with the manifest. The host test
`prompts_manifest_test` runs `--check` with the tokeniser fixture of [03](03-model-reference.md) work item R1, and
fails on any difference. The frozen files are the truth: a tokeniser or template change at an upstream sync that
changes a set's ids shows up as this test's failure, and the sets are rebuilt in a reviewed commit with a new
`builder_version`.

### 2.3 Test models

The model-label tests and the acceptance runs read models from `STRATA_TEST_MODELS`, which is
`/Users/Shared/strata-models` on the reference Mac. Each model has a folder named by its setup tag:

```
<STRATA_TEST_MODELS>/<tag>/
    shard 1 (and shard 2) GGUF files, as setup downloads them
    pack/            iq_pack.py's output ([03](03-model-reference.md), section 4.5)
    mtp/             the MTP runtime, as tools/mtp_fetch.py writes it
    config.json      the server config setup writes on this Mac: exe, args, cwd, tokenizer, model_name
```

- `tools/metal/test_models.py fetch <tag>...` imports `setup` and runs the same download, pack and MTP steps as
  setup's step 6 ([07](07-server-setup-port.md), section 4.8), with `setup.ROOT` set to the model's folder and the
  answers given by `--yes`. It then writes `config.json` with setup's own function and the engine plan of this Mac.
  The shared PLE shard 2 is downloaded once and hard-linked into each folder that needs it.
- For Q2_0 and Coder, `fetch` also downloads the image encoder's `mmproj` file, as setup does with images on the
  GPU, so their `config.json` has a `vision` entry. The server run's S.image check needs it (section 4.7).
- `test_models.py check` compares every file's size and SHA-256 with `tests/metal/models.json`. The model-label
  CTest tests and every acceptance run call `check --quick` (sizes, and the SHA-256 of the first and last 64 MiB of
  each file) before they start; the nightly run calls the full check once a week (section 7.2).
- `config.json`'s `args` are the **setup-written args** that the benchmarks use (section 5.1). The runs that need
  other settings change only the flags they name.

| Tag | Files | Needed by |
| --- | --- | --- |
| Q2_0 | ISTA shard 1 (35.03 GiB of tensors) and the PLE shard 2 | Every gate; nightly |
| Coder (IQ1_M) | Coder shard 1 and the PLE shard 2 | C1, C2, C4; nightly short form; release |
| IQ2_XS, IQ3_XXS, IQ3_S | ISTA shards 1 and the PLE shard 2 | C2 and C1.1 on `tf-code` in the weekly rotation (Wednesday); R5's full compare ([03](03-model-reference.md)); release `--plan-only` |
| Swift IQ2_XS, Swift IQ3_XXS | Both Swift shards | R5's full compare (R-4), R9 and R10 ([03](03-model-reference.md)); release `--plan-only` |
| UD-Q4_K_XL | Not in Phase 1 ([00](00-feasibility-study.md), section 5) | Added with its format |

Disk: the seven shard 1 files are about 352 GB, the PLE shard 28.8 GB, and the packs, MTP runtimes, golden set and
llama.cpp references (section 2.5) add about 80 GB. The reference Mac should have a 2 TB internal or Thunderbolt SSD
for `STRATA_TEST_MODELS` and `STRATA_RESULTS`, excluded from Spotlight (`mdutil -i off`) and from Time Machine.

### 2.4 The CUDA reference PC

The golden data is recorded on a PC with upstream's own build, not on a Mac:

| Item | Requirement |
| --- | --- |
| GPU | An NVIDIA GeForce RTX 5070 12 GB (sm_120), the card of upstream's published numbers (`bench/results/2026-09-29-speed-0126`). Any other sm_120 card with at least 12 GB may stand in; the card is recorded, and measurement L-9 ([04](04-metal-kernels.md), section 11) is rerun for its SASS. A card of another architecture is not allowed, because nvcc's contractions may differ by architecture. |
| Host | Any x86-64 PC with at least 64 GB of RAM and PCIe 4.0 x16 or better (upstream's: Ryzen 5 7600, 64 GB DDR5-5200, PCIe 5.0 x16, Windows 10) |
| Engine | Upstream's release asset for the version at `UPSTREAM_SYNC`, or `setup.py --build` at that commit when that source revision is not a release tag. Recorded with its `--version`, the CUDA runtime and the driver. |
| Models | The same files, with the same SHA-256 values, as `tests/metal/models.json` |

**The reference environment.** Every golden run uses the model's setup-written args on the PC, changed as follows:

| Setting | Value | Why |
| --- | --- | --- |
| `STRATA_ROPE_TABLE` | `1` | RoPE from the table, as the Mac computes it ([03](03-model-reference.md), section 6.7) |
| `STRATA_PREFILL_MMQ` | `0` | The prompt path's experts on the FP16 arm, the Mac's arithmetic ([04](04-metal-kernels.md), section 6.6) |
| `STRATA_PLE_BATCH` | unset | Upstream's default PLE block ([06](06-engine-host-port.md), section 5.4) |
| `STRATA_IQ_MT_MIN` | `1` for IQ formats (Coder, IQ2_XS, IQ3_XXS, IQ3_S) | Upstream issue #152: without it, CUDA's IQ runs are not repeatable (K5 section 6) |
| `--expert-cache`, `--pcie-frac`, `--adapt-swaps` | `4000`, `0`, `0` | Fixed experts: the same expert is computed the same way at every position, matching the retained teacher-forced reference run |
| `--prompt-cache` | `0` | No reuse between runs |
| `--spec` | `0` | Plain decode for every golden stream; speculation is checked Mac against Mac (C2) |
| `--prefill` | removed for the teacher-forced runs (every position through the token path); `auto` for the prompt-path runs (section 2.5) | Matches the retained teacher-forced reference run and the separate prefill yardstick |

### 2.5 The golden set (work item R12 of [03](03-model-reference.md))

`tools/metal/record_golden.py` runs on the CUDA PC, in a checkout of the project at the local project commit, and writes one
folder per sync: `golden/<UPSTREAM_SYNC[:12]>/<tag>/`. It is portable Python (paths through `pathlib`, the engine
named `strata` or `strata.exe`), because upstream's benchmark PC runs Windows.

```
python tools/metal/record_golden.py --engine <upstream engine> --models <dir> --out <dir> [--model TAG ...]
                                    [--llama-cpp <build>] [--only KIND --ids FILE]
```

`--models` has the layout of section 2.3. `--only` records one kind of file of the tables below (`tf`, `nocache`,
`ladder`, `router`, `pp`, `g1`, `lc-tf`, `lc-ladder`) for the ids in `--ids`, into a folder `extra/` beside the set,
with its own manifest entries: a bisection uses it for a position that the set does not cover (section 10.1).

For each of Q2_0 and Coder:

| Files | Input | Engine arguments added to the reference environment | Read by |
| --- | --- | --- | --- |
| `tf-code.logits`, `tf-doc.logits`, `tf-chat.logits` | The three `tf-*` sets | `--tokens-file <set>.ids --max-new 1 --dump-logits <file>` | C1.1, T-1, T-2 |
| `tf-*.nocache.logits` | The same | As above, without `--expert-cache` (every routed expert on the CPU pool) | T-1 (the noise floor) |
| `ladder-code.layers`, `ladder-doc.layers`, `ladder-chat.layers` | The first 65 ids of each `tf-*` set (64 positions) | `--tokens <ids> --max-new 1 --dump-layers <file>` | C1.3, R-5 |
| `router.layers`, `router.halves`, `router.routing` | The first 1,001 ids of `tf-doc` (1,000 positions) | `--tokens-file <ids> --max-new 1 --dump-layers <file> --dump-halves <file> --dump-routing <file>` | C1.4, R-1 |
| `pp-32k.logits` | `ca-32k` | `--prefill auto --prefill-until 32000 --tokens-file ca-32k.ids --max-new 1 --dump-logits <file>` | The prompt-path criterion (section 4.5) |
| `pp-32k.c6144.logits` | `ca-32k` | As above with `--prefill 6144` in place of `--prefill auto` | T-1: the prompt path's own yardstick, a chunk-size change of the unchanged CUDA engine |
| `g1-XX.out` | Each `g1-*` set | `--tokens-file g1-XX.ids --max-new 256 --greedy`; the `output  :` ids | G1 |
| `g1-XX.tf.logits` | Each `g1-*` set followed by its `g1-XX.out` ids | `--tokens-file <ids> --max-new 1 --dump-logits <file>` | G1 |
| `g1-XX.first` | Each `g1-*` set | `--prefill auto --tokens-file g1-XX.ids --max-new 1` with `STRATA_DUMP_FIRST_LOGITS=<file>` | C1.5 |

For IQ2_XS, IQ3_XXS and IQ3_S, the set holds only `tf-code.logits` and `tf-code.nocache.logits`, for the weekly
C1.1 check of those formats (section 7.2). This goes beyond [03](03-model-reference.md) work item R12, which names
Q2_0 and Coder; work item T2 records them.

From llama.cpp at `3cf03257`, built on the same PC with its CPU backend only (`GGML_CUDA=OFF`, so that its numbers
do not depend on a GPU), `record_golden.py --llama-cpp <build>` also writes:

| Files | Input | Tool | Read by |
| --- | --- | --- | --- |
| `lc-tf-code.logits`, `lc-tf-doc.logits`, `lc-tf-chat.logits` | The three `tf-*` sets | `lc_dump --logits`, in the engine's `--dump-logits` format (`int32 n_vocab, int32 rows`, then F32) | C1.2, T-3 |
| `lc-ladder-*.layers` | As `ladder-*` | `lc_dump --layers`, in the `--dump-layers` layout | C1.3, T-3 |

**`tools/metal/lc_dump`.** llama.cpp's own `llama-debug` (`examples/debug/debug.cpp`) is not enough: it saves only
the last position's logits (`llama_get_logits_ith(ctx, tokens.size() - 1)`) and logs tensors as text. `lc_dump`
is about 200 lines of C++ against `llama.h`:

```
lc_dump -m <shard 1> --tokens-file <set>.ids (--logits <out> | --layers <out>) [--threads N]
```

- It creates the context with `n_ctx = n_batch = n_ubatch = 8192`, so that one `llama_decode` takes the whole set,
  and a `llama_batch` whose `logits[i]` is true for every position.
- `--logits` writes `int32 n_vocab, int32 rows` with `rows = n`, then the rows of positions 0..n−1 from
  `llama_get_logits_ith`: the same rows the engine's teacher-forced `--dump-logits` writes for `--max-new 1`, whose
  count is `n_prompt − 1 + max_new` (`generate.cpp:3283`).
- `--layers` sets `cb_eval` in `llama_context_params`. The callback asks for the nodes named `hc_init` and
  `l_last-<l>` (the names in llama.cpp's model file for this architecture), reads each with
  `ggml_backend_tensor_get`, and writes per position the 49 slots of `4 × 2560` F32, as
  [03](03-model-reference.md), section 9.1, lays out `--dump-layers`.
- It fails if a node is missing or has another shape, so that a llama.cpp change of the graph cannot pass silently.
- Its build is a `CMakeLists.txt` that takes `LLAMA_CPP_DIR`; it is not part of the engine build and is not shipped.

**The manifest.** `tests/metal/golden/MANIFEST.json`, in git, schema 1: `upstream_sync`, `valid_for` (the list of
later sync commits it still applies to, below), `engine_version`, `gpu` (name, PCI id, VRAM), `driver`, `cuda`,
`os`, `llama_cpp` (commit, build flags), and one entry per file: `path`, `bytes`, `sha256`, `command` (the full
argument list and every `STRATA_*` variable set), `seconds`. `tools/metal/accept.py` checks every file it reads
against the manifest first and fails with `golden: <path>: sha256 <found> does not match the manifest` otherwise.

**Size.** A logits row is `248,320 × 4` B = 0.99 MB. Per model: the `tf-*` logits and their no-cache twins are
5.1 GB, the G1 logits about 6 GB, the two `pp-32k` files 0.5 GB of rows that the comparison reads, the ladders
0.4 GB, and the router's halves and ladder 4.7 GB. With the other formats' `tf-code` files and llama.cpp's files, the set is about 40 GB
(measurement T-7).

**Storage and transfer.** The files are not in git. `record_golden.py` ends by printing the manifest; the files go
to the reference Mac with `rsync` over SSH (or on a disk), to
`<STRATA_TEST_MODELS>/golden/<UPSTREAM_SYNC[:12]>/`. The previous reference revision's folder is kept until the next one passes
C1, so a regression can be compared with both.

**When it is recorded again.** Whenever the comparison baseline advances ([10](10-implementation-plan.md), section 4),
`tools/metal/check_drift.py` lists the upstream files the compatibility review finds. The golden set is recorded again when any
changed file is under `src/` or `include/` and is not on `check_drift.py`'s reviewed `no_arithmetic` list. That list
currently contains only `src/platform/**`. Every other source or header change, including server, option and logging
code, conservatively requires a new golden set. Add a path to `no_arithmetic` only after review proves that changes to
that path cannot affect numerical outputs through any supported execution path; record the reason and a regression test.
Otherwise append the local project commit to `valid_for`. If a compatibility change needs a new golden set,
run C1 against the previous set for reporting but do not merge the project pull request to `main` until the new set
exists and C1 passes against it.

### 2.6 What the references are not

- **Not a bitwise target.** Upstream's CUDA build itself is not bitwise repeatable across expert-cache settings
  (`bench/results/2026-09-27-cache-parity/README.md`: 2.3-5% of positions change their top-1 when the cache is
  turned on). The Mac is held to the same standard, not a stricter one.
- **Not upstream's live speed.** Upstream's published speeds (RTX 5070: 87.3 tok/s decode at 1K, 2,171 tok/s
  prefill at 32K) are recorded for the comparison of section 5.3. No criterion requires the Mac to match them.

---

## 3. The test surface

### 3.1 What the tests read from the engine

The acceptance runs read only what upstream's engine already prints or writes. [06](06-engine-host-port.md) keeps
these outputs with upstream's formats, and [03](03-model-reference.md), section 9.1, lays out the dump files.

| Output | Form | Read by |
| --- | --- | --- |
| `prompt  :` and `output  :` lines of a one-shot run | The prompt's and the generated ids, each preceded by one space (`generate.cpp:6331-6335`) | C2, C4, G1, the benchmark |
| `decode` line | `decode` padded to 24 columns, then `N tokens in X ms  ->  Y tok/s` (`:6337`) | The benchmark |
| `prefill` line | `prefill` padded to 24 columns, then `N tokens in X ms  ->  Y tok/s  (time to first token Z ms)`. It is printed only when the prompt has more than one token (`:6339-6342`). | The benchmark, PF-1, PF-2 |
| `speculation` line | `speculation` padded to 24 columns, then `R rounds of K, drafts accepted A of D (r), X tokens per round` (`:6240`) | The benchmark, C2 (oracle arms) |
| `strata generate: prompt chunk auto: N tokens` (stderr) | The chunk size `--prefill auto` chose | The prompt-path criterion, PF-1, PF-2 |
| `--dump-logits` file | `int32 n_vocab, int32 rows`, then F32 rows | C1, G1, the prompt-path criterion |
| `STRATA_DUMP_FIRST_LOGITS` file | `n_vocab` F32, no header | C1.5 |
| `--dump-layers`, `--dump-halves`, `--dump-routing` files | [03](03-model-reference.md), sections 9.1-9.2 | C1.3, C1.4, R-1, R-5 |
| `strata serve: STATE_HASH L=<n> gdn=<16 hex> ple=… tail=… pooled=… kv=… mtp=… stale=… dead=… pooled_full=… ple_prev=<a>,<b>` (stderr, with `STRATA_STATE_HASH=1`) | One line after each serve request (`:5558-5564`) | C3, through upstream's tools |
| `strata serve: STATE_HASH_GDN <hashes>` (stderr, with `STRATA_STATE_HASH_GDN=1`) | One hash per GDN layer | Bisection (section 10.1) |
| The server's `/v1/status`, `/metrics` and `DONE` counters | [07](07-server-setup-port.md), section 6 | The server run (section 4.7) |
| `strata --plan-only` | The PLAN line ([05](05-memory-and-residency.md), section 3) | The release run (section 8) |

**A logits dump after a prompt path is short.** With `--prefill`, upstream's header still says
`n_prompt − 1 + max_new` rows, but the loop writes rows only from the first token-path position `n_batched` on
(`generate.cpp:5654`, `:5739`, `:5873-5889`). The Mac keeps the format byte for byte, so it writes the same short file.
`tf_compare.py` therefore takes the row count from the file size, `(size − 8) / (4 · n_vocab)`, which must be a whole
number, and the first row's position as the header's count minus the rows present.

**Parsing.** `tools/metal/accept/engine.py` holds the one parser of these lines. It fails with
`accept: <file>: no <label> line` when a line it needs is missing, rather than reading a default. Its unit test
parses captured output of upstream's CUDA engine and of the fake engine (section 3.2).

### 3.2 The runner: `tools/metal/accept.py`

```
python3 tools/metal/accept.py <run> [--engine PATH] [--vision PATH] [--models DIR] [--golden DIR] [--out DIR]
                              [--model TAG ...] [--prompt SET ...] [--short] [--day N] [--keep] [--dry-run]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--engine` | `build-dev/strata` | The engine under test. The release run passes the installed engine (section 8). |
| `--vision` | `strata-vision` next to `--engine`, else `build-vision/bin/strata-vision` | The image encoder for S.image (section 4.7). A run that includes S.image and finds neither exits 2. |
| `--models` | `$STRATA_TEST_MODELS` | The test models (section 2.3) |
| `--golden` | `<models>/golden/<UPSTREAM_SYNC[:12]>/` | The golden set (section 2.5) |
| `--out` | `$STRATA_RESULTS/<run>/` | Where the result folder goes |
| `--model` | The run's own list | Restricts a run to some models |
| `--prompt` | The run's own list | Restricts a run to some prompt sets, for a bisection (section 10.1). A run restricted this way never counts for a gate or a release. |
| `--short` | off | The short form, where a run has one (C1) |
| `--keep` | off | Keeps the dumps after a pass (they are always kept after a failure) |
| `--day N` | `date -u +%u` | For `weekly`: the slice to run, 1-7 (section 7.2) |
| `--dry-run` | off | Prints the commands and checks the inputs; loads no model |

The runs:

| Run | What it checks | Section |
| --- | --- | --- |
| `c1` | C1.1-C1.5, or the short form with `--short` | 4.1 |
| `c2` | Speculation and every Mac path against plain decode | 4.2 |
| `c3` | The conversation-cache tools, the re-read check and the repeats | 4.3 |
| `c4` | Tiered against full mode at the acceptance level | 4.4 |
| `pp` | The prompt-path criterion | 4.5 |
| `needle` | Needles at the lengths the schedule names | 4.6 |
| `server` | The server end to end | 4.7 |
| `bench` | The decode benchmark and the regression A/B | 5.1, 5.4 |
| `pf` | PF-1 and PF-2 | 5.2 |
| `g1`, `g2`, `g3` | The gates | 6 |
| `nightly`, `weekly`, `release`, `family` | The sets of runs of sections 7 and 8 | 7, 8 |

**Exit codes.** 0: every criterion passed. 1: a criterion failed. 2: the run could not start, because an input
is missing or does not match its manifest, another `strata` process is running (`pgrep -x strata`; one engine per
Mac, [07](07-server-setup-port.md), section 6.8), or the engine did not start. CI treats 1 and 2 alike. The last
line of output is `accept <run>: PASS (<n> criteria, <minutes> min)`, or
`accept <run>: FAIL <criterion>: <value>, want <op> <threshold>` for the first failed criterion, or
`accept <run>: NOT RUN: <reason>`.

**Before any model loads**, a run checks:
- the prompt sets' `ids_sha256` against `tests/metal/prompts/MANIFEST.json`;
- every golden file it reads against the golden manifest (section 2.5);
- `test_models.py check --quick` for each model it uses;
- that the engine starts: `strata --version` and `strata --list-devices`.

**Engine processes.** Each engine run is a fresh process, started after the previous one has exited. Its command
line is the model's `config.json` args with the changes the run names, and nothing else. Every engine process
gets `STRATA_METAL_ARCHIVE=<out>/archive` (a folder per result, so no run reads another run's pipeline cache) unless
a run sets it itself. A step that runs longer than three times its expected time (measurement T-6), and at least 10
minutes, is sent SIGTERM, then SIGKILL 30 s later, and its criteria fail with the note `timeout`. A crash report
that appears in `~/Library/Logs/DiagnosticReports/` during a step is copied into the result folder.

**The result folder** is `<out>/<UTC yyyymmddThhmmssZ>-<version>-metal.<metal_rev>-<commit[:12]>/`:

```
result.json
steps/<nnn>.cmd       the argument list and every STRATA_* variable, one per line
steps/<nnn>.out       stdout
steps/<nnn>.err       stderr
steps/<nnn>.*         dumps (deleted after a pass unless --keep)
```

`result.json`, schema 1:

| Key | Content |
| --- | --- |
| `run`, `status`, `started`, `seconds` | The run, `pass`, `fail` or `not-run`, the UTC start, the duration |
| `engine` | `path`, `sha256` of the binary, `version`, `metal_rev`, `metallib_sha256` (from `strata --version` and BUILD.json, [08](08-build-and-distribution.md), section 8.4) |
| `source` | `commit` (the project's HEAD), `upstream_sync`, `prompts_builder_version` |
| `host` | `macos` (`sw_vers` product version and build), `chip` (`sysctl machdep.cpu.brand_string`), `memory_gib` (`hw.memsize`), `gpu_cores` (`system_profiler SPDisplaysDataType`), `power` (`pmset -g batt` source), `low_power_mode` |
| `models` | Per model: `tag` and the quick hashes of `check --quick` |
| `golden` | The folder and its manifest's `upstream_sync` |
| `criteria` | A list of `{id, value, op, threshold, pass, gating, note}`. `gating` false marks a reported-only number. |
| `steps` | A list of `{n, seconds, exit, model, prompt}` |

The run's folder is uploaded as a workflow artifact (`actions/upload-artifact`, kept 90 days) without the dumps. The
release run writes its folder under `$STRATA_RESULTS/release/<tag>/` as well, and that folder is kept on the
reference Mac for good.

**Criterion IDs.** `tools/metal/accept/criteria.py` holds one row per ID below: the operator, the threshold and the
gating flag. These are the IDs that `result.json`'s `criteria`, the `FAIL` line and the `accept:` checks of
`tests/metal/requirements.json` (section 9.4) use.

| Group | IDs | Section |
| --- | --- | --- |
| C1 | `C1.1.top1`, `C1.1.kl`, `C1.1.safe`, `C1.1.band`, `C1.1.ppl`, `C1.1.p99`, `C1.2`, `C1.3`, `C1.4.ids`, `C1.4.weights`, `C1.5` | 4.1 |
| C2 | `C2.b` to `C2.j`: each arm's output against arm (a)'s. `C2.d.accept`: arm (d)'s share of accepted drafts. | 4.2 |
| C3 | `C3.reuse`, `C3.pressure`, `C3.oversized`, `C3.exchange`, `C3.admission`, `C3.growth`, `C3.isolation`, `C3.soak`, `C3.http`, `C3.disabled`, `C3.reread`, `C3.repeat` | 4.3 |
| C4 | `C4.tiered`: the `--adapt-swaps 0` arm against full mode. `C4.swaps`: the arm with adaptive swaps against full mode, or against `C4.tiered`'s arm on a Mac without full mode. | 4.4 |
| Prompt path | `PP.top1`, `PP.kl`, `PP.ppl`, `PP.chunk` | 4.5 |
| Needles | `NEEDLE.32k`, `NEEDLE.128k`, `NEEDLE.262k`: every needle of that length found | 4.6 |
| Server | `S.models`, `S.status`, `S.openai`, `S.stream`, `S.tool`, `S.anthropic`, `S.image`, `S.cancel`, `S.unload` | 4.7 |
| Decode | `BENCH.mtp`, `BENCH.mtp.target`, `BENCH.plain`, `BENCH.other` | 5.1 |
| Prefill | `PF-1`, `PF-2` | 5.2 |
| Regression | `REG.decode`, `REG.prefill`, `REG.trend` | 5.4 |
| Robustness | `PRESSURE.smoke`: the `metal_memory_pressure_smoke` step of the Sunday slice | 5.5 |
| Gates | `G1.tf`, `G1.free`, `G2.speed` | 6 |
| Release | `PLAN.models`: step 1 of the release run | 8 |

**Planned runner tests.** The not-yet-created `tools/metal/test_accept.py` will run in the macOS Python test layer. It will use
`tools/metal/accept/fake_engine.py`, a script that takes the engine's arguments, prints the one-shot lines and writes
dump files, with differences planted on request (`FAKE_ENGINE_PLANT=<kind>:<where>`). Each criterion of sections 4
and 5 has a test that it passes on equal inputs and fails on its planted difference, with its message.

### 3.3 Comparing logits: `tools/metal/tf_compare.py`

```text
python3 tools/metal/tf_compare.py <reference.logits> <test.logits> --ids <set>.ids
                                  [--rows A:B] [--margins m0,m1] [--json OUT]
```

The old upstream comparison launcher embedded Windows checkout paths and is removed. `tf_compare.py` reads the two
logits files directly; it does not launch the engine. It keeps the eight-byte little-endian header and the F32 row
format used by the retained measurements.

- The header is `int32 n_vocab, int32 declared_rows`, followed by complete `n_vocab`-wide F32 rows. The file length
  determines how many rows were written. A short prompt-path dump begins at `declared_rows - rows_present`, as in
  section 3.1. Both dumps must cover the same positions and vocabulary.
- A row at position `p` predicts `ids[p + 1]`. Rows without a next-token ID are omitted; selected IDs outside the
  vocabulary fail with the token position. `--rows A:B` selects zero-based rows of the present dump, with B excluded.
- Log probabilities use stable row-wise F64 log-softmax. Same top-1 uses the lowest token ID on ties. KL is
  KL(reference ‖ test), summarized by mean, median, p99 and maximum. The p99 uses NumPy's 99th percentile.
- Perplexity is `exp(mean(-log p(next token)))` for each dump. The paired test-reference NLL delta is
  `mean(-log p_test + log p_reference)` in nats per token; its standard error is `sd / sqrt(n)` for at least two
  rows, and is reported as unavailable for a one-row comparison.
- Reference margins divide rows into `[0,m0)`, `[m0,m1)` and `[m1,infinity)`; defaults are 0.5 and 1.0. Every
  top-1 disagreement includes its position, reference margin, both top-1 IDs and both next-token log probabilities.
- NaN and infinity are rejected. The full-file validation and comparison read bounded row blocks so long-context
  dumps do not require both full vocabularies in memory at once.

`tools/metal/test_tf_compare.py` uses synthetic dumps with hand-checked outcomes. It covers the short-file rule,
equal inputs, paired deltas, top-1 ties and disagreements, row selection, vocabulary errors (including oversized IDs),
empty comma fields, non-finite values, chunk boundaries, one-row uncertainty reporting, and protection against JSON
outputs aliasing inputs through either the same path or a hard link.

---

## 4. Correctness criteria

### 4.1 C1: the Mac computes the same model as CUDA

C1 compares the Mac with the references of section 2. It gates on Q2_0 and Coder. The other formats run C1.1 on
`tf-code` in the weekly rotation (section 7.2).

**The Mac's settings.** The model's `config.json` args, with `--prompt-cache 0` and `--spec 0`, `--prefill` removed
except where a step names it, and `--adapt-swaps 0`. The reference Mac runs Q2_0 and Coder in full mode
([05](05-memory-and-residency.md), section 8). C4 (section 4.4) shows that tiered mode gives the same bits, so C1 is not
repeated there.

**The yardstick.** For each model and text t, from the golden set (measurements T-1 and T-3):
- the floor F(t): CUDA with `--expert-cache 4000` against CUDA with no expert cache (`tf-t.logits` against
  `tf-t.nocache.logits`). This is how much upstream's own valid paths differ;
- the cross reference L(t): CUDA against llama.cpp (`tf-t.logits` against `lc-tf-t.logits`);
- Y(t), the worse of the two: the lower same top-1, the higher median KL.

**The safe margin m₁.** The smallest margin above which the floor shows no flip in any text of either model, times
1.5, and at least 1.0 logit (measurement T-2). Upstream reports that most flips between its own paths have a margin
under 0.5 logits (`bench/results/2026-09-27-cache-parity/README.md`).

**C1.1, teacher-forced against CUDA.** For each `tf-*` set, the Mac runs
`--tokens-file <set>.ids --max-new 1 --dump-logits <file>`, and `tf_compare.py` compares it with `tf-<set>.logits`.

| Criterion | Pass when | Gating |
| --- | --- | --- |
| C1.1.top1 | Same top-1 ≥ Y(t) − 1.0 percentage point | yes |
| C1.1.kl | Median KL ≤ 1.5 × Y(t) | yes |
| C1.1.safe | Same top-1 is 100% in the band [m₁, ∞) | yes |
| C1.1.band | Same top-1 in the band [0.5, m₁) is at least the floor's in the same band minus 3 points | yes |
| C1.1.ppl | \|Δ\| ≤ max(3 standard errors, 0.005 nats per token) | yes |
| C1.1.p99 | p99 KL, against 2 × the floor's | no |

C1.1.p99 does not gate: the p99 of a text rests on its worst 1% of rows, so a few hard positions move it. It is
reported so that a change in the tail shows in review.

Until T-1 and T-3 are recorded, the thresholds are derived from upstream's published floor for Q2_0, and the same
numbers are used for Coder:

| Text | Upstream floor: same top-1, median KL | Provisional: same top-1 ≥, median KL ≤ |
| --- | --- | --- |
| `tf-code` | 97.7%, 3.0e-4 | 96.7%, 4.5e-4 |
| `tf-doc` | 96.2%, 4.5e-3 | 95.2%, 6.8e-3 |
| `tf-chat` | 95.0%, 4.4e-3 | 94.0%, 6.6e-3 |

The provisional m₁ is 1.0 logit. Upstream's numbers come from `--expert-cache auto` with a profile, not from
`--expert-cache 4000`, which is why T-1 measures the floor again with the golden set's own settings.

**C1.2, against llama.cpp.** The Mac's `tf-*` dumps against `lc-tf-*.logits`: same top-1 ≥ L(t)'s same top-1 − 1
point, and median KL ≤ 1.5 × L(t)'s median KL. The Mac must be no further from the independent implementation than
CUDA is. Gating once T-3 is recorded; reported until then.

**C1.3, the residual ladder.** The Mac runs `--tokens <the first 65 ids of the set> --max-new 1 --dump-layers <file>`
for each `tf-*` set. `tools/metal/ladder_compare.py <reference.layers> <test.layers> [--third <lc-ladder>]` computes,
for each slot s = 0..48, the median over the 64 positions of the relative L2 distance ‖test − ref‖₂ / ‖ref‖₂ over the
slot's 4 × 2,560 floats.
- The criterion: for every slot, the Mac-to-CUDA distance is at most the CUDA-to-llama.cpp distance at that slot, and
  at most 1e-6 at slot 0 (the embedding, which has no arithmetic but the dequantisation).
- Reported, not gating, until T-3 is recorded. This is the drift that R-5 ([03](03-model-reference.md), section 10)
  expects to grow smoothly with depth. A step of more than 4 times from one slot to the next is printed as a note;
  it is where a bisection starts (section 10.1).

**C1.4, the router replay.** The model-label test `metal_router_replay` runs for Q2_0 and Coder and is R-1's
measurement. The halves dump does not hold the router's input, so the test rebuilds it from CUDA's data, for each of
the 1,000 positions p and 48 layers l of the `router.*` files:
1. R = slot l of `router.layers` (R after layer l − 1, complete, because upstream dumps with the fused read off);
   `bo` = the attention half's block output (offset 0 of the halves record of layer l); `inject` = the attention
   `inject` (offset 5,120).
2. The Mac's FFN read on that input: `strata_gr_norm`, `strata_gr_down`, `strata_gr_up` with `apply = 1`
   ([04](04-metal-kernels.md), section 5.1, step 5), which gives `mixed`.
3. The router mat-vec and `strata_moe_route` on `mixed` give the ids and weights.
4. They are compared with the record of `router.routing` for (p, l). The records come in the order the token loop
   wrote them, one per layer and position.

| Criterion | Pass when |
| --- | --- |
| C1.4.ids | The id sets are equal at every (p, l), except that two experts may swap across the top-k boundary where the Mac's own k-th and (k+1)-th scores differ by at most 2^-16 relative. Such swaps are counted and reported. |
| C1.4.weights | For each id in both sets, \|w_mac − w_cuda\| ≤ 1e-5 · w_cuda |

The weight tolerance is R-1's provisional expectation. If R-1 measures a larger deviation for Coder's 256-expert
router, the tolerance becomes twice the measured maximum, recorded in appendix A, and
[03](03-model-reference.md), section 10, decides whether Coder needs the CPU FP64 normalisation.

**C1.5, the first token after the prompt path.** For each `g1-*` set the Mac runs
`--prefill auto --tokens-file <set>.ids --max-new 1` with `STRATA_DUMP_FIRST_LOGITS=<file>`, and compares the row with
`g1-XX.first`. It gates on the same top-1 where CUDA's margin is at least m₁; the KL is reported. This is the only check
of the prompt path that reaches every short prompt. The prompt-path criterion (section 4.5) checks a long one.

**The short form** (`accept.py c1 --short`): C1.1 on `tf-code` and C1.4, for Q2_0 and Coder. The nightly and the
release runs use it. The full form runs in the weekly rotation and whenever the compatibility baseline advances.

### 4.2 C2: speculation and every Mac path give the plain-decode tokens

C2 is Mac against Mac, so it is bitwise (P1): every arm's `output  :` line must equal arm (a)'s, character for
character. It is the acceptance check of the exactness contract of [06](06-engine-host-port.md), section 10.5, and
the "Done when" of [06](06-engine-host-port.md) work item H8.

**Prompts.** `g1-01` .. `g1-10`, `ca-1k`, `ca-4k` and `ca-32k`, each with `--max-new 256`, at the context of
section 2.1. **Base args**: the model's `config.json` args with `--prompt-cache 0`.

| Arm | Added to the base args | What it shows | Nightly |
| --- | --- | --- | --- |
| (a) | `--greedy --spec 0` | The reference: plain decode, T = 1 write-through windows | yes |
| (b) | `--greedy`, with the speculation settings of `config.json` (MTP and the suffix drafter) | The shipped configuration | yes |
| (c) | `--greedy --spec 8 --suffix-draft 0 --spec-min-p 0` | The widest MTP windows, drafted at every step | yes |
| (d) | `--greedy --spec 4 --spec-oracle <(a)'s output as a file>` | Accepted drafts: every window keeps all its rows. The `speculation` line must show at least 95% of the drafts accepted, or the arm is not testing what it should. | no |
| (e) | (d) with `--spec-corrupt 3` | Every third draft wrong: rollback after a partly accepted window | yes |
| (f) | `--greedy` with the MTP and `--spec-corrupt 2` | Rollback of the MTP drafter's own cells | no |
| (g) | `--seed 42 --temperature 0.7 --top-p 0.95`, `--spec 0`; then the same with the speculation of (b); then with `--coupled-draft` | Sampling: the output is a function of the seed, whatever the drafts were (rule 2). Each sampled arm is compared with the sampled `--spec 0` arm. | yes |
| (h) | (b) with `STRATA_METAL_ARCHIVE=0` | Pipelines compiled at start give the same bits as the archive's | no |
| (i) | (b) with `STRATA_METAL_DISPATCH=concurrent` | The concurrent encoders give the same bits ([04](04-metal-kernels.md), section 8.2) | no |
| (j) | (b) with the other value of `STRATA_METAL_WAIT` | The host's wait does not change anything | no |

- Q2_0 runs nightly on `g1-01..10` and `ca-4k` in the arms marked yes. Every arm and every prompt runs in the weekly
  rotation: Q2_0 and Coder on Tuesday, the three IQ formats on Wednesday (section 7.2).
- A difference fails with the first differing index, both tokens, and the window that produced it (from
  `--stats`). The bisection is in section 10.1.

### 4.3 C3: the conversation cache changes nothing but the time

C3 runs upstream's six conversation-cache tools unchanged (P7), with `--run`. Five of them take
`--config <model>/config.json`, `--engine` (the disabled tool takes `--upstream` and `--candidate` instead) and an
`--output` folder that must not exist yet. Each of the five loads private engines one after another, sets
`STRATA_STATE_HASH=1` itself, compares a cache-off baseline with a cache-on candidate token for token and
fingerprint for fingerprint, and exits non-zero on any difference. The Mac's job is to provide the same serve
protocol, `DONE` fields and fingerprint lines ([06](06-engine-host-port.md), section 11; [07](07-server-setup-port.md),
section 6), which is the "Done when" of [06](06-engine-host-port.md) work item H9.

| Check | Command (`python3 tools/<tool>.py --run --config <model>/config.json --engine <engine> --output <new folder>`, then) | When |
| --- | --- | --- |
| C3.reuse | `conversation_cache_parity.py --scenario reuse --spec 1`, then `--spec 4` | Nightly (Q2_0); release |
| C3.pressure | `--scenario pressure --cache-mib <cc_pressure_mib>` | Weekly |
| C3.oversized | `--scenario oversized --cache-mib 1` | Weekly |
| C3.exchange | `--scenario exchange --cache-mib <cc_pressure_mib>` | Weekly |
| C3.admission | `--scenario admission --min-free-mib <hw.memsize in MiB + 4096>`, a floor above any available memory | Weekly |
| C3.growth | `conversation_cache_growth.py --spec 1`, then `--spec 4` | Weekly |
| C3.isolation | `conversation_cache_isolation.py --scenario image`, `add`, `project` (synthetic images and control vectors; no encoder needed) | Weekly |
| C3.soak | `conversation_cache_soak.py` with its defaults (`--prompt-tokens 2048,40000,120000 --cycles 30`) and a config whose `--max-context` is 131,072 | Weekly |
| C3.http | `conversation_cache_http_smoke.py --run --url http://127.0.0.1:<port> --output <new file>` (no `--config` or `--engine`) against the server of section 4.7 | With the server run |
| C3.disabled | `conversation_cache_disabled.py --run --config … --upstream <previous release engine> --candidate <engine under test> --output <new folder>` | Release |

- `cc_pressure_mib` is a per-model field of `tests/metal/models.json`: a budget that holds each of the tool's
  snapshots but not two adjacent ones. The pressure scenario prints the range when it fails for want of pressure
  (`pressure_budget_hint`); measurement T-9 records the middle of that range for each model, and it is recorded
  again when a sync changes the size of a snapshot.
- **C3.disabled across releases.** Upstream's tool compares "untouched upstream" with cache-disabled builds. The
  Mac has no untouched upstream, so `--upstream` is the previous Mac release (section 5.4 says where it comes from).
  Its comparison is bitwise: tokens and fingerprints. A release whose kernels change rounding cannot pass it, so the
  annotated tag of such a release carries the line `Numerics: changed (<reason>)`. The run then records the failure
  as expected and C1 is that release's evidence. For the first release, `--upstream` is the engine under test.

Two checks of the Mac's own:

**C3.reread.** `STRATA_CKPT_REREAD=1` makes the serve loop read a checkpoint's tokens again from position 0, in one
batched read, instead of restoring it (`generate.cpp:4950-4983`); with the expert set fixed the answer must match.
No upstream tool sets it. The check runs `conversation_cache_parity.py --scenario reuse --spec 1` twice, the second
time with `STRATA_CKPT_REREAD=1` in the environment. The second run's verdict is not used, because its `reused`
counts differ by design. The check passes when the second log contains
`STRATA_CKPT_REREAD: reading <n> tokens again` at least once, and every candidate record of the second
`results.json` has the same `ids` and the same `state` as the first. On the Mac the batched read is independent of
the chunk size (`prefill_chunk_independent`), so the fingerprints must be bitwise equal. Nightly on Q2_0.

**C3.repeat.** The server of section 4.7, started with `--prompt-cache 0` added to the args, gets the same
non-streaming request (`g1-01`'s text, `temperature 0`, 64 tokens) 8 times, and `g1-02`'s 8 times with
`temperature 0.7` and `seed 42`. Each group must give one distinct answer. It runs in full mode, and in tiered mode
with `--gpu-budget-mib 37888` and `--adapt-swaps` as setup writes it, so that the experts move between requests.
On the Mac where an expert lives changes nothing (C4), so the answers must not change either. Upstream's CUDA build
does not promise this: its expert cache changes rounding (section 2.6).

### 4.4 C4: tiered mode gives the bits of full mode

C4 has two parts.
1. The model-label CTest tests `metal_full_vs_tiered` ([05](05-memory-and-residency.md), section 13: 50% and 90%
   residency, greedy and seeded, the tokens and `DONE` counts) and `tiered_eq_full` ([04](04-metal-kernels.md),
   section 10.4), for Q2_0 and Coder. They run in the nightly `ctest --preset mac-model` step.
2. `accept.py c4`: one-shot runs of `g1-01..10` and `ca-4k`, 256 tokens, greedy and with `--seed 42
   --temperature 0.7`, with the speculation of `config.json`, in three modes:
   - full mode (the reference);
   - `--gpu-budget-mib 37888 --adapt-swaps 0`, the 48 GB plan of [05](05-memory-and-residency.md), section 8.2;
   - `--gpu-budget-mib 37888` with `--adapt-swaps` as setup writes it, so that experts are swapped during the run.
   Every `output  :` line must equal full mode's.

This is the "Done when" of [06](06-engine-host-port.md) work item H10 and of [04](04-metal-kernels.md) work item L12.
On a Mac that cannot run the model in full mode, part 2 compares the two tiered arms with each other and records
`full mode not available on this Mac`.

### 4.5 The prompt-path criterion

The prompt path computes the same function as the token path with a different order of additions (chunked GEMMs,
[04](04-metal-kernels.md), section 6), so it is a tolerance comparison against CUDA, measured the way upstream measured
its own prompt-path changes (`bench/results/2026-09-28-prefill-speed/README.md`): the 32K prompt goes through the
prompt path up to its last 256 tokens, and those go through the token path with their logits dumped.

- The Mac runs `--prefill auto --prefill-until 32000 --tokens-file ca-32k.ids --max-new 1 --dump-logits <file>` at
  `--max-context 32768`. The dump holds the rows from position 32,000 on (section 3.1).
- `tf_compare.py golden/pp-32k.logits <file> --ids ca-32k.ids` compares the 255 rows whose next token is in the
  prompt.
- **The yardstick** Y_pp is CUDA against CUDA with another chunk size: `pp-32k.logits` against `pp-32k.c6144.logits`
  (T-1). Upstream measured 89.8% same top-1 and a mean KL of 0.33 for the same change (chunk 8,192 against 6,144),
  which are the provisional values. The CUDA run's chunk size is in its manifest command entry; if `auto` chose
  6,144 there, the yardstick file is recorded at 4,096 instead.

| Criterion | Pass when | Gating |
| --- | --- | --- |
| PP.top1 | Same top-1 ≥ Y_pp − 3 points (256 rows: one row is 0.4 points) | yes |
| PP.kl | Mean KL ≤ 1.5 × Y_pp's mean KL | yes |
| PP.ppl | The Mac's perplexity on the 255 rows within 3% of CUDA's | yes |
| PP.chunk | The chunk size the Mac chose, from its stderr line | no |

Mac against Mac across chunk sizes is bitwise, not a tolerance: that is the model-label test
`prefill_chunk_independent` ([04](04-metal-kernels.md), section 10.4). This criterion is the last condition of
[04](04-metal-kernels.md) work item L13. It runs nightly on Q2_0, and for Coder in the weekly rotation (Friday).

### 4.6 Needles

`tools/needle_bench.py` runs unchanged against the server of section 4.7, started with a config whose
`--max-context` holds the length: `needle_bench.py --url http://127.0.0.1:<port> --lengths <L> --depths <D>
--out <result folder>/needles.json`. The tool builds its text from the checkout's own files (docs, sources, and
the llama.cpp docs under `third_party/llama.cpp/docs` when they are present), asks for the hidden word with thinking off and greedy decoding, and exits 0 only when every
needle is found.

| Length | Depths | Context | When |
| --- | --- | --- | --- |
| `32k` | 10, 50, 90 | 32,768 | Nightly (Q2_0); release |
| `128k` | 10, 90 | 131,072 | Weekly (Saturday) |
| `262k` | 50 | 262,144 | Weekly (Saturday), on a Mac whose plan holds that context |

The tool skips a length the server cannot hold and prints it. The wrapper fails a skipped length that the schedule
requires, so a skip cannot pass as a success. Upstream reports 5 of 5 at 1K, 32K, 128K (two depths) and 262K with
Q2_0 (`bench/results/2026-09-28-prefill-speed/README.md`). The Mac must find every needle it is asked for.

### 4.7 The server end to end

`accept.py server` starts the server as setup's run script does,
`python3 serve/server.py --engine strata --config <model>/config.json --port <free port>`, with `--engine`'s
binary placed in the config's `exe` and `--vision`'s in `vision.exe`. It waits for `/health`, then checks:

| Check | Request | Pass when |
| --- | --- | --- |
| S.models | `GET /v1/models` | The config's `model_name` is listed |
| S.status | `GET /v1/status`, `GET /metrics` | Both answer 200 with JSON; `/v1/status` names the Metal engine and its plan |
| S.openai | `POST /v1/chat/completions` with each `g1-*` prompt's messages, non-streaming, `temperature 0`, `max_tokens` 256 | Each answer contains the "Expected in the answer" of section 2.2; `usage.prompt_tokens` equals the set's token count |
| S.stream | The same request for `g1-01` with `stream: true` | The concatenated deltas equal the non-streaming answer; the stream ends with `data: [DONE]` |
| S.tool | `g1-06`'s request with its `tools` | `finish_reason` is `tool_calls`, with a call to `get_weather` whose `city` argument is Osaka |
| S.anthropic | `POST /v1/messages` with `g1-02`'s prompt, then `POST /v1/messages/count_tokens` | The answer contains `360`; the count equals the set's token count |
| S.image | `POST /v1/chat/completions` with `tests/metal/images/red.png` (448 × 448, one colour, `#D02020`) as a `data:` URL and "What colour is this image? Answer with one word.", then with `tests/metal/images/osaka.png` (the word OSAKA in black on white, 64-pixel letters) and "What word is written in this image?"; `temperature 0` | The answers contain `red` and `osaka` (any case). The server log shows the encoder started on the GPU (`vision.gpu` true in the config). |
| S.cancel | `g1-07` streamed, the connection closed after 8 deltas, then `g1-02` | The second request answers correctly within 30 s, and `/v1/status` shows no request running |
| S.unload | `POST /v1/unload`, then `POST /v1/load`, each with `Content-Type: application/json`, body `{}` and no `Origin` header; then `g1-02` | The engine process exits and a new one starts; the answer is correct |
| C3.http | `conversation_cache_http_smoke.py` (section 4.3) | Its exit code is 0 |
| C3.repeat | Section 4.3 | One distinct answer per group |

The server's log is kept in the result folder. The run records the engine's pid (`pgrep -x strata`) before and
after each check; an engine that exits anywhere but in S.unload fails the run, with the last lines of the server's
log. The browser page and the menu bar item are not tested here: they are [07](07-server-setup-port.md) work item S14's screenshots,
reviewed by a person (section 7.6).

---

## 5. Performance criteria

### 5.1 The decode benchmark

`accept.py bench` measures what a user gets: the engine as setup configures it, on the code-agent prompts that
upstream's speed tables use (`bench/results/2026-09-29-speed-0126/README.md`: setup's args, MTP, greedy, 256
generated tokens).

**Arms.** For each model and prompt:

| Arm | Args | Reports |
| --- | --- | --- |
| `mtp` | The model's `config.json` args as setup writes them on this Mac, with `--greedy --max-new 256 --prompt-cache 0` | `decode` and `prefill` rates, time to first token, the `speculation` line's acceptance |
| `plain` | The same with `--spec 0 --suffix-draft 0` | `decode` rate |

**Prompts.** `ca-1k`, `ca-4k` and `ca-32k` nightly; `ca-64k` and `ca-128k` weekly (Thursday); `ca-262k` weekly
(Saturday) on a Mac whose plan holds a 262,144 context. Each runs at the context of section 2.1.

**Method.**
1. One warm-up run, then three measured runs, each a fresh process (T-4 confirms three).
2. The result of an arm is the median of the three runs. The spread, (max − min) / median, is reported with it.
3. If the spread is above 5%, the three runs are made once more and the median of all six is used. A spread still
   above 5% is recorded as the note `noisy`. A noisy result cannot fail the regression check of section 5.4; it is
   reported, and two noisy nights in a row are filed as a bug (section 10.2).
4. If T-5 finds thermal drift, a pause of the length it records goes between runs.

**Before the first run** the benchmark checks the host and records each value in `result.json`. A failed check
stops the run with exit 2, rather than measuring a slowed Mac:
- `pmset -g batt` reports `AC Power`;
- `pmset -g` reports `lowpowermode 0` (or `powermode 0`, where macOS names the key so);
- `pmset -g therm` reports no thermal warning and no performance warning;
- no other `strata` process is running (`pgrep -x strata`);
- `sysctl -n kern.memorystatus_level` is at least 50.

The reference Mac's runner also has a fixed setup. Its displays sleep, and nobody uses it interactively. Spotlight and
Time Machine exclude `/Users/Shared/strata-models` and `STRATA_RESULTS`. Automatic software updates are off
([08](08-build-and-distribution.md), section 12.4).

**Criteria.** The absolute numbers are requirements of [01-requirements.md](01-requirements.md), checked here on the
reference Mac (M5 Pro, 64 GB) with Q2_0 in full mode:

| Criterion | Pass when | Gating |
| --- | --- | --- |
| BENCH.mtp | `mtp` decode on `ca-4k` ≥ 50 tok/s | At G3 and in the release run |
| BENCH.mtp.target | `mtp` decode on `ca-4k` ≥ 60 tok/s | No: the target, reported |
| BENCH.plain | `plain` decode on `ca-4k` ≥ 20 tok/s | At G1, at G3 and in the release run |
| BENCH.other | Every other model, prompt and arm | No: reported, and compared by section 5.4 |

### 5.2 Prefill speed: PF-1 and PF-2

The rate comes from the `prefill` line of the benchmark's `mtp` arm, which goes through `--prefill auto` as setup
writes it. The chunk size the engine chose is read from its stderr line and recorded.

| Criterion | Pass when | When |
| --- | --- | --- |
| PF-2 | The `ca-32k` prefill rate ≥ 450 tok/s, on the reference Mac, Q2_0, full mode | Nightly; release |
| PF-1 | The `ca-128k` prefill rate ≥ 70% of the `ca-32k` rate, on the same Mac and model, in the same run | Weekly (Thursday); release, from the latest Thursday run of the same engine sources (section 8) |

- PF-2 is the low end of the projection in [00-feasibility-study.md](00-feasibility-study.md), section 3.3.
- PF-1 checks that the prompt path does not slow down with depth. Upstream keeps 97% (2,107 against 2,171 tok/s on
  the RTX 5070).
- A PF-1 failure starts work item L19 of [04](04-metal-kernels.md), section 12.2: the batched prompt attention.
- Time to first token on `ca-32k` is reported with PF-2.

### 5.3 Comparison with upstream

The result file puts the Mac's `ca-*` numbers next to upstream's for the same model and length, from the table of
`bench/results/2026-09-29-speed-0126/README.md` (for 262K, `2026-09-29-speed-0122`). The upstream numbers are
copied into `tests/metal/upstream_speed.json` with their source file. No criterion depends on them (section 2.6),
but the release notes quote both.

### 5.4 The regression check

Every nightly run, and the release run, compares the engine under test with the previous Mac release on the same
Mac, in the same run:
- **The previous release.** It is the highest published `v*-metal.*` release below the engine under test
  (`gh release list --exclude-drafts`). `tools/metal/accept/engines.py` downloads its `strata-macos-arm64.zip` and
  `SHA256SUMS` once (`gh release download <tag>`), checks the hash, and unpacks it into
  `$STRATA_RESULTS/engines/<tag>/`, which it then makes read-only. C3.disabled (section 4.3) uses the same engine.
- **Order.** The warm-up runs of the previous engine and of the engine under test come first. The measured runs then
  alternate: previous, under test, three times. Drift during the run then falls on both engines.
- **Args.** The same `config.json` args for both. If the previous engine refuses them (`unknown argument: <flag>`,
  which upstream's parser makes an error, `generate.cpp:1248-1250`), or exits while loading the model or pack, the
  check is skipped. The skip is reported with the previous engine's last stderr line.
- **The bar.** It is 0.95, unless T-4's coefficient of variation is above 2%. Then it is 1 − 2.5 × the coefficient.

| Criterion | Pass when | Gating |
| --- | --- | --- |
| REG.decode | For each of `ca-1k`, `ca-4k`, `ca-32k`: the `mtp` decode median ≥ the bar × the previous release's | Yes, unless skipped or noisy |
| REG.prefill | The `ca-32k` prefill median ≥ the bar × the previous release's | Yes, unless skipped or noisy |
| REG.trend | The change against the previous night's result of the engine under test | No: a change of more than 3% is printed |

Before the first release there is no previous engine. REG.* is then skipped with the note `no previous release`.

### 5.5 Robustness

| Test | Label or run | What it does | Pass when |
| --- | --- | --- | --- |
| `metal_gpu_time_test` | model | Records `GPUEndTime − GPUStartTime` of every command buffer during T = 1..8 decode windows (full mode, and tiered with `--gpu-budget-mib 37888`), and during `ca-32k` with `--prefill auto`. On a Mac whose plan allows it, also during a prompt at `--prefill 32768`. | Until L-12 ([04](04-metal-kernels.md), section 11) records the limit: no `MTLCommandBufferErrorTimeout`, and the times are reported. After L-12: every command buffer below half the limit. A T = 8 window is also below 1 s, far from any limit. |
| `metal_watchdog_test` | model | `strata serve` with `STRATA_TEST_VERIFY_STALL=3`, then one request of `g1-02`, in full mode and in tiered mode; then the same with `STRATA_WATCHDOG_S=5` as well ([06](06-engine-host-port.md), section 14, and work item H11) | Without the watchdog limit: the engine prints `ERR gpu error: window timeout` and exits 1 within 20 s + 5 s, measured from the request to the exit. With it: the engine prints the issue #29 line, the stall report and the crash-report hint, and ends by SIGABRT within 5 s + 5 s; a new `strata-*.ips` file is in `~/Library/Logs/DiagnosticReports/`. Each case in full and in tiered mode. |
| H-5 | Measurement | [06](06-engine-host-port.md), section 15 | Recorded once in appendix A, and again when tiered encoding changes |
| `metal_memory_pressure_smoke` | Weekly (Sunday) | The M-4 procedure of [05](05-memory-and-residency.md), section 10 | Decoding continues while a second process fills memory. The run records `vm_stat`, the footprint and the decode rate, and must end with no GPU error and no engine exit. |
| C3.soak | Weekly (Monday) | Section 4.3 | Its exit code is 0 |
| The server's engine restarts | With the server run | Section 4.7, S.cancel and S.unload | As stated there |

---

## 6. The gates

[00-feasibility-study.md](00-feasibility-study.md), section 7, names three gates. Each is a run of `accept.py`
with exit codes, and its result folder is the gate's evidence. `docs/mac/gates/G<n>.md` in the project records the
decision: the result folder's name, pass or fail, and what was decided when it failed.

### 6.1 G1: end of phase 2

The question: does the Mac compute the model correctly, at a speed that makes the rest worth doing?
`accept.py g1`, on the reference Mac with Q2_0 in full mode:

| Step | Criterion | Pass when |
| --- | --- | --- |
| 1 | `ctest --preset mac-model` | Every test passes |
| 2 | C1 in full (section 4.1) | C1.1, C1.4 and C1.5 pass. C1.2 and C1.3 are reported, and gate once T-3 is recorded. |
| 3 | G1.tf | For each `g1-*` set, the Mac runs `--tokens-file <set + g1-XX.out> --max-new 1 --dump-logits` and compares with `g1-XX.tf.logits` over the 256 generated positions. Same top-1 is 100% where CUDA's margin is at least m₁, and at least 95% over all 2,560 positions. |
| 4 | G1.free | For each `g1-*` set, the Mac's `--greedy --spec 0 --max-new 256` output equals `g1-XX.out`, or first differs at a position where CUDA's margin is below m₁. After the first difference the two texts are not compared, because they legitimately diverge from there. |
| 5 | C2 (section 4.2), Q2_0, the nightly arms | Every arm equals arm (a) |
| 6 | BENCH.plain (section 5.1) | ≥ 20 tok/s |
| 7 | R-1 and R-5 ([03](03-model-reference.md), section 10) | Recorded in appendix A |

This is how doc 00's "greedy output matches the reference for 256 tokens on 10 prompts" is made exact. CUDA is not
bitwise repeatable against itself (section 2.6). So "matches" means: equal wherever the reference's own choice is
safe, and teacher-forced agreement as close as the reference's own noise.

If G1 fails:
- **On correctness:** bisect (section 10.1). Phase 3 does not start until G1 passes.
- **On speed between 15 and 20 tok/s:** profile with a GPU capture ([08](08-build-and-distribution.md),
  section 10.2) and the H-4, L-13 and L-14 measurements. G1 runs again after the fixes.
- **Below 15 tok/s with no clear cause:** doc 00's stop rule applies. The decision and the profile go into
  `docs/mac/gates/G1.md`, and the work goes to llama.cpp instead.

### 6.2 G2: end of phase 4

The question: is the 48 GB class worth shipping? `accept.py g2`:

| Step | Criterion | Pass when |
| --- | --- | --- |
| 1 | G2.speed | Q2_0 with MTP, in tiered mode, on `ca-4k` at `--max-context 8192`, with the config setup writes for the Mac below: `mtp` decode median ≥ 20 tok/s (method of section 5.1) |
| 2 | C4 (section 4.4) | Passes at `--gpu-budget-mib 37888` |
| 3 | M-9 | The hit rate at that budget is reported, next to [05](05-memory-and-residency.md)'s forecast |

Where it runs:
- **On a 48 GB Mac (M5 Pro 48 GB, or any Apple10 Mac with 48 GB).** As setup configures it. This is the
  authoritative form.
- **On the 64 GB reference Mac, until a 48 GB Mac is available.**
  - The run uses `--gpu-budget-mib 37888` (37 GiB), so that the plan is the 48 GB plan of
    [05](05-memory-and-residency.md), section 8.2.
  - The plan alone is not enough. In tiered mode the page cache is the RAM tier ([05](05-memory-and-residency.md),
    section 5.2), and a 64 GB Mac has 16 GiB more of it. The run therefore also starts `tools/metal/balloon.py --gib 16`:
    - The balloon maps 16 GiB of anonymous memory, fills it with random bytes so that the memory compressor
      cannot shrink it, and writes one byte of each 16 KiB page every 5 s so that the pages stay active.
    - With `--lock`, used when M-7 shows that `mlock` holds 16 GiB, it also wires them.
    - It prints `balloon: 16 GiB held` and holds until SIGTERM.
  - The run reads `vm_stat`'s swap-outs before and after. If they grew by more than 1 GiB, the balloon did not hold
    its memory, and the run ends with exit 2.
  - The result is marked `64 GB with a 16 GiB balloon`. It counts for G2 until a run on a 48 GB Mac replaces it.

If G2 fails, doc 00's fallback applies: the 48 GB class ships as beta, and the work turns to product parity and long
context on 64 GB. That decision is recorded in `docs/mac/gates/G2.md`.

### 6.3 G3: end of phase 6

The question: is the port a release? `accept.py g3` passes when all of these hold:
- **Every requirement holds.** Each requirement of [01-requirements.md](01-requirements.md) is listed in
  `tests/metal/requirements.json` with its checks. A check is an `accept.py` criterion, a CTest test, a Python test
  module, a CI job, or a review of section 7.6 (section 9.4, check 3). Every check has passed on the candidate
  commit. A check "passed on the candidate commit" when a result file shows it passing on a commit whose engine
  sources (`src/`, `include/`, `CMakeLists.txt`, `cmake/`) and whose `serve/`, `tools/` and `setup.py` are identical
  to the candidate's (`git diff --quiet`). For a review, it is the review's record (section 7.6). For a CI job
  `ci:<workflow>/<job>`, it is a run of `<workflow>` on such a commit, found with
  `gh run list --workflow <workflow> --commit <sha> --json databaseId,conclusion`, in which every job named `<job>`
  (for example, the current `mac-host` job) has the conclusion `success` in `gh run view <id> --json jobs`. The
  planned `mac-release.yml/verify` job does not exist yet; release reproducibility cannot pass until that workflow and
  its release tools are implemented ([08](08-build-and-distribution.md), section 12.3).
- **The release run passes.** The release run of section 8 passes on a release-candidate tag.
- **The family runs pass.** Section 7.4 has a passing family run for each of M1-M4 on the candidate commit.

`accept.py g3` starts with `test_map.py --final` (section 9.4). It fails G3 when a test is still planned, when a
requirement has no check, or when a check it names does not exist.

If G3 fails, the release goes out as a beta, as doc 00 says. Its notes list each requirement whose check failed or
did not run, with the check's name.

---

## 7. Where and when the tests run

### 7.1 On every push

| Where | What | Defined in |
| --- | --- | --- |
| GitHub-hosted macOS arm64 runner (`ci.yml`) | The Python layer through `run_python_tests.py`, `ctest --preset mac-host`, and the host-independent Metal tooling checks | [08](08-build-and-distribution.md), section 12.1 |
| The reference Mac (`mac-hardware.yml`, push to main) | `ctest --preset mac-host` and `ctest --preset mac-metal` | [08](08-build-and-distribution.md), section 12.2 |

### 7.2 Nightly and weekly

**Nightly.** `mac-hardware.yml`'s scheduled run (`17 3 * * *`) builds `mac-dev` and runs the host and metal
labels ([08](08-build-and-distribution.md), section 12.2). Then:
1. `ctest --preset mac-model`.
2. The validation-layer run: `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 ctest --preset mac-metal`
   ([04](04-metal-kernels.md), section 10.1).
3. The Python layer on the built engine. The job builds `strata-vision` into `build-vision/` with
   `setup.metal_vision_defs` ([08](08-build-and-distribution.md), sections 8.2 and 9). It installs
   `build-dev/strata`, `build-dev/strata.metallib` and `build-vision/bin/strata-vision` into the checkout's
   `engine/` with the copy loop of [08](08-build-and-distribution.md), section 13, and runs
   `run_python_tests.py` with `STRATA_TEST_MODELS` set. This runs `test_geometry_plans` and
   `test_geometry_equals_files` ([07](07-server-setup-port.md), section 10.3), which a hosted Mac skips: the
   parity run of [07](07-server-setup-port.md), section 4.4.3.
4. `accept.py nightly`: the nightly core, at most 2 hours. In order:

| Run | Content |
| --- | --- |
| `c1 --short` | C1.1 on `tf-code` and C1.4, Q2_0 and Coder (section 4.1) |
| `c2` | Q2_0, the arms marked nightly (section 4.2) |
| `c3` | C3.reuse (`--spec 1` and `--spec 4`) and C3.reread, Q2_0 (section 4.3) |
| `c4` | Q2_0, `g1-01..03` and `ca-4k` (section 4.4); the model tests of C4 run in step 1 |
| `pp` | Q2_0 (section 4.5) |
| `bench` | Q2_0, `ca-1k`, `ca-4k`, `ca-32k`, with the regression check; PF-2 (sections 5.1, 5.2, 5.4) |
| `needle` | 32K, Q2_0 (section 4.6) |
| `server` | Q2_0, every check of section 4.7, with C3.http and C3.repeat |

The five steps fit `timeout-minutes: 240`. T-6 measures each run. If the nightly core grows past 2 hours, runs move
to the weekly slices below by a change to this table, never by a timeout.

**Weekly.** A second job of the same workflow, `weekly`, runs one slice of the weekly rotation each day:
- `needs: hardware`, on the same runner, with `if: github.event_name == 'schedule'` and `timeout-minutes: 600`;
- it checks out the same commit with `clean: false` and rebuilds `mac-dev`, which finds the first job's build up to
  date;
- the command is `accept.py weekly`, which picks the slice by `date -u +%u`;
- `workflow_dispatch` takes an input `day` (1-7) to run a chosen slice on demand.

The job runs after the nightly job has finished, so the two never load models at the same time. The patch to
[08](08-build-and-distribution.md), section 12.2, is work item T14.

| Day (`%u`) | Slice |
| --- | --- |
| 1 Monday | C3: pressure, oversized, exchange, admission, growth (`--spec 1` and `4`), isolation (three scenarios), soak; Q2_0, with T-9's budget |
| 2 Tuesday | C2, every arm, Q2_0 and Coder; C4 in full, Q2_0 and Coder |
| 3 Wednesday | C2, every arm, IQ2_XS, IQ3_XXS and IQ3_S; C1.1 on `tf-code` for the same three |
| 4 Thursday | `bench` on `ca-64k` and `ca-128k` for Q2_0 and Coder; PF-1 |
| 5 Friday | C1 in full (C1.1-C1.5) for Q2_0 and Coder; `pp` for Coder |
| 6 Saturday | `needle` at 128K (two depths) and 262K; `bench` on `ca-262k`. A Mac whose plan does not hold 262,144 records `not-run` for the 262K parts. |
| 7 Sunday | `metal_memory_pressure_smoke`; `test_models.py check` (full SHA-256 of every model); the `metal_<area>_model_test` set under the validation layer |

The slice's result folder goes under `$STRATA_RESULTS/weekly/`.

### 7.3 After an upstream compatibility change

The review process is [10-implementation-plan.md](10-implementation-plan.md), section 4. Run these checks on the
project branch before merging the local change:
1. Every push-level test (section 7.1).
2. If section 2.5's rule says so, record the golden set again on the CUDA PC for the local project commit.
3. C1 in full against the new golden set (or against the old one, if the rule keeps it), for Q2_0 and Coder.
4. `accept.py nightly` and the Tuesday and Friday slices, run on demand.
5. T-9 again, if the compatibility change altered a snapshot's size (section 4.3).

### 7.4 Family runs: M1 to M4

The reference Mac is an M5 (Apple10). The port also supports M1-M4 ([00-feasibility-study.md](00-feasibility-study.md)),
whose GPUs differ in `simd_sum` order, tile defaults, `mpp` support and memory sizes. Kernel results there need their
own evidence.

`tools/metal/family_run.sh <commit> [<asset zip>]` runs on one Mac of each family:
1. `git worktree add` at the commit. Then the toolchain check, and a `mac-dev` build
   ([08](08-build-and-distribution.md), section 13).
2. `ctest --preset mac-host` and `ctest --preset mac-metal`, then the validation-layer run of the `metal` label.
3. `strata --list-devices`; the output is kept.
4. `accept.py family`, if the Mac has at least 32 GB. Its `--engine` is the `strata` of the asset zip, unpacked into a
   temporary folder, when one is given, and the `mac-dev` build otherwise. It runs:
   - the models its plan holds (Q2_0 in tiered mode, at least);
   - C2's nightly arms;
   - C4 between its two tiered arms (section 4.4);
   - `bench` on `ca-1k` and `ca-4k`;
   - the needle at 32K.
5. Writes `family-<chip>-<commit[:12]>.json`: the CTest results, the device lines, the `accept.py` result and, when
   an asset was given, its SHA-256.

Family runs are made before each release, with the draft's asset (`gh release download <tag> --repo {REPOSITORY}`), so that
the engine REQ-PLT-03 asks about is the one that runs. They are also made after any change under `src/metal/kernels/`
or to `src/metal/manifest`, at least once per change set, without an asset. Their result files are attached to the
release-tracking issue. The reviewer of the `release` environment ([08](08-build-and-distribution.md), section 12.3)
checks them before approving (section 7.6). Family Macs are not CI runners: they have no runner service and no
repository secrets. Where they come from is an open question of
[11-risks-and-open-questions.md](11-risks-and-open-questions.md).

The measurements that say "each family" in [04](04-metal-kernels.md), section 11, run on the same Macs, as separate
programs.

### 7.5 Hosted Macs (measurement X-8)

A GitHub-hosted macOS arm64 runner may expose a paravirtualised GPU. The current `ci.yml` workflow is push-only;
its Metal self-test is explicitly best-effort with `continue-on-error`. The separate hardware workflow runs the
Metal suite on the reference Mac for pushes to `main` and on its scheduled run. No current workflow tests pull
requests. X-8 ([07](07-server-setup-port.md), section 14) is a future measurement for deciding whether hosted Metal
tests can become a required check. A hosted run is not a family run, and no model test runs there.

### 7.6 Reviews that need a person

Every other criterion is a command (P6). These need a person, and each leaves a record:

| Review | Who | Record |
| --- | --- | --- |
| The family results of a release (section 7.4) | The `release` environment's reviewer | The approval of the deployment, with a comment that names the family files |
| Monitor and About screenshots ([07](07-server-setup-port.md), work item S14) | A reviewer | The screenshots and an approving comment on the pull request |
| The planned Mac setup guide alone (S20), followed by a double-click on `START-HERE.command` in Finder | A reviewer who has not read the port documents | A comment with the Mac, the macOS build and the result |
| The `strata-ref` runner's rules ([08](08-build-and-distribution.md), work item B21) | A second person | A checklist issue, closed |
| The licence table ([08](08-build-and-distribution.md), work item B20) | A reviewer | An approving comment on the pull request |

`requirements.json` names a review by its row here, as `review:family`, `review:S14`, `review:S20`, `review:B21` or
`review:B20`. `accept.py g3` reads its record from the issue or pull request named in `tests/metal/reviews.json`, in
which the reviewer adds an entry with the commit and the link (section 9.4).

---

## 8. The release acceptance runs

Step 6 of `verify_release.py` ([08](08-build-and-distribution.md), section 11.8) runs
`accept.py release --engine <installed engine> --out $STRATA_RESULTS/release/<tag>/` on the reference Mac. The
engine under test is the one setup installed from the draft release, not a build of the checkout. It should take
about 2 hours.

| Step | Check | Section |
| --- | --- | --- |
| 1 | `strata --plan-only` for every model of `models.json` that the Mac has, and `strata --plan-only --geometry` for every file of `data/geometry/`: a PLAN line, exit 0, within the time limits of [06](06-engine-host-port.md), section 2.3 (1 s from the files, 200 ms from a geometry file) | 3.1 |
| 2 | C1 in its short form, Q2_0 and Coder | 4.1 |
| 3 | G1.tf (section 6.1, step 3), Q2_0 | 6.1 |
| 4 | C2, Q2_0, the nightly arms | 4.2 |
| 5 | C4 at `--gpu-budget-mib 37888`, Q2_0 | 4.4 |
| 6 | C3.reuse, C3.disabled against the previous release, and C3.http with the server of step 11 | 4.3 |
| 7 | `bench` on `ca-1k`, `ca-4k` and `ca-32k`, with BENCH.mtp, BENCH.plain and the regression check | 5.1, 5.4 |
| 8 | PF-2; PF-1 from the latest Thursday slice whose engine sources equal the tag's | 5.2 |
| 9 | The 32K needles | 4.6 |
| 10 | The weekly slices: each of the seven has passed on a commit whose engine sources equal the tag's (section 6.3's rule) | 7.2 |
| 11 | The server run | 4.7 |

- **If the kernels' rounding changed since the previous release**, C3.disabled cannot pass. Such a release is
  recognised by the line `Numerics: changed (<reason>)` in its annotated tag's message
  ([08](08-build-and-distribution.md), section 11.1). The run then records C3.disabled as an expected failure, and
  steps 2 and 3 are that release's evidence. Without the line, a C3.disabled failure fails the release.
- **Step 10's weekly slices can be run on demand** before tagging (`workflow_dispatch` with `day`). The usual
  release therefore waits for the seven slices of its commit, or runs the missing ones.
- **Any failed step** makes `verify_release.py` exit 1, and the release stays a draft. The result folder stays on
  the reference Mac for good, and the release-tracking issue links it.

---

## 9. The test map

This section lists every test that the documents define, with its label, its origin and what it serves. Names are
the CTest names. Where a work item of another document describes a test without naming it, the name here is its
name, and that work item cites it. A property of [04](04-metal-kernels.md), section 10.4, is a CTest test of the same
name, except `row_t_independent`, which is step 6 of every twin and has no test of its own.

### 9.1 CTest tests

With every work item done, a `mac-dev` build registers 187 tests: 47 `host`, 112 `metal` and 28 `model`. The 28 tests
of the bootstrap ([08](08-build-and-distribution.md), section 6.4) are among them.

| Label | Preset | Runs on |
| --- | --- | --- |
| `host` | `mac-host` | Every push, on every runner; the family runs |
| `metal` | `mac-metal` | The reference Mac on every push to `main`; the nightly validation-layer run; the family runs; hosted Macs if X-8 allows (section 7.5) |
| `model` | `mac-model` | The reference Mac, nightly (section 7.2, step 1); the `metal_<area>_model_test` set also under the validation layer on Sunday |

**`host` (47).**

| Tests | Defined in | What it serves |
| --- | --- | --- |
| `conversation_cache_test`, `gguf_reader_test`, `gguf_split_test`, `suffix_drafter_test`, `controller_test`, `draft_policy_test`, `conv_cache_test`, `coupled_draft_test`, `bf16_bits_test`, `direct_file_async_test` | Mac host tests, labelled by `_strata_label_rest` ([08](08-build-and-distribution.md), sections 6.2 and 6.4) | Host-side unit tests. `direct_file_async_test` is one of [06](06-engine-host-port.md) work item H2's platform tests. |
| `platform_memory_test`, `ple_reader_selftest` | [08](08-build-and-distribution.md), section 6.4 | Mac twins of the upstream tests in the CUDA block. `platform_memory_test` is H2's other platform test. |
| `ple_oracle_layout_test` | [03](03-model-reference.md), work item R9 | A compact synthetic GGUF checks standalone Direct reads and a split-shard mapped PLE tensor at a nonzero parsed offset, with known IQ4_NL values |
| `dequant_oracle_test`, `metal_manifest_test` | [08](08-build-and-distribution.md), section 6.4 | The pattern of the parity tests; the manifest against the built library |
| `sampler_tail_test` | [04](04-metal-kernels.md), section 5.12 | Philox known answers, strict inverse-CDF boundary, top-p/min-p order, temperature ordering, CPU FP64 tail cases, and upstream fixture 15's host penalty-history rows |
| `metal_moe_real_inputs_unavailable` | [04](04-metal-kernels.md), section 10.5 | Confirms that an explicit `metal_moe_test --input-set real` invocation returns 77 and CTest reports it as skipped until REV-002's model pack and CUDA dump fixture are available |
| `strata_version` | [08](08-build-and-distribution.md), section 6.3 | The engine's identity |
| `model_metadata_test`, `model_types_test`, `ple_constants_test`, `native_experts_reader_test`, `pack_index_test`, `expert_address_test` | [03](03-model-reference.md), work items R2-R4 and R6-R8 | The seven header fixtures' metadata and tensor tables are parsed by the production C++ reader from compact GGUF headers; Python checks every tensor role/type, block geometry and FF = 640. `ple_constants_test` matches each fixture's PLE metadata and table shape to the n-gram kernel constants, compares the C++ parameter block and compile-time-asserts the Metal parameter block, checks the head prefix sums and 90 trailing table rows, and rejects value or JSON-type mutations, including a mistyped tokenizer EOS, with the key named. Sparse WeightStore negatives cover changed metadata keys, Q5_0 expert downs and partial quantization blocks. R6 builds a compact 1,224-tensor model in a temporary directory and exercises the production expert-table reader, so it needs no downloaded model. R7 has a dedicated compact-pack test for valid 19-field rows, canonical `code_bits`, `compat_bf16: true`, and truncated 18-field rows; the seven real catalogue indexes remain dependent on T3's pinned packs. R8 exercises every expert in all 48 layers and three roles for five sparse layouts (512-expert single shard, Swift IQ2_XS-like and IQ3_XXS-like two-shard splits, a synthetic v4 per-role split, and 256-expert Coder) through WeightStore and the window planner, including absolute source offsets, slab sizes, shard selection and window-relative bounds. The generated layouts validate the reader and planner contract; T3's pinned packs are still required to verify catalogue offsets. |
| `codebooks` | [04](04-metal-kernels.md), section 10.5, work item L1 | The codebook header against upstream's and ggml's |
| `kernel_env_dispositions_test` | [04](04-metal-kernels.md), work item L14 | Every kernel flag and kernel-scope variable has its disposition |
| `metal_plan_test` | [05](05-memory-and-residency.md), sections 2.3 and 13 | The memory planner; exact byte agreement between planner and QSA/MTP KV layouts for all formats, Qwen4exp session and Window scratch subtotals for 256/512 experts, MTP parameter-slot group sizing, full/tiered admission totals and append-only PLAN reporting |
| `metal_kv_layout_test` | [04](04-metal-kernels.md), sections 4.5-4.6 and 5.11 | Exact device-free main QSA K/V, indexer and RoPE regions plus the MTP drafter's single K/V layer for fp16, int8, q4_0 and k8v4; four-cell capacity rounding, K8V4-to-int8 drafter mapping, checked extents, non-overlap and invalid-input refusal |
| `metal_mtp_bind_layout_test` | [04](04-metal-kernels.md), section 5.11 | Device-free normal-mode MTP bind regions, 256-byte start alignment, vocabulary-sized logits, capacity-dependent attention scratch and invalid-input refusal; verifies the unassigned tail stays reserved |
| `metal_tiered_layout_test` | [05](05-memory-and-residency.md), section 13; work item U5 | Dense/resident copy ranges, profile-ordered resident slots, address/resident tables and alignment-budget checks, plus fail-closed handling when a required GPU base address is unavailable; this pure layout test does not read files or allocate Metal buffers |
| `metal_window_test` | [05](05-memory-and-residency.md), section 13; work item U2 | 16 KiB outward alignment, the strict 256 MiB merge boundary, device-independent `wrapped_unread`, maxBufferLength failures, aligned and unaligned expert-slab splits, preserved merged-range coverage across a forced oversized gap, unavailable GPU window address rejection, and Q2_0/Swift R1 header-fixture windows; excludes the PLE tensor mapping while allowing page-alignment padding at its edge |
| `metal_window_scratch_layout_test` | [04](04-metal-kernels.md), section 4.4; D-IMPL-182 | Device-free exact T=8 region extents and 256-byte starts; byte totals at 32,768, 65,536 and 262,144 cells for both 256- and 512-expert router shapes; unsupported model and capacity rejection; overlap and bounds invariants |
| `metal_window_ple_rows_test` | [04](04-metal-kernels.md), section 5.1; D-IMPL-185, D-IMPL-301; REV-098 | Device-free token-major PLE row construction for T=1..8; absent and partial history, token ID zero, oldest-first predecessors, EOS propagation, artifact hash fixture, fixed qwen4exp geometry and vocabulary checks; committed token suffix initialization, read-only row building, accepted-prefix advancement, zero acceptance, reset, invalid input and transactional state preservation |
| `metal_geometry_plan_cli_test` | [06](06-engine-host-port.md), work item H5 | Geometry-backed `--plan-only` fit/no-fit output, unsupported-type diagnosis and unknown-format exit |
| `metal_weight_store_test` | [06](06-engine-host-port.md), work item H5; D-IMPL-186 | A sparse synthetic 1,224-tensor Qwen GGUF and native pack: exact inventory and shapes, index/conversion consistency, expert table versions, geometry facts and MTP runtime validation; the mapped PLE reader reuses and retains the GGUF owner, gathers sparse rows after the caller releases its reference, and releases the owner on close |
| `mtp_reader_vocabulary_test` | [03](03-model-reference.md), work item R10 | Reads each checked-in draft vocabulary as little-endian int32 IDs, checks its expected count, uniqueness and model-vocabulary bounds, then confirms the exact sequences match and rejects reordered or substituted IDs; runs without model checkpoints |
| `metal_mtp_storage_fixture` | [06](06-engine-host-port.md), section 10.3; D-IMPL-360; REV-157 | Host safety self-tests for random per-run fixture directories, private-parent enforcement, directory replacement before/after descriptor pinning and during cleanup, refusal of path-traversing cleanup names, recursive cleanup without following symlinks, success and failure lifecycle cleanup, safe rollback after partial ownership-marker initialization, existing sentinel preservation, FIFO markers and private ownership markers |
| `metal_weight_store_cli_test` | [06](06-engine-host-port.md), work item H5; D-IMPL-361; REV-158 | Real-file `--plan-only` agrees with its exported geometry; speculative planning requires MTP facts; geometry export safely replaces its target without following a pre-existing `.tmp` symlink. A Python helper creates and removes each unique fixture using pinned parent descriptors, a per-run ownership marker, and the captured device/inode after the child completes; an injected failure after large-fixture generation verifies the external symlink target survives cleanup |
| `oneshot_output_test` | [06](06-engine-host-port.md), work item H7 | Byte-exact formatting of upstream-compatible prompt, output, decode, prefill, speculation, MTP and dump lines, plus the documented Metal `--stats` lines, including plain T=1 windows, zero-token and zero-draft edge cases; this formatter is not yet connected to production inference |
| `prefill_chunk_test` | [04](04-metal-kernels.md), section 6.1; [06](06-engine-host-port.md), section 1.5; D-IMPL-363; REV-160 | Device-free checks of the constant-space schedule, bounded contiguous and unpadded chunk ranges, absolute positions, empty prompt scheduling, out-of-range lookup, signed-position overflow rejection and unchanged output on failure; per-range checks cover copied token/image markers, `PrefillParams`, six query-batch slots, signed `StepRow` values and shader-index limits |
| `prefill_gemm_selector_test` | [04](04-metal-kernels.md), section 6.2; [06](06-engine-host-port.md), section 4.4; D-IMPL-359; REV-156 | Device-free selector policy checks for n=0/1/63/64/65, Apple10 versus earlier families, default MPP/SIMD choice, forced SIMD/MPP, narrow projection override behavior, invalid settings and refusal to force unvalidated MPP on earlier families |
| `window_params_test` | [04](04-metal-kernels.md), section 4.2 | Device-free `WindowParams` ABI checks for T=1..8, `nc`, token/sampler rows, position-derived `StepRow`s and Philox counters, inactive-row zeroing, write-through keep counts, signed-position limits and transactional rejection |
| `options_dispositions_test` | [06](06-engine-host-port.md), work item H4 | Every upstream and Metal CLI flag has its disposition, warning policy and argument count; geometry and simulated-memory validation exits 2 |
| `metal_cli_modes_test` | [06](06-engine-host-port.md), work item H4 | Empty invocation fails closed; storage-check requires `--native` and rejects plan-only before Metal opens; incompatible device-list, self-test, capture and plan-only combinations exit 2 |
| `serve_transcript_test` | [06](06-engine-host-port.md), work item H6 | The host serve loop, linked with `tests/metal/fake_window.cpp` (deterministic logits from the seed and position), replays the checked-in stdout/stderr transcripts byte for byte. Cases cover startup key order, GEN/GENI (including a spaced SVE1 path), request errors, context/vocabulary checks, EOS, and a fatal GPU error. SVE1 checks include pre-allocation bounds from image-pad runs, the 256 MiB aggregate limit and file length, partial headers, multiple records, and 2-D M-RoPE positions. Adversarial backends verify `max_new`, PP totals/progress/completion, EOS finish values, rejection of post-EOS tokens and cancellation invariants. Child processes cover a continuing 16 MiB overlong line, queue overflow with and without immediate EOF, CR-terminated EOF, queued STOP, STOP during decode, STOP immediately after DONE, active and preclosed EOF (including EOF with queued QUIT), and SIGINT/SIGHUP between completed requests; each process exits within 5 s where specified. The finish path also has a 64 MiB synchronous input-drain ceiling, which fails closed on excess ([06](06-engine-host-port.md), sections 2.4, 5 and 5.4). |
| `prompts_manifest_test` | Section 2.2 | `python3 -m unittest tools.metal.test_make_prompts`: the prompt sets against their manifest |
| `test_map` | Section 9.4 | `python3 tools/metal/test_map.py` with `STRATA_BUILD_DIR` set |

**`metal` (100).**

The `metal_ple_test` fixture chains the three weighted RMS norms, PLE gate,
value broadcast, convolution-residual and history update at T=1, 2 and 8. The
reference chain uses `encode_ple_post_projection` in a caller-owned segment
with distinct query and `ple_norm` views and checks each stage against an
ordered scalar oracle; downstream oracles consume the preceding GPU output
(U(64) for norms/gate, bitwise broadcast/history and U(16) residual). An
instrumented chain reuses `query` for `ple_norm`, snapshots the query norm
before overwrite and requires bitwise parity with the reference. The
coordinator also succeeds with that exact alias through its standalone
command-buffer wrapper. Its explicit `defer` history mode leaves committed
history unchanged, and a separate commit dispatch updates only the accepted
prefix for T=1, 2 and 8; invalid policies are rejected. A short history fails
after earlier dispatches have been encoded; the test verifies discard and
unchanged outputs. Partial query/`ple_norm` overlap is rejected. The
`encode_ple_block` fixtures cover Q2_0, Q8_0 and BF16 key formats, the
F32-activation BF16 value projection, padded compiled columns and a discarded
late failure after both projections were encoded. `metal_ple_test_concurrent`
runs the same fixture with concurrent dispatch enabled. The PLE fixture also
composes layer 0's GR post write in place before PLE at T=1 and T=2. Its
query-norm oracle consumes the GPU result from a separate GR-post probe; a
late short-history failure verifies that both the preceding write and PLE
dispatches are discarded. Concurrent mode rejects this ordered composition
when called inside an open dispatch group. The focused serial and concurrent
CTest cases pass.

| Tests | Defined in | What it serves |
| --- | --- | --- |
| `metal_context_test` | [08](08-build-and-distribution.md), section 6.4; [06](06-engine-host-port.md), work item H3 | Opening the device, the pipelines and the archive |
| `metal_buffers_test` | [08](08-build-and-distribution.md), section 6.4; [06](06-engine-host-port.md), work item H3 | Shared untracked buffers, bounded BufferRef slices and coherent-widening rejection, rollback/Context-destruction invalidation, unavailable GPU-address sentinel handling and overflow, fixed-address arena alignment, poison initialization, no-copy validation, residency registration, context-retained mmap ownership, and a page-rounded window over a partial final file page |
| `metal_kv_store_test` | [04](04-metal-kernels.md), sections 4.5-4.6 | All four QSA cache formats share one Context-owned allocation; per-layer K/V, scale, indexer and shared RoPE subviews preserve extents, handles and alignment, and indexer state is deterministically initialized |
| `metal_session_state_test` | [06](06-engine-host-port.md), section 1.5; L9 prerequisite; D-IMPL-302; REV-099 | Checks the fixed qwen4exp geometry, exact GDN/PLE extents, all-layer alignment and zero-on-open, row/update refusal before one-time token-history initialization, read-only row construction and duplicate-seed rejection, host-suffix clearing only on successful reset, preservation of token IDs and GPU bytes on pending-use and foreign-context reset refusal, tracked-use safety, outer-transaction rollback, and fail-closed model/layer/context validation |
| `metal_mtp_kv_store_test` | [06](06-engine-host-port.md), section 10.3 | All four MTP K/V formats use one Context-owned allocation with bounded K/V and scale subviews; writable edge markers verify the views, invalid capacities and formats allocate nothing, and an outer rollback followed by another allocation leaves the provisional store invalid |
| `metal_mtp_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-322/336/337/338; REV-119/133/134/135 | MTP stages 1–5 compose BF16 token gather, both weighted norms, Q8_1 quantization and Q8_0 projections for K/V-only T=1,2,3,8 and full-layer T=1; the selected-token CPU mapping starts OOV and is overwritten by an earlier GPU dispatch in the same Encoder segment; independent oracles check gather, norms, quantization and every active projected row. Stage 6 has T=1..8 bitwise scalar-oracle parity; stage 12 covers finite suffix positions, full identity IDs and finite→unwindowed reuse; stage 21 checks full and indexed draft-head parity for Q5_K/Q6_K/IQ4_XS with one-time unique/range map validation; stage 22 checks greedy full/subset IDs and argmax softmax probability, including a strict subset oracle that excludes logits beyond the active mapped rows. Checks active extents and guards, immutable inputs, invalid T/hid/token/image-marker/map/format/sampler/history, short/misaligned/foreign views, selected-token/MtpParams overlap, parameter and output aliases, concurrent-group refusal, and caller-owned rollback after encoded work. `metal_mtp_storage_test` verifies runtime bindings through stage 22 and dispatches the synthetic ordered chain through stage 20; it does not run the full-vocabulary head or establish model parity. Production scheduler integration and model-backed drafter parity remain open |
| `metal_mtp_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-336/337/338; REV-133/134/135 | Repeats stage 1–5 projections and stage 21–22 head/sampling flows with concurrent dispatch, including a preceding GPU token write over stale OOV CPU bytes; compares intermediates, logits and probabilities against independent oracles, and verifies open-group refusal and caller-owned rollback |
| `metal_mtp_gr_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-323; REV-120 | MTP stage 7 only: T=1..8 exact parity with the direct ordered GR read at apply=false, finite populated active outputs and output guards, immutable residual/params, invalid T, short/misaligned dead h2 scratch, foreign params, parameter/read-only-input and parameter/output alias rejection, and standalone/caller-owned rollback after a late GR validation failure. Runtime-storage wiring, later MTP stages and model-backed drafter parity are not covered |
| `metal_mtp_gr_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-323; REV-120 | Repeats the stage-7 chain with concurrent dispatch and verifies rejection inside an open group, with all buffers unchanged |
| `metal_mtp_kv_projection_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-324; REV-121 | MTP stage 8 only: T=1..8 Q8_1 activation and key/value Q8_0 projection results match independent scalar oracles; checks active output/sink ranges and guards, immutable inputs, invalid T, short key weights and padded sink, foreign weights, params/output and K/V output aliases, and caller-owned rollback after a later invalid call. Runtime-storage wiring, remaining MTP stages and model-backed drafter parity are not covered |
| `metal_mtp_kv_projection_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-324; REV-121 | Repeats stage 8 with concurrent dispatch and verifies rejection inside an open group with all fixture buffers unchanged |
| `metal_mtp_kv_append_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-325; REV-122 | MTP stage 9 for T=1..8: fp16, int8 unrotated/rotated, q4_0 rotated and K8v4-requested→int8 storage match scalar QSA norm/RoPE, FWHT and byte-exact append oracles. Verifies cache untouched cells/tails, K/V active-row bounds, immutable params/norm/RoPE, duplicate/out-of-capacity positions, forged layout, short/foreign views, invalid `kv_rot`, open-group rejection and caller-owned late rollback. Runtime setting of `kv_rot`, MTP runtime-storage wiring and model-backed drafter parity are not covered |
| `metal_mtp_kv_append_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-325; REV-122 | Repeats the stage-9 format matrix under concurrent dispatch and verifies open-group rejection with all fixture buffers unchanged |
| `metal_mtp_query_projection_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-326; REV-123 | MTP stage 10 requires T=1, reuses the Q8_1 row from stage 8 and compares the 2560→12288 Q8_0 query projection with an independent scalar oracle. Checks every value in the full 24×512 q_full tensor, allocation guards, untouched sink and immutable inputs, invalid T, short weights/sink, foreign weights, q_full/sink and partial q_full/xq plus sink/xq aliases, ordered-group rejection and caller-owned rollback. Runtime-storage wiring and model-backed drafter parity are not covered |
| `metal_mtp_query_projection_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-326; REV-123 | Repeats stage 10 with concurrent dispatch and verifies ordered-group rejection with all fixture buffers unchanged |
| `metal_mtp_query_norm_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-327; REV-124 | MTP stage 11 requires T=1 and compares qcur with independent scalar QSA normalization/RoPE and optional FWHT oracles in standalone and caller-owned segments. Checks finite outputs, guards, immutable inputs, malformed T/position, short and foreign views, q_full/qcur and params/input aliases, ordered-group rejection, and rollback after a later invalid call. Runtime-derived `kv_rot`, runtime-storage wiring and model-backed drafter parity are not covered |
| `metal_mtp_query_norm_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-327; REV-124 | Repeats stage 11 with concurrent dispatch and verifies ordered-boundary rejection with all fixture buffers unchanged |
| `metal_mtp_window_ids_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-328; REV-125 | MTP stage 12 compares active IDs for short, exact, truncated and full-history windows with absolute KV-cell expectations, including a 262,144-cell history with a 64-ID output capacity and a 4,097-ID window crossing threadgroup and grid-stride boundaries. Checks inactive-tail sentinels, guards, immutable params, signed-range edges, malformed T/window/n_kv/StepRow width, short/misaligned/foreign views, active output/params alias rejection and caller-owned rollback |
| `metal_mtp_window_ids_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-328; REV-125 | Repeats stage 12 with concurrent dispatch and verifies ordered-boundary rejection without changing fixture buffers |
| `metal_mtp_attention_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-329; REV-126 | MTP stage 13 compares dense chunk attention, ordered merge and optional output FWHT with scalar oracles for fp16, int8 with and without rotation, rotated q4_0 and K8v4-requested→int8 storage. Checks the finite-window stage-12→13 handoff in one Encoder segment, an ID stride larger than KV cell capacity, active outputs and guards, untouched reserved scratch tail, immutable params/query/KV, invalid stride and short views, format/rotation mismatch, forged layout, scratch/query and output/query/cache aliases, and rollback after stage 12 has already encoded |
| `metal_mtp_attention_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-329; REV-126 | Repeats the stage-13 format matrix under concurrent dispatch and verifies open-group rejection while preserving ordered chunk→merge→FWHT dependencies |
| `metal_mtp_output_projection_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-330; REV-127 | MTP stage 14 compares gate values with the accurate sigmoid oracle, Q8_1 bytes with the scalar quantizer and `bo` with the scalar Q8_0 mat-vec oracle. Checks stage-10/stage-13 input geometry, immutable inputs and weights, output guards and Q8_1 padding, malformed T and short views, parameter/input/output aliases, foreign weights and late rollback |
| `metal_mtp_output_projection_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-330; REV-127 | Repeats the stage-14 scalar-oracle path under concurrent dispatch and verifies open-group rejection while preserving gate→quantize→projection ordering |
| `metal_mtp_feed_forward_read_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-331; REV-128 | MTP stage 15 compares updated residual, normalization, scales, low-rank output, injection and mixed output from standalone and caller-owned forms with the direct GR `apply=true` chain. Checks T errors, parameter/view and output/input aliases, short views, foreign weights, immutable read-only inputs, guards and rollback |
| `metal_mtp_feed_forward_read_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-331; REV-128 | Repeats the stage-15 parity path under concurrent dispatch and verifies open-group rejection while preserving normalize→down→up ordering |
| `metal_mtp_router_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-332; REV-129 | MTP stage 16 compares router logits against the BF16 mat-vec scalar oracle and top-10 IDs/weights against the route oracle. Checks standalone/caller-owned/direct-component parity, T errors, short views, parameter/input and output aliases, foreign weights, read-only inputs, guards and rollback |
| `metal_mtp_router_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-332; REV-129 | Repeats the stage-16 oracle/parity path under concurrent dispatch and verifies open-group rejection while preserving projection→route ordering |
| `metal_mtp_routed_experts_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-333/344; REV-130/141 | MTP stage 17: a synthetic S2 expert image and ten route IDs, including expert 511 at the final blob boundary, exercise mixed-row Q8_0 quantization, gate/up, SwiGLU, hidden-row quantization and down. Compares Q8 codes, scales, gate/up and down against scalar references; verifies full-image/split-window byte parity, standalone/caller-owned parity, read-only input preservation, output guards, short/misaligned/foreign views, invalid coverage and conflicting expert inputs, input/output and output/output aliases, and caller-owned rollback after encoded work. The fixture allocates about 675 MiB in one or more buffers and does not establish model-backed parity |
| `metal_mtp_routed_experts_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-333; REV-130 | Repeats the stage-17 S2 oracle path under concurrent dispatch and verifies open-group rejection while preserving quantize→gate/up→SwiGLU→quantize→down ordering |
| `metal_mtp_shared_ffn_write_test` | [04](04-metal-kernels.md), section 5.11; D-IMPL-334; REV-131 | MTP stages 18–19 quantize `mixed` to Q8_1, run the shared expert, combine with routed parts/weights and apply GR post. Compares Q8_1 with the scalar quantizer, shared/combine outputs with the direct MoE path, and the residual with a GR-post oracle. Covers caller-owned late rollback, invalid T, short Q8_1 output, parameter/input aliases, Q8_1 output aliases with all shared weights, routed parts, route weights, scratch, residual and injection, read-only input preservation after dispatch, and guards |
| `metal_mtp_shared_ffn_write_test_concurrent` | [04](04-metal-kernels.md), section 5.11; D-IMPL-334; REV-131 | Repeats the shared-expert/combine/GR-post chain with concurrent dispatch using the standalone wrapper and checks open-group rejection and unchanged buffers |
| `window_params_ring_test` | [04](04-metal-kernels.md), section 4.2; work item L9 | T=1 and T=8 blocks are built into a fixed-address two-slot Shared ring with 256-byte strides; invalid construction preserves published bytes, slots remain bounded and generation-checked, outer rollback invalidates views, and a too-small live `maxBufferLength` is rejected before allocation |
| `metal_window_scratch_storage_test` | [04](04-metal-kernels.md), section 4.4; D-IMPL-183 | One Context-owned Shared allocation with exact logical size; bounded views for every named region; allocator poison, maxBufferLength refusal, unsupported-input refusal and outer-transaction rollback |
| `metal_window_storage_test` | [05](05-memory-and-residency.md), work item U3; L9 prerequisite | A nine-expert synthetic GGUF validates exact-plan union checking, full-mode tables, checked named dense tensor views for mapped and BF16-converted weights including a second row at nonzero offset, rejection of expert slabs, the separate per-layer embedding PLE table (`per_layer_token_embd.weight`), unknown and rolled-back views, and CPU/GPU warm-up; it checks that the full-mode PLE reader view has row access but no public open/close lifecycle, while tiered publishes no reader. The ExpertLeaseSet probe accepts its prepared command buffer and rejects an unrelated command buffer, verifies exact live slot coverage for selected tiered roles, and confirms a retired layer is no longer pinned. A synthetic Q2_0 consumer with an empty same-command lease set, nonzero inputs and nonzero Q2_0 weights confirms that an unpinned active address sets GPU status bit 2 and zeroes the affected output instead of computing from the address. The FullMode T=1 PLE handoff also clears its caller-supplied projection/norm/convolution views and uses a deliberately incorrect key format to verify that the wrapper resolves all six mapped model tensors itself |
| `metal_fullmode_handoff_test` | [06](06-engine-host-port.md), work item L9; D-IMPL-294/296/297/303/318/319; REV-091/093/094/100/115/116 | A sparse 256-expert synthetic GGUF exercises T=1 input preparation, FullMode routing and planning, routed/shared MoE output, then the typed no-PLE attention consumer in one caller-owned Encoder segment. It checks the prepared token, position, seed, every non-default sampler field and empty PLE input, routes 0–9, finite nonzero pending and mixed outputs, a finite nonzero producer injection distinct from its decoy, exact parity with a direct `apply=true` GR read, one-time residual application and unchanged pending output. It routes through the typed pending-consumer selector and rejects a mismatched branch tag. The same CTest covers generic FullMode production at T=1, T=2, T=3 (`nc=4`) and T=8, independent router/route/Q2_0 routed-expert/Q8_0 shared-expert/combine oracles, standalone T=2 and T=3 plus caller-owned T=8 generic no-PLE producer-to-consumer steps, and a T=8 consumer checked against an independent pending-output GR reference. Active output tails and unchanged FullMode usage are checked. Caller-owned T=2 and standalone T=3 late shared-FFN failures plus standalone T=2 and caller-owned T=3 late consumer failures verify rollback and byte preservation; a consumer output alias into the FullMode usage table is rejected before dispatch. Coverage is synthetic and does not establish production scheduler or model-backed parity |
| `metal_fullmode_ple_handoff_test` | [06](06-engine-host-port.md), work item L9; D-IMPL-295/296/299/303/319/320/321; REV-092/093/096/100/116/117/118 | The same sparse 256-expert FullMode producer feeds the typed PLE consumer in one caller-owned Encoder segment. T=1 preparation rejects an uninitialized session, out-of-range token and overflowing position without changing the ring slot or scratch, then builds parameters and stages rows derived from the committed suffix. The test checks staged values against a direct gather, verifies preparation leaves history unchanged, and resolves deterministic, correctly typed PLE projection, normalization and convolution tensors from the FullMode owner as read-only views. It checks finite nonzero PLE outputs, direct GR-post plus PLE-entry parity, one-time residual application, pending-output immutability and deferred history. It also runs the generic FullMode input-to-PLE transaction at T=3 (`nc=4`), compares embedding, broadcast, every PLE output and residual with direct row-wise references, checks inactive output guards, rejects a later invalid token and nonzero `n_keep`, and verifies both standalone rollback and caller-owned rollback when its enclosing `Encoder::encode` body propagates false. The T=3 host preparation builds speculative parameters and committed-history PLE rows, matches staged values against a direct batch table gather, and rejects an invalid final token without changing the ring slot or PLE scratch. Its T=3 success case also sources token and sampler spans from that same scratch region and confirms staging preserves their values in the published parameters. T=8 no-PLE preparation publishes all eight rows without requiring or changing PLE history. Generic T=8 embedding/broadcast matches direct row-wise references containing nonzero values and preserves padded output tails. The fixture has a one-token vocabulary, so its row-sensitive T=3 PLE input uses synthetic values after table staging. It rejects speculative `advance_rows`, PLE table-input aliasing the pending MoE output and mismatched branch tags before dispatch. Synthetic fixture values do not establish production scheduler integration or parity against a real model artifact |
| `metal_fullmode_output_projection_test` | [04](04-metal-kernels.md), sections 5.1 and 5.8; [06](06-engine-host-port.md), work item L9; D-IMPL-306; REV-103 | A sparse production-shaped IQ4_XS `output.weight` tensor exercises FullMode lookup and a successful Q8_1-to-logits projection. The nonzero first vocabulary row matches independently retained fixture bytes and the scalar oracle. It checks output row guards, rejects the Q8_1/read-only active-row shared-backing alias, and verifies late output-alignment failure discards quantization in standalone and caller-owned forms without changing scratch. Synthetic weights do not establish model-backed parity |
| `metal_fullmode_output_head_test` | [04](04-metal-kernels.md), sections 4.4, 5.1 and 5.8; [06](06-engine-host-port.md), work item L9; D-IMPL-307; REV-104 | A sparse FullMode GGUF resolves exact F32/BF16 mixer tensors and a nonzero IQ4_XS output row, then composes the mixer, Q8_1 quantization and logits projection. It checks the mixer output and first logit against independent scalar oracles, exact retained fixture weight bytes, unchanged residual, active-row bounds and untouched logits tail. A residual/logits scratch alias is rejected before encoding. Missing `output.weight` after earlier mixer dispatches proves both standalone and caller-owned rollback, including discarded command buffers, empty results and byte-preservation across all head scratch. This synthetic fixture does not establish model-backed parity |
| `metal_fullmode_output_sampling_test` | [04](04-metal-kernels.md), sections 5.1, 5.8 and 5.12; [06](06-engine-host-port.md), work item L9; D-IMPL-308; REV-105 | A sparse FullMode fixture composes mixer→logits→penalty-aware GPU top-k/merge in one command and runs the CPU tail after completion. A T=2 case uses different sampler settings, counters and history offsets for each row; it checks exact expected candidate IDs/logit bits and each sampled result against an independent scalar/Philox tail oracle. Residual/candidate scratch aliasing is rejected before dispatch with all scratch unchanged. Zero repeat penalty and a temperature with a non-finite reciprocal fail before head dispatch; NaN candidate, overflowed-probability and `T=UINT32_MAX` CPU-tail cases are rejected without publishing partial picks. Standalone and caller-owned late history-validation rollback preserve every head/sampler write region, including the matvec sink. The fixture is synthetic; history setup remains caller-owned and this does not establish production `Window::run` or model-backed parity |
| `metal_fullmode_ple_q8_handoff_test` | [06](06-engine-host-port.md), work item L9; D-IMPL-300/303/320; REV-097/100/117 | Repeats the mapped FullMode-to-PLE handoff with native Q8_0 key blocks. T=1 preparation stages the session-derived PLE input. A deliberately wrong BF16 caller tag must be replaced from GGUF metadata; the T=3 input-to-PLE composition matches direct embedding, broadcast and PLE references |
| `metal_fullmode_ple_bf16_handoff_test` | [06](06-engine-host-port.md), work item L9; D-IMPL-300/303/320; REV-097/100/117 | Repeats the mapped FullMode-to-PLE handoff with native BF16 key weights. T=1 preparation stages the session-derived PLE input. A deliberately wrong Q2_0 caller tag must be replaced from GGUF metadata; the T=3 input-to-PLE composition matches direct embedding, broadcast and PLE references |
| `metal_mtp_storage_test` | [06](06-engine-host-port.md), sections 1.5 and 10.3; D-IMPL-184/339/340/341/342/343/344/345/346/360; REV-136/137/138/139/140/141/142/143/157 | Creates a unique sparse fixture and passes its pinned directory descriptor and device/inode identity through both child processes; then maps and warms it, checking every expert and dense tensor view, draft-vocabulary order and device-map contents, identity initialization of MTP window IDs, zeroed bounded fixed/prefill parameter slots including the no-prefill short-read slot, page-bounded windows, same-size replacement and symlink refusal, rollback after partial buffer registration, geometry-derived named bind subviews, planner/owner agreement for exact session and Window scratch bytes, pre-allocation refusal for forged/underfunded subtotals and a low maxBufferLength, ModelStorage scratch views, and generation-lock retention through Context teardown. It dispatches stages 1–8 at T=5 and T=8, then stages 1–20 from a GPU-produced selected token at T=1: the immutable stages 9–14 attention pass uses a finite window of one and one shared int8 rotation mode, followed by feed-forward read, routing, split expert windows, shared expert/writeback and final mixer. It checks the stage-6 residual identity, distinct stage-7 attention and stage-15 MLP hyperconnection weights, every expert window and the stages 18–20 handoffs. Runtime bindings for stages 21–22 are checked against the caller-owned head, cached vocabulary map, logits and sampler outputs. After fixture generation, it cleans up on either test result only while the pinned identity and private ownership marker still verify; an unsafe or incomplete cleanup remains available for inspection. The fixture verifies binding and synthetic dispatch, not parity with an inspected model, production scheduling or end-to-end inference |
| `metal_model_storage_owner_test` | [05](05-memory-and-residency.md), work item U3; D-IMPL-188/265 | The startup owner selects full and tiered storage, exposes the full-mode shared-mapping PLE row reader through the lifecycle-restricted interface and no tiered reader, exposes named dense weights and expert address/resident tables through the borrowed const full-mode resolver with `ReadOnlyBufferRef` CPU mappings, statically rejects conversion to mutable `BufferRef`, verifies generation invalidation after rollback and returns null in tiered mode, warms full-mode mappings and the converted-router arena, verifies exact BF16 row bytes and 16 KiB padding, and accepts only one-to-one converted-router rows. A sparse full-size PLE GGUF checks T=1..8 gathering into the named Shared scratch region, row decode/order, bounded active views, untouched suffix rows and malformed-input rejection. It also rejects malformed rows, source-lock mismatches, geometry/plan undercounts, insufficient page-padding slack, missing inspection locks, duplicate resident IDs and planned MTP buffers before leaking allocations. A non-empty unlisted `dense.bin` is accepted when the model has no converted rows and is not read into an arena. |
| `metal_events_test` | [08](08-build-and-distribution.md), section 6.4; [06](06-engine-host-port.md), work item H3 | Monotonic host/GPU shared-event signaling, blocked GPU waits, timeouts and command-buffer ordering; condition-variable synchronization verifies completion callbacks return exactly once after success or report a discarded NotEnqueued command buffer; `release_gpu_waits` respects a bounded drain timeout and remains blocked at a second command-buffer gate until that gate is released, poisons the event, rejects later operations and exposes no completion value after release |
| `metal_encoder_test` | [04](04-metal-kernels.md), section 8.2 | The generic encoder runs a dependent/independent self-test sequence in default, serial and concurrent modes with bitwise-equal outputs; checks inline-byte and dynamic threadgroup-memory bindings, manifest geometry, Context ownership, buffer extent/alignment, duplicate indices, device limits, group balance and failure/exception discard. A const MSL input copy is bound with `EncoderReadOnlyBufferBinding`, and overlapping writable aliases are rejected. Staging-range checks reject earlier dispatch reads/writes and indirect resources, plus later output writes to reserved scratch. This tests the shared wrapper only; Window/MTP/Prefill production dispatch parity remains under L-14. |
| `metal_embed_test_concurrent` | [04](04-metal-kernels.md), section 10.5 | Runs the embedding, optional image-copy and GR broadcast suite with `STRATA_METAL_DISPATCH=concurrent`, including the caller-owned Encoder chain and bitwise oracle checks |
| `metal_prefill_test_concurrent` | [04](04-metal-kernels.md), section 10.5; D-IMPL-378/380/381; REV-175/176/177 | Runs Prefill GR normalization, fused GR write/next-read normalization, GDN gates, tiled convolution, history update and q/k L2 normalization, SiLU and mix plus embedding-to-normalization composition with `STRATA_METAL_DISPATCH=concurrent`; checks fused-versus-separate stage bit parity, scalar gate and convolution oracles across 64-row tile boundaries, history oracles at T=1/2/3/8/65, same-segment old-history ordering and both-overload rollback, q/k L2 ordered-reduction and epsilon sentinels at T=1/2/3/8/65, unchanged v heads, guards and rollback, reduction and conversion oracles, BF16X2 ties, late-failure discard and open-group rejection. The recurrence stage checks scalar output/state parity at T=1/2/3/8/65, L2→recurrence composition at T=1/3/8, ordered reduction, input immutability, guards, view validation, rollback and open-group refusal. Output normalization compares F16 output with a scalar oracle at T=1/2/3/8/65, checks epsilon and a four-partial-order sentinel whose regrouping negative control changes F16 bits, and covers exact halfway RNE bits plus unsaturated signed overflow to infinity. The full GDN block compares all projections, transform outputs, persistent state and reused native dequant scratch with the direct component sequence in modes 0/1/2 at T=1 and mode 2 at T=3; the caller-owned and standalone forms are both exercised. It also checks complete active-range preflight for a short `bo`, invalid final projection format, cross-stage output alias, missing/unexpected BF16 remainder and open concurrent-group rejection. Both API forms, transform composition, validation, guards and input immutability are checked. An invalid selector after the first projection dispatches dequantization then fails its GEMM; standalone and caller-owned paths both discard the unsubmitted block and preserve all output, scratch and persistent-state bytes. The attention-half GR/GDN transition also runs in modes 0/1/2 at T=1 and mode 2 at T=3/65 against the direct chain; it checks model-weight, persistent-state and reused-normalization aliases, active-prefix arena reuse, late rollback and open-group refusal. The prepared-normalization GR read/read-mix APIs also consume the actual fused-transition output in the same Encoder segment in modes 0/1/2, then compare the updated-residual projections and mixer with a normal read/mix that recomputes normalization; a separate all-mode case verifies projection-only use without residual/norm views, and missing mixer inputs plus gated/injection-output aliases to residual/norm inputs are rejected before dispatch in every mode; mode 1 also rejects a low-rank remainder output aliased to the residual. |
| `metal_prefill_moe_grouping_test`, `metal_prefill_moe_grouping_test_concurrent` | [04](04-metal-kernels.md), sections 6.6 and 10.4-10.5; [06](06-engine-host-port.md); D-IMPL-395; REV-191 | T=1/7/25/26/64/300/4,096 with 256/512 experts compare all histogram columns, PfPlan fields and padding, tile lists, stable src/dst entries and F16 gathers against an independent host grouping oracle. Mixed, one-expert, all-expert, tile-boundary and invalid-ID inputs cover serial and concurrent dispatch. The fixtures check output tails, guards, input immutability, active-slice alias handling (including histogram reuse over an unselected residency row) and validation; production Prefill scheduling and real-model parity remain pending |
| `metal_prefill_moe_pipeline_test`, `metal_prefill_moe_pipeline_test_concurrent` | [04](04-metal-kernels.md), section 6.6; [06](06-engine-host-port.md); D-IMPL-408; REV-204 | Full-mode steps 1–6 compose the BF16X2 router, divide-mode top-10 route, GPU stable grouping and F16 row gather in modes 1 and 2. Synthetic T=1/7 cases with 256/512 experts compare logits and route results with scalar references; router inputs have nonzero high and low BF16 products at six K coordinates, including positions through K=2559. Histogram, PfPlan, tiles, source/destination maps and gathered rows use an independent grouping oracle. Tests check input immutability, guards, inactive tails, mode-0 and cross-stage alias rejection, and confirm late route validation discards the command buffer after router encoding. This component is now composed by the full-mode layer tests below; tiered scheduling and model parity remain pending |
| `metal_prefill_moe_shared_pipeline_test`, `metal_prefill_moe_shared_pipeline_test_concurrent` | [04](04-metal-kernels.md), section 6.6; [06](06-engine-host-port.md); D-IMPL-409; REV-205 | Shared-expert steps 7–10 compose native Q2_0 gate/up projections, F16 SwiGLU, native down projection and the BF16X2 shared gate in modes 1 and 2. Synthetic T=1/7 fixtures vary Q2_0 codes and scales by output row and K block; per-channel F32 projection outputs, all K-major F16 dequant scratch elements and shared-gate outputs use independent scalar references, while F16 SwiGLU outputs are checked by ULP. Tests also verify input immutability, guards, cross-stage writable-alias and mode-0 rejection, and serial/concurrent dispatch. A late GEMM-selector failure after dequantization verifies command-buffer discard in both the standalone and caller-owned forms. These component results are also exercised within the full-mode layer tests below; tiered scheduling and model parity remain pending |
| `metal_prefill_moe_full_pipeline_test`, `metal_prefill_moe_full_pipeline_test_concurrent` | [04](04-metal-kernels.md), section 6.6; [06](06-engine-host-port.md); D-IMPL-410; REV-206 | Full-residency steps 1–13 compose route/gather, shared FFN/gate, routed gate/up and down tiles, and weighted combine. Standalone T=1/mode-1/256-expert and caller-owned T=7/mode-2/512-expert synthetic cases compare every BF16X2 router logit, divide-mode route weight and shared-gate logit with independent FP64 oracles. They check shared projections, every grouped route and source gather, every routed F16 SwiGLU result and every routed down output with independent Q2_0 references whose codes and scales vary by output row, K block and packed lane. The tests also cover route destinations, combine, guards, unchanged gather tail, strict same-start producer/consumer aliases (including shifted-slice rejection), invalid routed/shared formats before dispatch, unsupported mode rejection and rollback after late address validation. The fixture reuses a single full-residency weight backing view for all expert addresses to bound memory; it does not establish model-specific expert-address parity. Tiered scheduling, model-backed parity and production `Prefill::run` remain pending |
| `metal_prefill_moe_post_test`, `metal_prefill_moe_post_test_concurrent` | [04](04-metal-kernels.md), sections 6.6 and 10.4-10.5; [06](06-engine-host-port.md); D-IMPL-396; REV-192 | T=1/2/3/7/65 shared-expert SwiGLU outputs match an independent scalar oracle within one half ULP, including positive and negative F16 saturation. Ten-rank combine matches ordered host `fma` evaluation, with a cancellation sentinel that fails under reversed rank order. Both APIs cover standalone and caller-owned encoding, serial/concurrent dispatch, input immutability, active output tails and allocation guards; validation covers zero rows, foreign Context inputs, output/input overlap and concurrent-group policy. These synthetic post-op fixtures do not implement expert tiles, production Prefill scheduling or model parity |
| `metal_prefill_moe_expert_test`, `metal_prefill_moe_expert_test_concurrent` | [04](04-metal-kernels.md), sections 6.6 and 10.4-10.5; [06](06-engine-host-port.md); D-IMPL-397/401/402/403/404/405/406/407; REV-193/197/198/199/200/201/202/203 | Grouping→F16 row gather→Q2_0/IQ4_XS/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_M gate/up tiles→SwiGLU runs at T=1/7 through standalone and caller-owned APIs in serial/concurrent dispatch. T=7 routes 65 entries to expert 0 and 5 to expert 3, exercising a tile split at 64 while resolving two separate weight addresses. Every hidden output is compared with an independent format-aware FP64-dot oracle within four F16 ULPs; IQ4_XS uses `strata::dequantize_iq4_xs`, IQ2 formats use their canonical `dq_iq2_*` contracts, and IQ3_XXS/IQ3_S use independent packed-field decoders (including IQ1_M signed grid digits and both delta signs), before rounding every weight to F16. Literal fixtures anchor IQ4_XS nibble placement and IQ2/IQ3 packed fields. Fixed packed IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS blocks check frozen F16 SwiGLU goldens `0x2c29`, `0x2738`, `0x1d33` and `0x40d3`; IQ3_S checks a fixed `0x0000` GPU golden from an exactly cancelling rounded gate pair. A GPU decoder probe compares every F16 bit for all four Q2_0 codes; IQ4_XS covers all 16 nonlinear codes and 64 packed scales, IQ2_XXS covers all 256 grid indices, 16 scale codes and 128 sign indices, IQ2_XS covers all 512 grid indices, both 16-value scale nibbles and 128 sign indices, IQ2_S covers all 1,024 grids, both 16-value scale nibbles and all 256 sign masks, IQ3_XXS covers all 256 grids, 128 sign indices and 16 scales, and an eight-block IQ3_S GPU decoder probe checks every F16 result bit against a CPU oracle that traverses packed quartets independently, covering all 512 grids, both values at each `qh` high-bit position, 256 sign masks and 16 scale nibbles. A 128-block IQ1_M probe checks all 2,048 grids under both delta signs and all scale fields against packed CPU decoding; a literal signed-zero fixture checks `+1.875` and `−1.875`. Unrounded-weight negative controls for all six prior nonlinear formats change the result beyond four ULPs; IQ3_S's cancellation sentinel differs by over 30,000 ULPs, and IQ1_M differs at its fixed output-0 golden by over 10,000 ULPs. Signed inputs, missing active addresses, malformed-plan status, duplicate/out-of-order expert IDs, a malformed tile list with a valid header, missing-tile status, zeroed failed tiles, an earlier grouping write that overwrites the source address row while dedicated snapshot scratch preserves oracle output, rejection and rollback when two pending layer dispatches reuse scratch, the following expert's preserved output after a short failed group, input/weight immutability, output tails and guards, Context ownership, active output/input aliases, snapshot-scratch aliases, bounded indirect GPU addresses, tiered-descriptor rejection without a matching lease and ordered-group policy are checked. A late expert preflight failure and a later caller-owned failure verify rollback of queued work and status writes. The separate `metal_window_storage_test` covers live slot coverage and fail-closed status bit 2. Q2_0/IQ4_XS/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_M gate/up and Q2_0/IQ4_NL down tiles are implemented in isolation; production scheduling and model parity remain pending |
| `metal_prefill_moe_down_test`, `metal_prefill_moe_down_test_concurrent` | [04](04-metal-kernels.md), sections 6.6 and 10.4-10.5; [06](06-engine-host-port.md); D-IMPL-399/400; REV-195/196 | Grouping→Q2_0/IQ4_NL F16-activation down tiles at T=1/7 through standalone and caller-owned APIs in serial/concurrent dispatch. T=7 routes 65 entries to expert 0 and 5 to expert 3, crossing the 64-entry tile boundary. Every F32 route output and output column is compared with format-specific scalar Q2_0 and IQ4_NL dot oracles; the destination map is checked against independently sorted route IDs. Missing down addresses set status bit 0 and zero only that expert's original routes. A malformed tile with a valid plan header, an out-of-range destination and a duplicate destination each set status bit 1 and zero the complete active parts output. Active hidden/output aliases are rejected; output and status tails plus buffer guards are checked. This validates the isolated tiles only; tiered scheduling, model-backed parity and production `Prefill::run` remain pending |
| `metal_prefill_gemm_test`, `metal_prefill_gemm_test_concurrent` | [04](04-metal-kernels.md), sections 6.2 and 10.5; D-IMPL-356/357/362; REV-153/154/159 | Apple10-only isolated F16 and BF16 MPP GEMMs plus the selected BF16X2 pair. F16 uses K-major weights; BF16 uses file-layout weights. Edge dimensions T,n=1/63/65 at K=32 plus T=n=65,K=2560 test overwrite and accumulate against FP64 products under U(2K) and R(2^-16). The BF16X2 coordinator has caller-owned Encoder and standalone CommandBuffer forms. Tests cover T=1/3/8, n=1/63/65, both output modes, backend reporting, high-plus-low FP64 products, a caller-owned prefix probe and rollback after late low-input and low/Y-alias failures at n=64. The tests also cover output padding, read-only input immutability, guards, short/misaligned/foreign buffers, extent/stride validation, exact/partial X/Y and partial W/Y alias rejection, adjacent-view acceptance, rollback and open concurrent-group refusal. On other GPU families the MPP CTest returns the configured skip code. These synthetic cases do not establish production Prefill integration or model-backed inference. |
| `metal_prefill_gemm_simd_test`, `metal_prefill_gemm_simd_test_concurrent` | [04](04-metal-kernels.md), sections 6.2 and 10.5; D-IMPL-358/362; REV-155/159 | Isolated F16 and BF16 SIMD-group matrix GEMMs and the selected BF16X2 pair, tested on Apple10. F16 uses K-major weights; BF16 widens row-major file-layout operands into F32 threadgroup tiles. Edge dimensions T,n=1/63/65 at K=32 plus T=n=65,K=2560 test overwrite and accumulate against FP64 products under U(2K) and R(2^-16); BF16X2 tests T=1/3/8, n=1/63/65 and both output modes. The tests also cover output padding, input immutability, guards, short/misaligned/foreign buffers, extent/stride validation, exact/partial X/Y and partial W/Y alias rejection, adjacent-view acceptance, rollback and open concurrent-group refusal. Other Apple GPU families, production selector/integration and model-backed inference are not established. |
| `metal_prefill_gemm_auto_test`, `metal_prefill_gemm_auto_test_concurrent` | [04](04-metal-kernels.md), sections 6.2 and 10.5; D-IMPL-359/362; REV-156/159 | Runs the selected F16/BF16 path and BF16X2 pair with an unset/empty `STRATA_METAL_GEMM`: narrow projections use SIMD and wide projections use the Apple10 default MPP backend. Each successful dispatch checks the backend reported by the encoder against the active family/width policy, then checks the FP64 numerical oracle, bounds, validation and rollback in serial and concurrent modes; this isolated selector is not wired into production Prefill |
| `metal_prefill_gemm_override_simd_test` | [04](04-metal-kernels.md), sections 6.2 and 10.5; D-IMPL-359/362; REV-156/159 | Runs the selected F16/BF16 paths and BF16X2 pair with `STRATA_METAL_GEMM=simd` and verifies the GPU result against the FP64 oracle, including width and validation boundaries |
| `metal_prefill_gemm_invalid_setting_test` | [04](04-metal-kernels.md), sections 6.2 and 10.5; D-IMPL-359; REV-156 | Rejects an unknown `STRATA_METAL_GEMM` value and verifies that the caller-owned unsubmitted command buffer is discarded |
| `metal_prefill_dequant_test`, `metal_prefill_dequant_test_concurrent` | [04](04-metal-kernels.md), sections 6.2, 6.5 and 10.5; D-IMPL-364/365/366/367/368/369/370/371/372/373/374/388/389; REV-161/162/163/164/165/166/167/168/169/170/171/184/185 | IQ4_NL, Q2_0, Q3_K, IQ4_XS, Q4_0, Q4_K, Q5_K, Q5_0, Q6_K, Q8_0 and IQ3_S `[N=257,K=512]` weight blocks expand to K-major F16; independent logical-code oracles verify exact RNE-half output across format block and output-column boundaries, including Q2_0's 64-value packing, Q3_K's hmask/two-bit codes and per-row 16 packed scales, IQ3_S's qh/qs grid index, four-byte lanes, signs and packed scale nibbles, IQ4_XS's eight nonlinear 32-value groups, nibble halves and packed low/high scale fields, Q4_K/Q5_K's packed six-bit scale/min groups, Q3_K/IQ4_XS/Q4_K/Q5_K/Q6_K's 256-value blocks, Q4_K's independent nibble fixtures, Q5_K's fifth-bit plane, Q6_K's ql nibble chunks, qh two-bit fields and signed 16-value scales, Q5_0's fourth-bit plane and both IQ4 formats' nonlinear codebook. Each format checks immutable inputs, guards, short/misaligned views, invalid signed geometry, exact and partial overlap, foreign input/output Context rejection and standalone/caller-owned rollback. The native projection coordinator also composes all 11 formats with F16 GEMM in one segment; tests compare the exact scratch and F32 output with the direct component chain, preserve activations and native weights, check output padding, alias rejection and late-failure discard. The caller-owned IQ3_S path checks success and propagates a late-failure discard. The QSA output coordinator tests fixed 6144→2560 gating and native-weight projection with synthetic Q8_0 weights, both without a transform and with in-place attention FWHT256; standalone/caller-owned results match explicit component chains, finite nonzero output and transformed-attention parity are checked, FWHT/Qf and FWHT/weight aliases are rejected, foreign weights fail preflight, and an invalid GEMM selector after FWHT, gate and dequantization encoding discards both API forms. Production Prefill wiring remains open |
| `metal_gdn_test_concurrent` | [04](04-metal-kernels.md), section 10.5; D-IMPL-263/264/268/269/272/313/314/315/316; REV-110/111/112/113 | Runs the fixture set with `STRATA_METAL_DISPATCH=concurrent`; T=1..8 covers the generic router→top-10 route→Q8_1 prefix for 512- and 256-expert routers, including padded-column sink oracles, T=3 undersized padded-sink rejection, T=2 caller-owned rollback after a late GDN active-row failure, and output-alias rejection; T=1..8 generic router→plan composition checks full/tiered plan and usage oracles, with T=3 short-plan rejection and T=2 caller-owned rollback after the plan prefix succeeds but a later consumer fails; T=1 covers attention GR-read→GDN→FFN GR-read parity, exact mixed/scratch/GDN-output handoffs, swapped injection, distinct attention/FFN weights, residual folding, speculative state/history, and late FFN rollback; the router prefix checks standard 512-expert and Coder 256-expert BF16 logits against the bitwise scalar oracle, route IDs and U(16) weights against the route oracle, and requires prefix weights to bitwise match a direct reciprocal-mode GPU dispatch while a division-mode negative control differs on the same logits. It also checks Q8_1 input bytes against the Q8_1 oracle, read-only router weights, scratch guards, short-sink and router-output/parameter alias rejection, and caller-owned segment discard. The attention→GDN fixture also rejects invalid nc/n_keep/token/StepRow, split handoffs and cross-stage aliases. T=1..8 covers speculative and write-through GDN blocks; write-through output matches the speculative output bitwise, recurrent state is checked against the scalar oracle, convolution history advances exactly, and mismatched T/n_keep/nc plus parameter overlap with writable state are rejected before dispatch. An undersized final Q8_1 output proves standalone and caller-owned write-through rollback after state/history dispatch encoding, preserving every buffer. Speculative attention→GDN and attention→GDN→FFN-read coordinators run for T=1..8 in standalone and caller-owned forms, match direct component chains and keep GDN state/history unchanged; malformed T/nc/n_keep/token/StepRow and cross-stage alias cases are rejected before dispatch. At T=2, undersized GDN active-row and FFN up-weight views cause late failures after earlier stages encode; standalone and caller-owned forms discard the segment and preserve output, residual, injection, recurrent state and history. The shared Encoder path runs T=2 and T=8, including open-group rejection. T=2 covers caller-owned GDN block and accepted-prefix commit, T=3 covers padded sink writes and n_keep=2 replay, cross-stage aliases are rejected, late block/commit gamma validation discards prior dispatches, and coordinators reject calls inside an open concurrent group |
| `metal_window_gdn_commit_batch_test` and `metal_window_gdn_commit_batch_test_concurrent` | [04](04-metal-kernels.md), section 5.5; [06](06-engine-host-port.md), L10; D-IMPL-309/310/312; REV-106/107/109 | Commits `n_keep=2` across all 36 session GDN ordinals, PLE history and all 12 QSA indexers in one ordered transaction. The fixture snapshots each QSA layer, speculatively appends three rows, then restores and replays the accepted two; it compares QSA tails/block positions exactly and dead/pooled rows against per-layer scalar oracles. The rejected third row leaves stale `pooled[n_bid]`, which is ignored because selection reads `dead` for that tail block. It also compares GDN histories and recurrent states against independent layer-specific oracles, advances PLE history from the two accepted rows, allows the QSA raw-key read to share the GDN qkv-replay read view, rejects a QSA tail/PLE-history write/write alias and a QSA raw-key read alias with writable PLE history, and verifies a malformed final QSA view discards earlier GDN/PLE work. Replay inputs and unused commit scratch remain unchanged. Serial and concurrent CTests exercise the same full state commit |
| `metal_qsa_projection_test`, `metal_qsa_projection_test_concurrent` | [04](04-metal-kernels.md), section 10.5; D-IMPL-393; REV-189 | T=2 Q4_0 K and Q8_0 V/Q projections plus BF16 indexer K/Q projections match independent FP64 scalar oracles in modes 0/1/2. Standalone and caller-owned Encoder forms check serial/concurrent ordering, shared dequant scratch reuse, input immutability, guards, invalid extents, cross-stage aliases and open-group rejection. An invalid GEMM selector after native dequant encoding proves both APIs discard the unsubmitted segment |
| `metal_qsa_indexer_commit_batch_test` and `metal_qsa_indexer_commit_batch_test_concurrent` | [04](04-metal-kernels.md), sections 4.6/5.6; [06](06-engine-host-port.md), L10; D-IMPL-311; REV-108 | Runs T=8 from position 1 with `n_keep=3` across all 12 QSA layers through snapshot, speculative append and accepted-prefix replay in one Encoder segment. The accepted position 3 completes block 0; rejected position 7 writes pooled row 1 at the accepted `n_bid` and spare row 2 beyond it, while replay restores the visible prefix. Per-layer raw keys/gamma vary; tail bits and block positions are exact, pooled/spare rows match independent scalar oracles, and replay inputs remain unchanged. It rejects a short layer list, distinct valid StepRow/RoPE/snapshot views, a cross-layer output/input alias and a malformed final-layer view; it permits output reuse beyond active StepRow and RoPE extents, and caller-owned failure discards an earlier encoded replay. Serial and concurrent dispatch modes exercise the same batch |
| `metal_gr_test_concurrent` | [04](04-metal-kernels.md), sections 5.1/5.4/5.11; D-IMPL-335; REV-132 | Runs the GR kernel suite with `STRATA_METAL_DISPATCH=concurrent`; shared Encoder GR norm, down/injection, up, cvec, post, down-SiLU, pre-gated and ordered GR-read composition fixtures are checked against their scalar oracles, including a caller-owned Encoder T=1 chain; the final Window head mixer runs T=1..8 through norm, BF16 down, SiLU, BF16 up and pre-gated dispatch with synthetic sparse weights, exercises caller-owned Encoder T=8, and verifies late-stage command-buffer discard; the GR-read case rejects an open concurrent group; its pending-MoE no-PLE selector matches the direct GR oracle for T=1..8 and rejects T=0/T=9 |; the MTP stage-20 caller-owned T=1 mixer matches its scalar head-mixer oracle and checks alias rejection, rollback and open-group refusal
| `metal_list_devices_test` | [08](08-build-and-distribution.md), section 6.4; [06](06-engine-host-port.md), work item H3 | Device and OS facts without compiling pipelines; ignores the capture environment setting |
| `metal_self_test` | [08](08-build-and-distribution.md), section 6.4; [06](06-engine-host-port.md), work item H4 | The explicit `strata --self-test` CLI path builds pipelines and dispatches the smoke-test kernels |
| `metal_fault_test` | [06](06-engine-host-port.md), work item H3 | With API and shader validation enabled by the test, a test-only kernel compiled at run time reads 1 TiB past its buffer. `MTLCommandBuffer.logs` must identify the bad offset and kernel, and Strata must print it with the command-buffer label. The faulting status may be `Completed` or `Error`; a following command buffer must complete and write its result. |
| `metal_twin_selftest` | [04](04-metal-kernels.md), work item L2 | The three planted faults of L2, compiled at run time like `metal_fault_test`: each must fail its twin with the seed and the first failing index. Its shared dispatch harness uses the Context fence helpers, so twin dispatch follows the same untracked-buffer ordering rule as engine encoders. It keeps the harness honest after L2 is done. |
| `metal_embed_test` | [04](04-metal-kernels.md), section 10.5 | Q3_K, IQ3_S, IQ4_XS and BF16 token gathers against ggml CPU, image-row copies and GR stream broadcast on `random`, `edge` and `fixture` sets; checks embedding T=1/7, image-copy T=1..8 under row-zero and after-text flag patterns (including T=1 with and without an image), image-token no-write behavior, exact output bits and buffer guards. A same-command-buffer GPU token producer replaces a stale out-of-range mapping before gather; the gather safely signals a GPU-produced OOV ID with quiet NaNs. `encode_embed_rows` rejects zero/overflowing rows, empty or flag-colliding vocabulary sizes, unsupported formats, undersized/misaligned/out-of-allocation table/token/output views, insufficient table rows, non-image IDs outside the vocabulary, same-handle and cross-handle output aliases, missing CPU backing metadata, foreign Context resources and enqueued/committed command buffers; it accepts overlapping read-only table/token inputs and skips vocabulary lookup for image flags. `encode_embed_image` rejects zero rows, shader-index overflow, short T=2 token/image/output views, misalignment, out-of-allocation views, exact and partial output overlaps with each input, valid same-handle slice overlap checks and separate forged CPU/GPU metadata rejection, distinct no-copy handles over shared backing, missing or inconsistent CPU backing pointers, foreign Context resources and enqueued/committed command buffers; overlapping read-only inputs are accepted. `encode_gr_broadcast` preserves stream bits at T=1/3/8 and rejects zero/overflowing rows, short/misaligned/out-of-allocation views, exact/partial input-output overlap including cross-handle aliases, missing CPU backing metadata, foreign Context resources and enqueued/committed command buffers. The `encode_prefill_embedding` coordinator is exercised end to end for text-only and image chunks through both its standalone wrapper and caller-owned Encoder form: it matches ggml-decoded embeddings, image-copy payloads and all four broadcast streams bitwise, rejects token-buffer drift, and verifies failed image and broadcast validation discards the command buffer before dispatch without changing output bytes. Concurrent mode rejects an open concurrent group and checks that a later token-metadata failure discards an earlier broadcast dispatch in the same segment. It also rejects an enqueued command buffer without committing it, races a failed encode against an enqueue to verify atomic discard, verifies `depends_on` rejects a consumer submitted before its producer, and checks an image-producer/broadcast-consumer chain across two command buffers on the Context fence. The T=1 Window input stage is compared bitwise with ggml CPU and exact four-stream broadcast; it checks read-only token/table inputs, rejects T=2 and out-of-vocabulary input, and verifies a late broadcast failure discards the earlier gather. |
| `metal_prefill_test` | [04](04-metal-kernels.md), section 10.5; D-IMPL-355/375/376/377/378/380/381; REV-152/172/173/174/175/176/177 | Prefill GR normalization T=1/3/8 covers BF16X2 modes 0/1/2 and random, edge (including an order-sensitive reduction sentinel), and fixture inputs. A scalar oracle reproduces the 256-thread FMA accumulation, per-simdgroup XOR butterfly, padded second butterfly, RNE high BF16 output, and scale (U(4)); the remainder is checked bitwise from the GPU scale and high output, avoiding amplification of the documented rsqrt deviation near zero. Fused GR write/next-read normalization covers T=1/3/8 in all modes and compares residual, scale, high and mode-1 low outputs bitwise with the separate GR-post plus normalization GPU chain. Prefill GDN gates cover T=1/3/8 against scalar softplus and sigmoid oracles, with branch/tail, input-overlap, immutability, guard, alias, ownership and rollback checks. Prefill GDN tiled convolution covers T=1/3/64/65/129 in both standalone and caller-owned Encoder forms against a scalar four-tap/FMA/SiLU oracle, including a cancellation-sensitive non-dyadic fixture whose negative control distinguishes reversed FMA order; history/QKV/weight inputs remain immutable, guards are checked, and validation covers row bounds, short views, output/input aliasing, foreign Context, rollback and open concurrent-group rejection. Convolution-history updates cover T=1/2/3/8/65 in both API forms against an exact oracle, with T=1/2/65 same-segment convolution→history ordering cases; history/QKV overlap, short/misaligned/foreign views, row bounds, rollback through both overloads and open-group rejection are checked. GDN q/k L2 covers T=1/2/3/8/65 in standalone and caller-owned Encoder forms against an ordered four-simdgroup oracle, with order-sensitive regrouping and omitted-epsilon negative controls, unchanged v heads, convolution→L2 composition, guards, validation, both-overload rollback and open-group rejection. GR SiLU T=1/3/8 covers modes 0/1/2 against a scalar `x / (1 + exp(-x))` oracle after the documented `lo / 4` scaling; high BF16 is U(1) and the low remainder is U(2). Exact high-output RNE halfway fixtures run in all modes; mode 1 includes exact positive and negative low-remainder halfway cases. GR mix covers the complete T=1/3/8 × BF16X2 mode 0/1/2 matrix: high BF16 and F16 outputs are checked against scalar oracles, while modes 1/2 check `high + low` reconstruction within `2^-15` of the absolute-term magnitude and require the low part to improve more than a quarter of high-only outputs. Exact binary-input midpoint fixtures assert high BF16 and positive/negative low-remainder RNE bits in modes 1/2. Standalone and shared Encoder forms check input immutability and guards. The embedding gather → GR broadcast → Prefill normalization chain and the GR read/mix coordinator run in one segment; the latter covers normalization → down → SiLU → up → injection → four-stream mix in modes 0/1/2 against the direct component sequence, checks independent FP64 projection and normalization/SiLU oracles, rejects aliases from every mixer output to all five read inputs, and checks mode-1/2 remainder aliases with active read outputs. Selector failures discard both API forms; a conservative full-view overlap case fails in the mixer after the complete read prefix has encoded, then verifies that standalone and caller-owned forms discard the prefix without changing allocations. Validation rejects zero/out-of-range rows, unknown modes, missing/unexpected remainder buffers, short and misaligned views, read/write and write/write aliases, and foreign Context buffers; concurrent mode rejects an open group. Late validation failures discard earlier encoded work. GDN recurrence checks raw outputs and final state against a scalar oracle at T=1/2/3/8/65, caller-owned Encoder cases at T=1/8 and L2→recurrence composition at T=1/3/8; an ordered-reduction sentinel, input immutability, guards, invalid views, rollback and open-group rejection are covered. Individual GDN kernels are tested independently and are now also composed through a complete one-chunk block; production GR-to-GDN scheduling and `Prefill::run` remain open. GDN output normalization checks scalar F16 parity at T=1/2/3/8/65 with one-half-ULP tolerance for Metal math approximations, epsilon, and an order-sensitive four-partial sentinel whose regrouping control changes F16 bits. Exact halfway RNE bits and unsaturated positive/negative overflow are checked on GPU; both Encoder forms, immutable inputs, guards, invalid views, rollback and L2→recurrence→out-normalization composition at T=1/3/8 are covered. The one-chunk GDN block also compares Q8_0 qkv/gate/ssm_out projections, BF16 alpha/beta projections, every intermediate, persistent history/state, final `bo` and reused F16 scratch against the direct chain. It runs mode 0/1/2 at T=1 and mode 2 at T=3, exercises both API forms, and checks short `bo`, late projection-format failure, aliases, remainder validation, input immutability, guards and open-group refusal. The attention-half GR/GDN transition composes GR read/mix, one GDN chunk and fused residual/next-read normalization, and matches the direct chain in modes 0/1/2 at T=1 plus mode 2 at T=3/65. Tests cover both API forms, GR/GDN model-weight and persistent-state aliases, preservation of `rs`/`xn16` for the next read, disjoint active prefixes in overlapping arena views, late rollback and open-group refusal. A same-segment fused-transition→prepared-read/mix chain consumes the transition output in all three modes and matches a regular read/mix on the updated residual. Additional tests match the prepared APIs to the regular path, permit projection-only calls without residual/norm views, preserve prepared state and reject invalid aliases/inputs; mode 1 specifically rejects its low-rank remainder output when aliased to the residual. The transition itself still does not schedule the following read. Individual native projection tests cover all 11 formats; these synthetic component tests do not establish model-backed inference or production `Prefill::run`.  |
| `metal_ple_test` | [04](04-metal-kernels.md), section 10.5 | `strata_rms_norm_w` participates in the T=1 composed chain; its scratch-reuse query norm is snapshotted and checked directly at U(64) before overwrite. `strata_ple_gate` runs through `encode_ple_gate` and is checked against the ordered lane and 16-simdgroup butterfly oracle at U(64), for T=1..8 on `random`, `edge` and `fixture`; a T=1 case dispatches the gate twice through one caller-owned `Encoder` segment and checks both outputs. It rejects linear lane and serial simdgroup reductions with crafted controls, checks first-row batch independence, read-only inputs and guards. The encoder rejects invalid T, short T=2 key/query/output views, misalignment, out-of-allocation ranges, exact and partial output/input overlaps, foreign Context/command buffers and enqueued/committed command buffers, while permitting overlapping read-only key/query inputs; valid same-handle subviews are checked for overlap; forged CPU/GPU metadata is separately rejected as unowned. `strata_ple_broadcast` runs through `encode_ple_broadcast` and is checked against a scalar bitwise oracle for T=1..8 with signed-zero cases, input/output guards and row independence; a T=1 case dispatches the broadcast twice through one caller-owned `Encoder` segment and checks both outputs bitwise. Encoder checks invalid T, undersized T=2 inputs and output, misalignment, out-of-allocation ranges, exact and partial output/input overlaps, foreign Context/command buffers, and enqueued/committed command buffers, while permitting overlapping read-only inputs. `strata_ple_conv_residual` runs through `encode_ple_conv_residual` and is checked at U(16) against sequential history advancement for T=1..8 with F16 weights; a T=1 fixture dispatches twice through one caller-owned `Encoder` segment, checks each output against the U(16) oracle and requires the two outputs to match bitwise. Exact controls reject fused accumulation, reordered taps and residual-parenthesization changes. Encoder tests reject invalid T, one-row T=2 `ple_norm`/`gated`/`residual` views, short history and F16 weight views, misalignment, out-of-allocation ranges, exact and partial residual/input overlap with each input, foreign Context/buffers and enqueued/committed command buffers; read-only input overlaps are covered; forged CPU/GPU metadata is separately rejected as unowned. In-place output guards, input immutability and first-row batch independence are checked. `strata_ple_history` runs through `encode_ple_history` and is bitwise checked against sequential updates at n=0/1/2/8/9/10/17; a T=1 fixture rebinds `n` from 1 to 2 within one caller-owned `Encoder` segment while advancing the same history buffer with consecutive norm slices, then checks the result against the three-row oracle. It separately dispatches n=1 to a second history buffer in the same segment and checks that result against the one-row oracle. Encoder checks signed-index overflow, short history and norm views (including 16 rows at n=17), misalignment, out-of-allocation ranges, exact/partial history/norm overlap, foreign Context/command buffers and enqueued/committed buffers, accepts an empty zero-row norm view, and checks valid same-handle subviews for overlap and separately rejects forged CPU/GPU metadata as unowned. Inputs and guards are checked. The T=2 and T=8 pending-consumer PLE fixtures verify selector dispatch and late caller-owned rollback. |
| `metal_ple_test_concurrent` | [04](04-metal-kernels.md), section 10.5 | Runs the same PLE kernel, full-block projection, scalar-oracle, alias and late-failure fixtures as `metal_ple_test`, with `STRATA_METAL_DISPATCH=concurrent` to exercise the shared-Encoder dependency chain under concurrent dispatch mode; the pending-consumer PLE branch runs at T=2 and T=8 and verifies late caller-owned rollback |
| `metal_gr_test` | [04](04-metal-kernels.md), sections 5.1/5.4/5.11; D-IMPL-335; REV-132 | Weighted RMS norm, `strata_gr_norm`, `strata_gr_down`, `strata_gr_up`, `strata_gr_cvec`, `strata_gr_post`, `strata_gr_down_silu` and `strata_gr_pre_gated` on `random`, `edge` and `fixture` sets; covers GR norm/down/up/cvec T=1..8, norm and up apply on/off, cvec project/add with write on/off, disabled flag, zero-scale and uncovered-layer paths with nonzero directions, active layer-3 project/add and distinct project scale, an FMA-sensitive folded-write bit control, exact no-op results and U(64) projection/write results, U(16) folded up residual writes, bit-exact GR down injection and U(16) low-rank SiLU outputs, weighted norm 2,560/10,240 columns and gamma periods 1/4, post rows=1/8, down-SiLU T=1..8 including partial-threadgroup tails, and pre-gated T=1..8 with ten full groups per row; a fixture dispatches the 2,560-column, gp=4, four-row weighted norm twice in one caller-owned `Encoder` segment, checks both results against the U(64) oracle and compares the GPU outputs bitwise; the ordered `encode_gr_read` composition checks T=1 without a pending write and T=8 with a write against per-stage scalar oracles, checks caller-owned Encoder parity at T=1, rejects T=0/T=9 and cross-stage aliases, and verifies late-stage failure through both overloads discards earlier dispatches without changing outputs; `encode_window_head_mixer` covers T=1..8 with sparse BF16 one-hot projections against an independent RMS/down-SiLU/up/pre-gated scalar oracle, checks residual and model-weight immutability, exercises caller-owned Encoder T=8, and rejects active-row mismatch, overlapping scratch and a short up projection after prior stages while proving the command buffer is discarded; exact fixture controls require `(R*g)*rs` and ascending stream accumulation, and reject reversed norm multiplication and reordered stream sums; `encode_gr_down` rejects invalid T, foreign-context buffers/command buffers, short/misaligned/out-of-allocation views, overlaps and enqueued/committed command buffers, and validates T × 512 × sizeof(float) dynamic threadgroup memory; `encode_gr_up` rejects invalid T, short/misaligned/out-of-allocation views, overlaps and enqueued/committed command buffers; `encode_gr_norm` rejects invalid T, undersized, misaligned or out-of-allocation views, overlaps and enqueued/committed command buffers; `encode_gr_cvec` rejects invalid T/layer/mode, undersized or misaligned buffers, overlapping views and submitted command buffers; `encode_gr_post` rejects invalid T, foreign-context buffers/command buffers, undersized, misaligned or out-of-allocation views, cross-role and partial residual/output overlap, and enqueued/committed command buffers, while accepting the exact residual/output alias; `encode_gr_down_silu` rejects invalid T, foreign-context buffer/command buffer, undersized (including one-row storage with T=2), misaligned or out-of-allocation views, and enqueued/committed command buffers; `encode_gr_pre_gated` rejects invalid T, foreign-context resources, undersized, misaligned or out-of-allocation views, output/input overlap and enqueued/committed command buffers, while allowing overlapping read-only inputs; `encode_weighted_rms_norm` rejects nonpositive rows/width/gamma period, unsupported width/gamma pairs, nonpositive or non-finite epsilon, undersized/misaligned/out-of-allocation views, unsafe output aliases and enqueued/committed command buffers, while accepting exact input/output in-place and overlapping read-only input/gamma views; all nine GR encoders check Context/queue ownership. Signed-zero FMA checks, exact-bit diagnostics, edge negative controls for fused and reordered pre-gated arithmetic, read-only inputs and buffer guards; the generic pending-MoE no-PLE selector matches the direct apply=true GR oracle for T=1..8 and rejects T=0/T=9. Window/Prefill production wiring remains open under REV-029 and REV-101. |; MTP stage 20 also runs as a standalone T=1 transaction against the scalar head-mixer oracle, checks residual/model-input immutability, and rejects parameter/residual and residual/weight aliases plus late failure
| `metal_math_test`, `metal_gdn_test`, `metal_qsa_test`, `metal_qsa_test_concurrent`, `metal_attn_test`, `metal_attn_test_concurrent`, `metal_mv_test`, `metal_mv_test_concurrent`, `metal_moe_test`, `metal_moe_test_concurrent`, `metal_sampler_parity_test`, `metal_sampler_parity_test_concurrent`, `metal_util_test` | [04](04-metal-kernels.md), section 10.5 | The remaining 10 twins cover the `random`, `edge` and `fixture` sets, except that `metal_moe_test_concurrent` uses only the `fixture` set; `metal_gdn_test` also runs all four GDN kernels through shared Encoder paths at T=1/T=8, checks their validation contracts, including both zero-row commit no-ops and the step-norm threadgroup-memory layout; the fixture-only concurrent GDN CTest also checks the composed Window GDN block and its open-group rejection; QSA norm/RoPE fixture cases use shared Encoder segments at T=1/T=8 for K, Q and indexer-Q and check the shared validation contract; the Prefill-specific entry adds 65-row scalar-oracle cases in all three modes through standalone and caller-owned forms, keeps the decode limit at 1..8, accepts the inclusive Prefill shader-row limit through preflight validation, and checks standalone rollback plus invalid Prefill row bounds; QSA FWHT256 uses shared Encoder segments for rows 1, 8, 9 and 192 and checks rejection cases; QSA indexer snapshots and appends run through standalone and shared Encoder forms, preserving raw tail bits and block positions and checking stride, extent, view, alias and ownership rejection; Prefill indexer append compares its p0={0,1,3,4,6}, T={1,2,3,4,5,9,300} first/blocks/tail batch with sequential Window replay and independent scalar oracles through both forms, and rejects malformed metadata, bounds, aliases and open concurrent groups; a late shared-Encoder failure after an earlier FWHT dispatch verifies segment discard and unchanged input bytes; a shared indexer segment covers snapshot→window append→commit replay, and a late validation failure verifies discard of earlier encoded work; all four QSA KV append formats run shared Encoder T=1/T=8 fixtures, with Q4 K/V FWHT and K8V4 V-only FWHT ordered before append; Prefill-specific APIs cover T=9 through standalone and shared Encoder forms for every format against scalar oracles, reject duplicate T=9 positions and over-limit rows through both overloads, and accept the inclusive row limit through preflight only; dispatch at the maximum is not exercised because it requires multi-gigabyte input and cache views; Q4 and K8V4 append reject an open concurrent dispatch group after FWHT and verify segment discard; QSA gate T=1/T=8 and attention chunk→merge fixtures use shared Encoder forms against scalar oracles, with validation and late-failure discard; Prefill attention covers T=1/31/32/33/64, all four KV formats at T=33, same-segment selection→attention, the 32-row scratch reuse boundary, p0=2047 with more than 2,051 live keys, malformed late-row preflight, aliases and downstream rollback; Prefill QSA gate covers T=1/9/65 in all input sets, optional FWHT→gate ordering, scalar F16 parity and exact unsaturated overflow/subnormal/signed-zero results, foreign/submitted commands, forged/out-of-range views, partial aliases and rollback both before and after a successful gate followed by downstream GEMM failure; the selection→attention coordinator feeds GPU-produced top-k IDs directly through chunk/merge in shared Encoder T=1/T=8 cases for all four KV modes; score/ID/attention/gate oracles and guards cover live widths 1/2/4/5/64/65/2048/2051 and untouched ID suffixes; an 8-cell cache also runs all four KV formats with the fixed 2051-ID stride; O-class FWHT is applied for q4_0/k8v4; invalid non-interleaved cell capacity, non-fixed stride, mismatched ID view, cross-stage alias and open concurrent dispatch group are rejected; serial and concurrent CTests cover the chain; the Prefill QSA tail composes selection→attention→optional FWHT→gate→Q8_0 native projection and tests all four KV formats through serial standalone and concurrent caller-owned forms with and without FWHT256, scalar attention/gate oracles, an independent FP64 Q8_0 projection oracle under U(2K) and R(2^-16), explicit output-component parity, gate/dequant/Y active-range canaries, mismatched handoffs, partial cross-stage aliases and late full-tail rollback; these synthetic Q8_0 weights do not establish model parity. Prefill QSA preparation covers all four KV modes, both optional-rotation settings and standalone/caller-owned APIs in serial and concurrent CTests, checking scalar norm/indexer results, exact cache bytes, fixed independent rotation expectations, PrefillParams/StepRow consistency, the in-place indexer-query contract, undersized producer/StepRow handoff rejection, cross-stage alias rejection and late-stage rollback. Sampler parity covers standalone and shared Encoder greedy, split-part/merge, composite sampled and coupled-penalty paths, checks shared validation/discard cases and a same-segment penalty→sampled dependency, and exercises model/maximum-vocabulary merge scratch; concurrent mode rejects dependent calls inside an open group and discards a prior split-part dispatch; `metal_mv_test` and its concurrent-mode CTest run shared Q8_1→Q2_0 plus BF16 projections through the same binary, with nonzero BF16 scalar-oracle and guard coverage for 256-thread and 160-thread variants. The read-only BF16 API also compares standalone and caller-owned dispatches with the scalar oracle, accepts overlapping read-only activation/weight views, rejects every input/output and output/sink overlap (including partial CPU-backing overlap across distinct no-copy handles), short or misaligned views and a mismatched active-row count, and proves late shared-segment failure discards earlier encoded work; a T=1, 512-expert synthetic router shape checks the 2560-wide/256-thread path. Both serial and concurrent CTests run these cases. `metal_moe_test` also chains the synthetic shared-expert stages on the GPU for T=1..8; the Prefill QSA projected-row layer coordinator checks all four cache formats from prepared rows through native output using independent per-stage scalar oracles and a nonzero-weight FP64 projection check, and verifies that late tail validation discards preparation work in serial and concurrent dispatch; a one-row QSA pipeline fixture connects Q8_0 K/V/Q and BF16 indexer projections through all 14 steps on f16 KV, uses dense channel-varying activations, distinct per-projection weights and independent FP64 projection references, checks scalar-oracle outputs, rejects broken producer handoffs and cross-stage aliases before dispatch, and verifies full standalone/caller-owned rollback |
| `mv_multi_eq_single`, `ple_batch_eq_sequential`, `idx_batch_eq_sequential`, `pf_gdn_eq_step_norm`, `pf_sort_eq_host_grouping`, `samp_coupled_eq_host` | [04](04-metal-kernels.md), section 10.4 | The six `metal` properties |
| `metal_never_wrap_test` | [05](05-memory-and-residency.md), work item U4 | The live full-mode loader checks a valid F32-router exclusion and shard identity, exact per-tensor inclusion/exclusion inventory, rejects non-router conversion exclusions and overlapping payloads, checks planned `wrapped_unread_bytes` and exact window union, resolves the converted F32 router only to its separate BF16 arena binding, and rejects PLE boundary overreach. `metal_tiered_copy_test` also checks that tiered buffer address ranges do not overlap the inspected GGUF mapping. |
| `metal_tiered_copy_test` | [05](05-memory-and-residency.md), section 13; work items U5-U6 | Reads dense and resident ranges through bulk descriptors, then runs synthetic ExpertStore resolution through the cached descriptor. Checks exact role bytes and GPU addresses, resident-hit and lookup counters, second-pass slot-cache hits, and prefill whole-layer prefetch. A GPU-produced layer-0 plan passes through the shared-event host resolver; an actual grouped IQ2_XS kernel reads the published resident and miss-slot addresses before the layer-complete signal, and a later layer-2 plan retires those leases while the command buffer remains in flight. The failure-cleanup fixture clears the plan's group only after plan-ready, then drains the interleaved GPU/host gates and verifies the guarded consumer leaves outputs untouched. Skip-header validation and malformed-plan no-signal behavior are also checked. The test rejects malformed source generations before allocation and verifies tiered buffer address ranges do not overlap the source GGUF mapping. Production Window scheduling and full/tiered parity remain open under REV-024. |
| `metal_slot_pool_test` | [05](05-memory-and-residency.md), section 13; work item U6 | Loading-to-Ready publication, duplicate-loading refusal, abandoned-load retry, ping-pong halves and LRU selection; a delayed GPU reader pins its slot, ExpertStore retires it after the layer-complete event, and the completion callback releases any leases left over. |
| `metal_qsa_selection_test` and `metal_qsa_selection_test_concurrent` | [04](04-metal-kernels.md), section 10.5; work item L6 | `strata_sel_scores` and both top-k forms through validated encoders and independent oracles. Scoring: T=1..8 random/edge/fixture inputs, per-head ReLU, tail blocks, untouched columns and 256-group stride at 131,072 cells; shared Encoder T=1/T=8 fixtures match the same oracle and shared validation covers invalid rows/capacity, short views, inconsistent StepRows, aliases, foreign Context, forged metadata and late-failure discard. Top-k: standalone and shared-Encoder register/memory forms, including shared T=8 for both kernels; direct/radix paths; register/memory dispatch boundary; tied partial threshold blocks; NaN key 0 below −∞ and signed-zero ties at the selection boundary; highest-scoring tail widths 1..3 selected and width 0 ignored; a high-id fixture verifies all selected ids exceed 65,535; infinities and finite scores; register/memory equivalence; shared score→top-k in one segment. Shared top-k validation covers invalid rows/capacity, short score/StepRow/id views, inconsistent StepRows, id/input aliases, foreign Context and forged metadata; a late failure discards the previously encoded top-k work. Invalid StepRows, capacities, extents, alignment, aliases, foreign Context resources and submitted command buffers are rejected; input immutability, output suffixes and guards are checked. Same-segment Prefill indexer append→selection fixtures check appended pool/dead data, scores and IDs against scalar oracles; `p0=3` covers the identity path and `p0=2047` forces top-k truncation for `n_kv > 2051`. Selection scores batches of at most eight rows within groups of 256 using serially reused scratch, then runs top-k once per group. T=1/7/8/9 and 255/256/257/300 cover score and group boundaries; a long-history fixture exercises memory top-k above 33,792 blocks, while another checks the fixed 2051-ID stride and clipped tail. Scalar score/top-k parity, final scratch contents including retained rows from the prior batch, untouched ID suffixes, input immutability, guards, metadata/extent/alias/ownership rejection, open concurrent-group refusal, and late-failure segment rollback are checked through standalone and caller-owned forms. The inclusive shader-row maximum is checked through buffer preflight only because a full dispatch needs multi-gigabyte views. Both CTests run the same fixtures in serial and concurrent dispatch modes. |

**`model` (28).**

| Tests | Defined in | What it serves |
| --- | --- | --- |
| `dense_vs_gguf_test` | [03](03-model-reference.md), work item R5 | The full compare of every unlisted float row and complete source-float inventory against the GGUF for each of the seven catalogue identities; required pack-form tensors cannot be hidden as native rows, recorded conversion bytes are recomputed, and `native_experts.txt` must exactly match the expert layout derived from every GGUF shard; setup's deterministic shard-salted 32-tensor sample uses the same code |
| `ple_oracle_model_test` | [03](03-model-reference.md), work item R9 | Scans all model GGUFs for PLE tensors and compares the frozen head/tail samples and full-row L1 sums for rows 0, 12345 and 320001535; requires canonical `qwen-Q2_0` split shard 2 and `swift-IQ2_XS` split shard 1, validates their split metadata and distinct extent/offset rules, compares their packed probe rows byte-for-byte, and checks that each reader returns the same decoded values as its raw tensor bytes |
| `mtp_reader_test` | [03](03-model-reference.md), work item R10 | Checks each tag’s file type, expert count, Swift marker and two-shard/1,224-tensor identity, then runs `WeightStore::inspect` on its `mtp/rt/` directory. It validates the paths and sizes of the three binary files and the parsed, 1 MiB-bounded `dense.txt` inventory, including all 29 tensor names, formats, shapes and aligned extents; draft-vocabulary IDs and order must exactly match one of the three shipped vocabulary files. Inspection may leave persistent `.strata-artifact-set.lock` files in the parent directories of the resolved GGUF parent, selected pack directory and MTP runtime directory. Those lock-containing parent directories must be writable; the inspected model, pack and runtime directories may remain read-only. `metal_weight_store_test` separately rejects a one-byte-truncated `experts.bin` |
| `dump_formats_test` | [03](03-model-reference.md), work item R11 | `tools/metal/tf_compare.py` reads a Mac `--dump-logits` file, and `tools/make_profile.py` reads a Mac `--dump-routing` trace. The tokens part of R11 is `dump_eq_nodump`. |
| `metal_embed_model_test`, `metal_ple_model_test`, `metal_gr_model_test`, `metal_gdn_model_test`, `metal_qsa_model_test`, `metal_attn_model_test`, `metal_mv_model_test`, `metal_moe_model_test`, `metal_mtp_model_test`, `metal_prefill_model_test` | [04](04-metal-kernels.md), section 10.1 | The `real` set of each twin whose kernels read tensors. `metal_math_test`, `metal_sampler_parity_test` and `metal_util_test` read no tensors and have no `real` set. |
| `window_eq_decode`, `state_after_commit_eq_decode`, `write_through_eq_commit`, `cvec_off_eq_stock`, `mtp_kv_t_independent`, `tiered_eq_full`, `dump_eq_nodump`, `metal_dispatch_modes_test`, `prefill_chunk_independent` | [04](04-metal-kernels.md), section 10.4 | The nine `model` properties |
| `metal_full_vs_tiered` | [05](05-memory-and-residency.md), section 13 | C4's model-level part (section 4.4) |
| `metal_router_replay` | Section 4.1 | C1.4, and measurement R-1 |
| `metal_gpu_time_test`, `metal_watchdog_test` | Section 5.5; [06](06-engine-host-port.md), work item H11 | Command-buffer times; the watchdog |
| `metal_alloc_fail_test` | [06](06-engine-host-port.md), section 2.3, step 9 | Q2_0 with `STRATA_TEST_CACHE_FAIL=1`, in full mode and with `--gpu-budget-mib 37888`: the engine exits 1, prints the PLAN line, and names the buffer it could not create. The hook makes the first long-lived buffer of step 9 fail as an allocation does. |

**Model tests and their inputs.** `dense_vs_gguf_test` reads all seven completed catalogue packs under
`STRATA_TEST_MODELS` and fails when inputs are missing, incomplete or mismatched. It checks the pack's recorded shard
sizes and split metadata, then compares the unconverted float rows and recomputes converted payloads. It does not yet
run the full first-and-last 64 MiB quick-hash preflight described for T3; that check must be added with the planned
`tools/metal/test_models.py` before the model directory can be treated as pinned catalogue input. The model PLE test
checks every PLE-bearing GGUF it finds and fails if the required Q2_0 or Swift IQ2_XS source is missing. It does not
stand in for the pinned-source quick-hash preflight. The other model acceptance tests and `tests/metal/model/`
fixtures remain planned.

**The twin's `--layer` option.** Every twin with a `real` set takes `--layer l` (and `--tensor <name>`), which
restricts the set to one layer's tensors and adds, as activations, the slot inputs of layer l from a
`--dump-halves` file named by `--halves <file>`. It is the last step of a bisection (section 10.1). Work item T6
adds it to `tests/metal/twin.{hpp,cpp}`.

### 9.2 Tests outside CTest

| What | Where it runs | Why not in CTest |
| --- | --- | --- |
| `metal_memory_pressure_smoke` ([05](05-memory-and-residency.md), section 13) | `tools/metal/accept/pressure_smoke.py`, a step of the Sunday slice (section 7.2) | It starts a second process that fills the memory of the Mac, so nothing else may run at the same time |
| The acceptance runs (sections 4-8) | `accept.py` | Section 1.2 |
| `metal_layer_replay` | Built with the tests (`tests/metal/model/layer_replay.cpp`), not registered: a bisection tool (section 10.1) | It answers "where", not "whether" |
| `tests/metal/metal_shared_event_handoff` | Built with the tests, not registered: measurement M-1 ([06](06-engine-host-port.md), section 15) | Its output is a number for appendix A |
| The H-5 stall probe, the programs of `spikes/metal-tuning/` (the L measurements of [04](04-metal-kernels.md), section 11), `tools/metal/balloon.py` | By hand, or by the step that names them | Measurements and helpers |
| Upstream's `tools/needle_bench.py` and `tools/conversation_cache_*.py` | Through `accept.py` (sections 4.3 and 4.6) | P7: unchanged, and they need a running engine |
| `tools/metal/lc_dump` | The CUDA PC (section 2.5) | It links llama.cpp, not the engine |

A work item whose "Done when" is a one-off check, such as [06](06-engine-host-port.md) H13 (the thread count of
`strata-vision`) or [08](08-build-and-distribution.md) B9 (two identical `strata-vision` builds), is checked once by
its reviewer and is not in the map.

### 9.3 The Python layer

`tools/metal/run_python_tests.py` discovers current `test_*.py` modules under `tools/`, `tools/metal/` and `serve/`,
then runs each in a fresh process. It passed 35/35 modules on 2026-10-07. Rows marked planned are absent and are not
run. `tools/metal/test_map.py` is a CLI checker exercised through CTest, not a unittest module.

| Modules | Defined in | Runs on |
| --- | --- | --- |
| Mac-supported Python modules not listed individually (the separately listed modules below are excluded): `tools/test_*.py` (14 modules), `tools/metal/test_*.py` (0 modules), `serve/test_*.py` (6 modules) | [07](07-server-setup-port.md), section 10.1; current checkout discovery | Hosted macOS CI and reference Mac |
| `tools/test_artifact_set_lock.py`, `tools/test_mtp_vocab.py`, `tools/test_mtp_pack.py` | [05](05-memory-and-residency.md), sections 13-14 | macOS/POSIX shared and exclusive locks, symlink aliases, inherited lock descriptors, killed-writer partial-publication recovery, interrupted GGUF/MTP recovery, selected MTP revision matching, inventory/manifest/sidecar/hash validation, pending-marker refusal, runtime-size/index validation, and locked atomic draft-vocabulary replacement |
| `serve/test_macos.py` | [07](07-server-setup-port.md), section 10.3 | macOS Python suite; engine-and-model cases run on the reference Mac nightly, step 3 (section 7.2). |
| `tools/test_hit_curve.py` | [05](05-memory-and-residency.md), section 10.1 | macOS Python suite; tests distinct held-out routing data including aliases, copies, order and weight-only differences, snapshot consistency after source mutation, nonblocking FIFO refusal, partial runtime-compatible profiles and exact route-miss metrics |
| `tools/test_make_profile.py` | [05](05-memory-and-residency.md), section 10.1 | macOS Python suite; validates STRP dimensions, ranking bounds, producer route-count/uniqueness limits and finite non-negative weights |
| `tools/metal/test_check_drift.py` | [06](06-engine-host-port.md), work item H12 | Hosted macOS CI and the reference Mac. In a temporary copy of the tree it plants, one at a time, each kind of change of [10](10-implementation-plan.md), section 4.2: a new flag; a new `STRATA_*` variable; a new INFO key, ERR string and DONE field; a changed line in a copied range; a removed `# MAC:` marker; a changed line next to a marker in a patched Python file, and a new `sys.platform` branch there; a changed and a new CUDA reference kernel; a new model and a new pinned revision. `check_drift.py` must fail naming each. On the unchanged copy it passes, and after `--update` it passes on the changed copy. |
| `tools/metal/test_check_release_policy.py` | [08](08-build-and-distribution.md), section 11.1, work item B13 | macOS Python suite; mocks every `gh` call |
| `tools/metal/test_workflows.py` | [08](08-build-and-distribution.md), section 12 | macOS Python suite; checks pinned Actions, read-only permissions, reference-runner triggers and Dependabot cadence |
| `tools/metal/test_macos_paths.py` | [07](07-server-setup-port.md), section 8.1 | macOS Python suite; uses `samefile()` to test filesystem path identity |
| `tools/metal/test_run_python_tests.py` | [08](08-build-and-distribution.md), section 12.1, work item B14 | macOS Python suite |
| `tools/metal/test_model_headers.py` | [03](03-model-reference.md), work item R1 | macOS Python suite. Checks shard sizes and tensor totals, fixed metadata and the shared PLE shard, rejects duplicate fixture JSON keys, and invokes `ggufhdr.py` on a generated header. |
| `tools/metal/test_dense_vs_gguf.py` | [03](03-model-reference.md), work item R5 | Synthetic single- and split-shard packs; deterministic sampling; source-hash and pack-form checks for converted rows; full comparison and rejection of a flipped byte, missing or native-marked float row, stale shard size, duplicate conversion, duplicate catalogue identity and unknown model identity. |
| `tools/metal/test_accept.py` | Section 3.2 | Planned; file absent |
| `tools/metal/test_tf_compare.py`, `tools/metal/test_ladder_compare.py` | Sections 3.3 and 4.1 | Planned; files absent |
| `tools/metal/test_make_prompts.py` | Section 2.2 | Planned; file absent |
| `tools/metal/test_map.py` | Section 9.4 | macOS host CTest preset, without check 1; through the `test_map` CTest test with it |
| `tools/metal/test_test_map.py` | Section 9.4 | macOS Python suite; the extra CMake-test and `--final` failure cases |
`tools/metal/check_drift.py` itself runs in CI as a check, not as a test module ([08](08-build-and-distribution.md),
section 12.1). The format of the one-shot lines (work item H7 of [06](06-engine-host-port.md)) is checked by
`test_accept.py`'s parser cases on captured CUDA output, and again by every acceptance run, which fails on a missing
line (section 3.1).

### 9.4 `tools/metal/test_map.py`

The map has a machine form, `tests/metal/test_map.json`, schema 1:

```json
{
  "schema": 1,
  "tests": {
    "metal_mv_test": {"label": "metal", "origin": "04 10.5"},
    "tiered_eq_full": {"label": "model", "origin": "04 10.4", "planned": "L12"},
    "...": {}
  },
  "runs": ["c1", "c2", "c3", "c4", "pp", "needle", "server", "bench", "pf", "g1", "g2", "g3",
           "nightly", "weekly", "release", "family"]
}
```

`test_map.py` makes four checks. Each failure prints one line and the script exits 1. With `--final`, which
`accept.py g3` uses (section 6.3), the notes of checks 1 and 3 are failures too.

1. **CTest against the map.** With `STRATA_BUILD_DIR` set, it runs
   `ctest --test-dir $STRATA_BUILD_DIR --show-only=json-v1` and reads each test's name and `LABELS` property. The
   names and labels must equal `tests` in `test_map.json`:
   `test_map: <name>: in CTest but not in tests/metal/test_map.json`,
   `test_map: <name>: in tests/metal/test_map.json but not in CTest`, or
   `test_map: <name>: label <found>, the map says <label>`.
   An entry with `"planned": "<work item>"` is a test whose work item has not registered it yet
   ([10](10-implementation-plan.md), section 1.3). It must not be in CTest
   (`test_map: <name>: in CTest, remove "planned"`), and it prints the note `test_map: <name>: planned (<work item>)`.
   When `STRATA_BUILD_DIR` is unset, check 1 is skipped and says so.
2. **This document against the map.** The backticked test names in the tables of section 9.1 of
   `docs/mac/09-testing-and-validation.md` must equal the names in `tests`, and each test's table label must match
   its label in the map. The counts in the first paragraph must equal the counts per label. A test is therefore
   added in three places in one commit: CMake, the JSON and this document.
3. **Requirements.** `tests/metal/requirements.json`, schema 1,
   `{"schema": 1, "requirements": {"<ID>": ["<check>", ...]}}`, must list every requirement ID of
   `docs/mac/01-requirements.md` exactly once and no other ID. The IDs are the first cells of that document's table
   rows, found with the pattern `^\| (REQ-[A-Z]{3}-[0-9]{2}) \|`. Each ID has a list of checks, or the string
   `"withdrawn"` when its row says `Withdrawn` (doc 01, section 1.1); `accept.py g3` skips a withdrawn ID. An empty
   list prints the note `test_map: <ID>: no check yet`: during the phases, a check is added with the work item that
   creates it ([10](10-implementation-plan.md), section 1.3). Every check named must exist. A check is one of:
   - `ctest:<name>`, a name in `tests`;
   - `accept:<criterion>`, a criterion ID in `tools/metal/accept/criteria.py`;
   - `python:<path>`, a module of section 9.3;
   - `ci:<workflow>/<job>`, a job `<job>` under `jobs:` in `.github/workflows/<workflow>`
     ([08](08-build-and-distribution.md), section 12);
   - `review:<id>`, one of the reviews of section 7.6 (`family`, `S14`, `S20`, `B21`, `B20`).
4. **Reviews.** `tests/metal/reviews.json`, schema 1,
   `{"schema": 1, "reviews": [{"review": "<id>", "commit": "<40 hex>", "url": "https://…", "by": "<login>",
   "date": "<yyyy-mm-dd>"}]}`. Every entry must name a review of section 7.6, a 40-hex commit and an `https` URL.
   `accept.py g3` then reads, for each review a requirement names, the newest entry whose commit has the candidate's
   sources (section 6.3).

---

## 10. When a test fails

### 10.1 Bisection

**Mac against a reference** (C1, G1, the prompt-path criterion). The question is where the Mac's arithmetic leaves
the reference's.
1. `tf_compare.py --json` lists the disagreements with their margins. Take the position with the largest KL among
   those where CUDA's margin is at least m₁.
2. `ladder_compare.py` on the ladder of that text, with `--third` the llama.cpp ladder. Slot 0 is the embedding;
   slot s ≥ 1 is the residual after layer s − 1. The first slot whose Mac-to-CUDA distance is above the
   CUDA-to-llama.cpp distance, or that grows by more than 4 times from the slot before, names the layer. If the
   position is not among the ladder's 64, record a ladder of the first `position + 1` ids on both sides first
   (`record_golden.py --only ladder --ids <file>` on the CUDA PC).
3. The Mac's `--dump-halves` at the same ids, against CUDA's (`router.halves` when the position is among its 1,000,
   else a new recording), names the half: attention or FFN.
4. `metal_layer_replay --model <tag> --layer l --half attn|ffn|router --halves <CUDA file> --layers <CUDA file>`
   runs the one half of layer l on CUDA's input to that layer, and compares its output with CUDA's. Attention needs
   its state, so the replay first runs every earlier position through the layer's attention in order, from CUDA's
   slot inputs. A replay that agrees means that the error came in from an earlier layer; a replay that differs names
   the half. `metal_router_replay` is the same program's router mode.
5. Down to the kernel: a GPU capture of the window ([08](08-build-and-distribution.md), section 10.2) at that
   position, or the twin's `real` set with `--layer l` (section 9.1). The fix adds the failing input to the twin as a
   regression case in its `fixture` set.

**Mac against Mac** (C2, C3, C4 and the properties). The question is which state first differs.
1. The failure message names the first differing index; `--stats` names the window that produced it.
2. Run both arms in serve mode with `STRATA_STATE_HASH=1 STRATA_STATE_HASH_GDN=1`, and a request whose `max_tokens`
   stops at that index. The first `STATE_HASH` field that differs names the part: `kv`, a GDN layer, `mtp`, the
   `tail`, `pooled` or `dead` part of the indexer, or `ple`. If every field is equal, the cause is the last window's
   logits, not the state.
3. Compare the logits dumps of the two arms at that position.
4. Run the property that covers the path (section 9.1), with the prompt's tokens (`--tokens-file`). If it passes,
   the bug is in host code, not in a kernel.
5. Otherwise run the twin of the kernel the property names, with `row_t_independent` for a window of that T.

**Performance** (BENCH, PF, REG).
1. The host checks of section 5.1, from `result.json`.
2. `--stats`, the `gpu window` times and measurement H-4's split of a window.
3. A GPU capture compared with the previous release's capture of the same prompt.
4. `git bisect run python3 tools/metal/accept.py bench --model Q2_0 --prompt ca-4k` between the two commits. With
   `--prompt`, a run never counts for a gate (section 3.2).

### 10.2 Flaky results

There are no retries (P8). A test that fails once is reported as failed.
- A test that passes and fails on the same inputs is filed as a bug with the label `flaky`, both result folders
  attached. It stays in its gate. It leaves a gate only by a reviewed change of this document that gives the reason.
- Two `noisy` nights in a row on the same benchmark arm are filed as a bug (section 5.1).
- A NOT RUN fails the job (exit 2, section 3.2). Two NOT RUN nights in a row are a runner problem, filed as one.
- A crash is always a failure, whatever the criterion. A step over its time limit is a failure.
- A failed nightly is triaged on the next working day. A commit that turns a gate's check red is reverted if it is not
  fixed within 2 working days.

### 10.3 Changing a threshold

A threshold changes only from a recorded measurement of section 11, never because a result needs it (P3). The change
is one reviewed commit that changes this document, `tools/metal/accept/criteria.py` and appendix A together, and
nothing in the engine. `test_accept.py` then fails until the planted-difference case of the criterion matches the
new threshold.

---

## 11. Measurements

| ID | Question | How to measure | Provisional default | Decision it drives |
| --- | --- | --- | --- | --- |
| T-1 | How far apart are upstream's own valid CUDA paths? | `tf_compare.py` of `tf-*.logits` against `tf-*.nocache.logits` per text and model; `pp-32k.logits` against `pp-32k.c6144.logits` | Upstream's Q2_0 floor for both models (section 4.1); Y_pp 89.8% and 0.33 (section 4.5) | The C1.1 and PP thresholds |
| T-2 | Above which margin does the floor never flip? | From T-1's disagreements: the largest margin of a flip, over the three texts and both models | m₁ = 1.0 logit | m₁ = max(1.0, 1.5 × that margin), used by C1.1.safe, C1.1.band, C1.5, G1.tf and G1.free |
| T-3 | How far is CUDA from llama.cpp? | `tf_compare.py` of `tf-*.logits` against `lc-tf-*.logits`; `ladder_compare.py` per slot | C1.2 and C1.3 not gating | L(t) in Y(t); C1.2 and C1.3 become gating |
| T-4 | How much does a benchmark result vary from run to run? | 20 runs of each arm on `ca-4k` and `ca-32k`, each a fresh process | Coefficient of variation ≤ 2% | Three measured runs, the 5% spread rule, and the regression bar: 0.95, or 1 − 2.5 × CV |
| T-5 | Does the Mac slow down over back-to-back runs? | 10 back-to-back `mtp` runs on `ca-32k` | The 10th run within 3% of the first: no pause | Otherwise, the shortest pause that keeps the 10th run within 3% (section 5.1, step 4) |
| T-6 | How long does each run take? | `seconds` of each step in `result.json`, over 2 weeks of nightly and weekly runs | The split of section 7.2 | The nightly and weekly split; the step time limits of section 3.2 |
| T-7 | How big is the golden set, and how long does recording take? | The first recording's manifest | About 40 GB, about 2 h per model | The disk of the reference Mac; the CUDA PC's time for each compatibility review whose verdict is `golden: record again` |
| T-8 | Does a replacement reference GPU give the same golden set? | Record again on the new card; compare old with new with `tf_compare.py`; L-9 on its SASS | Run only when a card is replaced | Whether the old set stays valid; otherwise T-1 to T-3 again |
| T-9 | Which cache budget gives the pressure scenario its pressure? | `conversation_cache_parity.py --scenario pressure --cache-mib 65536`, which fails and prints the range (`pressure_budget_hint`, `tools/conversation_cache_parity.py:53-70`) | None: C3.pressure and C3.exchange do not run until it is recorded | `cc_pressure_mib` in `models.json`: the middle of the range |

**Items in other documents that change values here.**

| Item | Document | What it changes here |
| --- | --- | --- |
| R-1 | [03](03-model-reference.md), section 10 | C1.4's weight tolerance |
| R-5 | [03](03-model-reference.md), section 10 | C1.3's expectation of smooth drift |
| L-9 | [04](04-metal-kernels.md), section 11 | Rerun by T-8 for a new card |
| L-12 | [04](04-metal-kernels.md), section 11 | The half-limit of `metal_gpu_time_test` |
| L-13 | [04](04-metal-kernels.md), section 11 | The profile of a slow G1; PF-1 and L19 |
| L-14 | [04](04-metal-kernels.md), section 11 | The dispatch mode the benchmarks run in |
| M-4 | [05](05-memory-and-residency.md), section 10 | The procedure of `metal_memory_pressure_smoke` |
| M-7 | [05](05-memory-and-residency.md), section 10 | Whether the balloon of G2 uses `--lock` |
| M-9 | [05](05-memory-and-residency.md), section 10 | G2, step 3 |
| H-4 | [06](06-engine-host-port.md), section 15 | The profile of a slow G1 |
| H-5 | [06](06-engine-host-port.md), section 15 | The robustness checks of section 5.5 |
| X-8 | [07](07-server-setup-port.md), section 14 | Whether hosted Macs run the `metal` label (section 7.5) |

**Open items** ([11-risks-and-open-questions.md](11-risks-and-open-questions.md)): who owns and runs the CUDA
reference PC; where the M1-M4 family Macs come from; when a 48 GB Mac replaces the balloon of G2; whether a hosted
macOS runner is used.

---

## 12. Work items for this area

Each item becomes one task in [10-implementation-plan.md](10-implementation-plan.md). An item is done when its
"Done when" holds on the reference Mac (or, for T2, on the CUDA PC), with its result in the item's pull request.

| # | Item | Done when |
| --- | --- | --- |
| T1 | The prompt sets (sections 2.1-2.2): `tests/metal/prompts/src/`, `make_prompts.py`, the sets, `test_make_prompts.py` and `prompts_manifest_test` | Every set builds and round-trips; `--check` passes; a changed source byte fails `prompts_manifest_test` |
| T2 | `record_golden.py`, the golden manifest and the first golden set (section 2.5), with [03](03-model-reference.md) R12, and the IQ formats' `tf-code` files | The set is on the reference Mac, its manifest is in git, and T-7 is in appendix A |
| T3 | `models.json` with `cc_pressure_mib` and the `mmproj` entries, `test_models.py`, the pack fixtures of section 9.1, the disk of section 2.3 | `fetch` gives every model of section 2.3 on the reference Mac; `check` passes; a changed byte fails `check --quick` when it is in the first or last 64 MiB, and the full check otherwise |
| T4 | The planned `accept.py` framework (section 3.2): runs, options, exit codes, `result.json`, step limits, crash-report copies; `engine.py`, `criteria.py`, `fake_engine.py`, `test_accept.py` | Planned: the macOS Python suite covers planted differences; the runner dry-run lists each acceptance command |
| T5 | `tf_compare.py` and its synthetic comparison tests (section 3.3); `lc_dump` (section 2.5) | The tests pass; `lc_dump` writes files that `tf_compare.py` and `ladder_compare.py` read, and fails on a renamed graph node |
| T6 | C1 (section 4.1): `ladder_compare.py` and its test, `metal_router_replay`, `metal_layer_replay`, the twins' `--layer` option | `accept.py c1` passes on Q2_0 and Coder; a planted ulp error in one layer's `strata_gr_down` is found at that layer by the steps of section 10.1 |
| T7 | The gates (section 6): `accept.py g1`, `g2`, `g3`; `balloon.py`; `docs/mac/gates/` | Each gate's run gives its result folder; `balloon.py --gib 16` holds 16 GiB without swap-outs growing by more than 1 GiB |
| T8 | C2 (section 4.2), every arm | `accept.py c2` passes on Q2_0; arm (d) shows at least 95% accepted drafts |
| T9 | C3 (section 4.3): the tool wrappers, C3.reread, C3.repeat; T-9 | `accept.py c3` passes on Q2_0 with every check; `cc_pressure_mib` is in `models.json` for Q2_0 and Coder |
| T10 | C4 (section 4.4), part 2 | `accept.py c4` passes on Q2_0 and Coder |
| T11 | The benchmark, PF-1, PF-2 and the regression check (section 5); `engines.py`; `upstream_speed.json` | `accept.py bench` and `pf` give their numbers; the host checks stop a run on battery with exit 2; the regression check skips, with its note, against an engine that refuses a flag |
| T12 | The needles and the server run (sections 4.6 and 4.7), with S.image and `tests/metal/images/` | `accept.py needle` and `accept.py server` pass on Q2_0 |
| T13 | `metal_gpu_time_test`, `metal_watchdog_test` (with [06](06-engine-host-port.md) H11), `metal_alloc_fail_test`, the pressure smoke step (section 5.5) | The three tests pass; the smoke step's numbers are in its result |
| T14 | The nightly job's five steps and the `weekly` job (section 7.2), with the patch to [08](08-build-and-distribution.md), section 12.2 | A week of nightly and weekly runs has a result folder for every run and every slice |
| T15 | `accept.py release`, and step 6 of `verify_release.py` (section 8) | A test release in a test repository runs the eleven steps; a planted failure leaves the release a draft |
| T16 | `family_run.sh` and the first family runs (section 7.4) | One family file for each of M1-M4 |
| T17 | `test_map.py`, `test_map.json`, `requirements.json`, `reviews.json` (section 9.4) | `test_map` passes; a test added only in CMake fails it with its message; with `--final`, a planned test and a requirement without a check each fail it with theirs |
| T18 | The Python layer on the reference Mac's nightly run (section 7.2, step 3): the parity run of [07](07-server-setup-port.md), section 4.4.3 | `test_geometry_plans` and `test_geometry_equals_files` run, and pass, in the nightly log |
| T19 | Measurements T-1 to T-9 (section 11) | Each result is in appendix A, and the value it drives is updated here and in `criteria.py` (section 10.3) |

---

## 13. Where the details are

| Topic | This document | Audit and measurements |
| --- | --- | --- |
| Principles and layers | 1 | [K5 section 6.1](audit/K5-orchestration-vram-tests.md) |
| Prompt sets | 2.1-2.2 | Upstream `bench/results/2026-09-29-speed-0126/` |
| Golden set and the CUDA PC | 2.4-2.5 | [E.0-E.8](audit/A-model-and-kernels.md); upstream `bench/results/2026-09-27-esp/` |
| Engine outputs | 3.1 | [03](03-model-reference.md), section 9 |
| C1 | 4.1 | Upstream `bench/results/2026-09-27-cache-parity/`; [E.6](audit/A-model-and-kernels.md) |
| C2 | 4.2 | [06](06-engine-host-port.md), section 10.5 |
| C3 | 4.3 | Upstream `tools/conversation_cache_*.py` |
| C4 | 4.4 | [05](05-memory-and-residency.md), section 13 |
| Prompt path | 4.5 | Upstream `bench/results/2026-09-28-prefill-speed/`; [04](04-metal-kernels.md), sections 6 and 10.3 |
| Needles | 4.6 | Upstream `bench/results/2026-09-28-prefill-speed/` |
| Performance | 5 | Upstream `bench/results/2026-09-29-speed-0126/`; [appendix A](appendix-a-measurements.md) |
| Gates | 6 | [00](00-feasibility-study.md), section 7 |
| Schedule and CI | 7 | [08](08-build-and-distribution.md), section 12 |
| Release | 8 | [08](08-build-and-distribution.md), section 11 |
| Test map | 9 | [08](08-build-and-distribution.md), section 6; [04](04-metal-kernels.md), section 10; [05](05-memory-and-residency.md), section 13; [K5 sections 6.2 and 6.4](audit/K5-orchestration-vram-tests.md) |
| Measurements | 11 | [Appendix A](appendix-a-measurements.md) |
