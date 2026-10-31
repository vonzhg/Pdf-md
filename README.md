# PDF-MD

PDF-to-Markdown extraction on the NCSA Delta HPC cluster, using three converters
side-by-side for comparison: **marker-pdf**, **AllenAI olmOCR**, and **MinerU**.

## Layout

```
PDF-MD/
├── <tool>_share_<date>.zip      Zipped output snapshots for sharing (see below)
├── input/                       Source PDFs (shared by all converters)
├── output/
│   ├── marker/                  marker-pdf results: <name>/<name>.md (+ figures)
│   ├── olmocr/                  olmOCR results: flat <name>.md
│   └── mineru/                  MinerU results: <name>/vlm/<name>.md (+ images, JSON, layout.pdf)
├── workspace/
│   └── olmocr/                  olmOCR scratch (intermediate state, can be wiped)
├── logs/
│   ├── olmocr/                  setup + sbatch logs
│   └── mineru/                  install + model-download + sbatch logs
└── scripts/
    ├── marker/
    │   └── run_marker.sh        Run marker on input/ → output/marker/
    ├── olmocr/
    │   ├── setup_env.sh         One-time conda env install (runs idempotently)
    │   ├── run.sh               Run olmOCR on input/ → output/olmocr/
    │   └── run_single.sh        Single-PDF smoke test (for debugging)
    └── mineru/
        ├── setup_env.sh         One-time conda env install (runs idempotently)
        ├── run.sh               Run MinerU on input/ → output/mineru/
        └── run_single.sh        Single-PDF smoke test (for debugging)
```

All scripts hardcode `PROJECT_ROOT=/u/zfeng2/Github/PDF-MD` (overridable with
`PROJECT_ROOT=...` in the environment). See the gotcha about sbatch script
copying below.

## Status

- **olmOCR install**: verified on 2026-05-08, env at `/projects/bepc/zfeng2/envs/olmocr_env`.
- **olmOCR last batch run**: job 18136522, 10 historic econ papers, 7:03
  wall time (378 s of inference after vLLM warmup), all 10 `.md` files written
  to `output/olmocr/`. Throughput ~188 words/sec end-to-end on one A100.
- **marker**: existing pipeline, was previously run on 14 Chinese-named PDFs
  (now removed from `input/`); old per-PDF dirs still present in
  `output/marker/` — see Cleanup below.
- **MinerU install**: verified on 2026-05-21, conda env `mineru`
  (`~/.conda/envs/mineru`), MinerU 3.1.15 + vLLM 0.11.2 + torch 2.9.0+cu128.
- **MinerU smoke test**: job 18385366, `input/zhang_2025.pdf` (54-page econ
  paper), `gpuA100x4-interactive`, one A100-40GB. `time mineru` = 2 m 59 s,
  of which ~1 m 50 s is vLLM engine startup + model load + CUDA-graph capture
  and only **44 s is VLM inference for all 54 pages** (~1.2 pages/s). Output:
  22,195-word `zhang_2025.md`, 90 extracted images, an annotated `layout.pdf`,
  and JSON sidecars — see `logs/mineru/sbatch_single.log`.

## Environments

| Converter | Conda env | Python | Notes |
|---|---|---|---|
| marker  | `marker` (pre-existing)                  | (existing) | Already installed prior to this project |
| olmOCR  | `/projects/bepc/zfeng2/envs/olmocr_env`  | 3.11       | Created by `scripts/olmocr/setup_env.sh` |
| MinerU  | `mineru` (named env, `~/.conda/envs/mineru`) | 3.11   | Created by `scripts/mineru/setup_env.sh`; MinerU 3.1.15 + vLLM |

- The olmOCR model (`allenai/olmOCR-2-7B-1025-FP8`, ~9.5 GB) is downloaded on
  first run into `HF_HOME=/projects/bepc/zfeng2/hf_cache` and reused thereafter.
- The MinerU VLM model (`opendatalab/MinerU2.5-Pro-2604-1.2B`, ~2.2 GB) is
  pre-downloaded by `scripts/mineru/setup_env.sh` into `~/.cache/huggingface`;
  `~/mineru.json` records the local path so runs work offline. Note the MinerU
  env lives in **home** (`~/.conda/envs/`), unlike olmOCR's project-space env —
  see the gotcha below.

## Running on Delta

Delta requires SLURM. The login node (`dt-login01`) cannot run any GPU work and
suffers Lustre stalls under heavy I/O — always submit via `sbatch` (or `srun`
to an interactive GPU node). The regular `gpuA100x4` partition can have a queue
of many hours; the `gpuA100x4-interactive` partition (1 h limit) usually
schedules far sooner and is plenty for a single-paper job.

### marker (interactive, existing flow)

```bash
srun --account=bepc-delta-gpu --partition=gpuA100x4-interactive \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
     --time=00:30:00 --pty bash
bash /u/zfeng2/Github/PDF-MD/scripts/marker/run_marker.sh
exit
```

Writes `output/marker/<name>/<name>.md` per PDF, plus extracted figures.

### olmOCR — one-time install

Run once on a CPU or GPU node (a GPU node is fine; the install itself does not
need a GPU). Takes ~45 min (downloads ~5 GB of wheels, slow on Lustre):

```bash
sbatch --account=bepc-delta-gpu --partition=gpuA100x4 \
       --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
       --time=03:00:00 \
       --output=/u/zfeng2/Github/PDF-MD/logs/olmocr/sbatch_setup.log \
       --job-name=olmocr_install \
       /u/zfeng2/Github/PDF-MD/scripts/olmocr/setup_env.sh
```

Wait for `=== Setup done ===` in the log.

### olmOCR — full batch (all PDFs in `input/`)

```bash
sbatch --account=bepc-delta-gpu --partition=gpuA100x4 \
       --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=64g \
       --time=02:00:00 \
       --output=/u/zfeng2/Github/PDF-MD/logs/olmocr/sbatch_run.log \
       --job-name=olmocr_run \
       /u/zfeng2/Github/PDF-MD/scripts/olmocr/run.sh
```

Expected runtime depends on input size. For ~10 typical PDFs (~14 MB total):
~7 min wall time (~6 min vLLM startup + cache warm + ~1 min inference). For
~14 longer PDFs: ~25–40 min. Writes `output/olmocr/<name>.md`.

### olmOCR — single-PDF smoke test

Useful when debugging the install or trying a new model. Defaults to
`workspace/olmocr/single_pdf/LI Bo.pdf`:

```bash
sbatch --account=bepc-delta-gpu --partition=gpuA100x4 \
       --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
       --time=00:45:00 \
       --output=/u/zfeng2/Github/PDF-MD/logs/olmocr/sbatch_single.log \
       --job-name=olmocr_single \
       /u/zfeng2/Github/PDF-MD/scripts/olmocr/run_single.sh
```

### MinerU — one-time install

Creates the `mineru` conda env, installs `mineru[all]` (MinerU 3.x + vLLM +
torch CUDA wheels) and pre-downloads the VLM model. Needs no GPU — only
internet — so it can run on a login node, but a compute node avoids Lustre
stalls. Already done once (2026-05-21); re-run only to rebuild/upgrade:

```bash
sbatch --account=bepc-delta-gpu --partition=gpuA100x4 \
       --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
       --time=03:00:00 \
       --output=/u/zfeng2/Github/PDF-MD/logs/mineru/sbatch_setup.log \
       --job-name=mineru_install \
       /u/zfeng2/Github/PDF-MD/scripts/mineru/setup_env.sh
```

Wait for `=== Setup done ===` in the log.

### MinerU — full batch (all PDFs in `input/`)

```bash
sbatch --account=bepc-delta-gpu --partition=gpuA100x4-interactive \
       --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
       --time=01:00:00 \
       --output=/u/zfeng2/Github/PDF-MD/logs/mineru/sbatch_run.log \
       --job-name=mineru_run \
       /u/zfeng2/Github/PDF-MD/scripts/mineru/run.sh
```

`run.sh` stages all not-yet-converted PDFs and invokes `mineru` **once** over
them, so the vLLM engine is loaded a single time for the whole batch. Per-page
inference is ~1.2 pages/s on an A100; budget ~2 min of fixed vLLM startup on
top. Writes `output/mineru/<name>/vlm/<name>.md`.

### MinerU — single-PDF smoke test

Defaults to `input/zhang_2025.pdf`:

```bash
sbatch --account=bepc-delta-gpu --partition=gpuA100x4-interactive \
       --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
       --time=00:45:00 \
       --output=/u/zfeng2/Github/PDF-MD/logs/mineru/sbatch_single.log \
       --job-name=mineru_single \
       /u/zfeng2/Github/PDF-MD/scripts/mineru/run_single.sh
```

All run scripts skip already-converted work:
- **marker**: skips a PDF if `output/marker/<name>/<name>.md` exists.
- **olmocr**: skips a PDF whose hash has a `done_flags/` entry under
  `workspace/olmocr/run/`. Wiping that workspace forces a full re-OCR (deleting
  files in `output/olmocr/` alone does **not** trigger reprocessing — olmocr
  doesn't read its own output).
- **mineru**: skips a PDF if its `output/mineru/<name>/` folder exists. Delete
  that folder to force re-extraction.

## MinerU output structure

For each PDF, MinerU (VLM backend) writes a `<name>/vlm/` folder:

| File | Contents |
|---|---|
| `<name>.md`                  | Extracted Markdown (the main result) |
| `<name>_content_list.json`   | Ordered list of content blocks with page numbers |
| `<name>_content_list_v2.json`| Newer richer block schema |
| `<name>_middle.json`         | Full intermediate layout/parse representation |
| `<name>_model.json`          | Raw VLM model output per page |
| `<name>_layout.pdf`          | Original PDF annotated with detected layout boxes |
| `<name>_origin.pdf`          | Copy of the source PDF |
| `images/`                    | Extracted figures/tables as `.jpg`, referenced from the `.md` |

## Verifying results

```bash
# marker
ls output/marker/ | wc -l
wc -w output/marker/*/*.md | tail -5

# olmOCR
ls output/olmocr/*.md | wc -l
wc -w output/olmocr/*.md | tail -5

# MinerU
ls output/mineru/ | wc -l
wc -w output/mineru/*/vlm/*.md | tail -5
```

## Sharing results

Output snapshots are zipped to the project root, dated by day:

```bash
cd /u/zfeng2/Github/PDF-MD
zip -r mineru_share_$(date +%Y%m%d).zip output/mineru
```

Current snapshots:

| Archive | Contents |
|---|---|
| `olmocr_share_20260509.zip` | olmOCR output as of 2026-05-09 |
| `mineru_share_20260521.zip` | MinerU output (`zhang_2025`) as of 2026-05-21 — 101 files, 3.0 MB |

## Watching a running job

```bash
squeue -u $USER                                                  # running/pending
squeue -j <JOBID> --start                                        # estimated start time
tail -f /u/zfeng2/Github/PDF-MD/logs/mineru/sbatch_run.log              # live output
scancel <JOBID>                                                  # cancel
```

## Notes / gotchas

### olmOCR

- **`--max_concurrent_requests 64`** is set in `scripts/olmocr/run.sh`. The
  upstream default is 1600, which deadlocks vLLM on a single A100 (engine
  silently stalls while the API server stays bound) and also caused SLURM
  OOM-kills (~600 page images held in host RAM at once). Do not raise this
  above ~128 on a 40 GB A100.
- **`--mem=64g`** for the run sbatch. 32 GB was not enough at high concurrency;
  64 GB has plenty of headroom even if you raise `--max_concurrent_requests`.
- **FP8 model on A100**: the default model (`allenai/olmOCR-2-7B-1025-FP8`)
  is FP8-quantized. A100 lacks native FP8, so vLLM falls back to the Marlin
  kernel — correct results, ~30 % slower than on H100/H200. A non-FP8 variant
  can be selected with `--model allenai/olmOCR-7B-0225-preview` if needed.
- **olmocr output nesting**: olmOCR writes
  `workspace/markdown/<absolute-input-path>.md`. `scripts/olmocr/run.sh`
  flattens this with `find ... -exec cp` so `output/olmocr/` ends up flat.

### MinerU

- **Backend** is `vlm-auto-engine` (set in `scripts/mineru/run*.sh`,
  overridable via `MINERU_BACKEND=`). MinerU 3.x backends:
  `pipeline` (CPU-capable, classic OCR), `vlm-auto-engine` (VLM model on the
  local GPU via vLLM — what we use), `vlm-http-client` (VLM via a remote
  server), and `hybrid-auto-engine` / `hybrid-http-client` (v3's new default,
  "next-generation" accuracy — combines pipeline + VLM, needs the pipeline
  models too: `mineru-models-download -m all`).
- **Offline model use**: the run scripts export `MINERU_MODEL_SOURCE=local` and
  `HF_HUB_OFFLINE=1` so compute nodes never reach the network. `~/mineru.json`
  records the local model path written by `setup_env.sh`. To re-download or
  switch sources, re-run `setup_env.sh` (uses `-s huggingface` by default;
  `MINERU_MODEL_DL_SOURCE=modelscope` switches to ModelScope).
- **Single vLLM load per batch**: `run.sh` symlinks all pending PDFs into a
  temp dir and calls `mineru` once. Invoking `mineru` per file would reload the
  ~2 GB model and re-pay the ~2 min vLLM startup for every PDF.
- **`onnxruntime ... pthread_setaffinity_np failed` warnings** flood the log on
  startup. These are harmless — onnxruntime (used for an auxiliary layout step)
  cannot pin threads inside a SLURM cgroup and falls back fine. Output is
  unaffected.
- **GPU sizing**: the VLM model is only 1.2 B params; a 40 GB A100 (or A40) is
  more than enough. `--mem=32g` host RAM is sufficient.
- **Env location**: the `mineru` conda env is a *named* env in home
  (`~/.conda/envs/mineru`), unlike olmOCR's path-based env under
  `/projects/bepc/`. Home quota is finite (~100 GB) — `du -sh ~/.conda/envs/mineru`
  to check. To move it to project space, recreate with
  `CONDA_ENV=/projects/bepc/zfeng2/envs/mineru_env bash scripts/mineru/setup_env.sh`
  (conda envs cannot be safely relocated by `mv`).

### General

- **Lustre on the login node**: `pip install` can hang for hours in
  `cl_sync_io_wait` on `/projects/bepc/`. Always submit installs via `sbatch`
  (compute nodes have a faster Lustre client).
- **CJK + spaces in filenames**: `LC_ALL=en_US.UTF-8` is exported in the
  scripts; PDF paths with Chinese characters or spaces (`LI Bo.pdf`) work.
- **sbatch script copying**: SLURM copies the submitted script to
  `/var/spool/slurmd/job<id>/` before executing, so `BASH_SOURCE[0]` and
  `dirname` resolve to that temp path — not the original location. The scripts
  therefore hardcode `PROJECT_ROOT="${PROJECT_ROOT:-/u/zfeng2/Github/PDF-MD}"` rather
  than deriving it from script location. Override at runtime if you ever move
  the project, e.g. `PROJECT_ROOT=/elsewhere bash scripts/mineru/run.sh`.

## Cleanup

Safe to remove (leftovers from initial debugging / earlier input set):

```bash
# Failed-run scratch dir from the original 14-PDF deadlock
rm -rf /u/zfeng2/Github/PDF-MD/workspace/olmocr/run_failed

# Smoke-test artifacts (LI Bo from the old Chinese-PDF input set)
rm /u/zfeng2/Github/PDF-MD/output/olmocr/'LI Bo.md'
rm -rf /u/zfeng2/Github/PDF-MD/workspace/olmocr/single
rm -rf /u/zfeng2/Github/PDF-MD/workspace/olmocr/single_pdf

# Old marker per-PDF dirs from the previous Chinese-named input set
# (only delete if you don't need those marker results anymore)
rm -rf /u/zfeng2/Github/PDF-MD/output/marker/{'LI Bo',吴璟,孙昂,徐铭梽,李瑶,王自干,盖庆恩,端木昊随,裴建锁,许志伟,谢觐,车翼,高华声,黄炜}
```

The conda envs (`/projects/bepc/zfeng2/envs/olmocr_env`, `~/.conda/envs/mineru`),
HF caches (`/projects/bepc/zfeng2/hf_cache`, `~/.cache/huggingface`), `input/`,
and current `output/` should be kept.

## Logs

### olmOCR install/setup session (2026-05-08)

`logs/olmocr/` contains chronological logs of how the env got installed and how
the pipeline was tuned — useful as reference if you ever need to rebuild from
scratch:

| File | What's in it |
|---|---|
| `setup.log`, `setup_resume.log` | Login-node `pip install` attempts that hung in Lustre I/O |
| `sbatch_setup.log` | Successful install on a GPU node |
| `sbatch_single.log` | First successful single-PDF run (`LI Bo.pdf`) |
| `sbatch_run.log` | Most recent batch run (job 18136522) |

### MinerU install/test session (2026-05-21)

| File | What's in it |
|---|---|
| `install.log` | `uv pip install mineru[all]` — 221 packages, vLLM + torch CUDA wheels |
| `model_download.log` | `mineru-models-download` — VLM weights (2.2 GB) + `~/mineru.json` |
| `sbatch_single.log` | Smoke test on `zhang_2025.pdf` (job 18385366), 54 pages in 2 m 59 s |
