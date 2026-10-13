# ACSA-Project — working notes for Claude

MiKV-style KV-cache compression research (Yang et al. 2024, "No Token Left Behind"):
a hardware-faithful software model of an importance-scoring IPU — channel balancer,
fixed-point score words, LUT age decay, HIGH/LOW quantization tiers — evaluated on
Line Retrieval, GSM8K and HumanEval with `meta-llama/Llama-2-7b-chat-hf` (fp16, eager
attention). Owner: Ninaad Desai (cluster account 23EC10068, git `ninzzd`). Branches
`ninaad` / `sanjay`; `main` is the PR target. Deep design rationale lives in
`docs/ninaad-doc.md` — preserve the numerics comments in the code, they are the owner's.

## Where things run (read first)

- The local path `/home/ubu-nin/mnt/param` is an **sshfs mount** of
  `/home/others/23EC10068/kvcache/ACSA-Project` on the Slurm cluster. Edits write
  through, but **nothing executes locally**: the `.venv` is a remote binary, there is
  no Slurm/GPU on the laptop.
- Execute via `ssh 23EC10068@10.5.18.100` (login node, hostname `master`; the name
  `master` does not resolve from the laptop). Login node is fine for plans, dry runs,
  merges, tokenizer checks. Anything that executes model-generated code goes through sbatch.
- Local git needs `git -c safe.directory='*'` (uid mismatch over sshfs).
- Partition `gpupart_p100`: gnode2–gnode5, 2× Tesla P100-PCIE-16GB each (17.06 GB usable,
  SM 6.0 → no bf16). venv: Python 3.12, torch 2.7.1+cu126, **transformers 5.16.1**
  (`torch_dtype` deprecation warning is harmless), human-eval 1.0.3.
- `sacct` is dead (slurmdbd refuses connections). Use `squeue -u $USER` and job logs.
- Array tasks: never `--nodelist=gnode[2-5]` (means "all at once"); use `--exclude`.

## Known cluster gotchas

- **NVML mismatch on gnode2/gnode3**: `nvidia-smi` fails ("Driver/library version
  mismatch", NVML 580.178 vs older kernel module), torch warns "Can't initialize NVML".
  Harmless — CUDA goes through libcuda. Job scripts guard `nvidia-smi` under `set -e`
  and run a real CUDA kernel as the health gate.
- **A CUDA OOM surfaces as** `NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL
  ASSERT FAILED` (the allocator's OOM message needs NVML). Treat it as out-of-memory at
  the traceback line. `PYTORCH_NVML_BASED_CUDA_CHECK` cannot help. If a long-context run
  OOMs, try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` first.
- **NFS (fileserver2) can be very slow from compute nodes.** Probe 27665 (2026-09-14)
  sat in D state (`rpc_wait_bit_killable`) importing torch, then lost its whole 45 min
  in `model.to(cuda)` paging 13 GB of safetensors over NFS. Fix now in
  `jobs_humaneval.sh`: flock'd one-time sequential copy of the model to node-local
  `/tmp/23EC10068_hf_cache` (~2.5–4.5 min), then load in ~5 s. A long silence right after
  "testing CUDA directly" means NFS, not a code hang. (`jobs_gsm8k.sh` still loads from NFS.)
- Two tasks on one node share that node's NFS link — keep big arrays spread out.
- **int16 fake-quant overflowed in fp16** (fixed 2026-09-14, `quantize_kv`): qmax=65535 >
  fp16 max 65504 → inf/NaN cache. Now upcasts to fp32 only when qmax exceeds the input dtype's
  max; int2/3/4/8 verified bit-identical to the old path. Any earlier Line Retrieval / GSM8K run
  with LOW=16 or HIGH `int16` used the broken path — distrust those results.
- Remote Python via ssh: nested single quotes break `ssh '... <<EOF'`. Write the script to a
  local file and pipe it: `ssh host '.venv/bin/python -' < script.py`. To bring back remote
  outputs (PNGs), have the script write to login-node `/tmp/23EC10068_*` and `tar -cf - .` over
  ssh stdout (send python's stdout to stderr so it can't corrupt the stream).
- Log progress lines `[humaneval uN i/164] ... | pass@1 X% avg Ys/problem`: take `tail -1` per
  log; `sort -u` keeps the *first* line per unit.

## Memory / cost envelope

- Llama-2-7b-chat fp16 + KV ≈ 13.5 GB before sweep work; Line Retrieval t_p=1800 baseline
  peaked at 15.4 GB → <1.7 GB headroom for prefill temporaries (why the fixed-point
  quantizer avoids full-size t_p×t_p copies).
- Line Retrieval: ~7.5 s/sample-eval at t_p=1116 (`--num-records 40`); ~24 s with
  `--evict` or any degraded config (wrong answers generate to the cap); records 70 → t_p 1800, ~1.6×.
- GSM8K (8-shot CoT, ~88 gen tokens): baseline 6.3 s/problem, MiKV ~12.5 s/problem.
- HumanEval (measured on probe): baseline 3.7 s/problem, MiKV 7.4 s/problem (first 10
  easy problems; harder ones generate longer — size with `SEC_PER_PROBLEM=12`).

## Code layout (scripts/)

Import DAG: `mikv_runtime → mikv_config → {mikv_quant, mikv_grids} → mikv_policy → mikv_bench → mikv_sweep`.
`mikv_runtime` enforces two load-bearing import orders (HF_HUB_DISABLE_XET before
transformers; torch/transformers before matplotlib) — both otherwise segfault.

- Line Retrieval: `mikv.py` CLI, `run_sweep.sh`, presets in `mikv_grids.py`, `jobs.sh` / `jobs_sweep.sh`, `followup_sweeps.sh`.
- GSM8K: `mikv_gsm8k{,_bench,_report,_sweeps}.py`, `gsm8k_sweeps.sh` + `jobs_gsm8k.sh`.
  Pattern: `plan` (frozen units JSON) → `run` one unit per array task, per-problem
  checkpoint parts → `merge`. Runs keyed by hash of config + generation settings + model
  (`mikv_gsm8k.run_key`), so overlapping sweeps share runs and resubmits resume.
- HumanEval (added 2026-09-13/14, same pattern): `mikv_humaneval{,_bench,_sweeps}.py`,
  `humaneval_sweeps.sh` + `jobs_humaneval.sh`. Spec: `HUMANEVAL_README.md` (it assumes an
  `eval/` layout and `--policy` flags; the implementation maps onto SweepPoint instead).
- `_greedy_decode` (mikv_policy) is shared by baseline and MiKV paths; supports
  `max_new_tokens` and `stop_strings`.

## Reference results to reproduce

- **Job 27408** (2026-09-12, commit 8419f0a, 200 samples, `--num-records 40`, t_p=1116):
  fp64 ground truth for the fixed-point score path — incl. 61.0% / 60.5% at r=0.25 and
  49.5% / 45.5% at r=0.05. A numerically neutral change to the score path must reproduce
  these; prove it cheaply first by comparing old vs new expression on a random fp16 tensor
  on the login node. (`quantize_fixed_point` now picks fp64 above 24-bit words, fp32 at/below;
  verified bit-identical for (30,16) and (17,16).)
- **GSM8K baseline**: 24.8% strict (62/250, screen set seed 0), t_p ≈ 827, gen ≈ 89 tokens.
  GSM8K `all` array (baseline + S1–S6) essentially complete as of 2026-09-13.

## HumanEval: decisions and status

- Protocol: all 164, verbatim prompt (no chat template), greedy, `max_new_tokens=512`,
  stops `\nclass `, `\ndef `, `\n#`, `\nif __name__`, `\nprint(`, `\n```` + post-hoc truncation.
- **Completion extraction**: `decode(prompt+new)[len(decode(prompt)):]`. Decoding only the new
  ids drops a leading space (SentencePiece) → 3-space bodies → ~0% silently. Verified 164/164
  canonical solutions round-trip.
- Scoring runs **inside the array task** via `human_eval.execution.check_correctness` (3 s,
  same as the official CLI), TMPDIR on node-local scratch. `merge` never executes code.
  The `exec(...)` line in the venv's `execution.py` is already uncommented.
- **Window**: HumanEval t_p is 48–433 (median 136), so with the RTL's w=64 and r≤0.5,
  k≤64 on nearly all problems → pure recency (157/164 at r=0.2). Reference therefore uses
  **w = k/2** (`window_ratio=0.5`); H4 keeps w=64 as a control.
- Sweeps: `baseline`, `probe` (10 problems), `sanity` (LOW=16 → ≈ baseline, ±1 problem), `rtn` (uniform int8/int4), `H1` (LOW int3/int4 × r 0.2–0.5), `H2` balancer
  paper vs pow2, `H3` fp32+exact vs Q30.16+LUT512, `H4` window. Groups `tier1` (12 runs), `all` (20 runs, ~6.5 GPU-h).
  `H5` (added 2026-09-14, not in `all`): LOW int2/int3/int4 × r 0.05/0.2/0.5 — "is tiering
  needed at all?" (r=0.05 → k=6 at median t_p). Submit with `SEC_PER_PROBLEM=60` (int2 may cap).
- Fixed-length budget freezes k at prefill and demotes generated tokens as they leave top-k, so
  the *final* cache is only ~14% HIGH at r=0.2, ~37% at r=0.5, ~3–4% at r=0.05.
- A true uniform int2/int3 control is impossible: `HIGH_TIER_MODES` = fp16/int16/int8/int4
  (enforced in `mikv_sweep.py`). Add `int2`/`int3` there if needed.
- Not available: H2O hard eviction (only `scripts/line_retrieval_scripts/eviction_baseline.patch`,
  never ported to `scripts/mikv_policy.py`); "balancer off"; an undecayed score ranking;
  explicit lowest-index tie-breaking in the policy's topk.
- Determinism: seed 0, batch 1, `use_deterministic_algorithms(True, warn_only=True)` enabled
  *after* model load, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Status 2026-09-14 (21:10)** — all HumanEval work done; merged + plotted, queue empty.
  - Job 27673 (`all`, commit 00505e3): 19/20 valid. Baseline **24/164 = 14.6%**
    (task log shows 13.6% because 10 problems came from the probe). rtn int8 23 (−1/+0),
    rtn int4 22 (−3/+1), H1–H4 all 22–26. Its `sanity` rows were the int16 bug → `parts_invalid/`.
  - Job 27780 (`sanity,H5`, commit 00505e3 + uncommitted int16 fix): complete.
    Pass / lost / gained / syntax-broken (ast.parse of prompt+completion) vs baseline:
    sanity 23 / 1 / 0 / 0; int4 r=.05/.2/.5 → 25/24/23, lost 1/1/1, broken 0/0/1;
    int3 → 21/22/23, lost 4/3/1, broken 2/0/0; int2 → 14/18/20, lost 12/8/6, gained 2 each,
    broken 33/6/1 (int2 r=.05: 14 hit the 512 cap, ~18 s/problem).
    Conclusion: tiering only matters at int2; int4 with ~3–4% HIGH is baseline-equivalent.
  - **Sanity is not bit-identical to baseline**: 161/164 texts match; HumanEval/18, 41, 147
    diverge by one early token (18 pass→fail). Near-tie flips from int16 round-trip or the
    MiKV decode path's different fp16 kernels (not separated). Noise floor ≈ ±1 problem;
    compare variants against sanity, not "0 lost/0 gained".
  - **Plot bug (unfixed)**: `_plot` in `mikv_humaneval.py` (~l.483) includes the 10-problem
    `probe` run (`complete` is per its own problem set) → spurious 30% point on the
    int4 w=0.5k curve. Fix: skip `n_done < 164`. Corrected figures (pass vs KV with CIs, H5
    lost/gained, syntax-broken) were rendered ad hoc from the summary CSV + parts, not in repo.
  - **Uncommitted** (since 00505e3): int16 fix (`mikv_quant.py`), `H5`, the HumanEval scripts
    (`humaneval_sweeps.sh` untracked), plus changes in mikv/bench/policy/sweep.
  - Next: fix `_plot` probe filter; commit; optionally a HIGH-only fp16 MiKV run to tell
    int16 round-trip apart from the decode path as the source of sanity's flips.

## Common commands (on the login node, from the repo root)

```bash
./humaneval_sweeps.sh dry <group>        # plan + t_p/k preview; no GPU
./humaneval_sweeps.sh submit <group>     # array job; resubmit = only what is left
./humaneval_sweeps.sh merge | plot       # CSVs, pass matrix, samples/, figure
./humaneval_sweeps.sh rescore <run_key>  # official evaluate_functional_correctness as a job
./gsm8k_sweeps.sh {prep|nodecheck|dry|submit|merge} [group]
./scripts/run_sweep.sh <preset> --dry    # Line Retrieval; always --dry first
squeue -u $USER                          # sacct does not work
```
Env knobs: `MAX_PARALLEL`, `SEC_PER_PROBLEM`, `FORCE=1`, `EXCLUDE=gnodeN`, `HUMANEVAL_MODEL` (must be in hf_cache; jobs run `HF_HUB_OFFLINE=1`).

## How Ninaad wants replies

- Commands first, each with a one-line explanation; not prose walkthroughs. Non-runnable
  notes go in a short trailing section. Diagnoses still get their reasoning.
- Comfortable at allocator-internals / float-mantissa level; asked for plain-language
  explanations only when explicitly requested (e.g. "explain in layman language").
- **`/mem` — save context**: when Ninaad types `/mem`, write the session's durable context
  into this file: update the status sections (jobs, results, what is pending, next step),
  new decisions, gotchas and bugs found. Replace stale lines rather than appending history;
  absolute dates; keep it terse.
