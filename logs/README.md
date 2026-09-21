# Job log index

Which Slurm job ran which sweep. Job logs are `logs/sweep_<jobid>.{out,err}` (from
`jobs_sweep.sh`) and `logs/mikv_<jobid>.{out,err}` (from `jobs.sh`), and the id alone
says nothing about the experiment -- hence this file.

The logs themselves are **not tracked** (`logs/*.out`, `logs/*.err` are gitignored);
this index and `index.tsv` are. `index.tsv` is appended to automatically by both
launchers at job start -- before the model loads, so a job that dies in its first
seconds still leaves a row -- with columns `jobid, started, node, kind, preset, flags`.
The table below is the same information with the run's outcome added, and is
maintained by hand; regenerate the rows from the logs when they are still around.

## What a "type" is

A sweep is a **preset** plus the axes left open on the command line. The presets are
`legacy`, `ofat`, `greedy`, `exhaustive`, `score`, `budget`, `confirm`
(`./scripts/run_sweep.sh list`), and `--sweep-mode` decides how the open axes are
walked: `grid` (full cartesian product), `ofat` (one factor at a time from a
baseline), `greedy` (coordinate descent). Results go to `results/<preset>/sweep.csv`,
appended across runs, every row carrying its own `run_id` and full configuration.

The flags column below is shortened: these pinned-in-every-recent-job flags are
dropped, so what is left is the actual experiment.

    --sweep-mode grid --budget-modes fixed_length --score-decay-applications ranking
    --score-schemes fixed --score-length-bits 30 --score-frac-bits 16 --score-signed 0
    --balancers pow2 --score-decay-schemes pow2 --window-tokens 64

`rows` is the number of `Line Retrieval accuracy` lines the log contains, i.e. MiKV
configurations that actually completed and reached the CSV.

## Jobs

| job | started | node | preset | experiment (flags, shortened) | rows | outcome |
|---|---|---|---|---|---|---|
| 26284 | 2026-08-27 19:33 | gnode3 | `-` | (preset defaults) | — | running |
| 26376 | 2026-08-28 21:35 | gnode3 | `-` | (preset defaults) | 3 | finished |
| 26377 | 2026-08-28 22:22 | gnode3 | `-` | (preset defaults) | — | running |
| 26378 | 2026-08-28 22:34 | gnode3 | `-` | (preset defaults) | 3 | finished |
| 26381 | 2026-08-29 08:45 | gnode3 | `-` | (preset defaults) | 3 | finished |
| 26382 | 2026-08-29 09:11 | gnode3 | `-` | (preset defaults) | 3 | finished |
| 26384 | 2026-08-29 09:46 | gnode3 | `-` | (preset defaults) | 3 | finished |
| 26399 | 2026-08-29 11:45 | gnode3 | `-` | (preset defaults) | — | running |
| 26405 | 2026-08-29 12:06 | gnode3 | `-` | (preset defaults) | 3 | finished |
| 26421 | 2026-08-29 12:39 | gnode5 | `-` | (preset defaults) | 3 | finished |
| 26446 | 2026-08-29 14:59 | gnode5 | `-` | (preset defaults) | — | running |
| 26447 | 2026-08-29 19:43 | gnode3 | `-` | (preset defaults) | 6 | finished |
| 26703 | 2026-08-31 22:43 | gnode5 | `-` | (preset defaults) | — | running |
| 26704 | 2026-08-31 23:53 | gnode5 | `-` | (preset defaults) | 6 | finished |
| 27011 | 2026-09-09 10:26 | gnode2 | `greedy` | started : 2026-09-09T10:26:17+05:30 | — | aborted at `nvidia-smi` |
| 27255 | 2026-09-10 16:33 | gnode5 | `--greedy` | started : 2026-09-10T16:33:01+05:30 | — | running |
| 27262 | 2026-09-10 16:33 | gnode5 | `confirm` | --low-bits 2,4 --window-tokens 32,64 | 96 | finished |
| 27365 | 2026-09-11 14:26 | gnode2 | `--greedy` | started : 2026-09-11T14:26:11+05:30 | — | aborted at `nvidia-smi` |
| 27369 | 2026-09-11 14:42 | gnode2 | `greedy` | started : 2026-09-11T14:42:52+05:30 | — | aborted at `nvidia-smi` |
| 27381 | 2026-09-11 19:06 | gnode2 | `legacy` | grid --num-samples 40 --num-records 40 --seed 0 --high-tiers fp16 --low-bits 3,4 --score-decay-schemes lut --age-lut-entries 512 --budget-ratios 0.05,0.1,0.15,0.25 | 8 | finished |
| 27382 | 2026-09-11 19:06 | gnode2 | `legacy` | grid --num-samples 40 --num-records 40 --seed 0 --high-tiers fp16,int8,int4 --low-bits 2,3,4 --score-decay-schemes lut --age-lut-entries 512 --budget-ratios 0.25,0.5 | 18 | finished |
| 27383 | 2026-09-11 19:07 | gnode3 | `legacy` | grid --num-samples 40 --num-records 40 --seed 0 --high-tiers fp16 --low-bits 2,3 --score-decay-schemes exact,pow2,lut --age-lut-entries 128,256,512 --budget-ratios 0.25,0.5 | 20 | finished |
| 27384 | 2026-09-11 19:06 | gnode3 | `legacy` | grid --num-samples 40 --num-records 40 --seed 0 --high-tiers fp16 --low-bits 2,3 --window-tokens 64,128,256 --score-decay-schemes lut --age-lut-entries 512 --budget-ratios 0.25,0.5 | 12 | finished |
| 27385 | 2026-09-11 19:06 | gnode4 | `legacy` | grid --num-samples 40 --num-records 40 --seed 0 --balancers paper,pow2 --high-tiers fp16 --low-bits 2,3 --score-decay-schemes lut --age-lut-entries 512 --budget-ratios 0.25,0.5 | 8 | finished |
| 27406 | 2026-09-12 11:34 | gnode2 | `legacy` | grid --evict --num-samples 100 --num-records 40 --seed 1 --high-tiers int8,int4 --low-bits 4 --budget-ratios 0.05,0.25 | 2 | killed — walltime |
| 27407 | 2026-09-12 11:34 | gnode2 | `legacy` | grid --num-samples 40 --num-records 70 --seed 1 --high-tiers int8,int4 --low-bits 2,3,4 --budget-ratios 0.05,0.25 | — | crashed — masked OOM |
| 27408 | 2026-09-12 11:34 | gnode3 | `legacy` | grid --num-samples 200 --num-records 40 --seed 1 --high-tiers int8,int4 --low-bits 2,3,4 --budget-ratios 0.05,0.25 | 12 | finished |
| 27409 | 2026-09-12 12:28 | gnode2 | `legacy` | grid --num-samples 40 --num-records 70 --seed 1 --high-tiers int8,int4 --low-bits 2,3,4 --budget-ratios 0.05,0.25 | — | aborted at `nvidia-smi` |
| 27460 | 2026-09-12 17:26 | gnode2 | `legacy` | grid --num-samples 40 --num-records 70 --seed 1 --high-tiers int8,int4 --low-bits 2,3,4 --budget-ratios 0.05,0.25 | — | crashed — masked OOM |
| 27461 | 2026-09-12 17:26 | gnode2 | `legacy` | grid --evict --num-samples 100 --num-records 40 --seed 1 --high-tiers int8,int4 --low-bits 4 --budget-ratios 0.05,0.25 | — | running |

## Failure signatures seen here

- **crashed -- masked OOM.** `RuntimeError: NVML_SUCCESS == ... nvmlInit_v2_()
  INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp`. This is a CUDA
  out-of-memory, not a PyTorch bug: the allocator calls NVML only while composing
  the "used by other processes" clause of an OOM message, and NVML is broken on
  gnode2/gnode3 (userspace 580.178 against an older kernel module), so the assert
  replaces the real error. Read the traceback's last frame as the allocation that
  failed. Both instances were `--num-records 70` (`t_p~1800`), where the baseline
  alone peaks at 15.4 GB of a 17.06 GB P100.
- **aborted at `nvidia-smi`.** The log stops right after
  `Failed to initialize NVML: Driver/library version mismatch`, the `.err` is empty
  and Slurm records no failure. The launchers run `set -euo pipefail`, so the
  non-zero exit from `nvidia-smi` killed the job before it loaded anything. The
  call is now guarded in both launchers; jobs 27011, 27365, 27369 and 27409 were
  lost to it.
- **killed -- walltime.** `slurmstepd: error: *** JOB ... CANCELLED ... DUE TO TIME
  LIMIT`. Size `--time` from measured cost: roughly 7.5 s per sample-eval at
  `t_p~1116`, ~24 s once a config is degraded enough to answer wrong (a wrong
  answer generates to the token cap), and `(1 baseline + N configs) x num_samples`
  evals per job.
- **rejected -- bad preset arg.** `run_sweep.sh: unknown sweep type '--greedy'`:
  the preset is positional, so `sbatch jobs_sweep.sh greedy`, not `--greedy`.

Note also that `priority_tests.sh` and `followup_sweeps.sh` are login-node
launchers that call `sbatch` themselves -- `sbatch priority_tests.sh` submits the
launcher as a job and dies on its own usage check (that was job 27405).
