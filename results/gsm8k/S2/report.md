# GSM8K S2

Sizing basis: fixed context seq_len=4096 tokens, uncompressed KV = 2147.5 MB. Uncompressed baseline accuracy = 24.8%.
Configurations: 6 over 6 runs. Varying axes: high_tier, low_bits.

### HIGH fp16 / LOW int3

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.25 | 206.4 | 490.6 | 22.8% | 22.0% |

### HIGH fp16 / LOW int4

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.25 | 206.4 | 618.0 | 28.8% | 22.8% |

### HIGH int8 / LOW int3

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.25 | 206.7 | 436.5 | 20.3% | 21.1% |

### HIGH int8 / LOW int4

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.25 | 206.7 | 564.0 | 26.3% | 23.6% |

### HIGH int4 / LOW int3

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.25 | 206.6 | 409.4 | 19.1% | 19.6% |

### HIGH int4 / LOW int4

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.25 | 206.7 | 536.9 | 25.0% | 22.6% |

### Axis effects (marginal, at the footprint each value implies)

| axis | value | runs | mean accuracy | best accuracy | mean KV size |
|---|---|---|---|---|---|
| high_tier | HIGH fp16 | 2 | 22.4% | 22.8% | 25.8% |
| high_tier | HIGH int8 | 2 | 22.4% | 23.6% | 23.3% |
| high_tier | HIGH int4 | 2 | 21.1% | 22.6% | 22.0% |
| low_bits | LOW int4 | 3 | 23.0% | 23.6% | 26.7% |
| low_bits | LOW int3 | 3 | 20.9% | 22.0% | 20.7% |

Axes whose values sit at the *same* mean KV size (balancer, scoreboard, decay) are compared at equal cost -- an accuracy difference there is a real difference in selection quality, and a cheaper value that holds accuracy is free in area terms. Axes whose values sit at different KV sizes (r, w, the bit widths) are not: part of their accuracy difference is just keeping more cache.

### Pareto front (nothing else is both smaller and at least as accurate)

| KV size (%) | accuracy | ratio | k | configuration |
|---|---|---|---|---|
| 19.1% | 19.6% | 0.25 | 206.6 | HIGH int4 / LOW int3 |
| 20.3% | 21.1% | 0.25 | 206.7 | HIGH int8 / LOW int3 |
| 22.8% | 22.0% | 0.25 | 206.4 | HIGH fp16 / LOW int3 |
| 25.0% | 22.6% | 0.25 | 206.7 | HIGH int4 / LOW int4 |
| 26.3% | 23.6% | 0.25 | 206.7 | HIGH int8 / LOW int4 |

### Top configurations by accuracy

| accuracy | KV size (%) | ratio | k | configuration |
|---|---|---|---|---|
| 23.6% | 26.3% | 0.25 | 206.7 | HIGH int8 / LOW int4 |
| 22.8% | 28.8% | 0.25 | 206.4 | HIGH fp16 / LOW int4 |
| 22.6% | 25.0% | 0.25 | 206.7 | HIGH int4 / LOW int4 |
| 22.0% | 22.8% | 0.25 | 206.4 | HIGH fp16 / LOW int3 |
| 21.1% | 20.3% | 0.25 | 206.7 | HIGH int8 / LOW int3 |
| 19.6% | 19.1% | 0.25 | 206.6 | HIGH int4 / LOW int3 |

