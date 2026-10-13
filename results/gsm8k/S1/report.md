# GSM8K S1

Sizing basis: fixed context seq_len=4096 tokens, uncompressed KV = 2147.5 MB. Uncompressed baseline accuracy = 24.8%.
Configurations: 2 over 8 runs. Varying axes: low_bits.

### LOW int3

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.05 | 40.9 | 420.1 | 19.6% | 20.4% |
| 0.1 | 82.3 | 437.7 | 20.4% | 21.6% |
| 0.15 | 123.6 | 455.3 | 21.2% | 22.8% |
| 0.25 | 206.4 | 490.6 | 22.8% | 22.0% |

### LOW int4

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.05 | 40.9 | 553.0 | 25.7% | 25.2% |
| 0.1 | 82.3 | 569.2 | 26.5% | 24.4% |
| 0.15 | 123.6 | 585.5 | 27.3% | 23.2% |
| 0.25 | 206.4 | 618.0 | 28.8% | 22.8% |

### Axis effects (marginal, at the footprint each value implies)

| axis | value | runs | mean accuracy | best accuracy | mean KV size |
|---|---|---|---|---|---|
| low_bits | LOW int4 | 4 | 23.9% | 25.2% | 27.1% |
| low_bits | LOW int3 | 4 | 21.7% | 22.8% | 21.0% |

Axes whose values sit at the *same* mean KV size (balancer, scoreboard, decay) are compared at equal cost -- an accuracy difference there is a real difference in selection quality, and a cheaper value that holds accuracy is free in area terms. Axes whose values sit at different KV sizes (r, w, the bit widths) are not: part of their accuracy difference is just keeping more cache.

### Pareto front (nothing else is both smaller and at least as accurate)

| KV size (%) | accuracy | ratio | k | configuration |
|---|---|---|---|---|
| 19.6% | 20.4% | 0.05 | 40.9 | LOW int3 |
| 20.4% | 21.6% | 0.1 | 82.3 | LOW int3 |
| 21.2% | 22.8% | 0.15 | 123.6 | LOW int3 |
| 25.7% | 25.2% | 0.05 | 40.9 | LOW int4 |

### Top configurations by accuracy

| accuracy | KV size (%) | ratio | k | configuration |
|---|---|---|---|---|
| 25.2% | 25.7% | 0.05 | 40.9 | LOW int4 |
| 24.4% | 26.5% | 0.1 | 82.3 | LOW int4 |
| 23.2% | 27.3% | 0.15 | 123.6 | LOW int4 |
| 22.8% | 21.2% | 0.15 | 123.6 | LOW int3 |
| 22.8% | 28.8% | 0.25 | 206.4 | LOW int4 |
| 22.0% | 22.8% | 0.25 | 206.4 | LOW int3 |
| 21.6% | 20.4% | 0.1 | 82.3 | LOW int3 |
| 20.4% | 19.6% | 0.05 | 40.9 | LOW int3 |

