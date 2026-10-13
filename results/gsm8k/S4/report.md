# GSM8K S4

Sizing basis: fixed context seq_len=4096 tokens, uncompressed KV = 2147.5 MB. Uncompressed baseline accuracy = 24.8%.
Configurations: 2 over 2 runs. Varying axes: window.

### w=64

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.25 | 206.4 | 490.6 | 22.8% | 22.0% |

### w=128

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.8% |
| 0.25 | 207.6 | 491.1 | 22.9% | 19.4% |

### Axis effects (marginal, at the footprint each value implies)

| axis | value | runs | mean accuracy | best accuracy | mean KV size |
|---|---|---|---|---|---|
| window | w=64 | 1 | 22.0% | 22.0% | 22.8% |
| window | w=128 | 1 | 19.4% | 19.4% | 22.9% |

Axes whose values sit at the *same* mean KV size (balancer, scoreboard, decay) are compared at equal cost -- an accuracy difference there is a real difference in selection quality, and a cheaper value that holds accuracy is free in area terms. Axes whose values sit at different KV sizes (r, w, the bit widths) are not: part of their accuracy difference is just keeping more cache.

### Pareto front (nothing else is both smaller and at least as accurate)

| KV size (%) | accuracy | ratio | k | configuration |
|---|---|---|---|---|
| 22.8% | 22.0% | 0.25 | 206.4 | w=64 |

### Top configurations by accuracy

| accuracy | KV size (%) | ratio | k | configuration |
|---|---|---|---|---|
| 22.0% | 22.8% | 0.25 | 206.4 | w=64 |
| 19.4% | 22.9% | 0.25 | 207.6 | w=128 |

