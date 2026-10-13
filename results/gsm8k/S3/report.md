# GSM8K S3

Sizing basis: fixed context seq_len=4096 tokens, uncompressed KV = 2147.5 MB. Uncompressed baseline accuracy = 24.6%.
Configurations: 4 over 4 runs. Varying axes: score_tag.

### fixed30.16u

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.6% |
| 0.25 | 206.6 | 490.7 | 22.8% | 21.8% |

### fixed30.16u+decaypow2

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.6% |
| 0.25 | 208.3 | 491.4 | 22.9% | 12.5% |

### fixed30.16u+decaylut128

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.6% |
| 0.25 | 207.8 | 491.2 | 22.9% | 9.7% |

### fixed30.16u+decaylut512

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 24.6% |
| 0.25 | 206.4 | 490.6 | 22.8% | 22.0% |

### Axis effects (marginal, at the footprint each value implies)

| axis | value | runs | mean accuracy | best accuracy | mean KV size |
|---|---|---|---|---|---|
| score_tag | fixed30.16u+decaylut512 | 1 | 22.0% | 22.0% | 22.8% |
| score_tag | fixed30.16u | 1 | 21.8% | 21.8% | 22.8% |
| score_tag | fixed30.16u+decaypow2 | 1 | 12.5% | 12.5% | 22.9% |
| score_tag | fixed30.16u+decaylut128 | 1 | 9.7% | 9.7% | 22.9% |

Axes whose values sit at the *same* mean KV size (balancer, scoreboard, decay) are compared at equal cost -- an accuracy difference there is a real difference in selection quality, and a cheaper value that holds accuracy is free in area terms. Axes whose values sit at different KV sizes (r, w, the bit widths) are not: part of their accuracy difference is just keeping more cache.

### Pareto front (nothing else is both smaller and at least as accurate)

| KV size (%) | accuracy | ratio | k | configuration |
|---|---|---|---|---|
| 22.8% | 22.0% | 0.25 | 206.4 | fixed30.16u+decaylut512 |

### Top configurations by accuracy

| accuracy | KV size (%) | ratio | k | configuration |
|---|---|---|---|---|
| 22.0% | 22.8% | 0.25 | 206.4 | fixed30.16u+decaylut512 |
| 21.8% | 22.8% | 0.25 | 206.6 | fixed30.16u |
| 12.5% | 22.9% | 0.25 | 208.3 | fixed30.16u+decaypow2 |
| 9.7% | 22.9% | 0.25 | 207.8 | fixed30.16u+decaylut128 |

