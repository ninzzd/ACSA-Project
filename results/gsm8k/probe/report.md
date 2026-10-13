# GSM8K probe

Sizing basis: fixed context seq_len=4096 tokens, uncompressed KV = 2147.5 MB. Uncompressed baseline accuracy = 15.0%.
Configurations: 1 over 1 runs. A single configuration.

### fixed length (k = r*t_p) / hardware (pow-2)

| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |
|---|---|---|---|---|
| - (uncompressed) | - | 2147.5 | 100.0% | 15.0% |
| 0.25 | 207.8 | 618.6 | 28.8% | 20.0% |

### Pareto front (nothing else is both smaller and at least as accurate)

| KV size (%) | accuracy | ratio | k | configuration |
|---|---|---|---|---|
| 28.8% | 20.0% | 0.25 | 207.8 | fixed length (k = r*t_p) / hardware (pow-2) |

### Top configurations by accuracy

| accuracy | KV size (%) | ratio | k | configuration |
|---|---|---|---|---|
| 20.0% | 28.8% | 0.25 | 207.8 | fixed length (k = r*t_p) / hardware (pow-2) |

