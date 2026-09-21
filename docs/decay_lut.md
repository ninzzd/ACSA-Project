# Context brief: the `ipu_age_lut` reciprocal table

*A self-contained handoff document. Assumes no prior knowledge of this project,
of LLM inference, or of the surrounding RTL. Everything needed to reason about
the LUT design decision is here.*

---

## 1. The system this sits inside

### 1.1 What an LLM KV cache is

A transformer LLM generating text keeps, for every token it has already seen, a
pair of vectors (a "key" and a "value"). This store is the **KV cache**. At every
new generation step the model attends over the whole cache, so the cache grows by
one entry per token and is re-read in full every step. For a long context it
becomes the dominant consumer of memory bandwidth and capacity.

### 1.2 What this project does about it

This project is a **precision-tiered KV cache**. Instead of *evicting* old tokens
(deleting them, which loses information irrecoverably), it keeps everything but
stores less-important tokens at **lower numerical precision**:

- **high tier** — full precision, capacity `B_hi` tokens
- **low tier** — quantized, holds everything else, unbounded

Moving a token from high to low tier is called **demotion**. Demotion is
**irreversible**: `HIGH -> MIGRATING -> LOW`, never back.

### 1.3 What the IPU is

The **IPU (Importance Processing Unit)** is a hardware block, written in RTL,
whose entire job is to decide **which token to demote next**. Once per decode
step, per attention head, it must nominate one victim.

It does this by maintaining an **importance score** `S_i` for every token `i`:
the running sum of all attention weight that token has ever received.

```
every decode step:   S_i <- S_i + a_{t,i}        for all i
```

`a_{t,i}` is the attention probability the model just assigned to token `i`. The
victim is the **eligible token with the smallest score** (an argmin).

### 1.4 The three rules that shape everything downstream

1. **`W_recent = 64`.** The 64 most recently generated tokens are *exempt* — they
   can never be demoted, no matter how low their score. A separate block,
   `ipu_mask_gen`, masks them out before the argmin. This matters enormously
   below.
2. **`T_MAX = 4096`.** Longest supported sequence, so a token's **age**
   (`age = t_current - token_index`) lies in `1 .. 4096`.
3. **`P = 16` lanes.** The hardware processes 16 tokens per cycle, in a pipeline.
   Sixteen of everything, every cycle.

---

## 2. The specific problem: the `ipu_age_lut` module

### 2.1 Why it exists

Raw score `S_i` favours old tokens: a token alive for 3000 steps has had 3000
chances to accumulate attention; a token alive for 100 steps has had 100. In the
optional **age-normalized policy mode**, the quantity actually compared is

```
cmp_val = S / age
```

`ipu_age_lut` computes this and feeds it to the comparator tree. It sits at
pipeline stage S3, between the accumulator adders and the min-tree. It is
selectable at runtime via a CSR (`cfg_mode`: 0 = raw sum, 1 = age-normalized), and
wrapped in a generate guard so it synthesizes away entirely when unused.

### 2.2 Why a lookup table

A hardware divider is iterative and multi-cycle. Sixteen of them, on a stage that
must retire one beat every cycle, is not an option. Standard fixed-point solution:
**store scaled reciprocals in a ROM and multiply.**

```
ROM contents:   R[n] = round( 2^F / n )
runtime:        cmp_val = (S * R) >> F
```

`2^F` is a scale factor letting a sub-1 value (`1/n`) live in an integer. `F` is
"how many fractional bits of the reciprocal we keep."

### 2.3 Number formats in play

| Quantity | Format | Width | Note |
|---|---|---|---|
| score `S` | Q12.16 unsigned fixed point | 30 b | `1.0` = 65536; saturating, never wrapping |
| tier flag | — | 2 b | packed into the top of the same 32-bit word |
| age | integer | 12 b | `1 .. 4096` |
| ROM entry `R` | integer | **B = 16 b** | the design choice |
| `S * R` | integer | 46 b | 30 x 16 |
| `cmp_val` | Q12.16 | 30 b | after the shift |

Fixed point rather than FP16 is deliberate: attention weights have a long tail
around `1e-4`, and FP16 accumulation would round those to zero once `S` grows
("swamping"), destroying exactly the signal that distinguishes a marginal token
from a dead one.

---

## 3. The design decisions, and where the numbers come from

### 3.1 `W = 64` fixes the *bottom* of the table, which fixes `F`

`F` is bounded by the **largest** stored entry, which occurs at the **smallest**
age:

```
2^F / n_min  <=  2^B - 1
```

Naively `n_min = 1` and `F = 15` — half the ROM's dynamic range spent storing
`1/1`. But `ipu_mask_gen` already guarantees no token with `age <= 64` ever
reaches the comparator, so its `cmp_val` is computed and discarded. **The table
never needs ages 1-64.**

```
n_min = W + 1 = 65
F = floor( log2( (2^16 - 1) * 65 ) ) = floor( log2 4259775 ) = 22
```

Verified both directions: `2^22/65 = 64527.75` fits in 16 bits; `2^23/65 = 129055`
does not. **F = 22 is maximal.**

*Gain: F goes 15 -> 22, i.e. **+7** fractional bits, for free, from a masking rule
that already existed. (The source guide's prose says "six"; the intuition is
`65 ~ 2^6`, but the floor arithmetic yields 7. The guide's own table, 15 -> 22, is
correct.)*

### 3.2 The chosen table

```
512 entries x 16 bits = 1 KB,  F = 22,  covering ages 65 .. 576
address a = n - 65
R[a] = round( 4194304 / (a + 65) )     for a = 0..511
```

Verified spot values:

| age n | addr a | R | hex |
|---|---|---|---|
| 65 | 0 | 64528 | 0xFC10 |
| 66 | 1 | 63550 | 0xF83E |
| 100 | 35 | 41943 | 0xA3D7 |
| 128 | 63 | 32768 | 0x8000 |
| 256 | 191 | 16384 | 0x4000 |
| 511 | 446 | 8208 | 0x2010 |
| 512 | 447 | 8192 | 0x2000 |
| 576 | 511 | 7282 | 0x1C72 |

Table entries are monotonically non-increasing (required — otherwise an older
token could get a *larger* scale factor than a younger one, corrupting the
ranking) and contain zero adjacent duplicates. Both machine-verified.

**Round, never truncate,** when generating the table: truncation biases every
entry the same direction, and the relative bias grows with `n`, which
systematically favours old tokens — a silent policy change disguised as a
rounding choice.

### 3.3 Handling ages above 576: the "fold"

The table stops at 576 but ages reach 4096. The gap is closed by exploiting the
self-similarity of `1/n` under powers of two:

```
1/n = (1/n') * 2^-e      where  n ~ n' * 2^e
```

In plain terms: **you don't need an entry for 1000, because 1000 is 500 doubled,
and `1/1000` is `1/500` halved.** Halving is a shift. So the table only has to
cover one *octave*; anything larger is halved repeatedly until it lands back
inside.

```verilog
e   = max(0, msb_position(age) - 9);            // msb_position = bit_length, 1-based
n'  = (e == 0) ? age : (age + (1 << (e-1))) >> e;
R   = ROM[n' - 65];
cmp_val = (S * R) >> (22 + e);
```

Worked example, `age = 3000`:
1. 3000 needs 12 bits -> `e = 12 - 9 = 3`
2. `n' = round(3000/8) = 375`, address `375 - 65 = 310`
3. `R = ROM[310] = 11185`
4. `cmp_val = (S * 11185) >> 25`
5. Check: `11185/2^25 = 3.3337e-4` vs true `1/3000 = 3.3333e-4` -> **0.011% error**

Cost: a 12-bit priority encoder, one adder, one shifter, a 3-bit `e`.

**RTL gotchas:**
- `msb_position` must be the **bit count** (1-based), not the bit index. With a
  0-based index, age 1023 gives `e = 0` and an out-of-range address.
- `e = 0` needs an explicit special case: `1 << (e-1)` is a shift by -1.
- **Shift the product, not the reciprocal**: `(S*R) >> (22+e)`, never
  `S * (R >> e)`. Pre-shifting discards up to 4 bits of `R`.
- Round `n'` rather than truncating, for the same bias reason as the table itself.

---

## 4. Error analysis (machine-verified over all ages 65-4096)

There are exactly two error sources, and they do not interact:

**(a) Rounding of the table entry itself.** `|R/2^F - 1/n| <= 2^-(F+1)`, so
relative error `<= n / 2^23`. At n = 576 that is **0.007%** — negligible.

**(b) Rounding of `age` to `n'` during the fold.** This dominates, and it is the
*only* thing table depth controls. Halving an odd number costs half an LSB of
`n'`. After folding, `n'` lands in an octave whose top is `D + 64`, so:

```
worst-case relative error  ~=  1 / D        (D = table depth)
```

Measured, with `worst` over ages 65-4096:

| Depth | ROM | Direct to age | Worst err | RMS err |
|---|---|---|---|---|
| 128 | 256 B | 192 | 0.516% | 0.227% |
| 256 | 512 B | 320 | 0.309% | 0.134% |
| **512** | **1 KB** | **576** | **0.175%** | **0.073%** |
| 1024 | 2 KB | 1088 | 0.096% | 0.038% |
| 2048 | 4 KB | 2112 | 0.058% | 0.019% |

Two structural facts worth internalizing:

- The error **does not grow with age**. Each new octave halves the age but also
  halves the rounding error. Error is flat from 577 to 4096.
- Accuracy-per-byte is a **straight line on log-log — there is no knee.** Doubling
  the ROM always halves the error, forever. So 512 is not chosen because the curve
  bends; it is chosen against an error budget.

**Entry collapse (a non-binding constraint).** Adjacent entries differ by
`~ 2^F / n^2`, which reaches 1 LSB (entries becoming duplicates, table carrying no
information) at `n ~ 2^(F/2)`. At F = 22 that is `n ~ 2048`, far beyond the
576 top. The table is informative across its entire range.

---

## 5. The decisive argument for depth 512

Reciprocal error only matters if it changes **which token gets demoted**. A Monte
Carlo over 1500 simulated decode steps per depth (heavy-tailed scores, older
tokens accumulating more, a saturated BOS attention sink), comparing the LUT's
argmin against exact division:

| Depth | Wrong victim (LUT's fault) | How much worse that victim actually was |
|---|---|---|
| 128 | 0.73% | 0.24% |
| 256 | 0.27% | 0.14% |
| **512** | **0.07%** | **0.027%** |
| 1024 | 0.00% | 0 |
| 2048 | 0.00% | 0 |

Two conclusions:

1. **Even a wrong victim is a harmless victim.** A flip only happens between two
   tokens that were already a near-tie, and near-ties are policy-neutral by
   definition. Error in the reciprocal is not error in the decision.
2. **There is a ~1.2% floor that no table depth can touch.** Running the *full*
   integer datapath (including rounding the 30-bit `cmp_val` output) yields ~1.2%
   victim disagreement at **every** depth from 128 to 2048. That floor comes from
   output quantization, not the table. **Past 512 entries you are buying ROM that
   changes nothing**, because the output rounding is already ~10x noisier than the
   reciprocal. This, not the reciprocal's own accuracy, is the real justification
   for stopping at 512.

---

## 6. Implementation details of the module

**The ROM port problem.** Sixteen lanes each need a lookup, and 16 read ports is
unaffordable. But lane indices within a beat are always **consecutive**, so their
ages are consecutive too. Organize the ROM as **32 rows x 16 consecutive entries
x 16 bits**, read **two adjacent rows**, and align the 16-entry window with a
16-way barrel rotator. Two read ports instead of sixteen.

**Gate count.** The sixteen 30x16 multipliers are by a wide margin the largest
arithmetic block in the entire IPU — hence the generate guard, since the raw-sum
policy (no normalization) is the common case.

**Overflow.** `cmp_val = S/n` with `n >= 65` is always smaller than `S`, so the
30-bit result cannot overflow. The saturation logic is belt-and-braces.

**Testbench assertions:**
- `R[a] >= R[a+1]` for all `a` (ranking correctness)
- Exactness at powers of two: `R[128]=32768`, `R[256]=16384`, `R[512]=8192`, zero
  rounding error — free spot checks
- `n' - 65` always within `[0, 511]`
- Coverage bins on `e`

---

## 7. Open items / known discrepancies

1. **The MSB fold rule never addresses the top 64 entries.** With
   `e = max(0, bit_length(age) - 9)`, generated addresses only ever span 0-447
   (i.e. `n'` in 256-512). Entries 448-511, nominally ages 513-576, are never
   read. Two resolutions, both defensible:
   - Leave it. 512 is the power of two you want anyway; the top 64 entries are
     harmless headroom, and correctness is unaffected.
   - Add one 12-bit comparator: `e = (age <= 576) ? 0 : bit_length(age) - 9`. Ages
     to 576 become exact rather than folded, the full table is used, and worst-case
     error drops from 0.196% to 0.175%. (The measured 0.175% figures in this
     document use this tighter rule; the cheap MSB rule measures 0.196%.)
2. **Source guide, section 13.4 prose** says `W=64` buys "six extra bits of F."
   It is seven (15 -> 22). The table in the same section is correct.

---

## 8. Reference values for anyone re-deriving this

```
B      = 16      bits per ROM entry
W      = 64      recent-window size; tokens younger than this are masked out
n_min  = 65      = W + 1; the smallest age the LUT can ever see
F      = 22      = floor(log2((2^B - 1) * n_min))
depth  = 512     entries, 1 KB total
covers = ages 65 .. 576
R[a]   = round(4194304 / (a + 65)),  a = 0..511
fold   = e = max(0, bit_length(age) - 9)
         n' = (e==0) ? age : (age + (1 << (e-1))) >> e
         cmp_val = (S * R[n'-65]) >> (22 + e)
T_MAX  = 4096    so age in 1..4096, e in 0..4
P      = 16      lanes
```

**Errors:** table entry `<= n/2^23` (0.007% at n=576); fold `<= 1/depth`
(0.175% at depth 512); policy-level victim disagreement 0.07%, dwarfed by a ~1.2%
floor from 30-bit output quantization.