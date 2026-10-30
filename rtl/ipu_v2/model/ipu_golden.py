"""
Pure-integer golden model of the IPU v2 (spec v0.3, Sections 2, 4.1 and 4.3).

This is the executable definition the RTL must match bit for bit: same rounding,
same tie rules, and the same DRAM side effects (which rows are written, which lanes
are zeroed, the victim's single-word write, the prefill drain).
"""
from dataclasses import dataclass, field

TIER_EMPTY, TIER_HIGH, TIER_RSVD, TIER_LOW = 0, 1, 2, 3
PH_PREFILL, PH_DECODE = 1, 2
S_MAX = (1 << 30) - 1
CMP_W = 42
CMP_INELIG = (1 << CMP_W) - 1
SH_THRESH = (2, 3, 6, 12, 23, 46, 91, 182, 363, 725, 1449, 2897)

ERR_BAD_PROB, ERR_SAT, ERR_NO_ELIG, ERR_T_RANGE, ERR_SEQ, ERR_AXI = (1 << i for i in range(6))


def fp16_to_delta(h):
    """fp16 bit pattern -> (delta in unsigned Q1.16, bad). Round half to even."""
    s, e, m = (h >> 15) & 1, (h >> 10) & 0x1F, h & 0x3FF
    if e == 31:
        return 0, True                      # Inf / NaN
    if s and (e or m):
        return 0, True                      # negative non-zero
    if s:
        return 0, False                     # -0
    M = m | (0x400 if e else 0)
    E = e if e else 1
    if E >= 16:
        return (1 << 17) - 1, False         # value >= 2 saturates
    y = M << (E - 1)                        # value * 2^24, exact
    q0, g, st = y >> 8, (y >> 7) & 1, (y & 0x7F) != 0
    q = q0 + (1 if g and (st or (q0 & 1)) else 0)
    return min(q, (1 << 17) - 1), False


def sh_of_age(age):
    return sum(1 for th in SH_THRESH if age >= th)


def cmp_of(S, age, raw):
    return S if raw else S << (12 - sh_of_age(age))


def s_from_cmp(cmp, age, raw):
    return cmp if raw else cmp >> (12 - sh_of_age(age))


def w_eff(K):
    return min(64, max(1, K >> 1))


def word(tier, S):
    return (tier << 30) | S


def split(w):
    return (w >> 30) & 3, w & S_MAX


@dataclass
class Geometry:
    P: int = 16
    T_MAX: int = 4096
    NB: int = 1          # sequences used
    NL: int = 32
    NH: int = 32

    @property
    def rows(self):
        return self.T_MAX // self.P

    def head_index(self, b, l, h):
        return (b * self.NL + l) * self.NH + h

    @property
    def n_heads(self):
        return self.NB * self.NL * self.NH


@dataclass
class IPUModel:
    g: Geometry
    K: list = field(default_factory=lambda: [0] * 8)
    raw: bool = False
    sel_en: bool = False             # v2: end-of-prefill selection inside the IPU
    dram: list = None                # flat list of words: head_index * T_MAX + idx
    errors: int = 0
    victims: list = field(default_factory=list)
    n_bulk: int = 0
    # on-chip bank (prefill residency)
    bank: list = None
    bank_dirty: bool = False
    bank_head: tuple = None
    bank_t: int = 0

    def __post_init__(self):
        if self.dram is None:
            self.dram = [0] * (self.g.n_heads * self.g.T_MAX)
        if self.bank is None:
            self.bank = [0] * self.g.T_MAX

    # ---- DRAM helpers
    def _base(self, b, l, h):
        return self.g.head_index(b, l, h) * self.g.T_MAX

    def drain(self):
        if not self.bank_dirty:
            return
        b, l, h = self.bank_head
        if self.sel_en:
            for idx in select_lows(self.bank, self.bank_t, self.K[b], self.raw):
                tier, S = split(self.bank[idx])
                self.bank[idx] = word(TIER_LOW, S)
                self.victims.append((b, l, h, idx))
                self.n_bulk += 1
        base = self._base(b, l, h)
        nrows = -(-self.bank_t // self.g.P)
        for idx in range(nrows * self.g.P):
            self.dram[base + idx] = self.bank[idx]
        self.bank_dirty = False

    def flush(self):
        self.drain()

    # ---- one packet
    def packet(self, phase, b, l, h, t, fp16):
        g = self.g
        if t == 0 or t > g.T_MAX:
            self.errors |= ERR_T_RANGE
            return None
        assert len(fp16) == t
        if phase == PH_PREFILL:
            cont = self.bank_dirty and self.bank_head == (b, l, h) and t > 1
            if self.bank_dirty and not cont:
                self.drain()
            if cont and t != self.bank_t + 1:
                self.errors |= ERR_SEQ
            if not cont and t != 1:
                self.errors |= ERR_SEQ
            old = self.bank
        else:
            self.drain()
            base = self._base(b, l, h)
            old = [self.dram[base + i] if i < t - 1 else 0 for i in range(t)]

        K = self.K[b]
        W = w_eff(K)
        new_words = []
        best = None                                   # (cmp, idx)
        for idx in range(t):
            tier, S = split(old[idx])
            d, bad = fp16_to_delta(fp16[idx])
            if bad:
                self.errors |= ERR_BAD_PROB
            if idx == t - 1:
                tier, S = TIER_HIGH, d                # d <= 2^17-1, never saturates
            elif tier == TIER_HIGH:
                S = S + d
                if S > S_MAX:
                    S = S_MAX
                    self.errors |= ERR_SAT
            new_words.append(word(tier, S))
            age = t - 1 - idx
            if phase == PH_DECODE and tier == TIER_HIGH and idx != t - 1 and age >= W:
                c = cmp_of(S, age, self.raw)
                if best is None or c < best[0]:
                    best = (c, idx)

        nrows = -(-t // g.P)
        if phase == PH_PREFILL:
            for idx in range(nrows * g.P):
                self.bank[idx] = new_words[idx] if idx < t else 0
            self.bank_dirty, self.bank_head, self.bank_t = True, (b, l, h), t
            return None

        base = self._base(b, l, h)
        for idx in range(nrows * g.P):
            self.dram[base + idx] = new_words[idx] if idx < t else 0
        if t > K:
            if best is None:
                self.errors |= ERR_NO_ELIG
                return None
            c, vidx = best
            S = s_from_cmp(c, t - 1 - vidx, self.raw)
            assert S == split(new_words[vidx])[1]
            self.dram[base + vidx] = word(TIER_LOW, S)
            self.victims.append((b, l, h, vidx))
            return (b, l, h, vidx)
        return None

    # ---- host bulk selection at the end of prefill (testbench / host, not RTL)
    def host_select(self, b, l, h, t_p):
        """Returns the list of idx demoted (tier set LOW in DRAM)."""
        base = self._base(b, l, h)
        demoted = select_lows(self.dram[base:base + t_p], t_p, self.K[b], self.raw)
        for idx in demoted:
            tier, S = split(self.dram[base + idx])
            self.dram[base + idx] = word(TIER_LOW, S)
        return sorted(demoted)


def select_lows(words, t_p, K, raw):
    """End-of-prefill selection (v1: host, v2: ipu_psel). words[idx] for idx < t_p.
    Keep the W_EFF window and the K - W_EFF largest decayed scores; demote the rest.
    Ties keep the newer token. Returns the demoted idx, ascending."""
    if t_p <= K:
        return []
    W = w_eff(K)
    kH = max(0, K - W)
    cands = [(cmp_of(split(words[idx])[1], t_p - 1 - idx, raw), idx) for idx in range(t_p - W)]
    cands.sort(key=lambda x: (-x[0], -x[1]))          # largest first; ties keep newer
    keep = {idx for _, idx in cands[:kH]}
    return sorted(idx for _, idx in cands if idx not in keep)


def select_lows_radix(words, t_p, K, raw):
    """Bit-serial radix select exactly as ipu_psel does it (cross-check of select_lows)."""
    if t_p <= K:
        return []
    W = w_eff(K)
    n = t_p - W
    cmp = [cmp_of(split(words[i])[1], t_p - 1 - i, raw) for i in range(n)]
    r = n if K == 0 else t_p - K                      # D
    T = 0
    for b in range(29 if raw else CMP_W - 1, -1, -1):
        hi = ~((1 << (b + 1)) - 1)
        c0 = sum(1 for c in cmp if (c ^ T) & hi == 0 and not (c >> b) & 1)
        if r > c0:
            T |= 1 << b
            r -= c0
    out, eq = [], 0
    for i, c in enumerate(cmp):
        if c < T or (c == T and eq < r):
            eq += (c == T)
            out.append(i)
    return out


if __name__ == "__main__":
    import math
    import numpy as np
    # Self-check 1: ingress == clamp(round_half_even(float16 * 2^16)) as in mikv_quant.
    bad = 0
    for hbits in range(1 << 16):
        v = np.array([hbits], dtype=np.uint16).view(np.float16)[0]
        q, isbad = fp16_to_delta(hbits)
        if np.isnan(v) or np.isinf(v) or (v < 0):
            ok = isbad and q == 0
        else:
            ref = int(np.clip(np.round(np.float64(v) * 65536.0), 0, (1 << 17) - 1))
            ok = (q == ref) and not isbad
        bad += not ok
    print("ingress mismatches vs numpy RNE:", bad)
    # Self-check 2: sh == round(log2(age)) for every age 1..4095 (as in _apply_decay).
    mism = [a for a in range(1, 4096) if sh_of_age(a) != round(math.log2(a))]
    print("sh mismatches vs round(log2):", len(mism))
    # Self-check 3: max eligible cmp is below the all-ones 'ineligible' code.
    assert (S_MAX << 12) < CMP_INELIG
    # Self-check 4: radix select == sort-based selection (random + heavy ties).
    import random
    rng = random.Random(1)
    for trial in range(3000):
        t_p = rng.randint(1, 300)
        K = rng.randint(0, 320)
        raw = rng.random() < 0.3
        hi = rng.choice([3, 50, S_MAX])
        words = [word(TIER_HIGH, rng.randint(0, hi)) for _ in range(t_p)]
        assert select_lows(words, t_p, K, raw) == select_lows_radix(words, t_p, K, raw), trial
    print("radix select == sort select on 3000 random heads")
    print("ok")
