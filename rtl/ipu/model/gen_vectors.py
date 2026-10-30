"""
Test-vector generator for the IPU testbenches (spec v0.2 Section 8).

  python3 gen_vectors.py unit   -> vectors/unit/*.hex
  python3 gen_vectors.py top    -> vectors/top_<name>/*   (see gen_top below)

All expected values come from ipu_golden.py.
"""
import os
import random
import struct
import sys

import numpy as np

from ipu_golden import (CMP_INELIG, PH_DECODE, PH_PREFILL, S_MAX, TIER_HIGH,
                        TIER_LOW, TIER_RSVD, TIER_EMPTY, Geometry, IPUModel,
                        cmp_of, fp16_to_delta, sh_of_age, split, w_eff, word)

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(HERE, "..", "vectors")


def hexline(value, bits):
    return format(value, "0{}x".format((bits + 3) // 4))


def pack(fields):
    """fields: list of (value, width), first entry ends up in the MSBs."""
    v, w = 0, 0
    for val, wid in fields:
        assert 0 <= val < (1 << wid), (val, wid)
        v = (v << wid) | val
        w += wid
    return v, w


def f16(x):
    return int(np.array([x], dtype=np.float16).view(np.uint16)[0])


# ---------------------------------------------------------------- unit vectors
def gen_ingress(d):
    with open(os.path.join(d, "ingress_exp.hex"), "w") as f:
        for h in range(1 << 16):
            q, bad = fp16_to_delta(h)
            f.write(hexline((int(bad) << 17) | q, 18) + "\n")


def rand_fp16(rng):
    r = rng.random()
    if r < 0.05:
        return rng.choice([0x7C00, 0xFC00, 0x7E00, 0xBC00, 0x8000, 0x0000, 0x3C00, 0x0001, 0x03FF])
    if r < 0.15:
        return rng.randrange(0, 0x0400)                       # subnormals
    if r < 0.6:
        return f16(rng.random() ** 6)                          # mostly tiny probabilities
    return f16(rng.random())


def gen_datapath(d, P=16, IDX_W=12, n=4000, seed=1):
    """Vectors for ipu_lane_ctl + ipu_ingress + ipu_accum + ipu_cmp_shift (one beat)."""
    rng = random.Random(seed)
    fin = open(os.path.join(d, "dp_in.hex"), "w")
    fex = open(os.path.join(d, "dp_exp.hex"), "w")
    T_MAX = 1 << IDX_W
    for v in range(n):
        t = rng.choice([rng.randrange(1, T_MAX + 1), rng.randrange(1, 80)])
        nb = -(-t // P)
        beat = rng.randrange(nb)
        a0 = t - 1 - P * beat
        K = rng.randrange(1, t + 2)
        W = w_eff(K)
        decode = rng.random() < 0.8
        raw = rng.random() < 0.15
        keep = (1 << P) - 1 if rng.random() < 0.9 else rng.randrange(1 << P)
        fp = [rand_fp16(rng) for _ in range(P)]
        words = []
        for i in range(P):
            tier = rng.choice([TIER_HIGH] * 6 + [TIER_LOW] * 2 + [TIER_RSVD, TIER_EMPTY])
            r = rng.random()
            S = S_MAX - rng.randrange(1 << 17) if r < 0.05 else (rng.randrange(1 << 24) if r < 0.9 else rng.randrange(S_MAX + 1))
            words.append(word(tier, S))
        inv, inw = pack([(a0, IDX_W), (W, 7), (keep, P), (int(decode), 1), (int(raw), 1)]
                        + [(fp[P - 1 - i], 16) for i in range(P)]
                        + [(words[P - 1 - i], 32) for i in range(P)])
        wout, elig, sat, bad, cmps, shs = [], 0, 0, 0, [], []
        for i in range(P):
            age = a0 - i
            valid = bool((keep >> i) & 1) and age >= 0
            isnew = age == 0
            q, b = fp16_to_delta(fp[i])
            bad |= int(b) << i
            tier, S = split(words[i])
            if isnew:
                tier, Sn, s_ = TIER_HIGH, q, False
            elif tier == TIER_HIGH:
                Sn = S + q
                s_ = Sn > S_MAX
                Sn = min(Sn, S_MAX)
            else:
                Sn, s_ = S, False
            e = decode and valid and not isnew and tier == TIER_HIGH and age >= W
            elig |= int(e) << i
            sat |= int(valid and s_) << i
            wout.append(word(tier, Sn) if valid else 0)
            sh = sh_of_age(age) if age >= 0 else 0
            shs.append(sh)
            cmps.append(cmp_of(Sn, age, raw) if e else CMP_INELIG)
        exv, exw = pack([(wout[P - 1 - i], 32) for i in range(P)]
                        + [(elig, P), (sat, P), (bad, P)]
                        + [(cmps[P - 1 - i], 42) for i in range(P)])
        fin.write(hexline(inv, inw) + "\n")
        fex.write(hexline(exv, exw) + "\n")
    fin.close()
    fex.close()
    return inw, exw


def gen_tree(d, P=16, n=3000, seed=2):
    """Packets of beats -> champion (cmp, idx) after the last beat."""
    rng = random.Random(seed)
    fin = open(os.path.join(d, "tree_in.hex"), "w")
    fex = open(os.path.join(d, "tree_exp.hex"), "w")
    for _ in range(n):
        nb = rng.randrange(1, 6)
        best = None
        for r in range(nb):
            mode = rng.random()
            cm = []
            for i in range(P):
                if mode < 0.2:
                    c = rng.choice([5, 7, CMP_INELIG])           # many ties
                elif mode < 0.3:
                    c = CMP_INELIG
                else:
                    c = rng.randrange(1 << 20) if rng.random() < 0.9 else CMP_INELIG
                cm.append(c)
                if c != CMP_INELIG and (best is None or c < best[0]):
                    best = (c, r * P + i)
            v, w = pack([(int(r == 0), 1), (int(r == nb - 1), 1), (r, 8)]
                        + [(cm[P - 1 - i], 42) for i in range(P)])
            fin.write(hexline(v, w) + "\n")
        found = best is not None
        bc, bi = best if found else (CMP_INELIG, 0)
        ev, ew = pack([(int(found), 1), (bc, 42), (bi if found else 0, 12)])
        fex.write(hexline(ev, ew) + "\n")
    fin.close()
    fex.close()


def gen_unit():
    d = os.path.join(VEC, "unit")
    os.makedirs(d, exist_ok=True)
    gen_ingress(d)
    print("datapath widths", gen_datapath(d))
    gen_tree(d)
    print("unit vectors written to", d)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "unit"
    if what == "unit":
        gen_unit()
    elif what == "top":
        from gen_top import gen_all_top
        gen_all_top()
