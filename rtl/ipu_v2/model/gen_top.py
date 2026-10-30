"""
Top-level scenarios for tb_ipu_top.sv (v2, spec v0.3 Section 8).

Each scenario directory vectors/top_<name>/ holds:
  tb_cfg.svh     shape / counts / expected error bits (included by the testbench)
  csr.hex        {addr[7:0], data[31:0]} register writes, terminated by addr 0xFF
  cmds.hex       32-bit commands: [31:28] op, [23:0] tuser for packets
                   1 = packet, 2 = flush + compare(mem_sel) + host LOW selection, 3 = final check, 0 = end
                 (v2: the IPU selects at the end of each head's prefill, so lowsel is empty and
                  victims.hex starts with the bulk demotes, head by head, idx ascending)
  beats.hex      {tkeep[15:0], tdata[255:0]} per beat, in packet order
  lowsel.hex     compact word indices to set LOW at each command 2; 0xFFFFFFFF ends a group
  victims.hex    expected demote packets {b, l, h, idx} in order
  mem_init.hex   initial DRAM image (random garbage: proves no clearing is needed)
  mem_sel.hex    expected DRAM image right after the first flush (before host selection)
  mem_exp.hex    expected final DRAM image
"""
import os
import random

import numpy as np

from ipu_golden import (ERR_BAD_PROB, ERR_T_RANGE, PH_DECODE, PH_PREFILL,
                        Geometry, IPUModel)

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(HERE, "..", "vectors")

P, T_MAX = 16, 256
B_W, L_W, H_W = 3, 5, 5
IDX_W = 8
T_W = IDX_W + 1
ROWS = T_MAX // P
BASE_HI = 0x5                     # SCORE_BASE[39:23]


def f16bits(a):
    return np.asarray(a, dtype=np.float16).view(np.uint16).astype(int).tolist()


class Scenario:
    def __init__(self, name, nb, nl, nh, K, raw=False, seed=0, sel=True):
        self.name, self.raw = name, raw
        self.g = Geometry(P=P, T_MAX=T_MAX, NB=nb, NL=nl, NH=nh)
        self.rng = np.random.default_rng(seed)
        self.prng = random.Random(seed)
        init = [self.prng.getrandbits(32) for _ in range(self.g.n_heads * T_MAX)]
        self.sel = sel
        self.m = IPUModel(self.g, K=list(K) + [0] * (8 - len(K)), raw=raw, sel_en=sel, dram=list(init))
        self.init = init
        self.cmds, self.beats, self.lowsel = [], [], []
        self.mem_sel = None
        self.hh = {}                                  # per-head heavy hitters
        self.expect_err = 0

    # --- realistic attention row of length t for head (b, l, h)
    def row(self, b, l, h, t, kind="normal"):
        if kind == "zero":
            return [0] * t
        if kind == "uniform":
            return f16bits(np.full(t, 1.0 / t))
        key = (b, l, h)
        if key not in self.hh:
            self.hh[key] = self.rng.integers(0, 4096, size=6)
        idx = np.arange(t)
        age = (t - 1) - idx
        logit = self.rng.normal(0.0, 1.0, size=t)
        logit[0] += 6.0                                         # attention sink
        logit += 2.5 * np.exp(-age / 8.0)                       # recency
        for hpos in self.hh[key]:
            if hpos % 300 < t:
                logit[hpos % 300] += 4.0                        # heavy hitters
        logit *= self.rng.uniform(0.8, 2.5)                     # head temperature
        p = np.exp(logit - logit.max())
        p /= p.sum()
        return f16bits(p)

    def packet(self, phase, b, l, h, t, fp=None, kind="normal"):
        if fp is None:
            fp = self.row(b, l, h, t, kind) if t > 0 else []
        tuser = (phase << (B_W + L_W + H_W + T_W)) | (b << (L_W + H_W + T_W)) | \
                (l << (H_W + T_W)) | (h << T_W) | t
        self.cmds.append((1 << 28) | tuser)
        nb = max(1, -(-t // P))
        for r in range(nb):
            lanes = fp[r * P:(r + 1) * P]
            keep = (1 << len(lanes)) - 1 if t > 0 else 1
            lanes = lanes + [0x7E00] * (P - len(lanes))         # garbage (NaN) in unused lanes
            data = 0
            for i, v in enumerate(lanes):
                data |= v << (16 * i)
            self.beats.append((keep << (16 * P)) | data)
        if t > 0:
            self.m.packet(phase, b, l, h, t, fp)
        else:
            self.m.errors |= ERR_T_RANGE

    def prefill(self, b, t_p):
        for l in range(self.g.NL):
            for h in range(self.g.NH):
                for t in range(1, t_p + 1):
                    self.packet(PH_PREFILL, b, l, h, t)

    def flush_select(self, tps):
        self.cmds.append(2 << 28)
        self.m.flush()
        self.mem_sel = list(self.m.dram)
        if self.sel:                                  # v2: already done inside the IPU
            self.lowsel.append(0xFFFFFFFF)
            return
        for b, t_p in tps.items():
            for l in range(self.g.NL):
                for h in range(self.g.NH):
                    for idx in self.m.host_select(b, l, h, t_p):
                        self.lowsel.append(self.g.head_index(b, l, h) * T_MAX + idx)
        self.lowsel.append(0xFFFFFFFF)

    def finish(self):
        self.cmds.append(3 << 28)
        self.cmds.append(0)
        self.m.flush()

    # --- output
    def write(self, csr):
        d = os.path.join(VEC, "top_" + self.name)
        os.makedirs(d, exist_ok=True)

        def img(path, words):
            with open(os.path.join(d, path), "w") as f:
                for r in range(len(words) // P):
                    v = 0
                    for i in range(P):
                        v |= words[r * P + i] << (32 * i)
                    f.write(format(v, "0128x") + "\n")

        img("mem_init.hex", self.init)
        img("mem_sel.hex", self.mem_sel if self.mem_sel else self.init)
        img("mem_exp.hex", self.m.dram)
        with open(os.path.join(d, "cmds.hex"), "w") as f:
            f.writelines(format(c, "08x") + "\n" for c in self.cmds)
        with open(os.path.join(d, "beats.hex"), "w") as f:
            f.writelines(format(bt, "068x") + "\n" for bt in self.beats)
        with open(os.path.join(d, "lowsel.hex"), "w") as f:
            f.writelines(format(x, "08x") + "\n" for x in (self.lowsel or [0xFFFFFFFF]))
        with open(os.path.join(d, "victims.hex"), "w") as f:
            for b, l, h, idx in self.m.victims or [(0, 0, 0, 0)]:
                f.write(format((b << (L_W + H_W + IDX_W)) | (l << (H_W + IDX_W)) | (h << IDX_W) | idx, "06x") + "\n")
        with open(os.path.join(d, "csr.hex"), "w") as f:
            for a, v in csr + [(0xFF, 0)]:
                f.write(format((a << 32) | v, "010x") + "\n")
        n_pkts = sum(1 for c in self.cmds if (c >> 28) == 1)
        with open(os.path.join(d, "tb_cfg.svh"), "w") as f:
            f.write("// generated by gen_top.py -- scenario {}\n".format(self.name))
            for k, v in [("NB", self.g.NB), ("NL", self.g.NL), ("NH", self.g.NH),
                         ("N_CMDS", len(self.cmds)), ("N_BEATS", len(self.beats)),
                         ("N_LOWSEL", max(1, len(self.lowsel))), ("N_VICTIMS", len(self.m.victims)),
                         ("N_PKTS", n_pkts), ("N_BULK", self.m.n_bulk), ("EXP_ERR", self.m.errors), ("BASE_HI", BASE_HI)]:
                f.write("`define {} {}\n".format(k, v))
        print("scenario {:8s}: {:5d} packets {:6d} beats {:4d} demotes ({} bulk) {:4d} host-LOW, errors=0x{:02x}".format(
            self.name, n_pkts, len(self.beats), len(self.m.victims), self.m.n_bulk,
            len([x for x in self.lowsel if x != 0xFFFFFFFF]), self.m.errors))


def csr_for(s, nb, nl, nh, K, raw=False, credit=8):
    """CTRL: enable | raw << 1 | sel_en << 3."""
    regs = [(0x08, BASE_HI), (0x0C, (nb - 1) << 10 | (nh - 1) << 5 | (nl - 1)), (0x10, credit)]
    regs += [(0x20 + 4 * b, k) for b, k in enumerate(K)]
    regs += [(0x00, 1 | (int(raw) << 1) | (int(s.sel) << 3))]
    return regs


def gen_main():
    """3 sequences x 2 layers x 2 heads: prefill, host selection, 40 decode steps.
    seq0: t_p 150, K 37 (r 0.25). seq1: t_p 37, K 40 (> t_p: no demotion for 3 steps).
    seq2: t_p 5, K 1 (W_EFF 1, k_H 0). One NaN lane, one out-of-order head."""
    tps = {0: 150, 1: 37, 2: 5}
    K = [37, 40, 1]
    s = Scenario("main", 3, 2, 2, K, seed=11)
    for b, tp in tps.items():
        s.prefill(b, tp)
    s.flush_select(tps)
    for step in range(1, 41):
        order = [(b, l, h) for b in range(3) for l in range(2) for h in range(2)]
        if step == 10:
            order[2], order[3] = order[3], order[2]                 # out-of-order head
        for b, l, h in order:
            t = tps[b] + step
            fp = s.row(b, l, h, t)
            if step == 20 and (b, l, h) == (0, 1, 0):
                fp[3] = 0x7E00                                      # NaN -> BAD_PROB
            s.packet(PH_DECODE, b, l, h, t, fp=fp)
    s.finish()
    s.write(csr_for(s, 3, 2, 2, K))


def gen_raw():
    """RAW (undecayed) compare, 1 x 2 x 2, tighter credits."""
    tps = {0: 64}
    K = [20]
    s = Scenario("raw", 1, 2, 2, K, raw=True, seed=12)
    s.prefill(0, 64)
    s.flush_select(tps)
    for step in range(1, 30):
        for l in range(2):
            for h in range(2):
                s.packet(PH_DECODE, 0, l, h, 64 + step)
    s.finish()
    s.write(csr_for(s, 1, 2, 2, K, raw=True, credit=1))


def gen_edge():
    """Single head (successor == itself: prefetch must stay off), K = 1, ties from all-zero
    and uniform rows, an invalid header (t = 0), rows crossing many 16-token boundaries."""
    tps = {0: 16}
    K = [1]
    s = Scenario("edge", 1, 1, 1, K, seed=13)
    s.prefill(0, 16)
    s.flush_select(tps)
    kinds = ["zero", "uniform", "normal"]
    for step in range(1, 100):
        t = 16 + step
        if step == 7:
            s.packet(PH_DECODE, 0, 0, 0, 0)                          # t = 0: dropped, T_RANGE
        s.packet(PH_DECODE, 0, 0, 0, t, kind=kinds[step % 3])
    s.finish()
    s.write(csr_for(s, 1, 1, 1, K))


def gen_big():
    """v2 selection near T_MAX (256 here): seq 0 t_p 250, K 62 (r 0.25, W_EFF 31, D 188
    bulk demotes per head); seq 1 K 0 (window only) with one head of constant scores
    (every candidate ties, so the tie rule decides all of it)."""
    tps = {0: 250, 1: 90}
    K = [62, 0]
    s = Scenario("big", 2, 1, 2, K, seed=14)
    s.prefill(0, 250)
    for h in range(2):                               # seq 1: head 0 normal, head 1 uniform
        for t in range(1, 91):
            s.packet(PH_PREFILL, 1, 0, h, t, kind="normal" if h == 0 else "uniform")
    s.flush_select(tps)
    for step in range(1, 6):
        for b in range(2):
            for h in range(2):
                s.packet(PH_DECODE, b, 0, h, tps[b] + step)
    s.finish()
    s.write(csr_for(s, 2, 1, 2, K))


def gen_all_top():
    gen_main()
    gen_raw()
    gen_edge()
    gen_big()


if __name__ == "__main__":
    gen_all_top()
