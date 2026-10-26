import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT); _sys.path.insert(0, _ROOT)  # resolve cl/, vectors.json, hwpx_cracker

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate the second-generation hwpx_kernels.cl:

  * the renamed dictionary kernels  (dist_dict / partial_dict / document_dict)
    still reproduce the CPU reference, and
  * the new GPU-side mask kernels    (dist_mask / partial_mask / document_mask)
    enumerate candidates in the *same* order as Python itertools.product and
    find the planted password at the expected linear index, including when the
    keyspace is split across several dispatches (batch_start offset).

Run:  PYOPENCL_CTX='0' python3 _test_mask.py
"""
import numpy as np, pyopencl as cl, hashlib, base64, zlib, os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

ctx = cl.create_some_context(interactive=False)
q   = cl.CommandQueue(ctx)
mf  = cl.mem_flags
STRIDE = 64
INTMAX = 0x7fffffff

prg = cl.Program(ctx, open("cl/hwpx_kernels.cl").read()).build(
        options=["-I", "cl", f"-DPW_STRIDE={STRIDE}"])

# ----------------------------------------------------------------------------- helpers
def buf(b):
    """READ_ONLY device buffer from a bytes-like host buffer."""
    a = np.frombuffer(bytes(b), np.uint8)
    if a.size == 0:
        a = np.zeros(1, np.uint8)          # avoid zero-size buffer
    return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

def u32buf(vals):
    a = np.asarray(vals, np.uint32)
    if a.size == 0: a = np.zeros(1, np.uint32)
    return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

def u64buf(vals):
    a = np.asarray(vals, np.uint64)
    if a.size == 0: a = np.zeros(1, np.uint64)
    return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

def newfound():
    return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                     hostbuf=np.array([INTMAX], np.int32))

def found(fb):
    r = np.empty(1, np.int32); cl.enqueue_copy(q, r, fb); q.finish(); return int(r[0])

def pack(cands):
    n = len(cands); w = np.zeros((n, STRIDE), np.uint8); ln = np.zeros(n, np.uint32)
    for i, c in enumerate(cands):
        b = c if isinstance(c, bytes) else c.encode()
        w[i, :len(b)] = np.frombuffer(b, np.uint8); ln[i] = len(b)
    return (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w.ravel()),
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ln), n)

# ----------------------------------------------------------------------------- mask model
class Mask:
    """Host-side mixed-radix mask, laid out exactly as mask_decode() expects.

    slots: list (one per position) of lists of token byte-strings.
    Leftmost position varies slowest (matches itertools.product)."""
    def __init__(self, slots):
        self.slots = [[t if isinstance(t, bytes) else t.encode() for t in s] for s in slots]
        P = len(self.slots)
        pos_count, pos_base, tok_off, tok_len = [], [], [], []
        cset = bytearray()
        for s in self.slots:
            pos_base.append(len(tok_off))
            pos_count.append(len(s))
            for tok in s:
                tok_off.append(len(cset)); tok_len.append(len(tok)); cset += tok
        pos_weight = [0]*P; w = 1
        for i in range(P-1, -1, -1):
            pos_weight[i] = w; w *= pos_count[i]
        self.P = P
        self.total = w
        self.pos_count = pos_count
        self.pos_weight = pos_weight
        self.pos_base = pos_base
        self.tok_off = tok_off
        self.tok_len = tok_len
        self.cset = bytes(cset)

    def decode(self, lin):
        out = bytearray()
        for i in range(self.P):
            t = (lin // self.pos_weight[i]) % self.pos_count[i]
            out += self.slots[i][t]
        return bytes(out)

    def index_of(self, pw_bytes):
        for lin in range(self.total):
            if self.decode(lin) == pw_bytes:
                return lin
        return -1

    def gpu_args(self, batch_start, batch_n):
        # exactly the MASK_ARGS ordering
        return [np.uint32(self.P),
                u32buf(self.pos_count), u64buf(self.pos_weight),
                u32buf(self.pos_base), u32buf(self.tok_off),
                u32buf(self.tok_len), buf(self.cset),
                np.uint64(batch_start), np.uint32(batch_n)]

def run_mask_batched(kernel, mask, tail_args, batch=None):
    """Dispatch a mask kernel over the whole keyspace in batches, returning the
    globally-first matching linear index (or None), mirroring how the real host
    driver reconstructs a hit from (batch_start + gid)."""
    total = mask.total
    if batch is None: batch = total
    winner = None
    off = 0
    while off < total:
        bn = min(batch, total - off)
        fb = newfound()
        args = mask.gpu_args(off, bn) + list(tail_args) + [fb]
        kernel(q, (bn,), None, *args)
        idx = found(fb)
        if idx != INTMAX:
            lin = off + idx
            if winner is None or lin < winner:
                winner = lin
            break          # first hit in scan order is the answer
        off += bn
    return winner

allok = True
def check(tag, cond):
    global allok; allok &= bool(cond)
    print(f"[{tag}] {'OK' if cond else 'FAIL'}")

# ============================================================ 1. dict regression
print("== dictionary kernels (renamed) ==")

# distribution
pwlist = ["alpha", "bravo", "secret123", "한글암호", "zzz"]; target = "secret123"
verb = buf(hashlib.sha1(target.encode()).digest())
wb, lb, n = pack(pwlist); fb = newfound()
prg.dist_dict(q, (n,), None, wb, lb, np.uint32(n), verb, np.uint32(1), fb)
idx = found(fb)
check("dist_dict", idx == pwlist.index(target))
wb, lb, n = pack(["no", "match", "here"]); fb = newfound()
prg.dist_dict(q, (n,), None, wb, lb, np.uint32(n), verb, np.uint32(1), fb)
check("dist_dict/neg", found(fb) == INTMAX)

# partial (PoC ciphertext)
ct = base64.b64decode("RHJVmiVKRikvIEu78gPk3PwhPaZa1IXK8pvnPUqbE7lQKjoXNM5TesSqmMAgCXar"
                      "1HRORddCDCbPz9yYq7ywMN6DLl5OXH3XElQvSN1AqZ8=")
prefix = "Hwp Private Information Security;".encode("utf-16le")   # 66 bytes / 528 bits
nbits = len(prefix)*8
need = (nbits + 7)//8
ctb = buf(ct[:need]); exb = buf(prefix)
cands = ["00000", "11111", "54321", "12345", "99999"]
wb, lb, n = pack(cands); fb = newfound()
prg.partial_dict(q, (n,), None, wb, lb, np.uint32(n), ctb, exb, np.uint32(nbits), fb)
idx = found(fb)
check("partial_dict", idx == cands.index("12345"))

# document (synthetic)
def make_entry(password, iters=1024, big=False):
    raw = ("<hml>"+("데이터 payload "*400)+"</hml>").encode()
    plain = zlib.compress(raw)[2:-4]
    if len(plain) % 16: plain += b"\x00"*(16-len(plain) % 16)
    if big and len(plain) < 1104: plain += b"\x00"*(1104-len(plain))
    salt = os.urandom(16); iv = os.urandom(16)
    sk = hashlib.sha256(password.encode()).digest()
    key = hashlib.pbkdf2_hmac("sha1", sk, salt, iters, dklen=32)
    enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    ctd = enc.update(plain)+enc.finalize()
    checksum = hashlib.sha256(plain[:1024]).digest()
    return dict(salt=salt, iv=iv, iters=iters, ct=ctd, checksum=checksum)

def doc_tail(e):
    ctd = e["ct"]; probe = ctd[:min(len(ctd), 1024)]; probe = probe[:len(probe)-(len(probe) % 16)]
    nb = len(probe)//16; hash_len = min(nb*16, 1024)
    return ([buf(e["salt"]), np.uint32(16), np.uint32(e["iters"]), np.uint32(32),
             np.uint32(0), np.uint32(32), buf(e["iv"]), buf(probe),
             np.uint32(nb), np.uint32(hash_len), buf(e["checksum"])])

e = make_entry("openSesame")
cands = ["foo", "bar", "openSesame", "baz"]
wb, lb, n = pack(cands); fb = newfound()
prg.document_dict(q, (n,), None, wb, lb, np.uint32(n), *doc_tail(e), fb)
idx = found(fb)
check("document_dict", idx == cands.index("openSesame"))

# ============================================================ 2. mask kernels
print("\n== mask kernels (GPU-generated candidates) ==")

# ---- 2a. partial_mask : ?d?d?d?d?d must find 12345 ----
digits = [str(d) for d in range(10)]
m5 = Mask([digits]*5)
assert m5.total == 100000
exp_idx = m5.index_of(b"12345")
check("mask.index(12345)==12345", exp_idx == 12345)
win = run_mask_batched(prg.partial_mask, m5, [ctb, exb, np.uint32(nbits)])
check("partial_mask single-batch", win == 12345 and m5.decode(win) == b"12345")

# same keyspace, split into uneven batches -> global winner still 12345
win_b = run_mask_batched(prg.partial_mask, m5, [ctb, exb, np.uint32(nbits)], batch=7000)
check("partial_mask multi-batch", win_b == 12345 and m5.decode(win_b) == b"12345")

# negative: a digit mask that cannot contain 12345 (fixed leading 9)
m_neg = Mask([["9"]] + [digits]*4)
win_neg = run_mask_batched(prg.partial_mask, m_neg, [ctb, exb, np.uint32(nbits)])
check("partial_mask negative", win_neg is None)

# ---- 2b. dist_mask : Hangul + ASCII mixed multibyte charset ----
# positions: {가,나,다} x {라,마} x {0..9}  (3-byte UTF-8 syllables mixed with 1-byte ascii)
hangul = Mask([["가", "나", "다"], ["라", "마"], digits])
tgt = "나마7"
tgt_idx = hangul.index_of(tgt.encode())
verb_h = buf(hashlib.sha1(tgt.encode()).digest())
win_h = run_mask_batched(prg.dist_mask, hangul, [verb_h, np.uint32(1)])
check("dist_mask hangul index", win_h == tgt_idx)
check("dist_mask hangul decode", win_h is not None and hangul.decode(win_h).decode() == tgt)
# multibyte across batches
win_h2 = run_mask_batched(prg.dist_mask, hangul, [verb_h, np.uint32(1)], batch=7)
check("dist_mask hangul multi-batch", win_h2 == tgt_idx)

# ---- 2c. document_mask : ?d?d?d?d finds a planted numeric password ----
docpw = "2468"
ed = make_entry(docpw, iters=1024)
m4 = Mask([digits]*4)
di = m4.index_of(docpw.encode())
win_d = run_mask_batched(prg.document_mask, m4, doc_tail(ed))
check("document_mask index", win_d == di)
check("document_mask decode", win_d is not None and m4.decode(win_d).decode() == docpw)
# batched
win_d2 = run_mask_batched(prg.document_mask, m4, doc_tail(ed), batch=1000)
check("document_mask multi-batch", win_d2 == di)

print("\nALL MASK TESTS", "PASS" if allok else "FAIL")
