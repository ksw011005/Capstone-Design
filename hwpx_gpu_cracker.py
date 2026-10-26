#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hwpx_gpu_cracker.py
===================

OpenCL (Hashcat-style) GPU host driver for the three HWPX protection profiles.

Architecture (as recommended in the project README)::

    HWPX file ──▶ Python/host extractor ──▶ attack blob ──▶ OpenCL kernels
                  (reuses hwpx_cracker.py:                   (hwpx_kernels.cl)
                   manifest parsing, mode                     ├ document  SHA256→PBKDF2→AES-CBC→checksum
                   detection, settings.xml                    ├ partial   SHA1→ROL→SHA1→AES-CFB1→prefix
                   decryption, EncryptString)                 └ dist      SHA1 → digest compare

All file parsing / cryptographic *setup* runs once on the host using the
already-validated CPU reference (``hwpx_cracker.py``). The GPU only runs the
per-candidate inner loop, in one of two candidate-supply modes per profile:

  * dictionary  – the host encodes each wordlist entry to bytes and uploads a
                  batch (``*_dict`` kernels);
  * mask        – candidates are generated *on the device* from a mixed-radix
                  decode of the linear index (``*_mask`` kernels), so even the
                  cheap SHA-1-only distribution profile is not host-bound.
                  Mask tokens are raw byte strings, so Korean/UTF-8 multibyte
                  character sets work natively.

The candidate byte stream is built once per requested encoding, so ``--encodings
utf-8,cp949`` behaves like the CPU reference (first matching encoding wins).

Only your own / authorised test documents. Verified end-to-end against the CPU
reference on the POCL CPU runtime.

Dependencies::

    pip install pyopencl numpy cryptography
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

try:
    import pyopencl as cl
except Exception as exc:  # pragma: no cover
    print("[-] pyopencl import 실패. `pip install pyopencl` 후 다시 실행하세요.",
          file=sys.stderr)
    raise

# The CPU reference is reused verbatim for all parsing / extraction.
import hwpx_cracker as ref


def _resource_dir() -> Path:
    """Directory that holds the bundled `cl/` kernels. When frozen by PyInstaller
    the kernels are unpacked under sys._MEIPASS; otherwise they sit next to this
    source file."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


CL_DIR = _resource_dir() / "cl"
DEFAULT_BATCH = 1 << 20          # candidates per dispatch
INTMAX = 0x7fffffff


# ---------------------------------------------------------------------------
# GPU program management
# ---------------------------------------------------------------------------

class GPU:
    """Owns the OpenCL context/queue and the compiled kernel program.

    ``pw_stride`` (bytes reserved per candidate slot) is baked in at build time,
    so if a candidate is longer than the current stride the program is rebuilt.
    Kernel handles are cached to avoid pyopencl's RepeatedKernelRetrieval cost.
    """

    def __init__(self, pw_stride: int = 64, part_maxbits: int = 1024,
                 device_index: Optional[int] = None):
        self.ctx = self._make_context(device_index)
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.part_maxbits = part_maxbits
        self.pw_stride = 0
        self._prog = None
        self._kernels: dict[str, cl.Kernel] = {}
        self._src = (CL_DIR / "hwpx_kernels.cl").read_text()
        self.rebuild(pw_stride)

    @staticmethod
    def _make_context(device_index: Optional[int]) -> cl.Context:
        if device_index is None:
            # honour PYOPENCL_CTX if present, else first available platform/device
            try:
                return cl.create_some_context(interactive=False)
            except Exception:
                pass
        devices = []
        for p in cl.get_platforms():
            devices.extend(p.get_devices())
        if not devices:
            raise RuntimeError("사용 가능한 OpenCL 디바이스를 찾지 못했습니다.")
        idx = device_index or 0
        if idx >= len(devices):
            raise IndexError(f"device index {idx} 범위 초과 (found {len(devices)})")
        return cl.Context(devices=[devices[idx]])

    def rebuild(self, pw_stride: int) -> None:
        pw_stride = max(pw_stride, 16)
        if pw_stride == self.pw_stride:
            return
        self.pw_stride = pw_stride
        self._prog = cl.Program(self.ctx, self._src).build(
            options=["-I", str(CL_DIR),
                     f"-DPW_STRIDE={pw_stride}",
                     f"-DPART_MAXBITS={self.part_maxbits}"])
        self._kernels = {k.function_name: k for k in self._prog.all_kernels()}

    def kernel(self, name: str) -> cl.Kernel:
        return self._kernels[name]

    def device_name(self) -> str:
        return self.ctx.devices[0].name

    # ---- buffer helpers ----
    def ro(self, host_bytes) -> cl.Buffer:
        a = np.frombuffer(bytes(host_bytes), np.uint8)
        if a.size == 0:
            a = np.zeros(1, np.uint8)
        return cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=a)

    def ro_arr(self, arr, dtype) -> cl.Buffer:
        a = np.asarray(arr, dtype)
        if a.size == 0:
            a = np.zeros(1, dtype)
        return cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=a)

    def found_buf(self) -> cl.Buffer:
        return cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                         hostbuf=np.array([INTMAX], np.int32))

    def read_found(self, fb: cl.Buffer) -> int:
        r = np.empty(1, np.int32)
        cl.enqueue_copy(self.queue, r, fb)
        self.queue.finish()
        return int(r[0])


# ---------------------------------------------------------------------------
# Mask model  (mixed radix, leftmost position varies slowest == itertools.product)
# ---------------------------------------------------------------------------

class Mask:
    """Host-side layout of a Hashcat-style mask, matching mask_decode() in the
    kernel. ``slots`` is one list of token byte-strings per position."""

    def __init__(self, slots: Sequence[Sequence[bytes]]):
        self.slots = [[t if isinstance(t, bytes) else t.encode() for t in s]
                      for s in slots]
        P = len(self.slots)
        pos_count, pos_base, tok_off, tok_len = [], [], [], []
        cset = bytearray()
        for s in self.slots:
            if not s:
                raise ValueError("빈 mask 위치가 있습니다.")
            pos_base.append(len(tok_off))
            pos_count.append(len(s))
            for tok in s:
                tok_off.append(len(cset))
                tok_len.append(len(tok))
                cset += tok
        pos_weight = [0] * P
        w = 1
        for i in range(P - 1, -1, -1):
            pos_weight[i] = w
            w *= pos_count[i]
        self.P = P
        self.total = w
        self.pos_count = pos_count
        self.pos_weight = pos_weight
        self.pos_base = pos_base
        self.tok_off = tok_off
        self.tok_len = tok_len
        self.cset = bytes(cset)
        # longest candidate this mask can emit (to size PW_STRIDE)
        self.max_len = sum(max(len(t) for t in s) for s in self.slots)

    @classmethod
    def from_spec(cls, mask: str, custom: dict[str, str], encoding: str) -> "Mask":
        """Build from the reference's mask grammar. Each parsed slot is a string
        of characters; every character becomes one token encoded with ``encoding``
        (so a multibyte char is a multi-byte token)."""
        slot_strings = ref.parse_mask(mask, custom)  # list[str], validated grammar
        slots: list[list[bytes]] = []
        for s in slot_strings:
            slots.append([ch.encode(encoding) for ch in s])
        return cls(slots)

    def decode(self, lin: int) -> bytes:
        out = bytearray()
        for i in range(self.P):
            t = (lin // self.pos_weight[i]) % self.pos_count[i]
            out += self.slots[i][t]
        return bytes(out)

    def gpu_prefix_args(self, gpu: GPU, batch_start: int, batch_n: int) -> list:
        """Exactly the MASK_ARGS ordering expected by every *_mask kernel."""
        return [np.uint32(self.P),
                gpu.ro_arr(self.pos_count, np.uint32),
                gpu.ro_arr(self.pos_weight, np.uint64),
                gpu.ro_arr(self.pos_base, np.uint32),
                gpu.ro_arr(self.tok_off, np.uint32),
                gpu.ro_arr(self.tok_len, np.uint32),
                gpu.ro(self.cset),
                np.uint64(batch_start),
                np.uint32(batch_n)]


# ---------------------------------------------------------------------------
# Attack blob  (host-extracted, profile-specific GPU inputs)
# ---------------------------------------------------------------------------

class AttackBlob:
    """Everything the GPU needs for one document, extracted once on the host."""

    def __init__(self, mode: str):
        self.mode = mode
        # partial
        self.ct_prefix: bytes = b""
        self.exp_prefix: bytes = b""
        self.nbits: int = 0
        # distribution
        self.verifiers: list[bytes] = []
        # document
        self.salt: bytes = b""
        self.iv: bytes = b""
        self.iters: int = 0
        self.key_size: int = 32
        self.start_algo: int = 0      # 0 sha256, 1 sha1
        self.start_size: int = 32
        self.probe_ct: bytes = b""
        self.nb: int = 0
        self.hash_len: int = 0
        self.checksum: bytes = b""


def extract_blob(pkg: "ref.HwpxPackage", mode: str) -> AttackBlob:
    blob = AttackBlob(mode)

    if mode == "distribution":
        settings = ref.decrypt_distribution_settings(pkg, pkg.encrypted_infos())
        if settings is None:
            raise ValueError("배포용 고정 키로 settings.xml 복호화에 실패했습니다.")
        verifiers = ref.extract_sha1_verifiers(settings)
        if not verifiers:
            raise ValueError("settings.xml에서 20바이트 SHA-1 검증값을 찾지 못했습니다.")
        blob.verifiers = sorted(verifiers)          # deterministic order
        return blob

    if mode == "partial":
        strings = ref.find_encrypt_strings(pkg)
        if not strings:
            raise ValueError("EncryptString 암호문을 찾지 못했습니다.")
        _src, ciphertext = strings[0]
        prefix = ref.PARTIAL_PREFIX_UTF16LE                 # 64 bytes
        blob.nbits = len(prefix) * 8                        # 512
        need = (blob.nbits + 7) // 8
        if len(ciphertext) < need:
            raise ValueError("EncryptString 암호문이 prefix보다 짧습니다.")
        blob.ct_prefix = ciphertext[:need]
        blob.exp_prefix = prefix
        return blob

    if mode == "document":
        infos = pkg.encrypted_infos()
        info = ref.choose_document_probe(pkg, infos)
        ciphertext = pkg.read(info.full_path)

        # GPU checksum core is SHA-256 over the first 1024 bytes (HWPX sha256-1k).
        # Anything else — a full SHA-256/SHA-1 checksum, or an oversized salt the
        # single-block HMAC path can't hold — is handled by the CPU reference.
        kind = info.checksum_type.lower()
        is_1k = ("1k" in kind) or (kind == "")      # matches reference truncation rule
        if len(info.checksum) != 32 or not is_1k:
            raise ValueError(
                "이 문서의 checksum이 sha256-1k(32B)가 아닙니다. "
                "document 프로파일 GPU 경로는 sha256-1k만 지원합니다. "
                "CPU 레퍼런스(hwpx_cracker.py)를 사용하세요.")
        if len(info.salt) > 32:
            raise ValueError(
                f"salt 길이({len(info.salt)}B)가 GPU 커널 한계를 초과합니다. "
                "CPU 레퍼런스(hwpx_cracker.py)를 사용하세요.")

        probe = ciphertext[:min(len(ciphertext), 1024)]
        probe = probe[:len(probe) - (len(probe) % 16)]
        if not probe and ciphertext:
            probe = ciphertext[:16]
        blob.salt = info.salt
        blob.iv = info.iv
        blob.iters = info.iterations
        blob.key_size = info.key_size
        name = info.start_key_name.lower()
        blob.start_algo = 1 if ("sha1" in name and "sha256" not in name) else 0
        blob.start_size = info.start_key_size if info.start_key_size > 0 else 32
        blob.probe_ct = probe
        blob.nb = len(probe) // 16
        blob.hash_len = min(blob.nb * 16, 1024)
        blob.checksum = info.checksum
        return blob

    raise ValueError(f"지원하지 않는 mode: {mode}")


# ---------------------------------------------------------------------------
# Per-profile GPU dispatch
# ---------------------------------------------------------------------------

def _dict_tail(gpu: GPU, blob: AttackBlob) -> list:
    """Profile-specific trailing kernel args shared by dict + mask kernels
    (everything after the candidate-source args, excluding g_found)."""
    if blob.mode == "distribution":
        ver = b"".join(blob.verifiers)
        return [gpu.ro(ver), np.uint32(len(blob.verifiers))]
    if blob.mode == "partial":
        return [gpu.ro(blob.ct_prefix), gpu.ro(blob.exp_prefix), np.uint32(blob.nbits)]
    if blob.mode == "document":
        return [gpu.ro(blob.salt), np.uint32(len(blob.salt)), np.uint32(blob.iters),
                np.uint32(blob.key_size), np.uint32(blob.start_algo),
                np.uint32(blob.start_size), gpu.ro(blob.iv), gpu.ro(blob.probe_ct),
                np.uint32(blob.nb), np.uint32(blob.hash_len), gpu.ro(blob.checksum)]
    raise ValueError(blob.mode)


DICT_KERNEL = {"distribution": "dist_dict", "partial": "partial_dict",
               "document": "document_dict"}
MASK_KERNEL = {"distribution": "dist_mask", "partial": "partial_mask",
               "document": "document_mask"}


def run_dict_batch(gpu: GPU, blob: AttackBlob, cand_bytes: list[bytes]) -> Optional[int]:
    """Test one batch of pre-encoded candidates. Returns the in-batch index of the
    first match, or None."""
    n = len(cand_bytes)
    if n == 0:
        return None
    stride = gpu.pw_stride
    w = np.zeros((n, stride), np.uint8)
    ln = np.zeros(n, np.uint32)
    for i, b in enumerate(cand_bytes):
        w[i, :len(b)] = np.frombuffer(b, np.uint8)
        ln[i] = len(b)
    wb = cl.Buffer(gpu.ctx, gpu.mf.READ_ONLY | gpu.mf.COPY_HOST_PTR, hostbuf=w.ravel())
    lb = cl.Buffer(gpu.ctx, gpu.mf.READ_ONLY | gpu.mf.COPY_HOST_PTR, hostbuf=ln)
    fb = gpu.found_buf()
    k = gpu.kernel(DICT_KERNEL[blob.mode])
    k(gpu.queue, (n,), None, wb, lb, np.uint32(n), *_dict_tail(gpu, blob), fb)
    idx = gpu.read_found(fb)
    return None if idx == INTMAX else idx


def run_mask(gpu: GPU, blob: AttackBlob, mask: Mask, batch: int,
             progress_every: int, started: float) -> tuple[Optional[int], int]:
    """Enumerate a mask keyspace on the device in batches. Returns
    (winning linear index or None, candidates tested)."""
    k = gpu.kernel(MASK_KERNEL[blob.mode])
    tail = _dict_tail(gpu, blob)
    total = mask.total
    tested = 0
    off = 0
    while off < total:
        bn = min(batch, total - off)
        fb = gpu.found_buf()
        args = mask.gpu_prefix_args(gpu, off, bn) + tail + [fb]
        k(gpu.queue, (bn,), None, *args)
        idx = gpu.read_found(fb)
        tested += bn
        if idx != INTMAX:
            return off + idx, tested
        off += bn
        if progress_every and tested % max(progress_every, batch) < batch:
            _progress(tested, started)
    return None, tested


# ---------------------------------------------------------------------------
# Candidate sources
# ---------------------------------------------------------------------------

def iter_wordlist(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8-sig", errors="strict") as f:
        for line in f:
            cand = line.rstrip("\r\n")
            if cand:
                yield cand


def chunked(it: Iterable[str], size: int) -> Iterator[list[str]]:
    buf: list[str] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _progress(tested: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    print(f"[*] tested={tested:,}, elapsed={elapsed:.1f}s, "
          f"rate={tested/elapsed:,.0f} cand/s", file=sys.stderr)


def crack_document_cpu(args: argparse.Namespace, path: Path, mode: str,
                       encodings: Sequence[str], started: float) -> int:
    """문서 보호(document) 전용 CPU 크래커.

    실제 한컴 문서 보호의 sha256-1k checksum은 '압축을 해제한 평문' 앞 1024바이트
    기준이라, 후보 검증에 raw-DEFLATE 해제가 필요하다. GPU 커널에서 inflate 구현은
    범위가 크므로, PBKDF2가 지배적인 이 프로파일은 OpenSSL 기반 CPU 검증으로 처리한다
    (선택 문자/배포용은 GPU 유지). 수만 후보까지는 실용적인 속도가 나온다."""
    with ref.HwpxPackage(path) as pkg:
        infos = pkg.encrypted_infos()
        info = ref.choose_document_probe(pkg, infos)
        ct = pkg.read(info.full_path)

    cs_kind = "sha256" if len(info.checksum) == 32 else "sha1"
    print(f"[*] cpu verify: PBKDF2-HMAC-SHA1 x{info.iterations}, "
          f"probe={info.full_path}, checksum={cs_kind}-1k(압축해제 후)", file=sys.stderr)

    if args.wordlist:
        candidates = ref.iter_wordlist(Path(args.wordlist))
        print(f"[*] source    : wordlist {args.wordlist}", file=sys.stderr)
    else:
        custom = {"1": args.custom1 or "", "2": args.custom2 or "",
                  "3": args.custom3 or "", "4": args.custom4 or ""}
        candidates = ref.iter_mask(args.mask, custom)
        print(f"[*] source    : mask {args.mask}", file=sys.stderr)

    tested = 0
    for c in candidates:
        enc = ref.verify_document_password(ct, c, info, encodings)
        tested += 1
        if enc:
            return _hit(c, enc, mode, tested)
        if args.progress_every and tested % args.progress_every == 0:
            _progress(tested, started)

    print("[-] 지정한 범위에서 비밀번호를 찾지 못했습니다.")
    return 1


def crack(args: argparse.Namespace) -> int:
    path = Path(args.file)
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    if not encodings:
        raise ValueError("encoding 목록이 비었습니다.")

    with ref.HwpxPackage(path) as pkg:
        detection = ref.detect_mode(pkg)
        mode = args.mode if args.mode != "auto" else detection.mode
        if mode == "none":
            print("[-] 지원 대상 보호 기능을 자동 탐지하지 못했습니다.")
            return 2
        if mode != "document":
            blob = extract_blob(pkg, mode)

    print(f"[*] mode      : {mode}", file=sys.stderr)
    started = time.monotonic()

    # 문서 보호는 실제 한컴 checksum(압축 해제 후 sha256-1k) 검증에 DEFLATE 해제가
    # 필요하므로 GPU 대신 CPU에서 처리한다. 선택 문자/배포용은 아래에서 GPU로 처리.
    if mode == "document":
        return crack_document_cpu(args, path, mode, encodings, started)

    if mode == "distribution":
        print(f"[*] verifiers : {len(blob.verifiers)} SHA-1 digest(s)", file=sys.stderr)
    elif mode == "partial":
        print(f"[*] prefix    : {ref.PARTIAL_PREFIX_TEXT!r} / UTF-16LE "
              f"({blob.nbits} bits)", file=sys.stderr)

    tested_total = 0

    if args.wordlist:
        wl = Path(args.wordlist)
        # size the stride to the longest encoded candidate we will send
        max_len = 0
        for c in iter_wordlist(wl):
            for enc in encodings:
                try:
                    max_len = max(max_len, len(c.encode(enc)))
                except UnicodeEncodeError:
                    pass
        gpu = _gpu_for(args, max(max_len, 16))
        print(f"[*] device    : {gpu.device_name()} (PW_STRIDE={gpu.pw_stride})",
              file=sys.stderr)

        for enc in encodings:
            tested_enc = 0
            for batch in chunked(iter_wordlist(wl), args.batch):
                enc_bytes, keep = [], []
                for c in batch:
                    try:
                        enc_bytes.append(c.encode(enc)); keep.append(c)
                    except UnicodeEncodeError:
                        continue
                if not enc_bytes:
                    continue
                idx = run_dict_batch(gpu, blob, enc_bytes)
                tested_enc += len(enc_bytes)
                tested_total += len(enc_bytes)
                if idx is not None:
                    return _hit(keep[idx], enc, mode, tested_total)
                if args.progress_every and tested_enc % max(args.progress_every, args.batch) < args.batch:
                    _progress(tested_total, started)
        print("[-] 지정한 wordlist에서 비밀번호를 찾지 못했습니다.")
        return 1

    # mask mode
    custom = {"1": args.custom1 or "", "2": args.custom2 or "",
              "3": args.custom3 or "", "4": args.custom4 or ""}
    for enc in encodings:
        try:
            mask = Mask.from_spec(args.mask, custom, enc)
        except UnicodeEncodeError:
            print(f"[!] mask를 {enc}로 인코딩할 수 없어 건너뜁니다.", file=sys.stderr)
            continue
        gpu = _gpu_for(args, max(mask.max_len, 16))
        if enc == encodings[0]:
            print(f"[*] device    : {gpu.device_name()} (PW_STRIDE={gpu.pw_stride})",
                  file=sys.stderr)
        print(f"[*] keyspace  : {mask.total:,} ({enc})", file=sys.stderr)
        win, tested = run_mask(gpu, blob, mask, args.batch, args.progress_every, started)
        tested_total += tested
        if win is not None:
            pw = mask.decode(win).decode(enc, errors="replace")
            return _hit(pw, enc, mode, tested_total)
    print("[-] 지정한 mask 범위에서 비밀번호를 찾지 못했습니다.")
    return 1


_GPU_SINGLETON: Optional[GPU] = None


def _gpu_for(args: argparse.Namespace, min_stride: int) -> GPU:
    """Create (once) or resize the GPU program so the stride covers min_stride."""
    global _GPU_SINGLETON
    stride = 16 * ((min_stride + 15) // 16)          # round up to 16
    if _GPU_SINGLETON is None:
        _GPU_SINGLETON = GPU(pw_stride=stride, device_index=args.device)
    else:
        _GPU_SINGLETON.rebuild(max(stride, _GPU_SINGLETON.pw_stride))
    return _GPU_SINGLETON


def _hit(password: str, encoding: str, mode: str, tested: int) -> int:
    print(f"[+] password : {password}")
    print(f"[+] encoding : {encoding}")
    print(f"[+] mode     : {mode}")
    print(f"[+] tested   : {tested:,}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    # delegate to the reference's inspector for identical output
    return ref.cmd_inspect(args)


def cmd_selftest(args: argparse.Namespace) -> int:
    """Confirm the GPU partial pipeline reproduces the project's PoC key path
    (password 12345), then defer to the reference's own primitive self-test."""
    import base64
    ct = base64.b64decode(
        "RHJVmiVKRikvIEu78gPk3PwhPaZa1IXK8pvnPUqbE7lQKjoXNM5TesSqmMAgCXar"
        "1HRORddCDCbPz9yYq7ywMN6DLl5OXH3XElQvSN1AqZ8=")
    prefix = ref.PARTIAL_PREFIX_UTF16LE
    blob = AttackBlob("partial")
    blob.nbits = len(prefix) * 8
    blob.ct_prefix = ct[:len(prefix)]
    blob.exp_prefix = prefix
    gpu = GPU(pw_stride=16, device_index=args.device)
    idx = run_dict_batch(gpu, blob, [b"54321", b"12345", b"00000"])
    ok = (idx == 1)
    print(f"[{'+ ' if ok else '- '}] GPU partial PoC (12345) : {'OK' if ok else 'FAIL'}")
    print(f"[+] device : {gpu.device_name()}")
    rc = ref.cmd_selftest(args)
    return 0 if (ok and rc == 0) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hwpx_gpu_cracker.py",
        description="HWPX 문서/선택 문자/배포용 보호 OpenCL GPU password recovery")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("inspect", help="보호 방식 및 암호 파라미터 자동 탐지")
    sp.add_argument("file")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("crack", help="GPU 후보 비밀번호 검증")
    sp.add_argument("file")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--wordlist", help="UTF-8/UTF-8-BOM wordlist")
    g.add_argument("--mask", help="Hashcat subset: ?l ?u ?d ?s ?a ?1..?4 ??")
    sp.add_argument("--custom1"); sp.add_argument("--custom2")
    sp.add_argument("--custom3"); sp.add_argument("--custom4")
    sp.add_argument("--mode", choices=["auto", "document", "distribution", "partial"],
                    default="auto")
    sp.add_argument("--encodings", default="utf-8",
                    help="후보 인코딩, comma 구분 (기본 utf-8)")
    sp.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help=f"dispatch 당 후보 수 (기본 {DEFAULT_BATCH})")
    sp.add_argument("--device", type=int, default=None,
                    help="OpenCL device index (기본: 자동 / PYOPENCL_CTX)")
    sp.add_argument("--progress-every", type=int, default=1_000_000,
                    help="N개 후보마다 진행률 (0이면 끔)")
    sp.set_defaults(func=crack)

    sp = sub.add_parser("selftest", help="GPU partial PoC + primitive 자체 점검")
    sp.add_argument("--device", type=int, default=None)
    sp.set_defaults(func=cmd_selftest)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, OSError, IndexError, RuntimeError) as exc:
        print(f"[-] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
