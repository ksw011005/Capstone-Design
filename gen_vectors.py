import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT); _sys.path.insert(0, _ROOT)  # resolve cl/, vectors.json, hwpx_cracker

# Ground-truth vectors from the CPU reference, used to validate OpenCL kernels.
import hashlib, json
import hwpx_cracker as R

vecs = {}

# --- Profile 2: partial (selected-character) ---
part = {}
for pw in ["12345", "hancom", "가나다", "A", "correct horse battery staple 한글"]:
    hexs = hashlib.sha1(pw.encode("utf-8")).hexdigest().upper()
    obf  = R.hancom_obfuscate(hexs)
    loop = ((len(obf) - 1) >> 6) + 1
    buggy = obf * loop
    key  = R.derive_partial_key(pw, "utf-8")
    part[pw] = dict(sha1hex=hexs, obf=obf.hex(), obf_len=len(obf),
                    loop=loop, buggy_len=len(buggy),
                    buggy_sha1=hashlib.sha1(buggy).hexdigest(),
                    aes_key=key.hex())
vecs["partial"] = part

# --- Profile 3: distribution SHA-1 ---
dist = {pw: hashlib.sha1(pw.encode("utf-8")).hexdigest() for pw in
        ["12345", "password", "한글암호", "Yang WangSunv!!"]}
vecs["dist_sha1"] = dist

# --- Profile 1: document PBKDF2-HMAC-SHA1 over SHA256(pw) ---
doc = {}
salt = bytes(range(16))
for pw, it, kl in [("12345", 1024, 32), ("한글", 4096, 32), ("pw", 1000, 32)]:
    sk = hashlib.sha256(pw.encode("utf-8")).digest()
    dk = hashlib.pbkdf2_hmac("sha1", sk, salt, it, dklen=kl)
    doc[f"{pw}|{it}|{kl}"] = dict(start_key=sk.hex(), salt=salt.hex(),
                                  iters=it, klen=kl, dk=dk.hex())
vecs["document_pbkdf2"] = doc

# Known selftest anchor
assert vecs["partial"]["12345"]["aes_key"] == "22eaa1c841c2bde8283e9bcf9a840ca9"
print(json.dumps(vecs, ensure_ascii=False, indent=2))
