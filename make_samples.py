import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT); _sys.path.insert(0, _ROOT)  # resolve cl/, vectors.json, hwpx_cracker

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build synthetic .hwpx packages for the three profiles, matching exactly what
hwpx_cracker.py's parser/detector expects, so the GPU driver can be exercised
end-to-end without a proprietary sample."""
import base64, hashlib, os, zipfile, zlib

MANIFEST_NS = "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"
SHA256_CS = f"{MANIFEST_NS}#sha256-1k"
AES256 = "http://www.w3.org/2001/04/xmlenc#aes256-cbc"
SHA256_SK = "http://www.w3.org/2000/09/xmldsig#sha256"


def raw_deflate(data: bytes) -> bytes:
    return zlib.compress(data, 9)[2:-4]


def enc_entry(xml_bytes: bytes, password: bytes, iters=1024, key_size=32):
    """Return (ciphertext, salt, iv, checksum) for an ODF-style encrypted entry,
    using the same SHA-256 start-key + PBKDF2-HMAC-SHA1 + AES-CBC pipeline as the
    reference. Checksum is sha256 over the first 1024 bytes of the (padded) raw
    DEFLATE plaintext, i.e. the AES-CBC plaintext."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    comp = raw_deflate(xml_bytes)
    if len(comp) % 16:
        comp += b"\x00" * (16 - len(comp) % 16)
    salt = os.urandom(16)
    iv = os.urandom(16)
    sk = hashlib.sha256(password).digest()[:32]
    key = hashlib.pbkdf2_hmac("sha1", sk, salt, iters, dklen=key_size)
    enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    ct = enc.update(comp) + enc.finalize()
    checksum = hashlib.sha256(comp[:1024]).digest()
    return ct, salt, iv, checksum


def manifest_xml(entries) -> bytes:
    """entries: list of dict(full_path, iv, salt, iters, key_size, checksum)."""
    rows = []
    for e in entries:
        rows.append(f"""  <manifest:file-entry manifest:full-path="{e['full_path']}" manifest:media-type="text/xml" manifest:size="{e['size']}">
   <manifest:encryption-data manifest:checksum-type="{SHA256_CS}" manifest:checksum="{base64.b64encode(e['checksum']).decode()}">
    <manifest:algorithm manifest:algorithm-name="{AES256}" manifest:initialisation-vector="{base64.b64encode(e['iv']).decode()}"/>
    <manifest:key-derivation manifest:key-derivation-name="PBKDF2" manifest:salt="{base64.b64encode(e['salt']).decode()}" manifest:iteration-count="{e['iters']}" manifest:key-size="{e['key_size']}"/>
    <manifest:start-key-generation manifest:start-key-generation-name="{SHA256_SK}" manifest:key-size="32"/>
   </manifest:encryption-data>
  </manifest:file-entry>""")
    body = "\n".join(rows)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<manifest:manifest xmlns:manifest="{MANIFEST_NS}" manifest:version="1.2">\n'
            f'{body}\n</manifest:manifest>\n').encode("utf-8")


def write_zip(path, files: dict):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        # mimetype must be first and stored
        z.writestr("mimetype", b"application/hwp+zip")
        for name, data in files.items():
            z.writestr(name, data)


# --------------------------------------------------------------------------- distribution
def make_distribution(path, write_password: str):
    """settings.xml encrypted with the fixed distribution key; its plaintext holds
    the Base64 SHA-1 digest of the real write-password."""
    digest_b64 = base64.b64encode(hashlib.sha1(write_password.encode()).digest()).decode()
    settings = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<settings><item name="writePassword">{digest_b64}</item></settings>\n'
                ).encode("utf-8")
    ct, salt, iv, cs = enc_entry(settings, b'"Yang WangSunv!!"', iters=1024)
    man = manifest_xml([dict(full_path="settings.xml", iv=iv, salt=salt, iters=1024,
                             key_size=32, checksum=cs, size=len(settings))])
    write_zip(path, {"META-INF/manifest.xml": man, "settings.xml": ct})


# --------------------------------------------------------------------------- document
def make_document(path, doc_password: str):
    """A normal password-protected document: Contents/section0.xml encrypted with a
    user password (NOT the distribution fixed key), so detect_mode -> document."""
    xml = ("<?xml version='1.0'?><section>" + ("한글 본문 데이터 " * 300) +
           "</section>").encode("utf-8")
    ct, salt, iv, cs = enc_entry(xml, doc_password.encode(), iters=1024)
    man = manifest_xml([dict(full_path="Contents/section0.xml", iv=iv, salt=salt,
                             iters=1024, key_size=32, checksum=cs, size=len(xml))])
    write_zip(path, {"META-INF/manifest.xml": man, "Contents/section0.xml": ct})


# --------------------------------------------------------------------------- partial
def make_partial(path):
    """No manifest encryption; a content XML carries the project PoC EncryptString
    ciphertext (whose password is 12345)."""
    poc_ct_b64 = ("RHJVmiVKRikvIEu78gPk3PwhPaZa1IXK8pvnPUqbE7lQKjoXNM5TesSqmMAgCXar"
                  "1HRORddCDCbPz9yYq7ywMN6DLl5OXH3XElQvSN1AqZ8=")
    content = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
               f'<hml><body EncryptString="{poc_ct_b64}">text</body></hml>\n'
               ).encode("utf-8")
    # a minimal (unencrypted) manifest so it still looks like a package
    man = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<manifest:manifest xmlns:manifest="{MANIFEST_NS}" manifest:version="1.2">\n'
           f'  <manifest:file-entry manifest:full-path="Contents/section0.xml" manifest:media-type="text/xml"/>\n'
           f'</manifest:manifest>\n').encode("utf-8")
    write_zip(path, {"META-INF/manifest.xml": man, "Contents/section0.xml": content})


if __name__ == "__main__":
    import sys
    outdir = sys.argv[1] if len(sys.argv) > 1 else "samples"
    os.makedirs(outdir, exist_ok=True)
    make_distribution(f"{outdir}/dist.hwpx", "secret777")
    make_document(f"{outdir}/doc.hwpx", "9271")
    make_partial(f"{outdir}/partial.hwpx")
    print("wrote", outdir, os.listdir(outdir))
