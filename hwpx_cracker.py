#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hwpx_cracker.py
===============

HWPX 보호 기능 크래킹 CPU reference implementation.

프로젝트에서 리버싱/실험으로 확인한 3개 프로파일을 분리해 구현한다.

1. 문서 보호(문서 암호)
   password
     -> SHA-256 start key
     -> PBKDF2-HMAC-SHA1(salt, iteration)
     -> AES-256-CBC
     -> manifest checksum(SHA-256/1K 등) 비교

2. 선택 문자 보호
   password
     -> SHA-1
     -> uppercase 40-char hex
     -> 한컴 ROL 난독화
     -> 버그성 2차 SHA-1
     -> first 16 bytes = AES-128 key
     -> AES-CFB1, IV=00*16
     -> UTF-16LE prefix "Hwp Private Information Security;" 비교

3. 배포용 문서
   고정 내부 문자열 b'"Yang WangSunv!!"'
     -> SHA-256
     -> PBKDF2-HMAC-SHA1
     -> AES-256-CBC
     -> settings.xml raw-DEFLATE 복호화
   settings.xml에 저장된 배포용 쓰기 암호의 SHA-1(Base64)과 후보 SHA-1 비교

목표:
    사용자가 salt/IV/hash/offset을 직접 뽑지 않고 .hwpx 파일 자체를 입력하면
    필요한 값을 자동 추출하여 후보 비밀번호를 검증한다.

이 코드는 정확성 검증용 CPU 버전이며, 이후 OpenCL 커널의 기준 구현으로 사용한다.
본인이 소유하거나 분석 권한이 있는 테스트 문서에만 사용한다.

Dependency:
    pip install cryptography
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import re
import string
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HWPX_MIMETYPE = b"application/hwp+zip"
MANIFEST_PATH = "META-INF/manifest.xml"

# 프로젝트에서 확인한 HWPX 배포용 내부 고정 문자열.
# 양끝 큰따옴표도 키 입력 바이트에 포함된다.
DISTRIBUTION_FIXED_PASSWORD = b'"Yang WangSunv!!"'

PARTIAL_PREFIX_TEXT = "Hwp Private Information Security;"
PARTIAL_PREFIX_UTF16LE = PARTIAL_PREFIX_TEXT.encode("utf-16le")
ZERO_IV_16 = b"\x00" * 16


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def local_name(name: str) -> str:
    if "}" in name:
        return name.rsplit("}", 1)[1]
    if ":" in name:
        return name.rsplit(":", 1)[1]
    return name


def attr_local(elem: ET.Element, wanted: str) -> Optional[str]:
    wanted = wanted.lower()
    for k, v in elem.attrib.items():
        if local_name(k).lower() == wanted:
            return v
    return None


def first_attr_local(elem: ET.Element, *wanted: str) -> Optional[str]:
    for name in wanted:
        value = attr_local(elem, name)
        if value is not None:
            return value
    return None


def b64decode_loose(value: Optional[str]) -> bytes:
    if not value:
        return b""
    compact = "".join(value.split())
    try:
        return base64.b64decode(compact, validate=True)
    except Exception:
        try:
            return base64.b64decode(compact)
        except Exception:
            return b""


def aes_cbc_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    if len(key) not in (16, 24, 32):
        raise ValueError(f"잘못된 AES key 길이: {len(key)}")
    if len(iv) != 16:
        raise ValueError(f"잘못된 CBC IV 길이: {len(iv)}")
    if len(ciphertext) % 16 != 0:
        raise ValueError(
            f"AES-CBC 암호문 길이는 16의 배수여야 합니다: {len(ciphertext)}"
        )
    dec = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    return dec.update(ciphertext) + dec.finalize()


def raw_deflate_decompress(data: bytes) -> bytes:
    """
    HWPX/ODF 암호화 엔트리는 AES 복호화 후 raw DEFLATE 데이터이며,
    블록 정렬용 trailing bytes가 남을 수 있다. decompressobj는 deflate
    end-of-stream 뒤 데이터를 무시할 수 있어 이 용도에 적합하다.
    """
    obj = zlib.decompressobj(-15)
    return obj.decompress(data) + obj.flush()


def is_probably_xml(data: bytes) -> bool:
    probe = data.lstrip()
    if not probe.startswith(b"<"):
        return False
    try:
        ET.fromstring(data.decode("utf-8"))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Manifest encryption profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncryptionInfo:
    full_path: str
    checksum_type: str
    checksum: bytes
    algorithm_name: str
    iv: bytes
    kdf_name: str
    salt: bytes
    iterations: int
    key_size: int
    start_key_name: str
    start_key_size: int
    declared_size: Optional[int] = None

    def usable(self) -> bool:
        return (
            bool(self.full_path)
            and len(self.iv) == 16
            and bool(self.salt)
            and self.iterations > 0
            and self.key_size in (16, 24, 32)
            and bool(self.checksum)
        )


def parse_manifest(manifest_xml: bytes) -> list[EncryptionInfo]:
    try:
        root = ET.fromstring(manifest_xml)
    except ET.ParseError as exc:
        raise ValueError(f"{MANIFEST_PATH} XML 파싱 실패: {exc}") from exc

    infos: list[EncryptionInfo] = []

    for file_entry in root.iter():
        if local_name(file_entry.tag).lower() != "file-entry":
            continue

        full_path = first_attr_local(file_entry, "full-path", "fullpath") or ""
        size_text = attr_local(file_entry, "size")
        declared_size = None
        if size_text:
            try:
                declared_size = int(size_text)
            except ValueError:
                pass

        enc_data = None
        for child in file_entry:
            if local_name(child.tag).lower() == "encryption-data":
                enc_data = child
                break
        if enc_data is None:
            continue

        checksum_type = attr_local(enc_data, "checksum-type") or ""
        checksum = b64decode_loose(attr_local(enc_data, "checksum"))

        algorithm = None
        kdf = None
        start_key = None

        for child in enc_data:
            lname = local_name(child.tag).lower()
            if lname == "algorithm":
                algorithm = child
            elif lname == "key-derivation":
                kdf = child
            elif lname == "start-key-generation":
                start_key = child

        algorithm_name = (
            attr_local(algorithm, "algorithm-name") if algorithm is not None else ""
        ) or ""
        iv = b64decode_loose(
            first_attr_local(
                algorithm, "initialisation-vector", "initialization-vector"
            )
            if algorithm is not None
            else None
        )

        kdf_name = (
            attr_local(kdf, "key-derivation-name") if kdf is not None else ""
        ) or ""
        salt = b64decode_loose(attr_local(kdf, "salt") if kdf is not None else None)

        iterations = 1024
        if kdf is not None:
            text = first_attr_local(kdf, "iteration-count", "iterations")
            if text:
                try:
                    iterations = int(text)
                except ValueError:
                    pass

        key_size = 32
        if kdf is not None:
            text = attr_local(kdf, "key-size")
            if text:
                try:
                    key_size = int(text)
                except ValueError:
                    pass

        start_key_name = ""
        start_key_size = 32
        if start_key is not None:
            start_key_name = (
                attr_local(start_key, "start-key-generation-name") or ""
            )
            text = attr_local(start_key, "key-size")
            if text:
                try:
                    start_key_size = int(text)
                except ValueError:
                    pass

        infos.append(
            EncryptionInfo(
                full_path=full_path,
                checksum_type=checksum_type,
                checksum=checksum,
                algorithm_name=algorithm_name,
                iv=iv,
                kdf_name=kdf_name,
                salt=salt,
                iterations=iterations,
                key_size=key_size,
                start_key_name=start_key_name,
                start_key_size=start_key_size,
                declared_size=declared_size,
            )
        )

    return infos


def start_key_for_password(password_bytes: bytes, info: EncryptionInfo) -> bytes:
    """
    프로젝트 HWPX 프로파일은 SHA-256 start key를 사용한다.
    manifest의 start-key-generation 이름도 함께 읽어 SHA-1/SHA-256을 지원한다.
    """
    name = info.start_key_name.lower()

    if "sha1" in name and "sha256" not in name:
        digest = hashlib.sha1(password_bytes).digest()
    else:
        # 프로젝트에서 확인한 기본 프로파일
        digest = hashlib.sha256(password_bytes).digest()

    if info.start_key_size > 0:
        return digest[: info.start_key_size]
    return digest


def derive_hwpx_aes_key(password_bytes: bytes, info: EncryptionInfo) -> bytes:
    start_key = start_key_for_password(password_bytes, info)

    # 프로젝트에서 확인한 PBKDF2-HMAC-SHA1.
    # Python 표준 라이브러리 구현을 사용하여 외부 KDF 의존성을 없앤다.
    return hashlib.pbkdf2_hmac(
        "sha1",
        start_key,
        info.salt,
        info.iterations,
        dklen=info.key_size,
    )


def checksum_digest(data: bytes, checksum_type: str, expected_len: int) -> bytes:
    kind = checksum_type.lower()

    if "sha256" in kind:
        h = hashlib.sha256
    elif "sha1" in kind:
        h = hashlib.sha1
    else:
        # HWPX 프로젝트 샘플의 checksum은 SHA-256/1K.
        # URI가 생략되어도 저장 digest 길이로 보조 판별한다.
        h = hashlib.sha256 if expected_len == 32 else hashlib.sha1

    if "1k" in kind or not kind:
        data = data[:1024]

    return h(data).digest()


def checksum_matches(decrypted_compressed: bytes, info: EncryptionInfo) -> bool:
    if not info.checksum:
        return False
    got = checksum_digest(
        decrypted_compressed,
        info.checksum_type,
        len(info.checksum),
    )
    return got == info.checksum


def decrypt_encrypted_entry(
    ciphertext: bytes,
    password_bytes: bytes,
    info: EncryptionInfo,
) -> bytes:
    key = derive_hwpx_aes_key(password_bytes, info)
    return aes_cbc_decrypt(ciphertext, key, info.iv)


# ---------------------------------------------------------------------------
# HWPX package
# ---------------------------------------------------------------------------

class HwpxPackage:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.zf = zipfile.ZipFile(path, "r")
        except zipfile.BadZipFile as exc:
            raise ValueError("ZIP 기반 HWPX 파일이 아닙니다.") from exc

        self.names = self.zf.namelist()

    def close(self) -> None:
        self.zf.close()

    def __enter__(self) -> "HwpxPackage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self, name: str) -> bytes:
        return self.zf.read(name)

    def has(self, name: str) -> bool:
        return name in self.names

    def manifest_infos(self) -> list[EncryptionInfo]:
        if not self.has(MANIFEST_PATH):
            return []
        return parse_manifest(self.read(MANIFEST_PATH))

    def encrypted_infos(self) -> list[EncryptionInfo]:
        return [
            info
            for info in self.manifest_infos()
            if info.full_path in self.names and info.usable()
        ]

    def validate_mimetype(self) -> bool:
        if "mimetype" not in self.names:
            return False
        try:
            return self.read("mimetype").strip() == HWPX_MIMETYPE
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Selected-character protection
# ---------------------------------------------------------------------------

def hancom_obfuscate(text: str) -> bytes:
    """
    프로젝트 hwp_part_decrypt.py의 한컴 문자열 난독화 로직 그대로 구현.
    입력은 SHA-1 uppercase hex이므로 ASCII 문자만 들어온다.
    """
    out = bytearray(len(text) * 2)

    for i, ch in enumerate(text):
        if i == 0:
            al = 0xEC
        else:
            al = out[i * 2 - 1]

        al = ((al << 1) | (al >> 7)) & 0xFF

        code = ord(ch)
        high_byte = (code >> 8) & 0xFF
        low_byte = code & 0xFF

        out[i * 2] = high_byte if high_byte != 0 else al
        out[i * 2 + 1] = low_byte if low_byte != 0 else al

    return bytes(out)


def derive_partial_key(password: str, encoding: str = "utf-8") -> bytes:
    """
    1) SHA-1(password bytes)
    2) uppercase hex string
    3) 한컴 ROL 난독화
    4) 버그성 2차 SHA-1: 64바이트 단위 루프에서 포인터가 이동하지 않아
       동일 obfuscated buffer가 loop_count만큼 반복된 것으로 재현
    5) SHA-1 digest 앞 16바이트 => AES-128 key
    """
    first_sha1_hex = hashlib.sha1(password.encode(encoding)).hexdigest().upper()
    obfuscated = hancom_obfuscate(first_sha1_hex)

    length = len(obfuscated)
    loop_count = ((length - 1) >> 6) + 1
    buggy_final_data = obfuscated * loop_count

    return hashlib.sha1(buggy_final_data).digest()[:16]


def aes_cfb1_decrypt_prefix(
    ciphertext: bytes,
    key: bytes,
    wanted_plain_bytes: int,
    iv: bytes = ZERO_IV_16,
) -> bytes:
    """
    AES-CFB1 복호화. 필요한 prefix 바이트까지만 수행한다.
    CFB1 feedback에는 ciphertext bit를 넣는다.
    """
    if len(key) != 16:
        raise ValueError("선택 문자 보호 AES key는 16바이트여야 합니다.")
    if len(iv) != 16:
        raise ValueError("CFB1 IV는 16바이트여야 합니다.")

    max_bits = min(len(ciphertext), wanted_plain_bytes) * 8
    if max_bits <= 0:
        return b""

    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()

    shift_reg = int.from_bytes(iv, byteorder="big")
    mask128 = (1 << 128) - 1
    out = bytearray((max_bits + 7) // 8)

    for bit_no in range(max_bits):
        ct_byte = ciphertext[bit_no >> 3]
        bit_index = 7 - (bit_no & 7)
        ct_bit = (ct_byte >> bit_index) & 1

        reg_block = shift_reg.to_bytes(16, byteorder="big")
        enc_reg = encryptor.update(reg_block)
        ks_bit = (enc_reg[0] >> 7) & 1
        pt_bit = ct_bit ^ ks_bit

        if pt_bit:
            out[bit_no >> 3] |= 1 << bit_index

        shift_reg = ((shift_reg << 1) | ct_bit) & mask128

    encryptor.finalize()
    return bytes(out)


def verify_partial_password(
    ciphertext: bytes,
    candidate: str,
    encodings: Sequence[str],
) -> Optional[str]:
    for enc in encodings:
        try:
            key = derive_partial_key(candidate, enc)
        except UnicodeEncodeError:
            continue

        clear = aes_cfb1_decrypt_prefix(
            ciphertext,
            key,
            len(PARTIAL_PREFIX_UTF16LE),
        )
        if clear == PARTIAL_PREFIX_UTF16LE:
            return enc

    return None


def find_encrypt_strings(pkg: HwpxPackage) -> list[tuple[str, bytes]]:
    """
    평문 XML에서 EncryptString 계열 attribute를 찾아 Base64 ciphertext 추출.
    선택 문자 보호 문서는 전체 패키지를 암호화하지 않으므로 XML 스캔이 가능하다.
    """
    found: list[tuple[str, bytes]] = []

    for name in pkg.names:
        if not name.lower().endswith((".xml", ".hpf")):
            continue

        try:
            raw = pkg.read(name)
            text = raw.decode("utf-8")
            root = ET.fromstring(text)
        except Exception:
            continue

        for elem in root.iter():
            for attr_name, attr_value in elem.attrib.items():
                lname = local_name(attr_name).lower()
                # 자료에는 EncryptString, 메모에는 EcryptString 표기도 있어
                # 두 경우 모두 좁게 허용한다.
                if lname not in {"encryptstring", "ecryptstring"}:
                    continue

                decoded = b64decode_loose(attr_value)
                if decoded:
                    found.append((name, decoded))

    return found


# ---------------------------------------------------------------------------
# Distribution document
# ---------------------------------------------------------------------------

def settings_info(infos: Sequence[EncryptionInfo]) -> Optional[EncryptionInfo]:
    for info in infos:
        if info.full_path.lower() == "settings.xml":
            return info
    for info in infos:
        if info.full_path.lower().endswith("/settings.xml"):
            return info
    return None


def decrypt_distribution_settings(
    pkg: HwpxPackage,
    infos: Sequence[EncryptionInfo],
) -> Optional[bytes]:
    info = settings_info(infos)
    if info is None:
        return None

    ciphertext = pkg.read(info.full_path)

    try:
        decrypted_compressed = decrypt_encrypted_entry(
            ciphertext,
            DISTRIBUTION_FIXED_PASSWORD,
            info,
        )
    except Exception:
        return None

    # 고정 배포용 키가 맞는지 manifest checksum으로 먼저 검증한다.
    if not checksum_matches(decrypted_compressed, info):
        return None

    try:
        clear = raw_deflate_decompress(decrypted_compressed)
    except zlib.error:
        return None

    if not is_probably_xml(clear):
        return None

    return clear


def _collect_xml_strings(xml_bytes: bytes) -> list[str]:
    strings: list[str] = []
    try:
        root = ET.fromstring(xml_bytes.decode("utf-8"))
    except Exception:
        # 구조를 못 읽어도 Base64 후보는 정규식으로 보조 추출
        text = xml_bytes.decode("utf-8", errors="ignore")
        return re.findall(r"[A-Za-z0-9+/]{27,}={0,2}", text)

    for elem in root.iter():
        if elem.text and elem.text.strip():
            strings.append(elem.text.strip())
        if elem.tail and elem.tail.strip():
            strings.append(elem.tail.strip())
        for value in elem.attrib.values():
            value = value.strip()
            if value:
                strings.append(value)

    return strings


def extract_sha1_verifiers(settings_xml: bytes) -> set[bytes]:
    """
    프로젝트 실험상 settings.xml에는 배포용 쓰기 암호의 SHA-1 digest가
    Base64로 저장된다. 정확한 XML element 이름에 종속되지 않도록
    20바이트로 디코딩되는 Base64 값과 40자리 hex 값을 수집한다.
    """
    results: set[bytes] = set()

    for value in _collect_xml_strings(settings_xml):
        compact = "".join(value.split())

        if re.fullmatch(r"[0-9A-Fa-f]{40}", compact):
            try:
                results.add(bytes.fromhex(compact))
            except ValueError:
                pass

        if re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact):
            decoded = b64decode_loose(compact)
            if len(decoded) == 20:
                results.add(decoded)

    return results


def verify_distribution_password(
    candidate: str,
    verifiers: set[bytes],
    encodings: Sequence[str],
) -> Optional[str]:
    for enc in encodings:
        try:
            digest = hashlib.sha1(candidate.encode(enc)).digest()
        except UnicodeEncodeError:
            continue
        if digest in verifiers:
            return enc
    return None


# ---------------------------------------------------------------------------
# Document password
# ---------------------------------------------------------------------------

def choose_document_probe(
    pkg: HwpxPackage,
    infos: Sequence[EncryptionInfo],
) -> EncryptionInfo:
    usable = [x for x in infos if x.usable() and x.full_path in pkg.names]
    if not usable:
        raise ValueError("검증 가능한 encrypted file-entry가 없습니다.")

    # XML 계열 우선, 그 안에서는 암호문 크기가 작은 엔트리를 택한다.
    def rank(info: EncryptionInfo):
        lname = info.full_path.lower()
        xml_priority = 0 if lname.endswith((".xml", ".hpf")) else 1
        try:
            size = pkg.zf.getinfo(info.full_path).file_size
        except Exception:
            size = 1 << 60
        return (xml_priority, size, info.full_path)

    return sorted(usable, key=rank)[0]


def decrypt_checksum_probe(
    ciphertext: bytes,
    password_bytes: bytes,
    info: EncryptionInfo,
) -> bytes:
    """
    /1K checksum만 확인하므로 전체 엔트리를 복호화할 필요가 없다.
    CBC는 처음부터 필요한 블록까지만 복호화하면 첫 1024 bytes를 얻을 수 있다.
    """
    if len(ciphertext) % 16 != 0:
        raise ValueError("AES-CBC 암호문 길이가 16의 배수가 아닙니다.")

    full_needed = "1k" not in info.checksum_type.lower()
    if full_needed:
        probe_cipher = ciphertext
    else:
        # checksum은 first min(1024, plaintext_len). AES block 경계로 자른다.
        probe_cipher = ciphertext[: min(len(ciphertext), 1024)]

        # 1024는 이미 16의 배수이지만 짧은 파일을 안전하게 정렬.
        probe_cipher = probe_cipher[: len(probe_cipher) - (len(probe_cipher) % 16)]
        if not probe_cipher and ciphertext:
            probe_cipher = ciphertext[:16]

    key = derive_hwpx_aes_key(password_bytes, info)
    return aes_cbc_decrypt(probe_cipher, key, info.iv)


def verify_document_password(
    ciphertext: bytes,
    candidate: str,
    info: EncryptionInfo,
    encodings: Sequence[str],
) -> Optional[str]:
    for enc in encodings:
        try:
            password_bytes = candidate.encode(enc)
        except UnicodeEncodeError:
            continue

        try:
            decrypted_compressed = decrypt_checksum_probe(
                ciphertext,
                password_bytes,
                info,
            )
        except Exception:
            continue

        if checksum_matches(decrypted_compressed, info):
            return enc

    return None


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    mode: str
    infos: list[EncryptionInfo]
    encrypt_strings: list[tuple[str, bytes]]
    distribution_settings: Optional[bytes]
    distribution_verifiers: set[bytes]


def detect_mode(pkg: HwpxPackage) -> Detection:
    infos = pkg.encrypted_infos()

    # manifest encryption이 있으면 먼저 배포용 고정키로 settings.xml을 시험한다.
    if infos:
        settings = decrypt_distribution_settings(pkg, infos)
        if settings is not None:
            verifiers = extract_sha1_verifiers(settings)
            return Detection(
                mode="distribution",
                infos=infos,
                encrypt_strings=[],
                distribution_settings=settings,
                distribution_verifiers=verifiers,
            )

        # 배포용 고정키가 아니면 일반 문서 암호 프로파일로 분류.
        return Detection(
            mode="document",
            infos=infos,
            encrypt_strings=[],
            distribution_settings=None,
            distribution_verifiers=set(),
        )

    encrypt_strings = find_encrypt_strings(pkg)
    if encrypt_strings:
        return Detection(
            mode="partial",
            infos=[],
            encrypt_strings=encrypt_strings,
            distribution_settings=None,
            distribution_verifiers=set(),
        )

    return Detection(
        mode="none",
        infos=[],
        encrypt_strings=[],
        distribution_settings=None,
        distribution_verifiers=set(),
    )


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

BUILTIN_MASKS = {
    "l": string.ascii_lowercase,
    "u": string.ascii_uppercase,
    "d": string.digits,
    "s": string.punctuation,
}
BUILTIN_MASKS["a"] = (
    BUILTIN_MASKS["l"]
    + BUILTIN_MASKS["u"]
    + BUILTIN_MASKS["d"]
    + BUILTIN_MASKS["s"]
)


def parse_mask(mask: str, custom: dict[str, str]) -> list[str]:
    slots: list[str] = []
    i = 0

    while i < len(mask):
        if mask[i] != "?":
            slots.append(mask[i])
            i += 1
            continue

        if i + 1 >= len(mask):
            raise ValueError("mask 마지막의 단독 '?'는 허용되지 않습니다.")

        token = mask[i + 1]
        i += 2

        if token == "?":
            slots.append("?")
        elif token in BUILTIN_MASKS:
            slots.append(BUILTIN_MASKS[token])
        elif token in "1234":
            chars = custom.get(token, "")
            if not chars:
                raise ValueError(f"?{token} 사용자 문자셋이 비어 있습니다.")
            slots.append(chars)
        else:
            raise ValueError(f"지원하지 않는 mask token: ?{token}")

    return slots


def iter_mask(mask: str, custom: dict[str, str]) -> Iterator[str]:
    slots = parse_mask(mask, custom)
    for combo in itertools.product(*slots):
        yield "".join(combo)


def iter_wordlist(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8-sig", errors="strict") as f:
        for line in f:
            candidate = line.rstrip("\r\n")
            if candidate:
                yield candidate


def iter_candidates(args: argparse.Namespace) -> Iterable[str]:
    if args.wordlist:
        return iter_wordlist(Path(args.wordlist))

    custom = {
        "1": args.custom1 or "",
        "2": args.custom2 or "",
        "3": args.custom3 or "",
        "4": args.custom4 or "",
    }
    return iter_mask(args.mask, custom)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def encodings_from_arg(text: str) -> list[str]:
    encs = [x.strip() for x in text.split(",") if x.strip()]
    if not encs:
        raise ValueError("encoding 목록이 비었습니다.")
    return encs


def print_progress(tested: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = tested / elapsed
    print(
        f"[*] tested={tested:,}, elapsed={elapsed:.1f}s, rate={rate:,.1f} cand/s",
        file=sys.stderr,
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.file)

    with HwpxPackage(path) as pkg:
        detection = detect_mode(pkg)

        print(f"file                  : {path}")
        print(f"valid_hwpx_mimetype   : {pkg.validate_mimetype()}")
        print(f"mode                  : {detection.mode}")
        print(f"encrypted_entries     : {len(detection.infos)}")
        print(f"encrypt_string_count  : {len(detection.encrypt_strings)}")

        if detection.infos:
            for info in detection.infos[:10]:
                print(
                    f"  - {info.full_path}: "
                    f"iter={info.iterations}, key={info.key_size}, "
                    f"salt={info.salt.hex()}, iv={info.iv.hex()}, "
                    f"checksum={info.checksum.hex()}"
                )

        if detection.encrypt_strings:
            for stream, ciphertext in detection.encrypt_strings[:10]:
                print(
                    f"  - EncryptString @ {stream}: "
                    f"{len(ciphertext)} bytes"
                )

        if detection.mode == "distribution":
            print(
                f"distribution_sha1_verifiers: "
                f"{len(detection.distribution_verifiers)}"
            )

    return 0


def cmd_crack(args: argparse.Namespace) -> int:
    path = Path(args.file)
    encodings = encodings_from_arg(args.encodings)

    with HwpxPackage(path) as pkg:
        detection = detect_mode(pkg)

        if args.mode != "auto":
            mode = args.mode
        else:
            mode = detection.mode

        if mode == "none":
            print("[-] 지원 대상 보호 기능을 자동 탐지하지 못했습니다.")
            return 2

        if mode == "document":
            if not detection.infos:
                detection = Detection(
                    mode="document",
                    infos=pkg.encrypted_infos(),
                    encrypt_strings=[],
                    distribution_settings=None,
                    distribution_verifiers=set(),
                )

            probe_info = choose_document_probe(pkg, detection.infos)
            ciphertext = pkg.read(probe_info.full_path)

            print("[*] mode: document-password", file=sys.stderr)
            print(f"[*] verifier entry: {probe_info.full_path}", file=sys.stderr)

            verifier = lambda candidate: verify_document_password(
                ciphertext, candidate, probe_info, encodings
            )

        elif mode == "distribution":
            settings = detection.distribution_settings
            verifiers = detection.distribution_verifiers

            if settings is None:
                settings = decrypt_distribution_settings(pkg, pkg.encrypted_infos())
            if settings is None:
                raise ValueError(
                    "배포용 고정 키로 settings.xml을 복호화하지 못했습니다."
                )
            if not verifiers:
                verifiers = extract_sha1_verifiers(settings)
            if not verifiers:
                raise ValueError(
                    "복호화된 settings.xml에서 20바이트 SHA-1 검증값을 찾지 못했습니다."
                )

            print("[*] mode: distribution-password", file=sys.stderr)
            print(
                f"[*] settings SHA-1 verifier candidates: {len(verifiers)}",
                file=sys.stderr,
            )

            verifier = lambda candidate: verify_distribution_password(
                candidate, verifiers, encodings
            )

        elif mode == "partial":
            encrypt_strings = detection.encrypt_strings or find_encrypt_strings(pkg)
            if not encrypt_strings:
                raise ValueError("EncryptString 암호문을 찾지 못했습니다.")

            source_name, ciphertext = encrypt_strings[0]

            print("[*] mode: selected-character protection", file=sys.stderr)
            print(f"[*] verifier source: {source_name}", file=sys.stderr)
            print(
                f"[*] prefix: {PARTIAL_PREFIX_TEXT!r} / UTF-16LE",
                file=sys.stderr,
            )

            verifier = lambda candidate: verify_partial_password(
                ciphertext, candidate, encodings
            )

        else:
            raise ValueError(f"지원하지 않는 mode: {mode}")

        started = time.monotonic()
        tested = 0

        for candidate in iter_candidates(args):
            tested += 1

            if args.max_candidates is not None and tested > args.max_candidates:
                break

            matched_encoding = verifier(candidate)
            if matched_encoding is not None:
                print(f"[+] password : {candidate}")
                print(f"[+] encoding : {matched_encoding}")
                print(f"[+] mode     : {mode}")
                print(f"[+] tested   : {tested:,}")
                return 0

            if args.progress_every and tested % args.progress_every == 0:
                print_progress(tested, started)

        print("[-] 지정한 후보 범위에서 비밀번호를 찾지 못했습니다.")
        return 1


def cmd_extract_distribution_settings(args: argparse.Namespace) -> int:
    path = Path(args.file)
    output = Path(args.output)

    with HwpxPackage(path) as pkg:
        infos = pkg.encrypted_infos()
        settings = decrypt_distribution_settings(pkg, infos)

    if settings is None:
        raise ValueError("배포용 settings.xml 자동 복호화에 실패했습니다.")

    output.write_bytes(settings)
    verifiers = extract_sha1_verifiers(settings)

    print(f"[+] output    : {output}")
    print(f"[+] xml bytes : {len(settings):,}")
    print(f"[+] SHA-1 verifier candidates : {len(verifiers)}")
    for digest in sorted(verifiers):
        print(f"    {base64.b64encode(digest).decode()}  ({digest.hex()})")
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    # 사용자가 제공한 선택 문자 보호 PoC.
    ciphertext = base64.b64decode(
        "RHJVmiVKRikvIEu78gPk3PwhPaZa1IXK8pvnPUqbE7lQKjoXNM5TesSqmMAgCXar"
        "1HRORddCDCbPz9yYq7ywMN6DLl5OXH3XElQvSN1AqZ8="
    )
    key = derive_partial_key("12345", "utf-8")
    clear = aes_cfb1_decrypt_prefix(
        ciphertext,
        key,
        len(PARTIAL_PREFIX_UTF16LE),
    )

    assert key.hex() == "22eaa1c841c2bde8283e9bcf9a840ca9"
    assert clear == PARTIAL_PREFIX_UTF16LE
    assert verify_partial_password(ciphertext, "12345", ["utf-8"]) == "utf-8"
    assert verify_partial_password(ciphertext, "54321", ["utf-8"]) is None

    # PBKDF2 reproducibility sanity check
    info = EncryptionInfo(
        full_path="settings.xml",
        checksum_type="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0#sha256-1k",
        checksum=b"\x00" * 32,
        algorithm_name="http://www.w3.org/2001/04/xmlenc#aes256-cbc",
        iv=b"\x11" * 16,
        kdf_name="PBKDF2",
        salt=b"\x22" * 16,
        iterations=1024,
        key_size=32,
        start_key_name="sha256",
        start_key_size=32,
    )
    k1 = derive_hwpx_aes_key(DISTRIBUTION_FIXED_PASSWORD, info)
    k2 = hashlib.pbkdf2_hmac(
        "sha1",
        hashlib.sha256(DISTRIBUTION_FIXED_PASSWORD).digest(),
        b"\x22" * 16,
        1024,
        dklen=32,
    )
    assert k1 == k2

    print("[+] selftest OK")
    print(f"[+] partial known key : {key.hex()}")
    print(f"[+] partial prefix    : {clear.decode('utf-16le')}")
    return 0


def add_candidate_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wordlist", help="UTF-8/UTF-8-BOM wordlist")
    group.add_argument(
        "--mask",
        help="Hashcat-like subset: ?l ?u ?d ?s ?a ?1..?4, ??",
    )

    parser.add_argument("--custom1", help="?1 custom charset")
    parser.add_argument("--custom2", help="?2 custom charset")
    parser.add_argument("--custom3", help="?3 custom charset")
    parser.add_argument("--custom4", help="?4 custom charset")

    parser.add_argument(
        "--encodings",
        default="utf-8",
        help="후보 비밀번호 인코딩. 여러 개면 comma 구분 (기본 utf-8)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="N개 후보마다 진행률 표시, 0이면 끔 (기본 1000)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="테스트 후보 수 제한",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hwpx_cracker.py",
        description=(
            "HWPX 문서 보호 / 선택 문자 보호 / 배포용 문서 "
            "자동 추출 + CPU password recovery reference"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("inspect", help="보호 방식 및 암호 파라미터 자동 탐지")
    sp.add_argument("file")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("crack", help="후보 비밀번호 검증")
    sp.add_argument("file")
    add_candidate_args(sp)
    sp.add_argument(
        "--mode",
        choices=["auto", "document", "distribution", "partial"],
        default="auto",
        help="기본 auto",
    )
    sp.set_defaults(func=cmd_crack)

    sp = sub.add_parser(
        "extract-dist-settings",
        help="배포용 문서 settings.xml을 내부 고정키로 복호화",
    )
    sp.add_argument("file")
    sp.add_argument("-o", "--output", required=True)
    sp.set_defaults(func=cmd_extract_distribution_settings)

    sp = sub.add_parser("selftest", help="프로젝트 PoC 기반 암호 primitive 자체 점검")
    sp.set_defaults(func=cmd_selftest)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except (
        ValueError,
        FileNotFoundError,
        OSError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"[-] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
