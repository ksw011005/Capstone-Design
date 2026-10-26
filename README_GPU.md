# HWPX Cracker — OpenCL GPU 모듈

캡스톤 프로젝트의 Python CPU 레퍼런스(`hwpx_cracker.py`)를 Hashcat 스타일 OpenCL
모듈로 이식한 것입니다. 파일 파싱·암호 파라미터 추출은 검증된 CPU 코드를 그대로
재사용하고, **후보당 반복되는 무거운 연산만 GPU 커널**로 옮겼습니다. README에 적혀
있던 권장 구조(host extractor + attack blob + OpenCL)를 그대로 따릅니다.

세 가지 보호 프로파일을 모두 지원하며, 각 프로파일마다 **사전(wordlist)** 과
**마스크(mask)** 두 가지 후보 공급 방식을 제공합니다.

```
HWPX ──▶ Python host extractor ──▶ attack blob ──▶ OpenCL
         (manifest 파싱, mode 판별,                 ├ document : SHA256→PBKDF2-SHA1→AES-CBC→checksum
          settings.xml 복호화,                       ├ partial  : SHA1→ROL→SHA1→AES-CFB1→prefix 비교
          EncryptString 추출)                        └ dist     : SHA1 → digest 비교
```

---

## 1. 파일 구성

| 경로 | 설명 |
|------|------|
| `hwpx_gpu_cracker.py` | **메인 GPU 드라이버** (CLI: `inspect` / `crack` / `selftest`) |
| `hwpx_cracker.py` | CPU 레퍼런스. 파싱·추출에 재사용되고, GPU 미지원 케이스의 fallback |
| `cl/sha.cl` | SHA-1 / SHA-256 device 함수 |
| `cl/aes.cl` | AES-128/192/256 (암호화 블록=CFB1용, 복호화 블록=CBC용) |
| `cl/kdf.cl` | PBKDF2-HMAC-SHA1, 선택 문자 보호 키 유도(한컴 난독화 포함) |
| `cl/hwpx_kernels.cl` | 3 프로파일 × {dict, mask} = 6개 커널 + 마스크 디코더 |
| `tests/` | 검증 스크립트 및 샘플 생성기 (아래 6절) |
| `vectors.json` | CPU 레퍼런스에서 뽑은 ground-truth 중간값/키 |
| `requirements.txt` | 의존성 |

---

## 2. 설치

```bash
python -m pip install -r requirements.txt
```

GPU가 없거나 드라이버가 없는 환경에서는 CPU용 OpenCL 런타임(예: POCL,
`pip install pocl-binary-distribution`)만으로도 동일하게 동작합니다. 개발·검증도
POCL CPU 런타임에서 수행했습니다.

디바이스 선택은 표준 방식을 따릅니다.

```bash
export PYOPENCL_CTX='0'                 # platform:device 자동 선택
# 또는
python hwpx_gpu_cracker.py crack ... --device 0
```

---

## 3. 사용법

CLI는 CPU 레퍼런스와 최대한 동일하게 맞췄습니다.

### 보호 방식 자동 판별

```bash
python hwpx_gpu_cracker.py inspect sample.hwpx
```

### 사전 공격

```bash
python hwpx_gpu_cracker.py crack sample.hwpx --wordlist passwords.txt
```

### 마스크 공격 (GPU에서 후보 생성)

숫자 5자리:

```bash
python hwpx_gpu_cracker.py crack sample.hwpx --mask "?d?d?d?d?d"
```

한글 커스텀 문자셋 (multibyte 그대로 지원):

```bash
python hwpx_gpu_cracker.py crack sample.hwpx \
  --mask "?1?1?d?d" --custom1 "가나다abc"
```

마스크 토큰: `?l ?u ?d ?s ?a`, 커스텀 `?1~?4`, 리터럴 `??`
(레퍼런스와 동일 문법. 실제로 레퍼런스의 `parse_mask`를 재사용합니다.)

### 공통 옵션

| 옵션 | 기본 | 설명 |
|------|------|------|
| `--mode` | `auto` | `document` / `distribution` / `partial` 강제 |
| `--encodings` | `utf-8` | 후보 인코딩, comma 구분. 예: `utf-8,cp949` |
| `--batch` | `1048576` | dispatch 당 후보 수 |
| `--device` | 자동 | OpenCL device index |
| `--progress-every` | `1000000` | 진행률 출력 주기(0이면 끔) |

성공 출력은 레퍼런스와 동일한 형식입니다.

```
[+] password : 12345
[+] encoding : utf-8
[+] mode     : partial
[+] tested   : 100,000
```

---

## 4. GPU 마스크 엔진

`*_mask` 커널은 선형 인덱스를 혼합진법(mixed-radix)으로 분해해 **디바이스에서 직접
후보를 생성**합니다. 열거 순서는 Python `itertools.product`와 동일(가장 왼쪽 자리가
가장 느리게 변함)하므로, CPU 레퍼런스와 완전히 같은 순서로 후보를 훑습니다.

- 각 위치의 토큰은 **바이트 문자열**이라 한글/UTF-8 multibyte 문자셋이 그대로 동작합니다.
- 키스페이스는 `--batch` 단위로 쪼개 여러 번 dispatch 하며, 매칭 시
  `batch_start + gid`로 실제 비밀번호를 복원합니다. 따라서 2³¹을 넘는 키스페이스도
  분할 처리할 수 있습니다.
- 매칭 판정은 `atomic_min(g_found, gid)`로, 한 dispatch 안에서 **가장 작은 인덱스**가
  결정적으로 승리합니다.

이 덕분에 SHA-1 비교만 하는 값싼 배포용 프로파일도 host의 후보 생성 병목 없이
GPU를 포화시킬 수 있습니다.

---

## 5. 레퍼런스 대비 채운 부분 / 이식 시 반영한 디테일

CPU 레퍼런스의 동작을 **바이트 단위로 동일**하게 재현하도록 다음을 반영했습니다.

- **선택 문자 보호의 버그성 2차 SHA-1**: 80바이트 난독화 버퍼가 `loop_count`만큼
  단순 반복(포인터 미이동)되는 원 구현의 버그를 그대로 커널에 재현했습니다.
  PoC 비밀번호 `12345` → AES key `22eaa1c841c2bde8283e9bcf9a840ca9` 를 GPU에서 검증.
- **AES-CFB1**: prefix 전체(528비트, 66바이트)를 비트 단위로 복호화·비교. IV=0, feedback에
  ciphertext bit 주입. 한 비트라도 어긋나면 즉시 중단(early-abort)해 오답 후보를 빠르게 버립니다.
- **한컴 ROL 난독화**: `al` 초기값 `0xEC`, ROL(1), ASCII hex라 high byte가 0인 처리까지 동일.
- **문서 보호 checksum**: 압축 해제 *전* 의 compressed plaintext 앞 1024바이트에 대한
  `sha256-1k`. probe는 `ciphertext[:min(len,1024)]`를 16바이트로 정렬해 그만큼만
  CBC 복호화 → 레퍼런스의 `decrypt_checksum_probe`/`checksum_digest`와 정확히 일치.
- **start key 판별**: manifest의 `start-key-generation-name`으로 SHA-1/SHA-256 선택.
  start key 길이는 실제 digest 길이(SHA-1 20B / SHA-256 32B)로 clamp해 레퍼런스의
  `digest[:start_key_size]`와 동일하게 맞췄습니다.
- **배포용**: host에서 고정키 `"Yang WangSunv!!"`(따옴표 포함)로 `settings.xml`을 1회
  복호화해 저장된 SHA-1 digest를 뽑고, GPU는 후보 SHA-1 비교만 수행.
- **다중 인코딩**: 인코딩별로 후보 바이트 스트림을 다시 만들어 순차 시도(첫 매칭 인코딩 반환).
- **PW_STRIDE 자동 조정**: 후보(또는 마스크가 낼 수 있는 최장 후보) 길이에 맞춰 커널을
  재빌드하므로 긴 비밀번호도 안전.

---

## 6. 검증

`tests/`에서 아래를 모두 확인했습니다. (POCL CPU 런타임 기준, 어느 CWD에서 실행해도 됨)

```bash
export PYOPENCL_CTX='0'

# 1) 암호 primitive를 Python(hashlib/cryptography)과 대조
python tests/test_primitives_sha.py     # SHA-1 / SHA-256 (한글 UTF-8 포함, 길이 0~200)
python tests/test_primitives_aes.py     # AES-128/192/256 ECB, AES-256-CBC 64블록
python tests/test_primitives_kdf.py     # PBKDF2-HMAC-SHA1 + 선택문자 키 유도

# 2) 커널(dict/mask) 동작 및 마스크 열거 순서·배치 복원 검증
python tests/test_kernels_mask.py
#   - dict 커널 3종이 레퍼런스 결과 재현
#   - ?d?d?d?d?d 마스크가 12345를 인덱스 12345에서 발견(단일/분할 배치)
#   - 한글+ASCII 혼합 마스크(가/나/다 × 라/마 × 0-9) multibyte 검증
#   - document 마스크가 심어둔 숫자 비밀번호 발견

# 3) 전체 파이프라인: 세 프로파일 샘플 .hwpx 생성 후 실제 crack
python tests/make_samples.py samples
python hwpx_gpu_cracker.py crack samples/partial.hwpx --mask "?d?d?d?d?d"   # -> 12345
python hwpx_gpu_cracker.py crack samples/doc.hwpx     --mask "?d?d?d?d"     # -> 9271
printf 'letmein\nsecret777\n' > wl.txt
python hwpx_gpu_cracker.py crack samples/dist.hwpx --wordlist wl.txt        # -> secret777

# 4) PoC 앵커 + 레퍼런스 primitive 동시 점검
python hwpx_gpu_cracker.py selftest
```

`tests/gen_vectors.py`는 레퍼런스에서 `vectors.json`(ground-truth)을 다시 생성합니다.

> 샘플 `.hwpx`는 레퍼런스 파서/판별기가 기대하는 구조로 합성한 것입니다. 실제 한컴
> 문서에도 동일 파이프라인이 적용됩니다(본인 소유/분석 권한이 있는 문서에만 사용).

---

## 7. 성능 & 한계 / 향후 최적화

- **프로파일별 비용**: `distribution`/`partial`은 후보당 매우 저렴(SHA-1 몇 회, 또는
  512비트 CFB1). `document`는 PBKDF2 반복(iteration-count)이 지배적이라 가장 비쌉니다.
- **AES는 정확성 우선 구현**: 현재 커널은 바이트 단위 GF-곱(`xtime`/`gmul`) 방식입니다.
  처리량이 필요하면 **T-table(4×256 룩업)** 로 교체하면 블록당 연산이 크게 줄어듭니다
  (특히 document 프로파일의 CBC probe에 효과적).
- **document checksum**: GPU 경로는 HWPX 표준인 `sha256-1k`(checksum-type에 `1k` 포함,
  또는 미표기)만 처리합니다. full SHA-256(비-1k)이나 SHA-1 checksum, 32B 초과 salt인
  문서는 드라이버가 명확히 알리고 CPU 레퍼런스로 자동 위임하도록 게이팅합니다.
- **단일 디바이스**: 현재는 device 1개를 사용합니다. 멀티 GPU는 키스페이스를
  디바이스 수만큼 분할(각자 `batch_start` 오프셋)하면 그대로 확장됩니다.
- **early-abort**: `partial`/`document` 검증은 불일치 즉시 중단하므로 오답 후보의
  평균 비용이 낮습니다.

---

## 8. 라이선스 / 사용 범위

이 도구는 캡스톤 연구·교육 목적이며, **본인이 소유하거나 분석 권한이 있는 문서**에만
사용해야 합니다.
