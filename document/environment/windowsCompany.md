# Windows（社用 PC）実行環境仕様書（APA / paper_pipeline_v4）

最終更新: 2026-01-09

このドキュメントは、`C:\Users\takumi\develop\APA` 配下の **APA プログラム群**
（例: `paper_pipeline_v4.py`）を Windows（社用 PC）上で **誰でも再現できる**ことを目的に、
OS/ハード/ツール/Python/依存関係/実行確認手順をまとめた「環境仕様書」です。

> 注意（重要）
>
> - 本 PC では **システム環境を汚さない** 方針のため、Git/Miniconda は `C:\Users\takumi\develop\` 配下へローカル導入しています。
> - **PATH へ追加していません**。コマンドは **フルパス**で実行してください（後述）。
> - `python`（WindowsApps）/ `git`（別インストール）が紛れ込むため、**誤った実行ファイルを使わない**こと。

---

## 1. 対象リポジトリ

- リポジトリ: `APA`
- 作業ディレクトリ（本 PC）:
  - `C:\Users\takumi\develop\APA`

---

## 2. OS / ハードウェア（概要）

### 2.1 OS

- OS: Microsoft Windows 11 Pro
- Version / Build:
  - `10.0.26100` (Build `26100`)
- アーキテクチャ: 64-bit
- Python 側で確認したプラットフォーム文字列:
  - `Windows-11-10.0.26100-SP0`

### 2.2 ハードウェア

- メーカー: Dell Inc.
- モデル: Vostro 3668
- CPU:
  - Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
  - 4 cores / 8 logical processors
- メモリ:
  - 16,292 MB（`systeminfo`）
  - 17,083,355,136 bytes（PowerShell/CIM）
- BIOS:
  - Dell Inc. / SMBIOSBIOSVersion: `1.4.0` / ReleaseDate: `2017-07-19`

> セキュリティ注意
>
> - Serial Number 等の個体識別情報は、社内共有版の仕様書には原則マスクして記載してください。
> - 本ドキュメントは再現性に必要な情報（OS/CPU/メモリ/ツール構成）を優先しています。

---

## 3. シェル / CLI ツール

- 既定シェル: `cmd.exe`
- PowerShell: 利用可能（情報採取やバッチ実行で使用）

### 3.1 Git（Portable をローカル導入）

本 PC では **Portable Git** を `develop` 配下に展開し、**PATH 追加なし**で運用しています。

- 実体:
  - `C:\Users\takumi\develop\git\bin\git.exe`
- バージョン:
  - `git version 2.48.1.windows.1`
- グローバル設定:
  - `user.name=takumi0616`
  - `user.email=takumi0616.mrt@gmail.com`

> 補足（混入注意）
>
> - `where git` では `C:\Users\takumi\AppData\Local\Programs\Git\cmd\git.exe` が見える場合があります。
> - 本リポジトリ運用では **Portable 版のフルパス**を常に使用してください。

### 3.2 torch.hub / git 依存について（XFeat）

`paper_pipeline_v4.py` は XFeat を `torch.hub` 経由でロードします。
環境によっては `torch.hub` が内部で `git` を呼ぶため、
本プロジェクトでは `ensure_portable_git_on_path()` により **Portable Git を一時的に PATH に追加**して実行します。

---

## 4. Python 実行環境（最重要）

### 4.1 Miniconda（ローカル導入・PATH 追加なし）

本 PC では **Miniconda** を `develop` 配下にサイレント導入し、**PATH 追加なし**で運用しています。

- インストール先:
  - `C:\Users\takumi\develop\miniconda3`
- Python:
  - `C:\Users\takumi\develop\miniconda3\python.exe`
  - `Python 3.13.11`
- conda:
  - `C:\Users\takumi\develop\miniconda3\Scripts\conda.exe`
  - `conda 25.11.1`
- pip:
  - `C:\Users\takumi\develop\miniconda3\Scripts\pip.exe`
  - `pip 25.3 (python 3.13)`

### 4.2 conda 環境

`conda env list` の結果:

- `base` : `C:\Users\takumi\develop\miniconda3`
- `xfeat`: `C:\Users\takumi\develop\miniconda3\envs\xfeat`

> 混入注意（重要）
>
> - `python` は WindowsApps の `python.exe` が `where python` で見えます。
> - `conda` / `pip` は PATH に無いので、短縮コマンドは通りません。
> - 必ず **`C:\Users\takumi\develop\miniconda3\python.exe`** を直接使ってください。

---

## 5. 主要ライブラリと要件

以下は本 PC の Miniconda（base）上で import と動作が確認できた構成です（2026-01-09 時点）。

### 5.1 OpenCV（WeChat QR 有効）

- OpenCV:
  - `cv2.__version__ == 4.12.0`
  - `opencv_wechat_available: True`
  - インストール例（pip freeze から確認）:
    - `opencv-contrib-python==4.12.0.88`
    - `opencv-python==4.12.0.88`

`paper_pipeline_v4.py` では、フォーム B の QR が小さい/低解像度の時に標準 `QRCodeDetector` が失敗しやすいため、
WeChat エンジン（`wechat_qrcode_WeChatQRCode`）を優先的に使います。

そのため **opencv-contrib** ビルドが必須です。

### 5.2 PyTorch

- `torch==2.9.1+cpu`
- `torch.cuda.is_available() == False`（CPU 実行）

### 5.3 その他（抜粋）

- NumPy: `numpy==2.2.6`
- Pillow: `pillow==12.1.0`
- DocAligner:
  - `docaligner_docsaid @ git+https://github.com/DocsaidLab/DocAligner.git@...`
- capybara:
  - `capybara-docsaid==0.12.0`

### 5.4 互換性メモ（DocAligner / capybara）

Windows 環境では `capybara` が namespace package になっている影響などで、
`docaligner` の import が期待するシンボルが欠けて例外になることがあります。

本プロジェクトの `paper_pipeline_v4.py` では `patch_capybara_exports()` を実行して、
`docaligner` が期待するシンボルを `capybara` 側へ補うことで互換性を確保しています。

#### 参考: pip freeze（本 PC / base で採取）

再現性のため、pip の全インストール一覧（`pip freeze`）を付録に掲載します。

<details>
<summary>pip freeze 一覧（クリックで展開）</summary>

```txt
anaconda-anon-usage @ file:///opt/miniconda3/conda-bld/anaconda-anon-usage_1764636648062/work
anaconda-auth @ file:///C:/miniconda3/conda-bld/anaconda-cloud-auth-split_1765839853030/work
anaconda-cli-base @ file:///C:/miniconda3/conda-bld/anaconda-cli-base_1764888883727/work
annotated-types @ file:///C:/miniconda3/conda-bld/annotated-types_1761745107361/work
archspec @ file:///home/task_175812491784513/conda-bld/archspec_1758124989039/work
beautifulsoup4==4.14.3
blinker==1.9.0
boltons @ file:///C:/b/abs_e2_iokhxbp/croot/boltons_1751383740243/work
brotlicffi @ file:///C:/miniconda3/conda-bld/brotlicffi_1764961374486/work
capybara-docsaid==0.12.0
certifi @ file:///home/conda/feedstock_root/build_artifacts/certifi_1767500808759/work/certifi
cffi @ file:///C:/miniconda3/conda-bld/cffi_1761832792955/work
charset-normalizer @ file:///C:/miniconda3/conda-bld/charset-normalizer_1761744975868/work
click @ file:///C:/miniconda3/conda-bld/click_1764332372183/work
colorama @ file:///C:/Users/dev-admin/perseverance-python-buildout/croot/colorama_1729036581634/work
colored==2.3.1
coloredlogs==15.0.1
conda @ file:///C:/miniconda3/conda-bld/conda_1765567840841/work/conda-src
conda-anaconda-telemetry @ file:///croot/conda-anaconda-telemetry_1755883788794/work
conda-anaconda-tos @ file:///C:/b/abs_c9yejtsx9g/croot/conda-anaconda-tos_1755123332641/work
conda-content-trust @ file:///C:/Users/dev-admin/perseverance-python-buildout/croot/conda-content-trust_1729088072778/work
conda-libmamba-solver @ file:///C:/miniconda3/conda-bld/conda-libmamba-solver_1764245615235/work/src
conda-package-handling @ file:///C:/miniconda3/conda-bld/conda-package-handling_1762366515145/work
conda_package_streaming @ file:///C:/miniconda3/conda-bld/conda-package-streaming_1762361689679/work
contourpy==1.3.3
cryptography @ file:///C:/miniconda3/conda-bld/cryptography-split_1761932023265/work
cycler==0.12.1
dacite==1.9.2
dill==0.4.0
distro @ file:///C:/Users/dev-admin/perseverance-python-buildout/croot/distro_1729059153117/work
docaligner_docsaid @ git+https://github.com/DocsaidLab/DocAligner.git@16039d5e4a3a0565c24eaffd5c9c16cbf47e92a0
filelock==3.20.0
Flask==3.1.2
flatbuffers==25.12.19
fonttools==4.61.1
frozendict @ file:///C:/miniconda3/conda-bld/frozendict_1761750728192/work
fsspec==2025.12.0
humanfriendly==10.0
idna @ file:///C:/miniconda3/conda-bld/idna_1761912043388/work
ImageIO==2.37.2
itsdangerous==2.2.0
jaraco.classes @ file:///C:/b/abs_6erueoob1v/croot/jaraco.classes_1755516340851/work
jaraco.context @ file:///C:/Users/dev-admin/buildout/perseverance-python-buildout/croot/jaraco.context_1731721000658/work
jaraco.functools @ file:///C:/b/abs_a1jv6v_pzp/croot/jaraco.functools_1740408129620/work
Jinja2==3.1.6
jpeg4py==0.1.4
jsonpatch @ file:///C:/Users/dev-admin/perseverance-python-buildout/croot/jsonpatch_1729054776004/work
jsonpointer @ file:///C:/b/abs_73u73l7pl9/croot/jsonpointer_1753788460913/work
keyring @ file:///C:/miniconda3/conda-bld/keyring_1763637203252/work
kiwisolver==1.4.9
kornia==0.7.2
kornia_rs==0.1.10
libmambapy @ file:///C:/miniconda3/conda-bld/mamba-split_1763111608356/work/libmambapy
markdown-it-py @ file:///C:/b/abs_3espcrdl35/croot/markdown-it-py_1756299213703/work
MarkupSafe==3.0.3
matplotlib==3.10.8
mdurl @ file:///C:/miniconda3/conda-bld/mdurl_1758552277768/work
menuinst @ file:///C:/miniconda3/conda-bld/menuinst_1765382397455/work
ml_dtypes==0.5.4
more-itertools @ file:///C:/miniconda3/conda-bld/more-itertools_1761121564395/work
mpmath==1.3.0
msgpack @ file:///C:/b/abs_4b3t4uhz3r/croot/msgpack-python_1750958631084/work
natsort==8.4.0
networkx==3.6.1
numpy==2.2.6
onnx==1.20.0
onnxruntime==1.23.2
onnxslim==0.1.82
opencv-contrib-python==4.12.0.88
opencv-python==4.12.0.88
packaging @ file:///C:/miniconda3/conda-bld/packaging_1761049099114/work
pdf2image==1.17.0
piexif==1.1.3
pillow==12.1.0
pillow_heif==1.1.1
pkce @ file:///C:/Users/dev-admin/perseverance-python-buildout/croot/pkce_1729049216383/work
platformdirs @ file:///C:/miniconda3/conda-bld/platformdirs_1762356623609/work
pluggy @ file:///C:/b/abs_dfec_m79vo/croot/pluggy_1733170145382/work
protobuf==6.33.2
psutil==7.2.1
pybase64==1.4.3
pycosat @ file:///C:/b/abs_18nblzzn70/croot/pycosat_1736868434419/work
pycparser @ file:///C:/miniconda3/conda-bld/pycparser_1757496153123/work
pydantic @ file:///C:/miniconda3/conda-bld/pydantic_1764083582233/work
pydantic-settings @ file:///C:/miniconda3/conda-bld/pydantic-settings_1764165236385/work
pydantic_core @ file:///C:/miniconda3/conda-bld/pydantic-core_1764009799098/work
Pygments @ file:///C:/miniconda3/conda-bld/pygments_1762431428918/work
PyJWT @ file:///C:/miniconda3/conda-bld/pyjwt_1764332257117/work
pyparsing==3.3.1
pyreadline3==3.5.4
PySocks @ file:///C:/miniconda3/conda-bld/pysocks_1761753030965/work
python-dateutil==2.9.0.post0
python-dotenv @ file:///C:/b/abs_71cpoh9hpg/croot/python-dotenv_1745613639902/work
PyTurboJPEG==1.8.2
pywin32-ctypes @ file:///C:/Users/dev-admin/perseverance-python-buildout/croot/pywin32-ctypes_1729046491215/work
PyYAML==6.0.3
pyzbar==0.1.9
readchar @ file:///C:/miniconda3/conda-bld/readchar_1760613474723/work
requests @ file:///C:/miniconda3/conda-bld/requests_1762359611326/work
rich @ file:///C:/miniconda3/conda-bld/rich_1760375661587/work
ruamel.yaml @ file:///C:/miniconda3/conda-bld/ruamel.yaml_1762536064547/work
ruamel.yaml.clib @ file:///C:/miniconda3/conda-bld/ruamel.yaml.clib_1762530094515/work
semver @ file:///C:/miniconda3/conda-bld/semver_1761903323755/work
setuptools==80.9.0
shapely==2.1.2
shellingham @ file:///C:/miniconda3/conda-bld/shellingham_1761912227081/work
six==1.17.0
soupsieve==2.8.1
sympy==1.14.0
tomli @ file:///C:/b/abs_88e598m6o8/croot/tomli_1753774604115/work
torch==2.9.1+cpu
tqdm @ file:///C:/miniconda3/conda-bld/tqdm_1762863407729/work
truststore @ file:///C:/miniconda3/conda-bld/truststore_1762521027919/work
typer==0.20.0
typer-slim==0.20.0
typing-inspection @ file:///C:/miniconda3/conda-bld/typing-inspection_1760614188477/work
typing_extensions @ file:///C:/b/abs_ecq8gc0vbm/croot/typing_extensions_1756281142218/work
ujson==5.11.0
urllib3 @ file:///C:/miniconda3/conda-bld/urllib3_1765399243000/work
Werkzeug==3.1.4
wheel==0.45.1
win_inet_pton @ file:///C:/miniconda3/conda-bld/win_inet_pton_1761746278300/work
zstandard @ file:///C:/miniconda3/conda-bld/zstandard_1758189089298/work
```

</details>

---

## 6. WeChat QR モデルファイル

`paper_pipeline_v4.py` は以下の 4 ファイルを必要とします。

- `models/wechat_qrcode/detect.prototxt`
- `models/wechat_qrcode/detect.caffemodel`
- `models/wechat_qrcode/sr.prototxt`
- `models/wechat_qrcode/sr.caffemodel`

本 PC のリポジトリに `APA/models/wechat_qrcode/` が存在することを確認してください。

---

## 7. VS Code

- インストールパス（CLI）:
  - `C:\Users\takumi\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd`
- バージョン:
  - `1.107.1`（commit: `994fd12f8d3a5aa16f17d42c041e5809167e845a`）
- アーキテクチャ:
  - `x64`

---

## 8. 実行手順（再現用）

### 8.1 推奨（確実）実行方法

PATH を汚していないため、Python は必ずフルパスで指定します。

```bat
cd C:\Users\takumi\develop\APA
C:\Users\takumi\develop\miniconda3\python.exe paper_pipeline_v4.py --explain
```

### 8.2 最小実行（動作確認用）

```bat
cd C:\Users\takumi\develop\APA
C:\Users\takumi\develop\miniconda3\python.exe paper_pipeline_v4.py --limit 1 --src-forms A,B,C --degrade-n 1 --device cpu
```

---

## 9. 本 PC のローカル開発ディレクトリ構成（方針）

この PC では「会社 PC のシステム環境を汚さず、削除容易」を優先し、開発関連は `develop` 配下に集約します。

```
C:\Users\takumi\develop\
├─ APA\                 # 対象リポジトリ
├─ git\                 # Portable Git（PATH未登録）
├─ miniconda3\          # Miniconda（PATH未登録）
└─ work\                # 作業用（必要に応じて）
```

---

## 10. 別 PC へ展開するときのチェックリスト

1. OS: Windows 11 / x64
2. Git/Miniconda を **PATH 未登録のローカル配置**で導入できているか
3. `C:\Users\...\develop\miniconda3\python.exe -V` で想定 Python が動くか
4. `opencv_wechat_available: True` になっているか
5. `models/wechat_qrcode/` の 4 ファイルが存在するか
6. `paper_pipeline_v4.py --limit 1 ...` が完走するか

---

## 11. 採取コマンド例（社用 PC でも同様に記録する）

```bat
:: OS / ハード概要
ver
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Manufacturer" /C:"System Model" /C:"System Type" /C:"Processor(s)" /C:"Total Physical Memory"

:: 詳細（PowerShell）
powershell -NoProfile -Command "Get-CimInstance Win32_OperatingSystem | Select Caption,Version,BuildNumber,OSArchitecture,LastBootUpTime | Format-List"
powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select -First 1 Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | Format-List"

:: Portable Git
C:\Users\takumi\develop\git\bin\git.exe --version
C:\Users\takumi\develop\git\bin\git.exe config --global --list

:: Miniconda
C:\Users\takumi\develop\miniconda3\python.exe -V
C:\Users\takumi\develop\miniconda3\Scripts\conda.exe --version
C:\Users\takumi\develop\miniconda3\Scripts\pip.exe --version
C:\Users\takumi\develop\miniconda3\Scripts\conda.exe env list
C:\Users\takumi\develop\miniconda3\python.exe -m pip freeze

:: OpenCV WeChat
C:\Users\takumi\develop\miniconda3\python.exe -c "import cv2; print(cv2.__version__); print(hasattr(cv2,'wechat_qrcode_WeChatQRCode'))"
```
