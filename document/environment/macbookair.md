# MacBook Air 実行環境仕様書（APA / paper_pipeline_v4）

最終更新: 2026-01-09

このドキュメントは、`/Users/takumi0616/Develop/docker_miniconda/src/APA` 配下の **APA プログラム群**
（例: `paper_pipeline_v4.py`）を macOS（MacBook Air）上で **誰でも再現できる**ことを目的に、
OS/ハード/ツール/Python/依存関係/実行確認手順をまとめた「環境仕様書」です。

> 注意（重要）
>
> - 本リポジトリでは **conda(base)** と **プロジェクト専用 venv (`.venv`)** が混在し得ます。
> - 実行は **必ずこのリポジトリの `.venv` の Python** を使用してください。
>   - 例: `.venv/bin/python paper_pipeline_v4.py ...`

---

## 1. 対象リポジトリ

- リポジトリ: `APA`
- 作業ディレクトリ（本 Mac）:
  - `/Users/takumi0616/Develop/docker_miniconda/src/APA`

---

## 2. OS / ハードウェア（概要）

### 2.1 OS

- OS: macOS
- ProductVersion: `26.2`
- BuildVersion: `25C56`
- Darwin (uname -a):
  - `Darwin takumisMacBook-Air.local 25.2.0 Darwin Kernel Version 25.2.0: Tue Nov 18 21:09:34 PST 2025; root:xnu-12377.61.12~1/RELEASE_ARM64_T8112 arm64`
- アーキテクチャ: `arm64`
- Python 側で確認したプラットフォーム文字列:
  - `macOS-26.2-arm64-arm-64bit`

※ `sw_vers` や `system_profiler` の詳細（ビルド番号/メモリ容量など）は、
各社用 PC でも同様に採取し、本ドキュメントに追記してください（採取コマンド例は後述）。

### 2.2 ハードウェア

`system_profiler SPHardwareDataType` の出力（抜粋）:

- Model Name: `MacBook Air`
- Model Identifier: `Mac14,2`
- Model Number: `Z15T0004LJ/A`
- Chip: `Apple M2`
- Total Number of Cores: `8 (4 performance and 4 efficiency)`
- Memory: `24 GB`
- System Firmware Version: `13822.61.10`
- OS Loader Version: `13822.61.10`
- Activation Lock Status: `Enabled`

> セキュリティ注意
>
> - Serial Number / Hardware UUID / Provisioning UDID は個体識別情報のため、
>   **社内共有版の仕様書には原則マスク**して記載してください。
> - 本ドキュメントでは、再現性に必要な範囲として上記の非識別情報を優先しています。

---

## 3. シェル / CLI ツール

- シェル: zsh
- Git: 利用（torch.hub が内部で git を呼ぶ場合があるため必須になり得ます）

---

## 4. Python 実行環境（最重要）

### 4.1 使用 Python（プロジェクト venv）

- Python 実体:
  - `/Users/takumi0616/Develop/docker_miniconda/src/APA/.venv/bin/python`
- Python バージョン:
  - `Python 3.12.12`

### 4.2 conda(base) との混在について

ターミナル表示で `(base)` が付いていても、**実行に使う python を `.venv/bin/python` に固定**すれば問題ありません。

推奨:

```bash
cd /Users/takumi0616/Develop/docker_miniconda/src/APA
.venv/bin/python -c "import sys; print(sys.executable)"
```

---

## 5. 主要ライブラリと要件

### 5.1 主要依存（動作確認済み）

以下は本 Mac の `.venv` 上で import と実行が確認できた構成です（2026-01-09 時点）。

- OpenCV
  - `opencv-contrib-python==4.12.0.88`
  - `cv2.__version__ == 4.12.0`
  - `cv2.wechat_qrcode_WeChatQRCode` **利用可能**（WeChat QR 有効）
- NumPy
  - `numpy==2.2.6`
- PyTorch
  - `torch==2.9.1`
- Pillow
  - `Pillow==12.1.0`
- DocAligner 系
  - `docaligner` / `capybara`（※後述の互換パッチに依存）

#### 参考: pip freeze（本 Mac の `.venv` / 2026-01-09 採取）

再現性のため、pip の全インストール一覧（`pip freeze`）を付録に掲載します。
（社用 PC でも同様に採取して、このリストと差分を管理してください）

<details>
<summary>pip freeze 一覧（クリックで展開）</summary>

```txt
beautifulsoup4==4.14.3
blinker==1.9.0
bottlenose==1.1.8
capybara-docsaid==0.12.0
certifi==2026.1.4
charset-normalizer==3.4.4
click==8.3.1
colorama==0.4.6
colored==2.3.1
coloredlogs==15.0.1
contourpy==1.3.3
cycler==0.12.1
dacite==1.9.2
dill==0.4.0
docaligner_docsaid==1.1.0
filelock==3.20.2
Flask==3.1.2
flatbuffers==25.12.19
fonttools==4.61.1
fsspec==2025.12.0
humanfriendly==10.0
idna==3.11
itsdangerous==2.2.0
Jinja2==3.1.6
kiwisolver==1.4.9
kornia==0.8.2
kornia_rs==0.1.10
lxml==6.0.2
MarkupSafe==3.0.3
matplotlib==3.10.8
ml_dtypes==0.5.4
mpmath==1.3.0
natsort==8.4.0
networkx==3.6.1
numpy==2.2.6
onnx==1.20.0
onnxruntime==1.22.0
onnxslim==0.1.82
opencv-contrib-python==4.12.0.88
packaging==25.0
pdf2image==1.17.0
piexif==1.1.3
pillow==12.1.0
pillow_heif==1.1.1
protobuf==6.33.2
psutil==7.2.1
pybase64==1.4.3
pyparsing==3.3.1
python-amazon-simple-product-api==2.2.11
python-dateutil==2.9.0.post0
PyTurboJPEG==1.8.2
PyYAML==6.0.3
requests==2.32.5
setuptools==80.9.0
shapely==2.1.2
six==1.17.0
soupsieve==2.8.1
sympy==1.14.0
torch==2.9.1
tqdm==4.67.1
typing_extensions==4.15.0
ujson==5.11.0
urllib3==2.6.3
Werkzeug==3.1.4
wheel==0.45.1
```

</details>

### 5.2 OpenCV の WeChat QR を使う理由

`paper_pipeline_v4.py` では、フォーム B の QR が小さい/低解像度の時に
OpenCV 標準の `QRCodeDetector` が失敗しやすいため、
WeChat エンジン（`wechat_qrcode_WeChatQRCode`）を優先的に使います。

そのため **opencv-contrib** ビルドが必須です。

### 5.3 WeChat QR モデルファイル

`paper_pipeline_v4.py` は以下の 4 ファイルを必要とします。

- `models/wechat_qrcode/detect.prototxt`
- `models/wechat_qrcode/detect.caffemodel`
- `models/wechat_qrcode/sr.prototxt`
- `models/wechat_qrcode/sr.caffemodel`

本 Mac では上記ファイルの存在を確認し、初期化・空画像での `detect()` 呼び出しが成功しています。

---

## 6. 互換性メモ（DocAligner / capybara）

### 6.1 `docaligner` を単体 import すると落ちる場合

この環境では以下の症状がありました。

- `.venv` で `import docaligner` を直接実行すると
  `AttributeError: module 'capybara' has no attribute 'get_curdir'` が出ることがある

### 6.2 回避策（paper_pipeline_v4 側で実装済み）

`paper_pipeline_v4.py` には **`patch_capybara_exports()`** があり、
`docaligner` が期待するシンボルを `capybara` に生やすことで互換性を確保しています。

以下が **OK** なら DocAligner は使用可能です:

```bash
cd /Users/takumi0616/Develop/docker_miniconda/src/APA
.venv/bin/python -c "import paper_pipeline_v4 as p; p.patch_capybara_exports(); from docaligner import DocAligner, ModelType; print('OK')"
```

---

## 7. XFeat / torch.hub（キャッシュ・ネットワーク）

### 7.1 使用箇所

`test_recovery_paper.py` の `XFeatMatcher` で以下を使用しています:

```python
torch.hub.load(
    "verlab/accelerated_features",
    "XFeat",
    pretrained=True,
    top_k=top_k,
)
```

### 7.2 初回実行時の注意

- 初回は torch.hub がモデルをダウンロードすることがあります
- 環境によっては `git` が必要です
- キャッシュは通常 `~/.cache/torch/hub/` に作られます

本 Mac では `Using cache found in ~/.cache/torch/hub/...` が出ており、キャッシュ利用状態です。

---

## 8. 実行手順（再現用）

### 8.1 venv の有効化（任意）

```bash
cd /Users/takumi0616/Develop/docker_miniconda/src/APA
source .venv/bin/activate
```

### 8.2 推奨（確実）実行方法

activate を省略しても確実に動かすため、常に `.venv/bin/python` を直接指定します。

```bash
cd /Users/takumi0616/Develop/docker_miniconda/src/APA
.venv/bin/python paper_pipeline_v4.py --degrade-n 3 --template-topn 0
```

### 8.3 安全な疎通確認（引数説明のみ）

```bash
cd /Users/takumi0616/Develop/docker_miniconda/src/APA
.venv/bin/python paper_pipeline_v4.py --explain
```

### 8.4 最小実行（動作確認用）

```bash
cd /Users/takumi0616/Develop/docker_miniconda/src/APA
.venv/bin/python paper_pipeline_v4.py --limit 1 --src-forms A,B,C --degrade-n 1 --template-topn 0 --device cpu
```

本 Mac での実行結果（要点）:

- DocAligner: 読み込み成功
- XFeat: 読み込み成功（torch hub cache 使用）
- WeChat QR detector: 初期化成功
- A: warp まで成功、テンプレ一致
- B: warp まで成功、テンプレ一致
- C: `form_unknown` で停止（期待動作として OK）

---

## 9. OpenCV（WeChat）導入手順（再現用）

WeChat QR を有効化するために、以下のように **opencv-python → opencv-contrib-python** に入れ替えます。

```bash
cd /Users/takumi0616/Develop/docker_miniconda/src/APA

.venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless || true
.venv/bin/python -m pip install -U --no-cache-dir "opencv-contrib-python==4.12.0.88"

.venv/bin/python -c "import cv2; print('opencv_wechat_available:', hasattr(cv2,'wechat_qrcode_WeChatQRCode'))"
```

`opencv_wechat_available: True` になれば完了です。

---

## 10. トラブルシュート

### 10.1 `zsh: parse error near ')'`

原因: ターミナルの **プロンプト文字列**（例: `((.venv) ) (base) ...`）まで一緒にコピペすると、
zsh が `((` や `)` を算術式として解釈し、構文エラーになります。

対策: **プロンプトを含めず、コマンド行だけ**を実行してください。

### 10.2 `zsh: command not found: #`

原因: `# コメント行` まで含めて貼り付けると、状況により `#` をコマンドとして解釈してしまうことがあります。

対策: コメント行は貼らず、コマンドだけ実行。

### 10.3 `docaligner import` が落ちる

上記 6.2 の `patch_capybara_exports()` を経由すること。
（`paper_pipeline_v4.py` は内部で実施する設計です）

---

## 11. 別 PC（社用 PC）へ展開するときのチェックリスト

1. OS/CPU（x86_64 or arm64）
2. Python バージョン（推奨: 3.12 系で統一）
3. `.venv` の作成手順（venv + pip）
4. `opencv-contrib-python` が入り `opencv_wechat_available=True`
5. `models/wechat_qrcode/` の 4 ファイルが存在
6. `docaligner` が `patch_capybara_exports()` 後に import できる
7. XFeat が torch.hub でロードできる（ネットワーク/キャッシュ/ git）
8. `paper_pipeline_v4.py --limit 1 ...` が完走する

---

## 12. 採取コマンド例（社用 PC でも同様に記録する）

環境を仕様書として残すため、以下の出力を貼り付ける運用を推奨します。

```bash
# OS
sw_vers
uname -a

# ハード
system_profiler SPHardwareDataType

# Python / pip
cd /Users/takumi0616/Develop/docker_miniconda/src/APA
.venv/bin/python -V
.venv/bin/python -m pip -V
.venv/bin/python -m pip freeze

# OpenCV wechat
.venv/bin/python -c "import cv2; print(cv2.__version__); print(hasattr(cv2,'wechat_qrcode_WeChatQRCode'))"
```

> メモ
>
> - 上記コマンドは **コメント行（# ...）を含めず**に実行すると安全です。
> - `cd /path/to/APA` は「自分の環境の APA ディレクトリ」に置き換えてください。
