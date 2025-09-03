einsumはONNXのOpset version12以降で使用可能  
drp-ai-translatorは  
Onnx version 1.12.0  
Opset version 13   
のため使用可能。

# compressai-gdn-tools

CompressAI モデルの **エンコーダ内 GDN を置換**し、**ONNX へエクスポート**するためのツール群。  
重複コードを排し、`gdn_variants`（GDNの実装）、`replacer`（置換ロジック）、`model_io`（入出力）、`wrappers`、`onnx_utils` に分割して保守性を高めています。

---

## 特徴

- 置換対象: `model.g_a`（Analysis/Encoder）配下の **GDN（inverse=False）**
- 置換方式:
  - `einsum`：**等価置換**（1×1 Conv → MatMul/einsum）。通常**再学習不要**
  - `diag`：**対角近似**（軽量）
  - `lowrank`：**低ランク近似**（`--rank r` で調整）
- ONNX エクスポート（デフォルト **エンコーダのみ**）＋ **ONNX Runtime** による数値突き合わせ検証
- すべて CLI から実行可能

---

## ディレクトリ構成

```
compressai-gdn-tools/
├─ README.md
├─ requirements.txt
├─ pyproject.toml                # （任意）editable install 用
├─ compressai_gdn_tools/
│  ├─ __init__.py
│  ├─ gdn_variants.py            # einsum/diag/lowrank GDN 実装
│  ├─ replacer.py                # GDN 判定・置換（エンコーダ優先）
│  ├─ model_io.py                # ckpt 読み込み/保存、zoo モデル構築
│  ├─ wrappers.py                # Encoder/Full ラッパ（ONNX export 用）
│  ├─ onnx_utils.py              # ONNX エクスポート & ORT 検証
│  └─ verify.py                  # 置換前後の数値チェック（任意）
└─ cli/
   ├─ swap_gdn_in_encoder.py     # GDN 置換 → .pth 保存
   └─ export_onnx.py             # .pth → ONNX 出力（デフォルト: エンコーダ）
```

---

## 置換済みのonnx & pytorch
URL: https://drive.google.com/drive/folders/1yGw8BqEkoI445tGvdBaRUxGgEpjHUFyC?usp=drive_link

## セットアップ

```bash
git clone <this-repo> compressai-gdn-tools
cd compressai-gdn-tools

python -m venv .venv
source .venv/bin/activate  # Windowsは .venv\Scripts\activate

pip install -r requirements.txt
# （任意）開発用: pip install -e .
```

**要件（推奨）**
- Python 3.9+
- PyTorch 2.1+
- CompressAI 1.2+
- onnx 1.15+ / onnxruntime 1.18+

---

## 1) GDN 置換 → .pth 保存

### Zoo からモデルを構築して置換
```bash
python -m cli.swap_gdn_in_encoder \  
--arch bmshj2018_factorized --quality 8　--pretrained \
--approx einsum --out swapped_bmshj2018_q8_einsum.pth \
--check
```
- `--approx einsum` は **等価置換**（1×1 Conv→MatMul）。`--check` で置換前後の最大誤差を表示。

### 既存チェックポイントを読み込んで置換
```bash
python -m cli.swap_gdn_in_encoder   --ckpt original.pth --arch bmshj2018_factorized   --approx lowrank --rank 16   --out swapped_lowrank.pth
```
- `--arch` は ckpt に arch 情報が無い場合のヒントとして使用。

**主なオプション**
- `--approx {einsum,diag,lowrank}`：置換方式
- `--rank <int>`：`lowrank` のランク
- `--check`：等価置換時の簡易数値チェック
- `--device {cpu,cuda}`：実行デバイス

---

## 2) .pth → ONNX（デフォルト: エンコーダのみ）

```bash
python -m cli.export_onnx   --ckpt swapped_bmshj2018_q8_einsum.pth   --out swapped_bmshj2018_q8_einsum_only_enc.onnx   --height 224 --width 224 --batch 1
```
- 出力は潜在 `y`（エンコーダサブグラフ）。出力名は `y`。

### フルモデル（再構成 `x_hat`）を出力
```bash
python -m cli.export_onnx   --ckpt swapped_bmshj2018_q8_einsum.pth   --out full.onnx   --full
```
- `--no-dynamic` を付けると **固定形状 ONNX** を生成。

**主なオプション**
- `--opset 17`：ONNX opset（デフォルト 17）
- `--full`：フルモデルで `x_hat` を出力（辞書→Tensor に正規化済）
- `--skip-verify`：ONNX Runtime 検証をスキップ

---

# Update: `approx=matmul` が選べるようになりました

- 置換方式 `--approx` の選択肢に **`matmul`** を追加しました（`einsum` と数値的に等価／ほぼ等価）。
- `gdn_variants.py` に `class MatmulGDN` を追加しました（`torch.matmul` を用いた実装）。
- `replacer.py` の `make_replacement` / `replace_gdn_in_encoder` が `matmul` を受け付けます。
- `cli/swap_gdn_in_encoder.py` の `--approx` choices に `matmul` を追加しました。

最短実行例（Encoder 内 GDN を置換 → .pth 保存）:

```bash
python -m cli.swap_gdn_in_encoder \
  --arch bmshj2018_factorized --quality 8 --pretrained \  --approx matmul --out swapped_bmshj2018_q8_matmul.pth --check
```

> 注: `matmul` は `einsum` と同様に 1×1 Conv を等価に置換する実装です。各ランタイム/エクスポート環境で
> `einsum` より `matmul` が好まれる場合に選択してください。

---

## ワークフロー例（最短）

1. 置換して保存：
   ```bash
   python cli/swap_gdn_in_encoder.py --arch bmshj2018_hyperprior --quality 6 --pretrained      --approx einsum --out swapped_hp_q6_einsum.pth --check
   ```
2. エンコーダのみ ONNX：
   ```bash
   python cli/export_onnx.py --ckpt swapped_hp_q6_einsum.pth --out encoder.onnx
   ```
3. ORT 検証が `max|diff| ≈ 1e-5` 程度なら OK。

---

## よくある質問（FAQ）

- **Q. 等価置換（einsum）でも誤差が出ますか？**  
  A. 浮動小数の丸め差で `~1e-6〜1e-5` 程度の差は出る場合があります。学習再開は不要です。

- **Q. g_a が見つからないとエラーになります**  
  A. 一部モデルはエンコーダ名が異なる可能性があります。`replacer.py` の探索ルートを拡張してください。

- **Q. lowrank/diag で性能が落ちます**  
  A. 教師=元モデル、生徒=近似 GDN モデルで **GDN 出力の特徴蒸留**（L2/Huber）を 1〜3万 iter ほど行うと回復しやすいです。

---

## ライセンス

このリポジトリのサンプルコードは研究・検証目的の MIT ライセンスを想定しています。  
CompressAI 本体はそれぞれのライセンスに従ってください。
