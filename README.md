# compare_lora_epochs

[English](README.md)

**LoRAの最適なエポックを選ぶ**ためのコマンドラインツールです。
起動中の [ComfyUI](https://github.com/comfyanonymous/ComfyUI) に対して、LoRAファイル（と強度）ごとに
同じプロンプト・同じシードで画像を生成し、結果を比較表の画像にまとめます。
UIでLoRAを1つずつ選び直す必要がありません。

- 標準ライブラリのみで動作（比較表の作成にだけ Pillow を使用。なくても個別画像は生成されます）
- ComfyUIのHTTP APIで通信するので、スクリプトはどこに置いても動きます
- 行がLoRA×強度、列がテストプロンプト。先頭行はLoRAなしの比較用です
- SDXL系のチェックポイント向けです（既定のサイズやサンプラー設定がSDXL想定です）

## 必要なもの

- Python 3.8以上
- 起動中のComfyUI（既定 `127.0.0.1:8188`）
- ComfyUIの `models/loras` フォルダに入れたLoRAファイル（サブフォルダ可）
- 任意: 比較表用に `pip install pillow`

## クイックスタート

```bash
# 1. ComfyUIから見えるチェックポイントとLoRAを確認
python compare_lora_epochs.py --list-ckpts
python compare_lora_epochs.py --list

# 2. 実行内容の確認（生成はしません）
python compare_lora_epochs.py --ckpt YOUR_MODEL.safetensors --match my_lora --dry-run

# 3. 実行
python compare_lora_epochs.py --ckpt YOUR_MODEL.safetensors --match my_lora --trigger mychar
```

名前は `--list-ckpts` / `--list` で表示されたとおりに指定してください。`--match` は、名前に指定した文字列を含む
LoRAだけを対象にします（フォルダ名や学習名など）。省略すると、すべてのLoRAが対象です。

## 出力

- 個別画像: ComfyUIの出力フォルダ内 `lora_compare/`（`--save-prefix` で変更可）
- 比較表: スクリプトを実行したフォルダに `compare_<日時>_1.png`、`_2.png`…
  各シートの先頭行がLoRAなし、その下に最大 `--per-sheet` 行のLoRAが並びます。

## おすすめの使い方

1. **荒く全体を見る**: 全エポック・1つの強度・トリガーワードだけで生成（LoRA自体が何を覚えたかが分かります）
   ```bash
   python compare_lora_epochs.py --ckpt MODEL --match my_lora --trigger mychar --strengths 0.8
   ```
2. **候補を数本に絞り**、強度を変えて比べる
   ```bash
   python compare_lora_epochs.py --ckpt MODEL --match 000005 000007 000009 --strengths 0.6 0.8 1.0 --trigger mychar
   ```
3. **自由度を確認する**: 自分のプロンプト（服・背景・ポーズ違い）で試す
   ```bash
   python compare_lora_epochs.py --ckpt MODEL --match 000007 --trigger mychar --prompts-file example_prompts.txt
   ```

見るポイントは、キャラや画風が出ているか、プロンプトに従うか、構図が学習画像に固まっていないか（固まっていれば
過学習の兆候なので、手前のエポックを試します）の3つです。

## オプション

| オプション | 既定値 | 説明 |
|---|---|---|
| `--server` | `127.0.0.1:8188` | ComfyUIのアドレス |
| `--list`, `--list-ckpts` | | LoRA / チェックポイントの一覧を表示して終了 |
| `--ckpt` | （必須） | `--list-ckpts` で表示された名前 |
| `--match A B ...` | すべて | いずれかの文字列を名前に含むLoRAだけを対象にする |
| `--strengths` | `0.8` | LoRA強度（複数可。モデルとCLIPの両方に適用） |
| `--trigger` | 空 | LoRA行に入れるトリガーワード |
| `--tags` | 空 | LoRA行に毎回足すタグ（トリガーに焼き付けていない服のタグなど） |
| `--baseline-tags` | 空 | LoRAなし行に使うタグ |
| `--no-baseline` | | LoRAなし行を作らない |
| `--quality` | `masterpiece, best quality` | 先頭に入れる品質タグ |
| `--negative` | 汎用セット | ネガティブプロンプト |
| `--prompts-file` | 内蔵3種 | 1行1プロンプト（`tags \| 幅x高さ` でサイズ指定可） |
| `--seed` | `12345` | 全画像で共通の固定シード |
| `--steps`, `--cfg` | `30`, `5.5` | サンプラー設定 |
| `--sampler`, `--scheduler` | `euler_ancestral`, `normal` | サンプラー設定 |
| `--clip-skip` | `2` | CLIP skip |
| `--per-sheet` | `5` | 1枚の比較表に載せるLoRA行数 |
| `--thumb` | `448` | 比較表のサムネイル幅 |
| `--save-prefix` | `lora_compare` | ComfyUI側の保存先サブフォルダ / ファイル名の接頭辞 |
| `--timeout` | `3600` | 生成完了を待つ秒数 |
| `--dry-run` | | 実行内容の表示のみ |

## トラブルシューティング

- **接続できない**: 先にComfyUIを起動してください。ポートが8188以外なら `--server ホスト:ポート` を指定します。
- **LoRAが見つからない**: `--list` で表示された名前の一部を正確に指定します。Windowsではサブフォルダ内の名前に
  バックスラッシュが含まれることがありますが、名前の一部での絞り込みは動作します。
- **`--ckpt` が見つからない / HTTP 400**: `--list-ckpts` の表示どおりの文字列を指定します。
- **比較表ができず個別画像だけ**: Pillowを入れます（`pip install pillow`）。
- **比較表が大きすぎる**: `--thumb` や `--per-sheet` を小さくします。
- **どのエポックも同じに見える**: `--strengths 1.0` を試す、`--tags` を空にしてLoRAに頑張らせる、
  離れたエポック同士を比べる、のいずれかを試してください。

内部で組むワークフローは1種類だけです（チェックポイント → LoRA → CLIP skip → KSampler → VAE Decode → 保存）。
LoRA単体の効きを見るため、IP-Adapter・ControlNet・アップスケーラーは意図的に入れていません。

## ライセンス

公開前に、お好みのライセンス（例: MIT）を追加してください。
