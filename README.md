# ペルソナエージェント with Qwen2-VL

Qwen2-VL-2BをベースにLoRA finetuningした、**ペルソナの価値観を形式化するモジュール**です。

1000体のエージェントをリアルタイムで動かす軽量ポリシーの「教師・審判・解釈者」として機能します。

---

## このモデルの本質的な役割

直接1000体を動かすことはしません。このモデルが担うのは**「なぜそうするか」を知っていること**です。

```
軽量ポリシー  →  「何をするか」を高速に決める（1000体をリアルタイム制御）
このモデル    →  「なぜそうするか」を知っている（価値観の基準点）
```

### 3つの顔

**① 報酬関数**　ポリシー学習（PPO）に「ペルソナらしさ」の報酬を与える

**② データフィルター**　学習データの品質をスコアリングし、ノイズを除去する

**③ 解釈器**　軽量ポリシーの行動を人間が読める理由テキストに変換する

---

## アーキテクチャ

```
マップ画像 (256x256 PNG)
    ↓
Vision Encoder（Qwen2-VL内蔵ViT）
    ↓
状態トークン列 + ペルソナ指示テキスト
    ↓
LLM Transformer（Qwen2-VL-2B + LoRA）
    ↓
行動トークン列 + 理由テキスト
  <action_move> <x_0612> <y_0389> 「未探索エリアへの強い引力を感じた」
```

### トークン設計

座標値は0〜1の正規化値を1000分割した離散トークンで表現します（トークン化回帰）。

| 種別 | 例 | 意味 |
|---|---|---|
| 座標 | `<x_0512> <y_0234>` | 正規化座標 (0.512, 0.234) |
| ペルソナ | `<persona_explorer>` | 探索者ペルソナ |
| 行動 | `<action_move>` | 指定座標へ移動 |
| 構造 | `<state_start>` `<action_end>` | 入出力の区切り |

---

## ペルソナ定義

| ペルソナ | トークン | 行動哲学 |
|---|---|---|
| Explorer Rex | `<persona_explorer>` | 未知への好奇心が恐怖に勝る。未訪問エリアを常に優先する |
| Homebody Lily | `<persona_homebody>` | 安心感を最優先する。既知の安全地帯に留まりたがる |
| Social Marco | `<persona_social>` | 他者との接触が行動の原動力。常に誰かに近づこうとする |
| Businessman Cole | `<persona_businessman>` | 効率が正義。目標への最短経路しか眼中にない |

同じマップ・同じ位置でも、ペルソナによって出力が異なります。

```
Rex  → <action_move> <x_0800> <y_0200> 「北東の暗いエリアに引き寄せられる」
Lily → <action_rest>                    「ここは安全。動く理由がない」
Marco→ <action_move> <x_0200> <y_0700> 「Coleがあちらにいる。合流しよう」
Cole → <action_move> <x_0900> <y_0900> 「目的地への最短経路を取る」
```

**理由テキストがセットで出力される**ことが、このモデルの最大の特徴です。

---

## 学習データ

### 1サンプルの構造

**入力①: マップ画像（PNG）**

```
256x256ピクセル
  薄ベージュ: 通路
  黒        : 壁・障害物
  薄青      : 既訪問エリア
  赤い丸    : エージェントの現在位置
```

**入力②: 状況テキスト**

```
<state_start>
<time_morning>
<self>        <x_0300> <y_0500>
<agent_Lily>  <x_0700> <y_0300>
<agent_Marco> <x_0200> <y_0800>
<state_end>
```

**正解ラベル（出力）**

```
<action_start> <action_move> <x_0612> <y_0389> 未探索の北東エリアへ向かう <action_end>
```

### データ品質のポイント

**同じ状況で複数ペルソナの行動を全部記録する**ことが最重要です。対比が明確なサンプルが多いほど、モデルが「ペルソナごとの価値観の差」を学習できます。

### データ収集の3パターン

**パターンA: ルールベース自動生成**　コストゼロ。動作確認・初期実験向き。

**パターンB: 人間のデモ操作ログ（推奨）**　実際に人間がエージェントを操作したログを録ります。1ペルソナあたり50〜100デモから学習が始まります。ペルソナらしさの品質が最も高くなります。

**パターンC: 既存ゲームのプレイログ変換**　RPGや戦略ゲームのリプレイデータを変換します。スクリーンショットとプレイヤー座標ログがあればそのまま使えます。


### 必要データ量の目安

| 目的 | サンプル数 | T4での学習時間 |
|---|---|---|
| 動作確認 | 200〜500 | 約30分 |
| ペルソナの差が出る | 1,000〜2,000 | 2〜3時間 |
| 報酬モデルとして使える | 5,000〜10,000 | 半日〜1日 |

---

## finetuning手順

### 環境

- Google Colab T4（VRAM 15GB）
- Python 3.10+、transformers 4.45.0、peft、bitsandbytes

### インストール

```bash
pip install transformers==4.45.0 peft accelerate qwen-vl-utils bitsandbytes
```

### モデルロード（4bit量子化・T4対応）

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
```

### LoRA設定（学習パラメータ：全体の約0.45%）

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
```

### 学習実行

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='/tmp/persona_agent',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    optim='paged_adamw_8bit',
)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=dataset, data_collator=collate_fn)
trainer.train()
```

### 保存・ダウンロード

```python
model.save_pretrained('/tmp/persona_agent_lora')
processor.tokenizer.save_pretrained('/tmp/persona_agent_lora')

# Colabからローカルにダウンロード
import shutil
from google.colab import files
shutil.make_archive('/tmp/persona_agent_lora', 'zip', '/tmp/persona_agent_lora')
files.download('/tmp/persona_agent_lora.zip')
```

---

## HuggingFaceへのアップロード

```python
from huggingface_hub import login
from google.colab import userdata

login(token=userdata.get('HF_TOKEN'))  # ColabのSecretsにHF_TOKENを設定

# LoRAアダプターのみアップロード（数十MB）
model.push_to_hub('oggata/persona-agent-qwen2vl-lora')
processor.push_to_hub('oggata/persona-agent-qwen2vl-lora')
```

使用時：

```python
from peft import PeftModel
base = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
model = PeftModel.from_pretrained(base, 'oggata/persona-agent-qwen2vl-lora')
```

---

## 使い方：3つの用途

### 用途1：ポリシー学習の報酬関数

PPOでポリシーを学習するとき、「ペルソナらしさ」の報酬をこのモデルが判定します。人手で報酬関数を設計する必要がなくなります。

```python
def persona_reward(state_image, action, persona_id):
    prompt = f"ペルソナ({persona_id})にとってこの行動は適切か。0〜1のスコアで答えよ"
    score = lora_model.generate(state_image, prompt)
    return float(score)

# PPOの報酬として利用
reward = persona_reward(current_state, taken_action, 'explorer')
```

### 用途2：学習データの品質フィルター

収集したデータをスコアリングして、ノイズを除去します。1000体のポリシー学習に使うデータの品質を担保します。

```python
clean_dataset = []
for sample in raw_dataset:
    score = lora_model.score_consistency(
        persona=sample['persona'],
        action=sample['action'],
        reason=sample['reason'],
    )
    if score > 0.8:
        clean_dataset.append(sample)
```

### 用途3：軽量ポリシーの解釈器

1000体を制御する軽量ポリシーが出した行動を、人間が読める理由テキストに変換します。デバッグ・デモ・コンテンツ生成に使えます。

```python
# 軽量ポリシーが出した座標
action_coord = fast_policy(state, persona_id='rex')  # (0.80, 0.20)

# このモデルが理由を生成
explanation = lora_model.explain(state_image, action=action_coord, persona='explorer')
# → 「未探索エリアへの強い引力を感じた。Rexらしい判断。」
```

---

## 1000体スケールへの全体パイプライン

```
フェーズ1: このモデルで高品質な行動データを大量生成
  Qwen2-VL + LoRA → (状態, 行動, 理由) × 数万件

フェーズ2: 軽量ポリシーをBehavior Cloningで初期学習
  入力: 状態ベクトル + Persona Embedding
  出力: 行動座標
  ※ペルソナはEmbeddingベクトルで表現するためモデルは1つ

フェーズ3: PPOでさらに最適化
  報酬関数 = このLoRAモデルが判定

フェーズ4: 1000体をリアルタイム制御
  軽量ポリシー（MLP/小型Transformer）
  推論 <1ms/体、1000体同時実行が現実的
```

### 軽量ポリシーの構造

```python
class PersonaPolicy(nn.Module):
    def __init__(self, n_personas=1000):
        super().__init__()
        self.persona_embed = nn.Embedding(n_personas, 32)
        self.net = nn.Sequential(
            nn.Linear(64 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # (x, y) 出力
        )

    def forward(self, state, persona_id):
        p = self.persona_embed(persona_id)
        return self.net(torch.cat([state, p], dim=-1))
```

ペルソナをEmbeddingで表現するため、**1000体でもモデルは1つ**です。

---

## データフライホイールとの接続

```
このLoRAモデルが生成した行動データ
        ↓
   軽量ポリシー学習
        ↓
  1000体がシミュレーション上で行動
        ↓
  良い行動結果をフィードバック
        ↓
  LoRAのfinetuningデータに追加
        ↓
  より高品質な教師データが生成される  ← ループ
```

このLoRAモデルは**フライホイールの起点**です。最初の高品質データがなければポリシー学習が始まらず、フライホイールも回りません。

---

## ファイル構成

```
persona_agent_qwen2vl.ipynb   # Colabノートブック（全10セル）
README.md                     # このファイル
```

## ライセンス

ベースモデルのQwen2-VL-2BはApache 2.0ライセンスです。商用利用可能です。
