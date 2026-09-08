# Context Economy — 検証用ライブラリ v0.1

回答に必要なデータを保ちながら、モデルへ渡すトークンを減らすための試作。特定のAPI・コーディングエージェント・MCPに依存しません。Python 3.11+。

**初版は、全件保持する整形と、呼び出し側が明示した正確な操作を実装しています。実LLMの回答品質維持や、Headroomに対する総合的優位性はまだ実証していません。**

## 標準の使い方：アダプター不要

入力の準備だけを行います。モデル呼出し・自動再取得・自動再試行はしません。モデルやサブスク、APIの接続方法を変更せず、返った `text` を既存の送信先へ渡せます。出力・推論の長さの変動は許容し、その制御を標準機能の必須条件にはしていません。

```python
from context_economy import Economy, Store, TiktokenCounter, Router, prepare_input

router = Router(Economy(Store('originals.sqlite'), TiktokenCounter()))
prepared = prepare_input(
    router, inventory_json,
    'SKU-103に一致する全レコードを返して',
    kind='json', operation='select',
    arguments={'field': '/sku', 'equals_json': '"SKU-103"'},
)
# prepared.text を、普段使っているモデルの入力にそのまま渡す。
print(prepared.text)
print(prepared.saved_tokens)
```

検索等を指定しなければ、完全な情報を保つ圧縮だけを行います。質問から操作を自動推測しません。`prepared.input_tokens <= prepared.original_input_tokens` を、指定したトークナイザーで計数した最終文字列について確認します。質問・JSONエスケープ・説明を含めて増える候補は原文へ戻します。削減ゼロなら原文を優先します。

既存のプロンプト形式を使う場合は `render=lambda context: instructions + '\n' + question + '\n' + context` を渡せます。renderは同じ入力に同じ文字列を返す関数とし、質問もその中へ含めてください。追加の履歴等は計数後に付け足さず、このrenderへ含めます。既定のrenderはquestionとcontextを持つJSON文字列です。

この比較は選んだトークナイザーでの可視入力に限ります。接続先の隠れた指示・課金量・モデル出力の非増加を保証するものではありません。別のトークナイザーを使う接続先には適切な `count` を渡します。

CLIもモデル接続なしで使えます。

```powershell
python -m context_economy --db originals.sqlite prepare inventory.json --kind json --question '在庫の全体を確認して'
```

stdoutは計数済みの入力文字列そのものです。`--operation` と `--arguments` で明示的な操作を指定可能。`--metadata` を付けると統計付きJSONを返すので、モデルへ渡すのはその `text` フィールドだけにしてください。

任意の `budget` を超えた場合、Pythonは情報を保持して `within_budget=False` を返します。CLIの通常出力は終了コード2・stdout空で止まり、メタデータ出力なら全文と超過状態を返します。予算を満たすための抜粋・追加往復は行いません。

通常利用に `plan_call` や `SingleCallGate` は不要です。これらは接続先ごとの厳密な総量制限を必要とする場合だけ使う任意機能です。[標準入口の検証](results/preparation/RESULTS.md)。

## 実装したもの

| 操作 | 使う場面 | 品質を守る条件 |
|---|---|---|
| `pack` | 内容全体を読ませたい | 全行・全要素を保持。JSONは数値の元表記・型・キー順・欠損/nullを保持し、復号を検証 |
| `aggregate` | 合計・件数・最小・最大が必要 | 全件を計算。小数はDecimal、欠損や文字列を黙って除外しない |
| `select` | IDなど特定の値を探す | 全件に対する指定フィールドの等値検索。全一致行を返す |
| `focus` | 実験専用の部分抜粋 | 標準のprepare_input/Routerは採用しない。単独利用は非増加の保護対象外 |
| `read` / `retrieve` | 低水準の明示取得 | 標準入口は自動呼出ししない。原文は永続保存、毎回checksum検証 |

`pack` は元データ・JSON空白除去・列名共有・同一行の回数表現から、説明文まで含めたトークン数が最小の候補を選びます。数値配列・ファイルパス配列を間引きません。コードや `byte_exact=True` は文字列を変更しません。

予算が小さすぎても勝手にデータを捨てず、`within_budget=False` を返します。利用側は予算を増やすか、明示的な検索・集計の契約を指定します。低水準の探索語はliteralの大文字小文字非区別一致であり、意味検索や依存関係解析ではありません。

## 起動

```powershell
cd I:\Workspace\context-economy
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[tokens,test]"
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python benchmarks/measure.py --output results
```

このワークスペースでは既存の検証環境でも実行済みです。

```powershell
$env:TIKTOKEN_CACHE_DIR = 'I:/Workspace/headroom-audit-results/tiktoken'
& 'C:/Users/mxg01/AppData/Local/Temp/headroom-audit-20260908/env/Scripts/python.exe' benchmarks/measure.py --output results
```

初回のtiktokenは語彙データをダウンロードする場合があります。本体は標準ライブラリのみ。任意のモデル用トークナイザーを `Callable[[str], int]` として差し替え可能です。文字数÷4などへの暗黙の推定fallbackはありません。

## Python API

```python
from context_economy import Economy, Store, TiktokenCounter

eco = Economy(Store(".context-economy/originals.sqlite"), TiktokenCounter())

# 全体を渡す場合：元の数値500件を勝手に15件にはしない
import json
raw = json.dumps(list(range(500)))
view = eco.pack(raw, kind="json", budget=2000)
print(view.text)                   # モデルへ渡す本文
print(view.tokens, view.coverage)  # 計数と完全性はアプリ側でも確認可能
assert eco.retrieve(view.original_id) == raw

# 合計だけ必要な場合：LLMへ全数値を送る前に、全件を正確に計算
answer = eco.aggregate(view.original_id, op="sum")
print(answer.text)                 # result:124750、source_items:500、操作範囲・原文ID

# ネストしたJSON: /events の配列から /amount フィールドを合計
# eco.aggregate(ref, path="/events", field="/amount", op="sum")

# 探索は明示的に行う。原文を保存してから参照を発行
log_id = eco.store.put(large_log_text)
excerpt = eco.focus(log_id, terms=["FATAL", "inventory"], budget=600, radius=3)
print(excerpt.text)                # 部分取得だと明示される
more = eco.read(log_id, start=100, limit=50)
```

`large_log_text` は利用側で用意するログ文字列です。保存済みDBを `Store` で開き直せば、プロセス再起動後も原文を取得できます。TTLや自動削除は実装していません。ローカルの同一信頼境界で使うストアであり、マルチテナント認可サービスではありません。

`pack` の保存・検証に失敗した場合は元の全文と `reason` を返します。存在しない参照・checksum不一致・不正な問い合わせは明示的な例外です。`View.restore_semantic()` は整形前のJSONの意味内容または全文を復元します。元の空白・改行を含む完全な原文が必要なら `retrieve` を使います。

## CLI

```powershell
python -m context_economy --db demo.sqlite pack data.json --kind json --budget 2000
python -m context_economy --db demo.sqlite aggregate 'sha256:...' sum
python -m context_economy --db demo.sqlite read 'sha256:...' --unit items --start 0 --limit 100
python -m context_economy --db demo.sqlite focus 'sha256:...' --term FATAL --budget 600
python -m context_economy --db demo.sqlite retrieve 'sha256:...'
```

`pack`等のCLIはアプリ向けJSON envelopeを出力します。トークン数は `.text` の計数であり、メタデータまで丸ごとモデルへ送る場合はenvelope全体を別途計数してください。`retrieve`だけはUTF-8原文をそのまま出力します。

## 実測と品質の読み方

[結果](results/BENCHMARK.md)と[機械可読ログ](results/benchmark.json)を同梱しています。固定seedで、全件復号と正確な操作を検証しました。

- 辞書型JSONは14,403→4,543（68.46%削減）。単純な空白除去だけでは8,703トークンなので、それを基準にすると47.80%削減。
- 数値500件の正確な合計は、元1,500→結果75（95%削減）。これは同じ配列全体の表現ではなく、指定された問いへの全件計算結果です。
- ファイルパス全件やコードは0%削減。反復のない全文に0%と出ることも正常です。
- 抜粋後に全文を読み直す再生例では、総トークンが8.74%増えます。この不利なケースも計上しています。

反復ログ・日本語の高い削減率は、意図的に反復の多い入力に対するものです。一般的なログや日本語全体の削減率として扱わないでください。

全件の復元成功はモデルの理解力・回答精度の測定ではありません。表形式や回数表現をモデルが正しく読むか、必要時に原文取得を選べるかは次のpaired評価で確認します。

## 実モデルの評価を接続する

`context_economy.evaluation` はプロバイダー非依存です。モデルrunnerは `Task(question, context, tools)` を実行し、`ModelRun(output, calls)` を返します。`Task.id`も渡されます。toolsはこの課題の原文に束縛したretrieve/read/aggregate/selectです。read/aggregate/selectの戻り値はモデルへ渡せる文字列です。

```python
from context_economy.evaluation import ModelRun, CallUsage

def run(task):
    # ここで任意のモデルにquestion/contextと必要なtool schemaを渡す。
    # ツール呼び出しにはtask.toolsを使い、全呼出しusageを蓄積する。
    # 以下の値は実APIのusageから取得する（推定値で置き換えない）。
    return ModelRun(output=answer_json, calls=tuple(all_call_usage))
```

上記はアダプターの契約例です。`answer_json` と `all_call_usage` は利用側のモデル実行結果であり、ダミーの実測値は付けていません。

```powershell
python examples/evaluate_model.py --runner my_adapter:run --repetitions 3 --output results/model.json
```

評価は同じ課題・質問をpaired比較し、順序をseed付きで入れ替えます。部分一致・「それっぽい」回答ではなくJSON全体の厳密比較です。余計なキー、欠落要素、丸めを許しません。再取得・リトライ・モデルのreasoningを含む**全API呼出し**のusageをadapterが返してください。失敗してusageが不明なら0トークンとして節約に加えず、全体の節約率を未確定にします。

キャッシュread/writeと通常input/outputを分け、`CallUsage.cost()` に利用者が与えた実際の単価で費用を計算できます。cacheの初期値0はadapterが確認したうえで使用する必要があります。キャッシュは計算・費用の節約で、コンテキストの情報削減とは別です。既存履歴の表現を後から変更せず、準備した `view.text` をそのまま再利用してprefixを保持します。本ライブラリはAPI設定や履歴を勝手に変更しません。

少数例の成功は非劣性の証明ではありません。初版の4課題はアダプター確認用のsmokeセットです。本評価にはコード修正・長い依存関係・否定・例外・複数箇所の統合・長期会話を含む実タスク、必要なサンプル数、固定した成功基準が必要です。

## Astraサブスクでの実測（2026-09-08）

API契約を使わず、既存ChatGPTログインのCodex CLIで6問を原文／最適化の2回にまとめて実行しました。両条件6/6正解、実入力トークン13,303→11,134（16.3%削減）。小規模な合成課題の結果で、実務品質の保証ではありません。共通指示の負担、失敗しやすい部分取得の未検証、利用枠の計測限界を含む詳細は [results/subscription/RESULTS.md](results/subscription/RESULTS.md)。再現用スクリプトは `examples/subscription_smoke.py` です。各条件を個別起動し、間に利用枠を確認します。既存試行への上書き・自動再試行は拒否します。

## 操作に応じた経路選択

```python
from context_economy import Economy, Store, TiktokenCounter, Router

router = Router(Economy(Store('originals.sqlite'), TiktokenCounter()))
decision = router.route(
    inventory_json, kind='json', operation='select',
    arguments={'field': '/sku', 'equals_json': '"SKU-103"'},
)
model_context = decision.view.text
```

`complete`（既定）は全情報を残し、`aggregate`・`select`・`lookup`は明示された範囲を全件処理した結果が、完全圧縮より小さい場合だけ採用します。`lookup`は `arguments={'path': '/packages/0/runtimeArguments'}` のようにJSON Pointerを指定します。短い集計の説明負担が大きければ原文を選び、保存できなければ原文に戻ります。指定した項目が欠けるなど正確な操作結果を作れない場合も完全な情報に戻り、`reason`に理由を残します。

自然言語の意図分類器ではありません。呼出し側が操作を指定し、その指定自体にLLMを使う場合は使用量を別途合算する必要があります。小さいトークン予算でも情報を勝手に欠落させません。

`explore` は安全方針を変更しました。Routerは、再取得確率を0と指定しても部分抜粋を採用せず、完全な情報に戻します。`ExplorationCost` は呼出し互換性のため残していますが、許可判定には使いません。低水準の `Economy.focus/read` と過去の探索実験スクリプトはこの保護を通らない実験用APIです。通常の節約処理はRouterを使ってください。

### 任意機能：接続先を含めた厳密な実行制御

以下は標準利用には不要です。対応アダプターが必要なのは、この任意機能でモデル出力・SDK再試行まで制限する場合だけです。

`plan_call` は最終的に送る質問・データ・説明を組み立て、同じ出力上限を使う原文案より入力上限が増えれば原文案に戻します。`max_output_tokens` を含め `total_limit` に収まらなければ送信前に `BudgetRefused`。情報を切って予算に合わせません。

`count_input_upper_bound` は接続先の隠れた指示・フレーミングを含む最終入力の確実な上限を返す必要があります。不明ならNoneを返し、実行を拒否します。単なる文字数やo200k_baseのpayload計数を、未知の接続先の確実な上限として使ってはいけません。

`SingleCallGate('attempts.sqlite').run(stable_request_id, plan, send)` は送信前に永続台帳へ記録してから `send(plan)` を1回だけ呼びます。同じIDでの二重実行・再起動後の再試行・タイムアウト後の再送を拒否。使用量不明や失敗でも自動で原文を再送しません。呼出し側は論理的に同じ仕事に同じIDを使い、台帳を保持してください。出力に問題があれば未完了として扱い、再開判断を呼出し側に返します。

**接続条件:** sendアダプターは物理呼出し1回、SDK自動retryとツール無効、推論込みの出力上限を接続先で強制する必要があります。ライブラリは外部SDK内部の再試行を制御できません。返却usageの超過は検出・記録しますが、既に使ったトークンを取り消せません。これらを満たせない接続先はこの実行口に接続しないでください。従来のAstra CLI実験アダプターはこの上限強制を確認しておらず、まだ接続していません。

保証の対象は、再取得ループを発生させない経路選択と、契約を満たすアダプターの送信上限です。実際の原文回答より必ず総トークンが少ないことや、モデルの品質を保証するものではありません。[変更と検証結果](results/no-spike/RESULTS.md)。

混合6問のAstra実測は全3条件で6/6正解。総トークンは原文18,994、従来の完全圧縮14,419、経路選択12,028。今回の経路選択で原文比36.7%、従来圧縮比16.6%減。効果の大部分は合成在庫検索で、実務全体への一般化はできません。[結果と範囲](results/routing/RESULTS.md)。

## 検証履歴

追加の原文取得テストでは、Astraが不足3資料だけを取得し、4/4正解を維持しました。一方で全文取得と追加往復により総トークンは14,565→26,876（84.5%増）。抜粋が常に節約になるわけではありません。接続方式による固定費の限界も含む詳細は [results/retrieval/RESULTS.md](results/retrieval/RESULTS.md)。

中間データのフィルタ・集計をローカルへ移す方向は[AnthropicのMCP実行設計](https://www.anthropic.com/engineering/code-execution-with-mcp)を参照。履歴の既存prefixを維持する方針は[OpenAIのキャッシュ仕様](https://developers.openai.com/api/docs/guides/prompt-caching)を確認しました。これらの公開削減率を本実装の成績として使っていません。

詳細な範囲・完了条件・次段階は [PLAN.md](PLAN.md) に記録しています。
