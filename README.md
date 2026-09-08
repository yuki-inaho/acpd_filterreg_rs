# acpd-filterreg 0.3.0 — 論文・添付実装照合による是正版（DoD達成・精度/速度改善）

2次元・3次元の剛体FilterRegと逐次合成Analytic-CPDを、独立したC++数値核／nanobind接続とRust数値核／PyO3接続で実装します。
**前回版の全対／半径探索による代用を廃止し、実際のpermutohedral格子を実装しました。**

## 0.3.0で変わったこと

0.2.1で固定DoDを全て実行検証したうえで、**理論整合の是正**と**高速化**を行いました。詳細は `CHANGELOG.md`、
速度の実測と再実行手順は `validation/rerun_2026-09-08/SPEED.md` にあります。

| 是正 | 効果 |
|---|---|
| 次数継続が低次数の収束で段階全体を終えていた | 予算依存の非単調性が解消。最高次数まで到達する |
| 解析段の初期分散が段階と不整合（既定`auto`へ） | 剛体姿勢を捨てて潰れる／写像を1つも採用しないの両極端が解消 |
| 発散状態が最良として採用され得た | `numerical_divergence`で停止し最良状態を返す。返り値1.8e77 → 有界 |

合成精度監査54条件（両エンジン）: 改善 **54** / 悪化 **0** / 不変 **0**、
最終既知対応RMSの中央値 **1.02×10⁻⁷**（0.2.0は 改善17/悪化6/不変13、中央値 0.0408）。

速度は決定的な命令数（callgrind）で計測しています。壁時計時間は他プロセス負荷を受けるため比較に使いません。
格子のアロケーション除去で **1.75〜1.98倍**、M-stepの完全直交分解化で **最大9.3倍**、
解析段の合計で 3次元 n=200 **3.20倍**。数値は不変で、原コード比較のM-step一致度はむしろ改善しています。

## 受入状態

**総合PASS。作業前に固定したDoD 11項目を、条件を下げずに全て実行検証しました。**
0.2.0で環境要因によりBLOCKEDだったD07（実nanobind境界）とD09（実pixi依存解決）、および実行免除だったD08（Rust/PyO3）を、
2026-09-08にLinux x86_64・pixi 0.76.2の環境で実際にビルド・実行して解消しました。免除は使用していません。

| 実行した検証 | 結果 |
|---|---|
| pixi依存解決（default / cpp / rust / docs の4環境） | 全て成功。`pixi.lock` と `rust/Cargo.lock` を生成 |
| 実nanobind拡張のビルド・インポート・境界試験 | 成功。`_native.cpython-312-x86_64-linux-gnu.so` |
| C++エンジンのPython試験（境界13件を含む） | 164件合格 |
| Rust数値核 `cargo test -p acpd-core` | 90件合格（原コード比較オラクル込み） |
| 実PyO3拡張のビルドとRustエンジンのPython試験 | 124件合格 |
| C++対Rustの実装間比較（2次元・3次元 × 4方式） | 8件合格。最終変換点の一致は3e-7以内 |
| C++数値核の単体検査／ASan・UBSan・リーク検査 | 各100項目合格 |
| 原コード成分比較84条件 | 全条件が0.2.0の報告値を再現 |

**0.2.0からのコード変更は`cpp/CMakeLists.txt`のビルド移植性修正1点のみで、数値アルゴリズムは変更していません。**
同梱Eigen 3.3.90は`Transpositions.h`にupstreamが3.4.0で修正した欠陥を含み、GCC 15はテンプレート本体を実体化前に診断するため
ビルドが停止しました。第三者ヘッダーを1バイトも改変せず、その経路にのみ`-Wno-template-body`を付与して解決しています。

作業前に固定した条件は `docs/review/WORK_ORDER.md`、各条件の最終状態は `validation/DoD.json` にあります。
0.2.0時点の判定と証拠ログは `validation/as_delivered_v0.2.0/`、今回の実行ログは `validation/rerun_2026-09-08/` に保存しています。

## 処理

| method | 2D / 3D共通の内容 |
|---|---|
| `rigid` | FilterRegの逆向きGMM事後、格子モーメント、SE(2)/SE(3) twist最小二乗による回転・並進 |
| `analytic` | 直接CPD事後、階乗付きTaylor基底、無正則化SVD、現在点群への解析写像の逐次合成 |
| `nonrigid` | FilterRegを一度実行 → その回転・並進を固定 → Analytic-CPDで残差写像を逐次合成 |

引数は **`registration(fixed, moving, ...)`**、方向は **moving → fixed** です。点数一致や行順の対応を仮定しません。
二段目では回転・並進の独立変数を更新しません。ただし、原論文どおりTaylorの定数・一次項を残しており、
残差写像から剛体様の変位を厳密に排除する一意分解ではありません。剛体直交投影、隠れた正則化、変位上限、減衰はありません。

## E-stepバックエンドの区別

| backend | E-stepの計算 |
|---|---|
| `permutohedral`（既定） | `[moving,fixed]` と `[0,values]` の拡張入力。重心補間Splat → 全d+1軸Blur → Slice |
| `permutohedral_noblur` | 原FilterRegの観測専用格子。Splat後にBlurを省略、問い合わせ側の頂点は追加せずSlice。固定分散なら格子を再利用 |
| `probreg` | 添付probregと同じ格子数判定 `L > 0.015 N` により、拡張入力のBlur省略版へ切り替え。probregの利得係数を保持 |
| `direct` | 全点対ガウス和。式照合の基準であり、明示指定時だけ使用 |
| `fgt` | 明示選択のIFGT近似。格子中心まわりのTaylor展開。**両論文の実装にはありません** |

ACPD段のE-stepは `AnalyticOptions(backend=...)` で別に選びます。既定は厳密な `direct` で、`fgt` だけが代替です。
格子バックエンドはCPDと事後確率の正規化方向が逆のため、ACPD段では拒否されます。

`permutohedral_noblur` と `probreg` は同一ではありません。前者にはprobregの利得係数がなく、観測以外の頂点を構築しません。
**Blur省略は「半径探索への代用」ではありません。どちらも実際のpermutohedral格子の重心補間を使用します。**
ACPDの事後計算には、上記FilterReg設定と無関係に常に直接法を使います。原論文のモデル比較に合わせた選択です。

## pixiによる操作（4環境とも依存解決・実行を確認済み）

```sh
# C++だけ。Rustコンパイラやmaturinは要求しません。
pixi run -e cpp test-cpp

# Rustだけ。C++のビルドやnanobindは要求しません。
pixi run -e rust test-rust

# 両実装、実拡張の境界試験、実装間比較
pixi run test

# 理論文書
pixi run -e docs docs
```

初回はconda/PyPI/cratesの依存取得が必要です。`pixi.lock` と `rust/Cargo.lock` は同梱しています。
**実行確認はLinux x86_64・pixi 0.76.2・Python 3.12.14・conda-forge GCC 15.3.0・rustc 1.92.0-nightlyのみです。**
OS設定はmacOS x86_64/arm64、Windows x86_64も記述していますが、それらでの動作確認は行っていません。
CI定義は `.github/workflows/ci.yml` に収録しています。CIを外部で起動・合格させたという主張はありません。

## Pythonでの利用

```python
from pathlib import Path
import numpy as np
import acpd_filterreg as reg

fixed = np.ascontiguousarray(np.load("fixed.npy", allow_pickle=False), dtype=np.float64)
moving = np.ascontiguousarray(np.load("moving.npy", allow_pickle=False), dtype=np.float64)

result = reg.registration(
    fixed, moving,
    method="nonrigid",
    engine="cpp",                 # "rust" で独立Rust版
    backend="permutohedral",      # 実Splat/Blur/Slice
)
np.save("registered.npy", result.transformed)
result.save(Path("transform.npz"))

# 保存済み写像の評価はネイティブ拡張を必要としません。
restored = reg.load_result("transform.npz")
np.testing.assert_allclose(restored.transform(moving), result.transformed)
```

`(点数,2)` / `(点数,3)` を同じ操作で扱います。既定の`copy=False`ではfloat64・C連続・整列済み配列を要求します。
変換を許す場合のみ`copy=True`を指定します。read-only配列も受け入れ、入力を書き換えません。
`copy=False`は型や配置の暗黙変換を禁止する指定であり、GIL解放前の所有メモリへの複製まで禁止するゼロコピー契約ではありません。

解析段階の既定初期分散は、剛体変換済み点群から計算するCPDの全対平均二乗距離です。
`AnalyticOptions(initialization="filterreg")` は剛体段階の分散を引き継ぐ**明示的な統合拡張**です。
これは論文に必須の仕様ではなく、格子近似が小さな分散を返した場合、初期状態が内部指標上の最良値となり残差更新が採用されない場合があります。

## E-stepバックエンドの選び方（実測）

命令数、σ²=0.05、direct比の倍率（大きいほど速い）。全条件は `validation/rerun_2026-09-08/SPEED.md`。

| d | n | permutohedral | permutohedral_noblur | probreg | fgt |
|---|---:|---:|---:|---:|---:|
| 2 | 100 | 1.31 | 1.45 | 1.20 | 0.36 |
| 2 | 1000 | 23.00 | **26.02** | 16.64 | 0.39 |
| 3 | 100 | 1.04 | 1.37 | 0.94 | 0.34 |
| 3 | 1000 | 11.82 | **22.37** | 10.14 | 0.24 |

- 格子と direct の損益分岐は **n ≈ 100〜150**。それ未満なら `direct` が最速です。
- 大きい点群では `permutohedral_noblur` が最速です。
- **`fgt` は全条件で遅く、既定では使いません。** 明示選択のモードとして提供しています。
  1ペアあたり280 flops（項数56×値列5）対 directの約8 flopsで、ペア数を35分の1未満にしないと勝てず、
  精度に必要な小さい被覆半径と速度に必要な大きいセルが正面から対立します。損益分岐は n≈7×10⁴ と見積もられ、
  本ライブラリの想定規模から2桁離れています。ACPD段でも direct の 0.06〜0.09倍でした。
- 参考論文の**低ランクM-stepは採用していません**。本実装のTaylor設計行列は数値ランクが常にKと一致し（286/286等）低ランクではないためです。
  代わりに同じ最小ノルム解を保つ完全直交分解へ替えて高速化しました。

## 実行済みの検証

C++の原コード比較は、添付版から別にビルドした参照実行形式と照合しています。本実装を参照側へリンクしていません。
84条件の内訳は格子32、原FilterReg単体座標2、CPD事後6、Taylor基底12、重み付き解析更新8、次数割当24です。
格子の最大絶対差は約9.21e-7（原版float32／本版float64）、解析更新は約6.00e-15、次数割当は全条件一致しました。
C++単体試験とASan/UBSan検査、共通Python操作層を通した原コード比較・保存復元等の結果は `validation/` にあります。
**試験専用driverの実行成功は、nanobind/PyO3接続の成功ではありません。** 接続の証拠は実拡張を要求する境界13試験と実装間比較8試験です。

実装間比較では、2次元・3次元 × `permutohedral` / `permutohedral_noblur` / `probreg` / `direct` の8条件で、
C++版とRust版の回転・並進が2e-8以内、変換後座標が3e-7以内、最良反復番号が完全一致しました。
36条件の精度監査も両エンジンで実行し、最終の既知対応RMSは最大4.7e-10、最終分散は最大3.3e-12の差でした。
5条件では採用反復番号が最大3ずれますが、いずれも`stop_reason`が`stable_tolerance`で、最終RMSと分散は12桁一致しています。
収束後の平坦域における最良反復のタイブレーク差であり、発散ではありません。

36条件の小規模合成監査では、剛体段階より既知対応RMSが改善した条件17、悪化6、1e-8以内で不変13でした。
悪化例を除外・再調整していません。詳細は `validation/synthetic_accuracy.json` に全件保存しています。
原実装との成分一致は、非凸な位置合わせ全体の成功、正しい点対応、単調な真値誤差改善、可逆変形を保証しません。

## ネイティブC++だけをオフラインで検査

同梱Eigenを利用するため、C++コンパイラ・CMake・Python/NumPy/pytestが既にある環境では以下の数値核試験に外部依存取得は不要です。

```sh
python tools/manage.py core-cpp
python tools/manage.py test-core
```

これは意図的にnanobind接続試験を含みません。実拡張を検査する`test-cpp`と混同しないでください。
原出力の再生成には元の添付ZIPを別途指定します。

```sh
python tools/build_reference.py --attachments /path/to/original_uploads --build build/oracles
python tools/generate_fixtures.py --oracles build/oracles
python tools/generate_rust_fixtures.py
```

参照抽出・ビルドスクリプトは今回GNU C++で実行しました。原プロジェクト全体（CUDA/PCL/Windows UI）のビルドではありません。

## 構成と範囲

`cpp/` と `rust/` はそれぞれPython非依存の数値核と薄い接続層です。点群検証、格子、事後統計、剛体解法、解析写像、二段階制御を分離しました。
`python/` は共通設定・操作・所有結果・保存済み写像評価のみです。位置合わせをPythonへ代替する実装はありません。
`docs/review/` は厳格レビューと作業書、`docs/TRACEABILITY.md` は式・原コード・修正・試験の対応表です。
`docs/theory_ja.tex` は外部図ファイルやBibTeXを必要としない理論文書です。

GPU、関節モデル、ノードグラフ変形、特徴記述子による全域初期化、類似変換の拡大縮小、論文掲載の全実験・速度再現は今回の範囲に含めません。
原論文と原コードに差がある箇所は `docs/SOURCE_DIFFERENCES.md` で明示しています。
過去の3系統のZIPとの置換互換はありません。正本はこの0.2.1のみです。0.2.0との差分は`CHANGELOG.md`にあります。
