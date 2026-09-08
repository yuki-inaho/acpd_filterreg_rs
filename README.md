# acpd-filterreg 0.2.0 — 論文・添付実装照合による是正版

2次元・3次元の剛体FilterRegと逐次合成Analytic-CPDを、独立したC++数値核／nanobind接続とRust数値核／PyO3接続で実装します。
**前回版の全対／半径探索による代用を廃止し、実際のpermutohedral格子を実装しました。**

## 受入状態

**ソース是正版です。全DoDの完了宣言ではありません。** C++数値核と共通Python操作層を、試験専用実行形式を介して検証しました。
nanobind拡張は依存未取得で未ビルド、pixiは本環境に存在せず実際の依存解決も未実行です。
Rust/PyO3は実装・同一参照データに対する試験コードを用意しましたが未コンパイルです。
Rustのビルド必須免除はユーザー指定によるもので、数値アルゴリズムの省略を認めるものではありません。

作業前に固定した条件は `docs/review/WORK_ORDER.md`、各条件の最終状態は `validation/DoD.json` にあります。
**D07（実nanobind境界）、D09（実pixi依存解決）はBLOCKEDであり、代替の自己試験を合格根拠に読み替えていません。**

## 処理

| method | 2D / 3D共通の内容 |
|---|---|
| `rigid` | FilterRegの逆向きGMM事後、格子モーメント、SE(2)/SE(3) twist最小二乗による回転・並進 |
| `analytic` | 直接CPD事後、階乗付きTaylor基底、無正則化SVD、現在点群への解析写像の逐次合成 |
| `nonrigid` | FilterRegを一度実行 → その回転・並進を固定 → Analytic-CPDで残差写像を逐次合成 |

引数は **`registration(fixed, moving, ...)`**、方向は **moving → fixed** です。点数一致や行順の対応を仮定しません。
二段目では回転・並進の独立変数を更新しません。ただし、原論文どおりTaylorの定数・一次項を残しており、
残差写像から剛体様の変位を厳密に排除する一意分解ではありません。剛体直交投影、隠れた正則化、変位上限、減衰はありません。

## 実格子法の区別

| backend | E-stepの計算 |
|---|---|
| `permutohedral`（既定） | `[moving,fixed]` と `[0,values]` の拡張入力。重心補間Splat → 全d+1軸Blur → Slice |
| `permutohedral_noblur` | 原FilterRegの観測専用格子。Splat後にBlurを省略、問い合わせ側の頂点は追加せずSlice。固定分散なら格子を再利用 |
| `probreg` | 添付probregと同じ格子数判定 `L > 0.015 N` により、拡張入力のBlur省略版へ切り替え。probregの利得係数を保持 |
| `direct` | 診断・式照合用の全点対ガウス和。明示指定時だけ使用 |

`permutohedral_noblur` と `probreg` は同一ではありません。前者にはprobregの利得係数がなく、観測以外の頂点を構築しません。
**Blur省略は「半径探索への代用」ではありません。どちらも実際のpermutohedral格子の重心補間を使用します。**
ACPDの事後計算には、上記FilterReg設定と無関係に常に直接法を使います。原論文のモデル比較に合わせた選択です。

## pixiによる操作（環境解決自体は本配布時点で未確認）

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

初回はconda/PyPI/cratesの依存取得が必要です。`pixi.lock` と `Cargo.lock` は未生成です。
OS設定はLinux x86_64、macOS x86_64/arm64、Windows x86_64を記述していますが、全OSでの動作確認を意味しません。
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

## 実行済みの検証

C++の原コード比較は、添付版から別にビルドした参照実行形式と照合しています。本実装を参照側へリンクしていません。
84条件の内訳は格子32、原FilterReg単体座標2、CPD事後6、Taylor基底12、重み付き解析更新8、次数割当24です。
格子の最大絶対差は約9.21e-7（原版float32／本版float64）、解析更新は約6.00e-15、次数割当は全条件一致しました。
C++単体試験とASan/UBSan検査、共通Python操作層を通した原コード比較・保存復元等の結果は `validation/` にあります。
**試験専用driverの実行成功は、nanobind/PyO3接続の成功ではありません。**

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
過去の3系統のZIPとの置換互換はありません。正本はこの0.2.0のみです。
