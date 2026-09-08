# 第三者ソース・由来・再配布条件

## 新規の数値核・操作層

新規の統合処理、接続、試験、文書はルートLICENSEのMIT条件です。
C++とRustは互いを呼び出さない独立実装ですが、**第三者由来のアルゴリズム移植であることを否定する意味ではありません。**
格子移植には次のBSD-3-Clause条件を併存させます。

## permutohedral

`cpp/src/lattice.cpp` と `rust/crates/acpd-core/src/lattice.rs` は、添付probregの
`third_party/permutohedral/permutohedral.{h,cpp}` に由来するアルゴリズム移植です。
Copyright (c) 2013, Philipp Krähenbühl。BSD-3-Clauseの全文を `third_party/licenses/permutohedral-BSD-3-Clause.txt` に保持しています。
変更内容は倍精度化、64ビットキー、独立した所有配列、安全検査、正しいstart伝達、範囲外読出しの回避、2D/3D共通化です。
原FilterRegの観測専用no-blur方式は添付コードと論文を照合して実装し、原関数本体の再配布は行いません。

## Analytic-CPD / probreg

原Analytic-CPDのMIT条件は `third_party/licenses/Analytic-CPD-MIT.txt`、probregのroot MIT条件は `third_party/licenses/probreg-MIT.txt` に保持しました。
原コード成分試験はユーザー添付から原関数を抽出し別実行形式にする方法です。本ZIPには原プロジェクト全体や生成した参照バイナリを含めません。
抽出関数・元ファイルの由来は `validation/oracle_provenance.json` に記録しています。
FilterReg全体のルートライセンスは添付版で確認できませんでした。その不明な範囲をMITと称して再配布しません。

## 同梱Eigen

`cpp/third_party/eigen/Eigen/` と `COPYING.*` は、添付FilterRegの `external/eigen3/` から複製した未変更の第三者ヘッダーです。
ヘッダーの版表示は3.3.90です。主にMPL-2.0であり、個別のBSDその他の表記を保持します。
`COPYING.MPL2`, `COPYING.BSD`, `COPYING.LGPL`, `COPYING.GPL`, `COPYING.MINPACK`, `COPYING.README` を保持しています。
ビルドは `EIGEN_MPL2_ONLY` を定義します。新規コードのMIT表記はEigenの個別条件を上書きしません。

## 別途取得する依存

nanobind、PyO3、NumPy、nalgebra、ndarray、pixi、CMake、maturin、TeXは環境から取得します。
それらのバイナリ、フォントファイル、ユーザー添付論文PDF、元の大きなデータセットは本ZIPに同梱しません。
理論PDFで通常必要なフォント埋込みはありますが、再利用用のフォントファイルを配布しません。
