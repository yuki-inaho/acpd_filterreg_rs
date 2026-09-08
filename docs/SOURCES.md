# 根拠資料と版識別

アルゴリズムの根拠はユーザー添付版です。公開リポジトリの現在のブランチへ無断で置換していません。

[1] Wei Gao, Russ Tedrake. FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization. CVPR 2019; 添付arXiv:1811.10136v3, 16 July 2019, 10ページ。

[2] Wei Feng, Haiyong Zheng. Structured Analytic Coherent Point Drift for Non-Rigid Point Set Registration. 添付arXiv:2605.00934v2, 15 May 2026, 22ページ。

[3] `probreg-master(1).zip`: probreg/filterreg.py、probreg/cc/kabsch.cc、third_party/permutohedral/permutohedral.{h,cpp}。root MIT、格子ソースBSD-3-Clause。

[4] `FilterReg-master.zip`: corr_search/gmm/、geometry_utils/permutohedral_common.hpp、rigid twist関連処理。ルートのライセンス表記を確認できなかったため、比較に必要な原関数本体は本ZIPへ再配布せず、ユーザー添付から抽出するスクリプトを提供します。Eigenのみ元の個別ライセンスを保持して同梱します。

[5] `Analytic-CPD-main(1).zip`: Analytic_CPD/Algo.h、Fitting.h。原ファイルはGB18030として復号しました。MIT（Wei Feng, 2026）。

[6] `cpd3d_rs-main.zip`: 操作構成・独立数値核とPython接続の責務分離の参照。アルゴリズム実装をそのまま置換したものではありません。

補助的な理論背景として、[1]の参考文献[1]（Adams, Baek, Davis, 2010, permutohedral filtering）、[2]の参考文献[3]（Myronenko, Song, 2010, CPD）を引用します。
今回の比較は上記添付資料から読める範囲で行い、これら背景論文を別途完全再現したとは称しません。

外部参照は接続API／環境設定の仕様確認に限定しました。nanobind 2.12.0、PyO3 0.28.3、numpy crate 0.28.0、nalgebra 0.34.2、ndarray 0.17.2、pixi manifestの公式資料です。
これらを参照した事実は、該当依存を取得・ビルドできたことを意味しません。

全入力アーカイブ・論文のバイト数とSHA256はSOURCE_MANIFEST.json、原ファイルと抽出関数のハッシュはvalidation/oracle_provenance.jsonです。
