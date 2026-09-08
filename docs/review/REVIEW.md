# 前回納品物の厳格レビュー（修正前の判定）

対象：添付された hybridreg_2d3d_cpp_rust.zip、acpd_filterreg_suite.zip、hybridreg_rust_sources.zip。
基準：添付 FilterReg 論文（2019）、Structured Analytic CPD 論文（2026-05-15 v2）、
probreg-master(1).zip、FilterReg-master.zip、Analytic-CPD-main(1).zip、cpd3d_rs-main.zip。
外部の最新版に置き換えず、添付版を基準とする。重大度 P0 は要求不適合で受入不可、P1 は主要機能・数値・検証の欠落、P2 は再現性・保守性上の欠陥。

## 結論

**前回納品物は「論文と添付実装に忠実な FilterReg + Analytic-CPD」の受入基準では不合格です。**
多数の自己試験の成功は、未実装のアルゴリズムを実装済みにする根拠にはなりません。
範囲外という注意書きで主要構成の削除を追認した点も不適切でした。

| ID | 重大度 | 修正前の証拠 | 判定・要求する修正 |
|---|---|---|---|
| R01 | P0 | suite/cpp/include/acpd/types.hpp の Backend は Direct, Grid のみ。cpp/src/gaussian.cpp は全対評価／半径探索。hybridreg も同様。 | permutohedral の埋込み・単体・重心座標・格子拡散が存在しません。FilterReg §4 の中心技術を別手法で代用しています。実格子法へ修正します。 |
| R02 | P0 | suite/cpp/src/registration.cpp の fit_rigid は SVD のみ。 | 論文 §5 の twist/Gauss–Newton 更新がありません。SE(2)/SE(3) の点対点・点対線／面の更新を実装し、有限差分で符号を検証します。SVD は任意の別解として区別します。 |
| R03 | P0 | hybridreg は既定 fixed-reference、正則化と剛体直交制約。suite は別の逐次合成、既定正則化 1e-6、最大変位 0.25。Rust専用版は前者に由来。 | 同名手法の仕様が納品ごとに変化しています。ACPD §3.5 の無正則化の重み付き最小二乗と §3.6 の現在点への逐次合成を既定とし、無断の投影・減衰を除去します。 |
| R04 | P1 | suite/cpp/src/analytic.cpp の exponents は q<=8、既定 max_degree=3。 | 論文の標準 qmax=10 を指定できません。2D 66項、3D 286項まで対応・試験します。 |
| R05 | P1 | suite degree_schedule は剰余を単純巡回配分。添付 Algo.h/DegreeScheduleDecreasingStages は長さ D,D-1,... の接頭辞配分。 | 55/10 の整数倍だけでなく、非整数予算について原コードと一致を検証します。 |
| R06 | P1 | suite registration.cpp は最終状態を無条件に返却し、best-state と反発時の復元を持ちません。 | ACPD Fig.1、Appendix B.8、Fitting.h の内部最良状態を保存・復元します。写像列と分散も同時に復元し、点群だけの復元を禁止します。 |
| R07 | P1 | 二段目の分散継承を必須仕様として記述。 | 元のユーザー要求は初期姿勢です。論文のCPD式初期分散と、統合時のFilterReg分散継承を明示的に区別します。既定は論文式とします。 |
| R08 | P1 | 多数の自己試験、代替C ABI／試験実行形式の成功を主な検証根拠としています。 | 実際の格子出力、E-step、Taylor基底、M-step、次数計画を添付原実装と比較していません。独立参照実装との数値比較が必要です。 |
| R09 | P1 | nanobind/PyO3/pixi は未実行。Rust専用版の44件はPython層のみ。 | 未実行である旨は書かれていますが完成確認にはなりません。実行試験とソース確認と環境阻害を別の状態で記録します。Rustのビルド必須免除のみユーザー指定済みです。 |
| R10 | P2 | 3系統の名前・操作・非剛体制約・履歴形式が混在。 | 一つの正本、仕様差分、ファイル指紋、完了条件と試験の対応表を作成します。旧ZIPを正本として使いません。 |

## 原コードにもある不整合（黙って模倣／修正しない）

- `probreg/filterreg.py` は M-step に渡した更新前の `t_source_e` から分散を計算し、2Dでも分母を `3.0` としています。論文の次元 d と更新後の点群を使う分散式を本実装の基準とし、差分を明記します。
- `probreg/cc/kabsch.cc` は重心で weight、相互共分散で weight² を使用します。一方 `filterreg.py` は sqrt(responsibility/sigma²) を渡しています。本実装の任意Kabsch解は一貫した責任度による重み付き目的を解きます。
- `FilterReg-master/corr_search/gmm/gmm_permutohedral_*` は **本物のpermutohedral格子でBlurを省く**最適化です。全対／半径探索への代用とは違います。`updatedvar_withblur.h` は宣言ガードだけで完全実装ではありません。
- `FilterReg-master` の固定分散版は観測だけをSplatし、モデル側はSliceのみです。`probreg` は拡張入力 `[moving,fixed]` と0値を使い、格子数でBlurを切り替えます。この二つを同一コードと称しません。
- `Analytic_CPD/Fitting.h` は点数が等しいだけで外部RMSE用の対応を仮定します。本ライブラリは点数一致を対応の証拠にせず、順不同点群では内部指標だけを使用します。外部RMSEを最適化や停止に漏らしません。
- FilterReg 論文 §5 の `skew` の記述と式(16)の左更新から得られる回転ヤコビアンは符号の扱いに注意が必要です。実装は原コードと一致する `[-[p]_x,I]` とし、変換の有限差分で確認します。

## 受入不可だった理由

本件の問題はコード行数や努力量ではなく、指定された方法の識別性を失う置換、原資料への追跡不足、自己試験だけの検証です。
修正後も既存論文の全実験・速度・GPU・関節運動を再現したと称してはいけません。今回の範囲はユーザー要求の2D/3D剛体FilterRegと解析非剛体の二段階です。
