# 旧納品物からの移行

正本は `acpd-filterreg` 0.2.0です。`hybridreg`、`acpd-filterreg` 0.1.0、Rust専用hybridregを同じ仕様として混用しないでください。
0.2.0は破壊的な是正です。旧NPZ形式1や旧hybridreg保存形式を自動読込しません。

`grid`、`radius`は廃止しました。`permutohedral`は実際のSplat/Blur/Slice、`permutohedral_noblur`は原観測専用格子です。
`cutoff`、`regularization`、`max_step`、`allow_degree_reduction`、固定参照型や剛体直交投影は廃止しています。
ACPDは現在点群への逐次合成だけです。FilterRegのsolverは既定twistとなり、Kabschは明示選択です。
解析既定はT=55、qmax=10、無正則化で、最良状態へ点群・分散・写像列を同時に戻します。
分散継承は既定ではなく、`AnalyticOptions(initialization="filterreg")`で明示します。

元の `cpd3d_rs` の責務分離・明示的エンジン選択を参考にしていますが、完全置換互換の実装ではありません。
`cpd3d_rs` のIFGTや類似変換のスケールを、本ライブラリのFilterReg格子やSE(d)剛体更新と同一視しません。
`scale`プロパティは剛体倍率1、`normalization_scale`は共通数値座標の尺度です。
共通APIの順序はfixed,movingです。添付FilterReg論文のX=moving,Y=fixedの記号順と取り違えないでください。
