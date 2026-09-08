# 操作仕様 0.3.0

## 契約

`registration(fixed, moving, *, method="nonrigid", engine="cpp", backend="permutohedral", rigid=None, analytic=None, initial_rotation=None, initial_translation=None, target_normals=None, copy=False)`

移動→固定の方向です。float64の有限実数、2D又は3D、各点群は少なくともd+1点を要求します。
点群自体の階数が不足すれば、数値核は識別不能の剛体問題を例外にします。点数一致は必要なく、対応順を仮定しません。
SO(d)回転以外の初期行列（反射、拡大縮小を含む）は拒否します。二段階の途中で入力姿勢や座標系を勝手にリセットしません。

`registration_rigid` / `registration_analytic` / `registration_nonrigid` は上記methodを固定する入口です。
`engine_names()`はcpp,rust、`method_names()`はrigid,analytic,nonrigidです。
`backend_names()`はdirect,permutohedral,permutohedral_noblur,probregです。
選択した拡張がなければ`NativeExtensionUnavailableError`です。他言語やPythonへの暗黙切替はありません。

## 座標・単位

共通中心は固定点群の重心c、共通尺度は固定点群の重心からのRMS距離sです。両点群に同じ(p-c)/sを使用します。
回転・並進、出力点群、公開sigma2、履歴sigma2は入力座標の単位です。履歴step_rmsとfit_rmsも入力単位です。
`FilterRegOptions.sigma2`、`AnalyticOptions.sigma2`は入力単位の二乗です。
`min_sigma2`は正規化座標の二乗、toleranceは正規化座標又は相対変化の停止判定です。
`nll_before`は正規化座標での反復前対数評価であり、異なる手法や座標尺度の品質比較には使えません。
格子に支持がなくw=0で無限評価になる場合、Python側では`None`とします。

## 剛体設定

| 設定 | 既定 | 内容 |
|---|---:|---|
| max_iterations | 60 | EM反復上限 |
| tolerance | 1e-7 | 正規化変位RMSと分散相対変化の双方のしきい値 |
| w | 0.1 | 外れ値重み、0以上1未満 |
| sigma2 | None | 全対平均距離による初期分散、明示時は入力単位の二乗 |
| min_sigma2 | 1e-8 | 正規化二乗単位の下限 |
| update_sigma2 | True | 更新後座標・次元dで分散を更新 |
| solver | twist | twist又はkabsch。後者は点対点のみ |
| objective | point_to_point | point_to_planeは2Dでは点対線 |
| inner_iterations | 1 | E-stepを固定したGauss–Newton内反復数、1〜100 |

点対線・面には`target_normals`（固定点群と同形、各行単位法線）が必要です。
E-stepで平均した法線を単位長へ再正規化しません。低い平均法線長も式の重みに反映されます。

## 解析設定

| 設定 | 既定 | 内容 |
|---|---:|---|
| max_iterations | 220 | 全次数を合わせた反復予算の**上限**。次数継続により最高次数の収束で早期終了する |
| min_degree / max_degree | 1 / 10 | 総次数の範囲。実行次数は有効点数により低減 |
| tolerance | 1e-7 | 内部残差と相対変化の停止しきい値 |
| w | 0.1 | CPD外れ値重み |
| sigma2 | None | 明示時は入力単位の二乗。他の初期化指定より優先 |
| min_sigma2 | 1e-12 | 正規化二乗単位の下限 |
| rank_tolerance | 1e-12 | 最大特異値に対する切捨てしきい値 |
| min_mass | 1e-12 | 重み付き当てはめから除外する微小事後質量 |
| initialization | auto | 宣言methodで決定。nonrigid→filterreg（剛体段の分散を継承）、analytic→cpd（論文の全対初期化）。cpd/filterregの明示指定も可 |
| backend | direct | ACPD段のE-step。厳密な直接法が既定。fgtは明示選択のIFGT近似。格子backendは正規化の向きが逆のため拒否 |
| divergence_radius | 100.0 | 固定点群半径の何倍を超えた反復を発散として拒否するか。停止規則であり推定値への制約ではない |
| stable_patience | 5 | 小変化の連続回数／内部反発の待機回数 |
| no_improve_patience | 8 | 有意な改善がない反復の上限 |
| min_iterations | 6 | 待機型停止を許す最小反復数 |
| improvement_relative | 1e-6 | 有意改善の相対しきい値。最良値保存自体は真の最小値 |
| rebound_relative | 1e-3 | 内部最良残差からの反発比率 |

次数の重みはD,D-1,...,1です。総予算が三角数の整数倍でない場合も添付Algo.hの接頭辞配分に従います。
予算不足、点数不足、早期停止によりmax_degreeに到達しない場合があります。
正則化、減衰、最大変位、剛体直交制約の設定はありません。削除した旧設定は例外にします。
`analytic`単独で分散継承を指定し、明示sigma2もない場合は入力エラーです。

## 結果・保存

`RegistrationResult`はrotation、translation、center、normalization_scale、transformed、rigid_transformed、steps、sigma2、両stageを持ちます。
`transform(points)`は全写像、`residual_displacement(points)`は固定姿勢との差、`displacement(points)`は入力点との差です。
`jacobian(points)`は(n,d,d)、行が出力座標、列が入力座標です。法線の変換や可逆写像の保証は提供しません。
`steps`は最良状態までに採用された補正係数列、`history`は反発後を含む全試行履歴です。
`best_iteration`と履歴長は一致しない場合があります。剛体stageは最後の反復を返し、解析stageだけ最良状態へ戻します。
`converged=False`でも有限な最良状態を返すことがあります。stop_reasonを必ず区別してください。

`save(path)`、`load_result(path)`は形式2のNPZ（JSONメタデータ＋数値配列、allow_pickle=False）です。
再評価にはネイティブ拡張不要です。保存済み写像評価は登録の代替実装ではありません。
入力外への多項式外挿、極端な桁の入力、巨大ファイルの資源消費まで保証しません。

## 低水準の検査用操作

`gaussian_sum(sources, queries, values, sigma2=..., backend="direct", engine="cpp")` は行正規化しないガウス和です。
直接法以外はガウス和の格子近似で、名称ごとの尺度・利得の違いを保持します。
`posterior_stats(fixed,moving,sigma2=...,w=0.1,kind="cpd",backend="direct")` はrho,px,x2,mass,nll,vertices,unsupported,lattice_modeを返します。
ここでは登録処理の共通座標正規化は行いません。呼び出し時の座標とsigma2で評価します。
`permutohedral_filter(features,values,with_blur=True,start=0,reverse=False)` は白色化済み1〜16次元特徴を受け取る汎用格子です。
start以降のみSplatし全行Sliceします。reverseはBlur軸順の逆転です。

## 例外と所有権

配置・型の不一致はTypeError又はValueError、識別不能・質量不足・オーバーフローはRuntimeError系を返します。
C++では`std::invalid_argument`と`acpd::NumericalError`、Rustでは`Error::Invalid`と`Error::Numerical`を使用します。
GIL解放の前にネイティブ所有の作業配列へコピーし、結果にも所有メモリを付けます。
Rust接続とnanobind接続のこれらの契約は実装済みですが、実拡張経由の受入試験は本配布時点では未実行です。

## FgtOptions

`backend="fgt"`（FilterReg段）または `AnalyticOptions(backend="fgt")`（ACPD段）を選んだときだけ読まれます。
既定では使われず、厳密計算が失敗したときの代替として自動選択されることもありません。

| 名前 | 既定 | 意味 |
|---|---|---|
| order | 5 | クラスタ中心まわりTaylor展開の打切り次数。項数は C(order+d, d) |
| max_clusters | 4096 | 中心数の上限。被覆半径が満たせなかったことは内部で記録される |
| cluster_radius | 0.25 | 目標被覆半径。単位は h = sqrt(2σ²)。小さいほど高精度・高コスト |
| cutoff_radius | 4.0 | h 単位で、この距離より遠いクラスタは問い合わせ点で無視する |

計測値（d=3、n=300/260、値2列、既定 order=5・cluster_radius=0.25）では、
σ²を0.5から0.0005まで動かしても厳密和に対する最大相対誤差は 5×10⁻⁶ 以下でした。

## 停止理由

`stop_reason` は次のいずれかです。

| 値 | 意味 |
|---|---|
| not_run | その段階を実行していない |
| iteration_limit | 反復上限に到達 |
| tolerance | 剛体段が移動量と分散変化の両方で収束 |
| residual_tolerance | 内部残差が tolerance を下回った |
| stable_tolerance | 最高次数で小変化が stable_patience 回続いた |
| internal_rebound | 最良値から rebound_relative を超えて悪化した |
| no_improvement | 有意な改善がないまま no_improve_patience 回経過 |
| insufficient_posterior_mass | 事後質量または有効行数が当てはめに足りない |
| numerical_divergence | 反復が divergence_radius を超えた、または数値誤差が発生した。最良状態を返す |
