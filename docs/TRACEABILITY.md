# 要求・数式・原コード・是正箇所・試験の対応

FR＝添付FilterReg論文、ACPD＝添付Structured Analytic CPD論文。ZIPの版識別はSOURCE_MANIFEST.jsonです。
個々の原関数のハッシュと行範囲は `validation/oracle_provenance.json` に保存しています。

| 要求 | 原資料 | C++ / Rustの対応箇所 | 実行又は用意した試験 |
|---|---|---|---|
| 実permutohedral | FR §4.1–4.2、probreg third_party/permutohedral | lattice.cpp / lattice.rs: enclosing_simplex, Permutohedral | test_upstream_lattice（32）、original_filterreg_simplex（2）、Rust同一84参照群 |
| 観測専用格子再利用 | FilterReg corr_search/gmm/gmm_permutohedral*、geometry_utils/permutohedral_common.hpp | FixedNoBlurLattice、registrationのキャッシュ | test_noblur_index_reused_with_fixed_variance、C++/Rust利得・再利用試験 |
| 逆向き事後とモーメント | FR (6),(7),(11),(13) | gaussian.cpp / gaussian.rs | test_filterreg_fused_moments、C++事後正規化 |
| 法線モーメント | FR (9),(10)、probreg/filterreg.py | moment_values, Statistics.normals | test_normals_registration、point/plane M-step |
| twist小行列 | FR (16)–(19)、FilterReg rigid*.cpp | rigid.cpp / rigid.rs: twist_jacobian, twist_pose, fit_rigid | 有限差分、既知姿勢、点対線・面、識別不能の例外 |
| 直接CPD事後 | ACPD (3),(4)、Algo.h ComputePosteriorP | posterior_statistics(filterreg=false) | test_upstream_exact_posterior（6）、w=0極端距離 |
| 重心集約・分散 | ACPD (7),(9),(27)、Algo.h UpdateSigma2FromStats | Statistics, variance_from_statistics | barycentric_identity_and_variance、原統計照合 |
| 階乗Taylor | ACPD (11),(13)、Algo.h/Fitting.h Taylor設計 | analytic.cpp / analytic.rs: exponents,basis | test_upstream_taylor_basis（12）、q10項数 |
| 無正則化重み付き更新 | ACPD (20)–(23)、AMVFF2D/3D_MStep | fit_analytic | test_upstream_analytic_mstep（8）、no_hidden_ridge、Rust同一出力 |
| 次数計画 | ACPD §3.7、Algo.h DegreeScheduleDecreasingStages | degree_schedule | 原コード24条件、予算1を含む |
| 逐次合成・最良復元 | ACPD §3.6 / Fig.1 / Appendix B.8 | registration.cpp / registration.rs | explicit_handoff_and_best_rollback、saved-map consistency |
| 姿勢固定 | ユーザーの二段階要求 | registration、Result.apply / RegistrationResult.apply | frozen_pose_composition_and_storage、initial_pose_coordinate_conjugation |
| 共通APIと所有権 | ユーザーの再利用・可搬性要求 | bindings.cpp、acpd-py/lib.rs、python/_api.py | test_native_boundary（未実行）、API試験はC++driverで実行 |
| 保存・微分 | 合成写像の連鎖律、今回の操作仕様 | python/_result.py, _mapping.py | 保存復元、未知点、Jacobian有限差分 |
| 原実装との差分明示 | 論文と添付コードの相違 | SOURCE_DIFFERENCES.md | ソースレビュー・oracle_provenance |
| Rust独立実装 | ユーザーのPyO3+Rust要求 | rust/crates/acpd-core、acpd-py | 90件のRust試験コード。未コンパイル、実行済みに数えない |

D01–D11の判定は `validation/DoD.json` を正本とします。ここで「試験あり」は実行成功と同義ではありません。
