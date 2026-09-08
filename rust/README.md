# Rust / PyO3版（ビルド・試験実行済み）

`crates/acpd-core` はnalgebraのみを使うPython非依存の数値核です。
permutohedral格子、ガウス和、両向き事後、twist、解析SVD、次数計画、最良状態復元、二段階処理をRustで独立実装しています。
C++ FFI、Python数値処理への代替、未実装のダミーはありません。
`crates/acpd-py` がPyO3とnumpy crateによる薄い接続です。

ユーザー指定によりRustのビルド必須は免除されていましたが、0.2.1では免除を使わず実際にビルド・実行しました。
84条件の原実装出力と6つの機能・異常系検査からなる90件のRust試験が合格しています（`validation/rerun_2026-09-08/cargo_test.log`）。
PyO3拡張もmaturinでビルドし、共通Python層経由で124件が合格しました（境界13件を含む）。
C++版との実装間比較8条件も合格しています。C++の成功をRustの成功と読み替えてはいません。両方を実行した結果です。
実行確認はLinux x86_64・rustc 1.92.0-nightly・PyO3 0.28.3のみで、他OSは未確認です。

```sh
# PythonやC++を呼ばない数値核
cargo test -p acpd-core --manifest-path rust/Cargo.toml

# Rust専用pixi環境（C++コンパイラ・nanobind不要）
pixi run -e rust test-rust
```

`crates/acpd-core/examples/register_points.rs` はRustから直接利用する例です。
独立したRust配布が必要な場合も共通python操作層と設定を同じ0.2.1に合わせて使用します。
格子アルゴリズムには添付probregのBSD-3-Clause条件を保持します。
