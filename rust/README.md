# Rust / PyO3版（ソース提供・未コンパイル）

`crates/acpd-core` はnalgebraのみを使うPython非依存の数値核です。
permutohedral格子、ガウス和、両向き事後、twist、解析SVD、次数計画、最良状態復元、二段階処理をRustで独立実装しています。
C++ FFI、Python数値処理への代替、未実装のダミーはありません。
`crates/acpd-py` がPyO3とnumpy crateによる薄い接続です。

ユーザー指定によりRustのビルド必須は免除されていますが、コンパイル成功と数値一致は未確認です。
84条件の原実装出力と6つの機能・異常系検査からなる90件のRust試験を同梱しています。
同じ固定参照データをC++側で照合しました。C++の成功をRustの成功と読み替えません。

```sh
# PythonやC++を呼ばない数値核
cargo test -p acpd-core --manifest-path rust/Cargo.toml

# Rust専用pixi環境（C++コンパイラ・nanobind不要）
pixi run -e rust test-rust
```

`crates/acpd-core/examples/register_points.rs` はRustから直接利用する例です。
独立したRust配布が必要な場合も共通python操作層と設定を同じ0.2.0に合わせて使用します。
格子アルゴリズムには添付probregのBSD-3-Clause条件を保持します。
