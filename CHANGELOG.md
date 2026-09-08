# 変更履歴

## 0.2.1 — 2026-09-08

固定したDoDの条件は一切変更していません。0.2.0で環境要因により未達だった項目を、実際にビルド・実行して解消した版です。

### 判定

| ID | 0.2.0 | 0.2.1 | 根拠 |
|---|---|---|---|
| D07 実nanobind境界 | BLOCKED | **PASS** | 拡張をビルド・インポートし、driverを使わない境界13試験を含む164試験が合格 |
| D08 Rust/PyO3 | SOURCE_ONLY（実行免除） | **PASS** | 免除を使わず`cargo test`90件、PyO3経由124件、実装間比較8件を実行 |
| D09 実pixi依存解決 | BLOCKED | **PASS** | default/cpp/rust/docs の4環境を解決し、`pixi.lock`と`rust/Cargo.lock`を生成 |
| その他8項目 | PASS | **PASS** | 本環境で全て再実行し、0.2.0の報告値を再現 |

0.2.0がBLOCKEDだった原因は、ビルドホストにnanobind・pixi・cargoが存在せずネットワークからも取得できなかったことです。
コードの欠陥ではありませんでした。0.2.0の判定と証拠ログは `validation/as_delivered_v0.2.0/` にそのまま保存しています。

### 修正

- `cpp/CMakeLists.txt`: 同梱Eigen 3.3.90の`Transpositions.h`は、存在しないメンバ`derived()`を
  `Transpose<TranspositionsBase<Derived>>`に対して呼び出します。upstream Eigenが3.4.0で修正した欠陥です。
  GCC 15はテンプレート本体を実体化前に診断するため、この潜在欠陥がビルド停止に変わりました。
  第三者ヘッダーを1バイトも改変せず、同梱Eigenを使う経路にのみ`-Wno-template-body`を付与します。
  Eigen 3.4以降を使う場合は`ACPD_USE_SYSTEM_EIGEN=ON`を指定してください。
- `.gitignore`: `validation/**/*.log` を除外対象から外しました。DoDの証拠だからです。

数値アルゴリズム、公開API、既定値、停止条件はいずれも変更していません。

### 実行環境

Linux x86_64 / pixi 0.76.2 / Python 3.12.14 / conda-forge GCC 15.3.0 / rustc 1.92.0-nightly /
nanobind 2.12.0 / PyO3 0.28.3。macOS・Windowsは設定のみで未実行です。

## 0.2.0

論文・添付原実装との照合による是正版。詳細は `docs/review/REVIEW.md` と
`validation/as_delivered_v0.2.0/VALIDATION.md` を参照してください。
