# C++ / nanobind版

数値核はC++17とEigenだけで動きます。Python接続はnanobind 2.12.0です。
数値核のビルド・試験・ASan/UBSan、およびnanobind拡張のビルド・インポート・境界試験まで実行済みです。
状態は上位READMEとvalidation/DoD.jsonを参照してください。

```sh
cmake -S cpp -B cpp/build/core -DACPD_BUILD_PYTHON=OFF -DACPD_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build/core --parallel 2
ctest --test-dir cpp/build/core --output-on-failure
```

再利用する場合、`ACPD_INSTALL_CORE=ON`を指定して`cmake --install`してください。
`find_package(acpd CONFIG REQUIRED)`、`acpd::acpd_core`で利用できます。
既定では同梱Eigenを使用し、`ACPD_USE_SYSTEM_EIGEN=ON`で環境のEigen3へ切替できます。
これはアルゴリズムの実装切替ではありません。

同梱Eigenは添付FilterRegから複製した未変更の3.3.90スナップショットで、`Transpositions.h`にupstreamが3.4.0で
修正した`trt.derived()`の欠陥を含みます。GCC 15以降はテンプレート本体を実体化前に診断するため、この経路では
`-Wno-template-body`をCMakeが自動付与します。第三者ヘッダーは改変していません。Eigen 3.4以降を使う場合は
`ACPD_USE_SYSTEM_EIGEN=ON`を指定してください。

Python拡張はリポジトリルートで`pixi run -e cpp test-cpp`です（0.2.1で164件合格）。数値核だけの試験を接続の代わりに数えません。
BSD-3-Clause由来の格子と同梱Eigenの個別条件はTHIRD_PARTY_NOTICES.mdを参照してください。
