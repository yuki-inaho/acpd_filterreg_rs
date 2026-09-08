# GPU向け再定式化の実装（`gpu` ブランチ）

参照設計は `三次元Coherent Point DriftのGPU向け再定式化`（2026-09-08、第1.0版）です。
この文書自体はユーザー提供の資料であり、本リポジトリには同梱していません
（`THIRD_PARTY_NOTICES.md` の「添付PDFを再配布しない」方針に従います）。節番号のみ引用します。
本ブランチは、その設計の**中核であるE-step作用素分解**を実装し、実機で測定したものです。
設計文書の全体（構造化ガウス共分散、FFT格子、行列非形成のPCG、近似E-stepの受理判定）のうち、
何を実装し、何を実装していないかを本書で明示します。

## 実装したもの：E-stepのガウス和作用素

設計文書 §4 は、対応行列 `P`（N×M）を作らずに、必要な十分統計量が
**前向き1成分と転置4成分**のガウス和だけで得られることを示します。

```
s  = K π                                    (前向き、N次元)
z_j = s_j + c_j,   v_j = a_j / z_j
p  = diag(π) Kᵀ v                           (転置、対応質量)
F  = diag(π) Kᵀ diag(v) X                   (転置、3成分)
```

本実装では、これを**単一の作用素**に帰着させています。

```
out(q, ch) = Σ_s exp(-|q - s|² / (2σ²)) · values(s, ch)
```

- **FilterReg方向**（既存の`inverse`経路）: 1回。`values` は `[1, x, |x|²]`（法線を使う場合は `+ n`）。
- **CPD方向**（ACPD段）: 2回。前向きに `values = 1` で分母 `z_j` を求め、
  次に `1/(z_j + C)` で重み付けした `[1, x, |x|²]` を転置方向へ変換します。

これは既存の `fgt` バックエンドと同じ2段構成で、変換の中身だけが厳密なGPU評価に替わります。
`cpp/cuda/gaussian_transform.cu` がカーネル、`cpp/src/cuda.cpp` がEigenラッパーです。

### 厳密であり、近似ではありません

デバイス側は全点対を厳密に評価します。格子・打切り・展開は使いません。
唯一の近似は `CudaOptions(single_precision=True)` で、指数と積をfloatで評価し**蓄積はdouble**で行います。

実測（d=3、n=400/350、5チャンネル、σ²=0.05、CPU直接法との比較）:

| 精度 | 最大相対誤差 |
|---|---|
| fp64 | 2.18×10⁻¹⁶ |
| fp32 | 1.75×10⁻⁷ |

fp64は丸め水準で一致します。`tests/test_api.py::test_cuda_matches_the_exact_cpu_sum` が固定しています。

## 測定（NVIDIA GeForce GTX 1070、CUDA 12.9、sm_61）

壁時計時間です。他プロセスの負荷を受けるため参考値ですが、桁の比較には十分です。
**GP104はFP64が1:32**なので、fp64カーネルは理論性能の1/32で動きます。

### CPD方向のE-step（格子が原理的に使えない唯一の場所）

| n | direct[s] | cuda fp64[s] | cuda fp32[s] | 対direct fp64 | fp32 |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.6624 | 0.0934 | 0.1338 | **7.1×** | 5.0× |
| 2000 | 4.4466 | 0.0898 | 0.0741 | **49.5×** | 60.0× |
| 8000 | 40.4147 | 0.4019 | 0.2218 | **100.6×** | 182.2× |

CPDの事後確率はFilterRegと正規化の向きが逆で、permutohedral格子を適用できません。
この方向では直接法しか選択肢がなく、GPUの利得がそのまま出ます。設計文書が指摘するとおりです。

### FilterReg方向のE-step（CPUにO(N)の格子がある）

| n | noblur[s] | perm[s] | cuda fp64[s] | cuda fp32[s] | 最速 |
|---:|---:|---:|---:|---:|---|
| 1000 | 0.0361 | 0.1151 | 0.1699 | 0.0316 | cuda fp32 |
| 4000 | 0.1521 | 0.3264 | 0.1917 | 0.1115 | cuda fp32 |
| 16000 | 0.4846 | 1.1296 | 1.7450 | 0.8655 | **noblur** |
| 50000 | 0.4087 | 0.7535 | 4.2864 | 2.2809 | **noblur** |
| 150000 | 1.6205 | 2.4215 | 36.3940 | 19.5744 | **noblur** |

**この方向ではGPUを既定にしません。** 格子はO(N)、GPUの厳密和はO(NM)なので、
点数が増えれば必ず格子が勝ちます。交差点はおよそ **n ≈ 5000〜10000** です。

### end-to-end（`nonrigid`、d=3、解析20反復）

| n | CPU（noblur + direct） | CUDA fp64 | CUDA fp32 | 短縮 |
|---:|---:|---:|---:|---:|
| 500 | 0.464 s | 0.283 s | 0.271 s | 1.6× |
| 2000 | 5.052 s | 1.059 s | 0.873 s | 4.8× |
| 6000 | 44.476 s | 4.209 s | 3.716 s | **10.6×** |

剛体段の反復数が異なります（n=6000で48対25）。CUDA経路の剛体段は**厳密**なガウス和、
`permutohedral_noblur` は格子近似なので、収束の経路が違うためです。同一反復数の比較ではありません。

## 100〜1000点での最速構成：段ごとに使い分ける

このライブラリが実際に使われる規模での測定です。end-to-end `nonrigid`（剛体＋解析40反復）、
1回あたりミリ秒、5試行の最小値（他プロセス負荷に対して頑健な統計量）。

### d=3

| 剛体段 | ACPD段 | n=100 | n=200 | n=500 | n=1000 |
|---|---|---:|---:|---:|---:|
| permutohedral | direct | **32.0** | 161.3 | 743.9 | 2525.2 |
| permutohedral_noblur | direct | 38.2 | **136.8** | 651.9 | 2566.5 |
| direct | direct | 69.6 | 210.2 | 964.2 | 2876.0 |
| probreg | direct | 66.2 | 211.5 | 775.2 | 3001.2 |
| fgt | direct | 42.2 | 181.2 | 876.5 | 3790.0 |
| cuda | direct | 63.5 | 156.0 | 661.8 | 2289.6 |
| **permutohedral_noblur** | **cuda** | 91.9 | 151.0 | 335.7 | **587.7** |
| cuda | cuda | 100.5 | 170.0 | **323.1** | 666.3 |
| cuda | cuda（fp32） | 180.3 | 274.5 | 457.6 | 990.4 |

### d=2

| 剛体段 | ACPD段 | n=100 | n=200 | n=500 | n=1000 |
|---|---|---:|---:|---:|---:|
| **permutohedral_noblur** | direct | **26.1** | **87.0** | 466.4 | 1787.9 |
| permutohedral | direct | 32.0 | 98.8 | 510.3 | 1965.3 |
| direct | direct | 36.3 | 138.5 | 966.5 | 4014.8 |
| fgt | direct | 34.7 | 141.1 | 1135.7 | 4743.8 |
| cuda | direct | 71.6 | 137.6 | 506.9 | 2056.4 |
| **permutohedral_noblur** | **cuda** | 97.2 | 119.5 | **200.7** | **350.1** |
| cuda | cuda | 115.1 | 142.3 | 239.8 | 427.8 |

### 選び方

| 点数 | 最速の構成 |
|---|---|
| **n ≲ 200** | **純CPU**。剛体 `permutohedral_noblur`（3次元 n=100 のみ `permutohedral`）＋ ACPD `direct` |
| **n ≳ 300〜1000** | **混成**。剛体 `permutohedral_noblur`（CPU）＋ **ACPD の E-step だけ `cuda`** |

```python
reg.registration(fixed, moving, method="nonrigid",
                 backend="permutohedral_noblur",                 # 剛体段: CPU の O(N) 格子
                 analytic=reg.AnalyticOptions(backend="cuda"))   # ACPD段: GPU
```

n=1000 で 2525 → 588 ms（4.3倍、d=3）、d=2 で 1788 → 350 ms（5.1倍）です。

**「全部CPU」でも「全部GPU」でもないのは、段ごとに計算構造が違うからです。**
剛体段は permutohedral 格子が O(N) なので CPU が勝ち、
ACPD 段は正規化の向きが逆で格子が使えず O(N²) のままなので GPU が勝ちます。
設計文書が「対応推定と変形推定を別々に解決する」と述べているのは、この構造そのものです。

補足:

- **n ≲ 200 でGPUが負けるのは転送コスト**です。変換呼び出しごとにH2D/D2Hしており、固定費が30〜100 msあります。
  GPU常駐化（下記の未実装項目）で下がる余地があります。
- **fp32 はこの規模では効きません。** n=1000・d=3 で 666→990 ms と逆に遅くなります。
  反復数（12+40）も対応付きRMS（4.254e-07）もfp64と同一なので収束差ではありません。
  この規模ではカーネルがFP64スループット律速になっておらず、共有メモリのタイルをdoubleで置いたまま
  要素ごとにfloatへ変換する分だけ増えるためです。fp32が効くのはn=8000級（そこでは1.8倍速）からです。
- `fgt` はこの規模でも最下位クラスで、既定にしない判断は変わりません。

再実行:

```sh
python tools/benchmark_speed.py --binary cpp/build/cuda/acpd_bench \
  --stages nonrigid --sizes 100,200,500,1000 --dims 2,3 \
  --backends direct,permutohedral,permutohedral_noblur,probreg,fgt,cuda \
  --analytic-backends direct,cuda --trials 5
```

## 実装していないもの

設計文書のうち、本ブランチが**扱っていない**部分です。速度・精度の主張もしていません。

| 設計文書の項目 | 状態 |
|---|---|
| §6 構造化ガウス共分散 `Ĝ = Sβ Cβ Sβᵀ`、FFT格子 | 未実装。本実装の非剛体変形はTaylor解析写像のままです |
| §6.3 行列非形成の対称双対系とPCG | 未実装。M-stepは完全直交分解による直接解法のままです |
| §7 疎な精度行列 `R = ρ0 I + ρ1 Lg + ρ2 Lg²` | 未実装 |
| §8 近似E-stepの目的関数区間による受理判定 | 未実装。本ブランチのGPU E-stepは厳密なので受理判定が不要です |
| §10 GPU常駐、ホスト転送の最小化 | 未実装。現状は呼び出しごとにH2D/D2H転送します |
| 複数GPU、混合精度の失敗状態管理 | 未実装 |

特に**GPU常駐化は未着手**です。現在は変換1回ごとに入力を転送して結果を戻すため、
小さい点群では転送が支配的になります（n=500でfp32がfp64より遅い理由）。
反復間で点群と統計量をデバイスに常駐させれば、この定数は下がります。

## 使い方

CUDAは既定で無効です。有効化するには次のように構成します。

```sh
# C++数値核とベンチ
cmake -S cpp -B cpp/build/cuda -G Ninja \
  -DACPD_BUILD_TESTS=ON -DACPD_BUILD_BENCH=ON \
  -DACPD_ENABLE_CUDA=ON -DACPD_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DACPD_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release

# nanobind拡張
pip install --no-deps --no-build-isolation -e ./cpp \
  -C cmake.define.ACPD_ENABLE_CUDA=ON \
  -C cmake.define.ACPD_CUDA_HOST_COMPILER=/usr/bin/g++-13
```

`ACPD_CUDA_HOST_COMPILER` は、C++側のコンパイラがCUDAツールキットの対応範囲より新しい場合に必要です。
カーネルは平文のC ABIを持つ独立した翻訳単位なので、両者が別のコンパイラでもABI上の問題は起きません。

```python
import acpd_filterreg as reg

reg.cuda_available()        # True/False。Falseなら cuda 選択は例外になります
reg.cuda_device_name()      # 'NVIDIA GeForce GTX 1070'

reg.registration(fixed, moving, method='nonrigid',
                 backend='cuda',                                  # FilterReg段のE-step
                 analytic=reg.AnalyticOptions(backend='cuda'),    # ACPD段のE-step
                 cuda=reg.CudaOptions(single_precision=False))
```

## 選択規則（既定にしない理由を含む）

- **既定では選ばれません。** デバイスが無いビルド・マシンで `cuda` を選ぶと**例外**になります。
  CPUへ黙って切り替えることはありません（`test_cuda_backend_never_falls_back_to_cpu`）。
- **Rustエンジンにはデバイス経路がありません。** `engine="rust"` で `cuda` を選ぶと明示的な例外です。
  実装間比較（parity）に `cuda` は含めません。両エンジンが同じ backend を持つ条件だけが比較対象です。
- FilterReg段は、点数が数千を超えるなら `permutohedral_noblur` の方が速くなります。
- ACPD段は、格子が使えないため `cuda` が有効な唯一の高速化手段です。

## 再実行

```sh
cpp/build/cuda/acpd_bench --stage estep --n 8000 --d 3 --backend cuda --sigma2 0.05
cpp/build/cuda/acpd_bench --stage nonrigid --n 6000 --d 3 --backend cuda --analytic-backend cuda
cpp/build/cuda/acpd_bench --stage nonrigid --n 6000 --d 3 --backend cuda --analytic-backend cuda --cuda-single-precision 1
```
