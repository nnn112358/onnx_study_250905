# 第1章 ONNXのインストールと使用

> 本章の概要: ONNX Runtime（ORT）の概要、インストール、Python/C#/C++での基本使用、そして一部の高度なAPI活用までを段階的に解説します。必要なコマンドと最小限のコード例に絞り、実務に直結する手順を示します。

ONNX（Open Neural Network Exchange）は、異なる機械学習フレームワーク間でモデルを共有するための標準化フォーマットです。ONNX Runtimeは、ONNXモデルを効率的に実行するためのクロスプラットフォーム推論エンジンです。

## 1.1 ONNX Runtime（ORT）のインストール

ONNX Runtimeは、高性能な機械学習推論を実現するライブラリです。さまざまなハードウェア（CPU/GPU/専用アクセラレータ）をサポートし、本番環境での展開に最適化されています。

### 1.1.1 環境要件

ONNX Runtimeを正しく動作させるには、適切なシステム環境が必要です。以下を事前に確認してください。

**システム要件:**
- **Python**: 3.7以上（Python使用の場合）
- **OS**: Windows 10、macOS 10.15以上、またはLinux
- **メモリ**: 最低4GB RAM（推奨8GB以上）
- **ディスク容量**: 最低1GB

**GPU使用時の追加要件:**
- **NVIDIA GPU**: CUDA Compute Capability 6.0以上
- **CUDA**: バージョン11.6以上
- **cuDNN**: 対応するバージョン

これらを満たすと、ONNX Runtimeの機能を最大限に活用できます。特にGPUを使う場合は、CUDA環境の正しいセットアップが重要です。

### 1.1.2 Pythonでのインストール

ONNX Runtimeは充実したPython APIを提供しています。以下はPython環境でのインストール手順です。

#### pipを使用したインストール

ONNX RuntimeはPyPI（Python Package Index）で配布されています。`pip`でインストールします。

**CPU版のインストール:**
```bash
# CPU版のONNX Runtimeをインストール
pip install onnxruntime
```

CPU版は最も軽量で互換性が高く、ほとんどの環境で動作します。特別なハードウェア要件がなく、初学者におすすめです。

**GPU版のインストール:**
```bash
# GPU版のONNX Runtimeをインストール（CUDA対応）
pip install onnxruntime-gpu
```

GPU版は、NVIDIA GPUを使用した高速推論が可能ですが、CUDA環境が正しくセットアップされている必要があります。

**ONNXコアライブラリのインストール:**
```bash
# ONNXライブラリもインストール（モデルの作成・編集に必要）
pip install onnx
```

ONNXライブラリは、ONNXモデルの読み込み・作成・変更を行うツールです。ONNX Runtimeと組み合わせると、開発環境が整います。

> 注意: CPU版とGPU版は同時に共存できません。GPU版を入れる場合は、先にCPU版をアンインストール（`pip uninstall onnxruntime`）してください。

#### インストールの確認

インストールの成否とシステム構成を以下で確認します。

```python
import onnxruntime as ort
import onnx

# バージョン情報の表示
print(f"ONNX Runtime バージョン: {ort.__version__}")
print(f"ONNXバージョン: {onnx.__version__}")

# 利用可能な実行プロバイダーを確認
providers = ort.get_available_providers()
print(f"利用可能なプロバイダー: {providers}")

# システム情報の詳細表示
print("\n=== システム構成詳細 ===")
for provider in providers:
    print(f"✓ {provider}")

# GPU対応の確認
if 'CUDAExecutionProvider' in providers:
    print("\n🚀 GPU（CUDA）サポートが利用可能です")
else:
    print("\n💻 CPU推論のみ利用可能です")
```

実行例:

```
ONNX Runtime バージョン: 1.16.3
ONNXバージョン: 1.15.0
利用可能なプロバイダー: ['CUDAExecutionProvider', 'CPUExecutionProvider']

=== システム構成詳細 ===
✓ CUDAExecutionProvider
✓ CPUExecutionProvider

🚀 GPU（CUDA）サポートが利用可能です
```

**実行プロバイダーの説明:**
- `CPUExecutionProvider`: CPU上でモデルを実行
- `CUDAExecutionProvider`: NVIDIA GPU上でモデルを実行  
- `DmlExecutionProvider`: DirectML（Windows GPU）上でモデルを実行
- その他：TensorRT、OpenVINO等の専用アクセラレータ対応

### 1.1.3 C#/C/C++/WinMLでの導入

ONNX Runtimeは、Python以外にも.NET（C#）やC++などから利用できます。これにより、さまざまな開発環境や本番システムでONNXモデルを活用できます。

#### C#での使用

C#（.NET）は、マイクロソフトが開発したプログラミング言語で、Windowsアプリケーションや企業システムで広く使用されています。ONNX RuntimeのC# APIは、.NET Frameworkおよび.NET Coreの両方をサポートしています。

**NuGetパッケージのインストール:**

NuGet（ニューゲット）は、.NET用のパッケージ管理システムです。Visual StudioのPackage Manager Consoleまたはコマンドラインから以下を実行します：

```bash
# CPU版のインストール
Install-Package Microsoft.ML.OnnxRuntime

# GPU版のインストール（NVIDIA CUDA対応）
Install-Package Microsoft.ML.OnnxRuntime.Gpu

# 特定のバージョンをインストールする場合
Install-Package Microsoft.ML.OnnxRuntime -Version 1.16.3
```

**基本的な使用例:**

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;

class Program
{
    static void Main()
    {
        // ONNXモデルファイルのパス
        string modelPath = "model.onnx";
        
        // 推論セッションの作成
        using var session = new InferenceSession(modelPath);
        
        // モデル情報の表示
        Console.WriteLine("=== モデル情報 ===");
        Console.WriteLine($"入力数: {session.InputMetadata.Count}");
        Console.WriteLine($"出力数: {session.OutputMetadata.Count}");
        
        // 入力情報の詳細表示
        foreach (var input in session.InputMetadata)
        {
            Console.WriteLine($"入力名: {input.Key}");
            Console.WriteLine($"データ型: {input.Value.ElementType}");
            Console.WriteLine($"形状: [{string.Join(", ", input.Value.Dimensions)}]");
        }
        
        // 出力情報の詳細表示
        foreach (var output in session.OutputMetadata)
        {
            Console.WriteLine($"出力名: {output.Key}");
            Console.WriteLine($"データ型: {output.Value.ElementType}");
            Console.WriteLine($"形状: [{string.Join(", ", output.Value.Dimensions)}]");
        }
    }
}
```

**セッションオプションの設定:**

```csharp
// セッションオプションの詳細設定
var options = new SessionOptions();
options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
options.IntraOpNumThreads = 4;  // 並列処理スレッド数

// GPU使用の指定
options.AppendExecutionProvider_CUDA(0);  // GPU ID 0を使用

using var session = new InferenceSession(modelPath, options);
```

#### C++での使用

C++は、高性能が要求されるシステムやリアルタイムアプリケーションで広く使用されています。ONNX RuntimeのC++ APIは、最小限のオーバーヘッドで高速な推論を実現できます。

**環境構築:**

C++でONNX Runtimeを使用するには、ヘッダーファイルとライブラリが必要です。公式リリースの事前ビルドを利用するか、ソースからビルドします。

1. [ONNX Runtime リリースページ](https://github.com/microsoft/onnxruntime/releases)から対応プラットフォームのファイルを取得
2. インクルードパスとリンクライブラリの設定
3. 必要なヘッダーファイル：`onnxruntime_cxx_api.h`

**基本的な使用例:**

```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>

int main() {
    try {
        // ONNX Runtime環境の初期化
        // 第1引数：ログレベル（WARNING, INFO, ERROR等）
        // 第2引数：ログ識別用の名前
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXTutorial");
        
        // セッションオプションの設定
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);  // 並列処理スレッド数
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );
        
        // ONNXモデルファイルの読み込み
        const char* model_path = "model.onnx";
        // Windowsの場合：const wchar_t* model_path = L"model.onnx";
        Ort::Session session(env, model_path, session_options);
        
        // モデル構造の情報取得
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();
        
        std::cout << "=== モデル情報 ===" << std::endl;
        std::cout << "入力数: " << num_input_nodes << std::endl;
        std::cout << "出力数: " << num_output_nodes << std::endl;
        
        // 入力詳細情報の取得
        for (size_t i = 0; i < num_input_nodes; i++) {
            // 入力名の取得
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name = session.GetInputNameAllocated(i, allocator);
            
            // 入力型情報の取得
            auto input_type_info = session.GetInputTypeInfo(i);
            auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            
            auto input_shape = tensor_info.GetShape();
            
            std::cout << "入力 " << i << ": " << input_name.get() << std::endl;
            std::cout << "  形状: [";
            for (size_t j = 0; j < input_shape.size(); j++) {
                std::cout << input_shape[j];
                if (j < input_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // 出力詳細情報の取得
        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatorWithDefaultOptions allocator;
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            
            std::cout << "出力 " << i << ": " << output_name.get() << std::endl;
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNXランタイムエラー: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "モデル読み込みが正常に完了しました。" << std::endl;
    return 0;
}
```

**CMakeを使用したビルド設定例:**

```cmake
cmake_minimum_required(VERSION 3.15)
project(ONNXTutorial)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtimeのパスを設定
set(ONNXRUNTIME_ROOT_PATH "/path/to/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include")
set(ONNXRUNTIME_LIB_DIRS "${ONNXRUNTIME_ROOT_PATH}/lib")

# インクルードディレクトリの設定
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# 実行可能ファイルの作成
add_executable(onnx_tutorial main.cpp)

# ライブラリのリンク
target_link_directories(onnx_tutorial PRIVATE ${ONNXRUNTIME_LIB_DIRS})
target_link_libraries(onnx_tutorial onnxruntime)
```

## 1.2 ONNX Runtimeの使用

インストールが完了したら、実際にONNXモデルを使用した推論を行います。この節では、PythonとC++での具体的な推論実行方法を学習します。

### 1.2.1 PythonでのONNX Runtime使用

Pythonでの推論実行は、機械学習の研究開発において最も一般的なアプローチです。PythonのNumPyライブラリとの親和性が高く、データの前後処理も容易に行えます。

#### 基本的な推論の実行

ONNXモデルを使用した推論の基本的な流れは以下の通りです：

1. **モデルの読み込み**: ONNXファイルから推論セッションを作成
2. **入力データの準備**: モデルが期待する形式でデータを準備
3. **推論の実行**: 準備したデータを使用して推論を実行
4. **結果の取得**: 推論結果を取得・処理

```python
import onnxruntime as ort
import numpy as np

# Step 1: モデルの読み込み
print("ONNXモデルを読み込んでいます...")
session = ort.InferenceSession("model.onnx")
print("✓ モデルの読み込みが完了しました")

# Step 2: 入力データの準備
# モデルの入力仕様を取得
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print(f"入力仕様:")
print(f"  名前: {input_name}")
print(f"  形状: {input_shape}")
print(f"  データ型: {input_type}")

# 動的次元（-1やNone）の処理
# 例：[None, 3, 224, 224] → [1, 3, 224, 224]
actual_input_shape = []
for dim in input_shape:
    if dim is None or dim == -1:
        actual_input_shape.append(1)  # バッチサイズを1に設定
    else:
        actual_input_shape.append(dim)

# ランダムデータで入力データを作成（実際の使用時は実データを使用）
input_data = np.random.randn(*actual_input_shape).astype(np.float32)
print(f"入力データ形状: {input_data.shape}")

# Step 3: 推論の実行
print("推論を実行しています...")
outputs = session.run(None, {input_name: input_data})
print("✓ 推論が完了しました")

# Step 4: 結果の取得と表示
print(f"\n=== 推論結果 ===")
for i, output in enumerate(outputs):
    print(f"出力{i+1}:")
    print(f"  形状: {output.shape}")
    print(f"  データ型: {output.dtype}")
    print(f"  データ範囲: [{output.min():.4f}, {output.max():.4f}]")
    
    # 小さなテンソルの場合は実際の値も表示
    if output.size <= 10:
        print(f"  値: {output.flatten()}")
    else:
        print(f"  最初の5要素: {output.flatten()[:5]}")
```

**重要なポイント:**

- **動的形状の処理**: モデルの入力形状に`None`や`-1`が含まれる場合、実際の値に置き換える必要があります
- **データ型の一致**: 入力データの型がモデルの期待する型と一致している必要があります
- **メモリ効率**: 大きなモデルや大量のデータを扱う場合は、メモリ使用量に注意が必要です

#### 複数入力・複数出力の処理

```python
import onnxruntime as ort
import numpy as np

# セッションの作成
session = ort.InferenceSession("multi_input_model.onnx")

# 入力情報の取得
input_names = [input.name for input in session.get_inputs()]
input_shapes = [input.shape for input in session.get_inputs()]

# 入力データの準備
inputs = {}
for i, (name, shape) in enumerate(zip(input_names, input_shapes)):
    inputs[name] = np.random.randn(*shape).astype(np.float32)

# 推論の実行
outputs = session.run(None, inputs)

# 結果の表示
for i, output in enumerate(outputs):
    print(f"出力{i+1}の形状: {output.shape}")
```

### 1.2.2 C++でのONNX Runtime使用

#### 基本的な推論の実行

```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

int main() {
    // 環境とセッションオプションの初期化
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXTutorial");
    Ort::SessionOptions session_options;
    
    // セッションの作成
    const char* model_path = "model.onnx";
    Ort::Session session(env, model_path, session_options);
    
    // メモリ情報の取得
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    // 入力データの準備
    std::vector<float> input_data(1 * 3 * 224 * 224);  // バッチサイズ1, チャンネル3, 224x224
    std::fill(input_data.begin(), input_data.end(), 0.5f);
    
    std::vector<int64_t> input_shape{1, 3, 224, 224};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());
    
    // 入力・出力名の取得
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    
    // 推論の実行
    auto outputs = session.Run(Ort::RunOptions{nullptr}, 
                              input_names, &input_tensor, 1,
                              output_names, 1);
    
    // 結果の取得
    float* output_data = outputs[0].GetTensorMutableData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    
    std::cout << "推論が正常に完了しました" << std::endl;
    std::cout << "出力形状: ";
    for (auto dim : output_shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

## 1.3 ONNX Runtimeの構築

### 1.3.1 構築の方法

#### ソースからのビルド

特定のニーズに合わせてONNX Runtimeをカスタマイズする場合、ソースからビルドできます。

```bash
# リポジトリのクローン
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# ビルドスクリプトの実行
./build.sh --config Release --build_shared_lib --parallel

# Python用ホイールの構築
./build.sh --config Release --build_wheel --parallel
```

#### Docker環境での構築

```dockerfile
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip

WORKDIR /workspace
RUN git clone --recursive https://github.com/microsoft/onnxruntime.git
WORKDIR /workspace/onnxruntime

RUN ./build.sh --config Release --build_shared_lib --parallel
```

### 1.3.2 ONNX Runtime API概要

#### Python API の主要コンポーネント

```python
import onnxruntime as ort

# 1. InferenceSession - メインのインターフェース
session = ort.InferenceSession("model.onnx")

# 2. SessionOptions - セッション設定
options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 3. RunOptions - 実行時オプション
run_options = ort.RunOptions()
run_options.log_severity_level = 2

# 4. プロバイダー設定
providers = [
    'CUDAExecutionProvider',  # GPU使用
    'CPUExecutionProvider'    # CPUフォールバック
]
session = ort.InferenceSession("model.onnx", providers=providers)
```

#### モデル情報の取得

```python
# 入力情報の取得
for input in session.get_inputs():
    print(f"入力名: {input.name}")
    print(f"形状: {input.shape}")
    print(f"データ型: {input.type}")

# 出力情報の取得
for output in session.get_outputs():
    print(f"出力名: {output.name}")
    print(f"形状: {output.shape}")
    print(f"データ型: {output.type}")

# プロファイル情報の取得
prof = session.end_profiling()
print(f"プロファイルファイル: {prof}")
```

### 1.3.3 API詳細

#### セッションオプションの詳細設定

```python
import onnxruntime as ort

# セッションオプションの作成
options = ort.SessionOptions()

# 並列処理の設定
options.intra_op_num_threads = 8  # オペレーター内並列数
options.inter_op_num_threads = 4  # オペレーター間並列数

# メモリ使用量の最適化
options.enable_mem_pattern = True
options.enable_cpu_mem_arena = True

# グラフ最適化レベルの設定
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# ログレベルの設定
options.log_severity_level = 2  # 0:詳細, 1:情報, 2:警告, 3:エラー, 4:致命的

# プロファイリングの有効化
options.enable_profiling = True

# セッションの作成
session = ort.InferenceSession("model.onnx", options)
```

#### 実行時オプション

```python
# 実行時オプションの設定
run_options = ort.RunOptions()
run_options.log_severity_level = 1
run_options.log_verbosity_level = 0

# タグを設定して実行をトラッキング
run_options.run_tag = "inference_batch_1"

# 推論の実行
outputs = session.run(None, inputs, run_options)
```

### 1.4 実行プロバイダー関連API

#### プロバイダー管理

```python
import onnxruntime as ort

# 利用可能なプロバイダーの確認
available_providers = ort.get_available_providers()
print(f"利用可能なプロバイダー: {available_providers}")

# デバイス情報の取得
if 'CUDAExecutionProvider' in available_providers:
    print("CUDA対応デバイスが利用可能です")
    # GPU情報の取得（利用可能な場合）
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorchがインストールされていないため、詳細なGPU情報を取得できません")
```

#### モデルメタデータの管理

```python
# モデルのメタデータを取得
def print_model_info(session):
    """モデルの詳細情報を表示"""
    print("=== モデル情報 ===")
    
    # 基本情報
    try:
        metadata = session.get_modelmeta()
        print(f"プロデューサー: {metadata.producer_name}")
        print(f"バージョン: {metadata.version}")
        print(f"説明: {metadata.description}")
    except:
        print("メタデータの取得に失敗しました")
    
    # 入力情報
    print("\n--- 入力情報 ---")
    for i, input_meta in enumerate(session.get_inputs()):
        print(f"入力 {i+1}:")
        print(f"  名前: {input_meta.name}")
        print(f"  形状: {input_meta.shape}")
        print(f"  型: {input_meta.type}")
    
    # 出力情報
    print("\n--- 出力情報 ---")
    for i, output_meta in enumerate(session.get_outputs()):
        print(f"出力 {i+1}:")
        print(f"  名前: {output_meta.name}")
        print(f"  形状: {output_meta.shape}")
        print(f"  型: {output_meta.type}")

# 使用例
session = ort.InferenceSession("model.onnx")
print_model_info(session)
```

## まとめ

この章では、ONNX Runtimeの基礎から実践的な使用方法までを概観しました。

### 🎯 学習した内容

1. **ONNX Runtimeの概要**
   - 異なるフレームワーク間でのモデル共有を実現する標準フォーマット
   - クロスプラットフォーム対応の高性能推論エンジン
   - CPU、GPU、専用アクセラレータでの最適化実行

2. **環境構築とインストール**
   - Python（pip）、C#（NuGet）、C++（ソースビルド）での導入方法
   - GPU対応の設定とCUDA環境の要件
   - インストール確認とトラブルシューティング

3. **基本的な推論実行**
   - モデル読み込みからデータ準備、推論実行、結果取得までの流れ
   - 動的形状やデータ型の適切な処理
   - エラーハンドリングとデバッグ手法

4. **高度なAPI活用**
   - セッションオプションによる性能チューニング
   - 実行プロバイダーの選択と最適化
   - プロファイリングと性能測定

### 💡 実践のポイント

- **開発段階**: Python APIで迅速なプロトタイピング
- **本番運用**: C++やC# APIで高性能システム構築
- **最適化**: 適切なプロバイダー選択とオプション設定
- **デバッグ**: 詳細なログとプロファイリングの活用

### 🚀 次のステップ

次章では、さらに高度なONNX Runtimeの機能を扱います。
- カスタムオペレーターの作成
- マルチGPU環境での分散推論
- モデルの量子化と最適化
- 実際のアプリケーション開発事例

ONNX Runtimeは、研究から本番システムまで幅広く活用できる強力なツールです。この基礎知識を土台として、次章でより実践的なアプリケーション開発技術を習得していきましょう。

---

> **📚 追加学習リソース**
> - [ONNX Runtime公式ドキュメント](https://onnxruntime.ai/docs/)
> - [ONNXモデルズー](https://github.com/onnx/models)
> - [コミュニティサンプル](https://github.com/microsoft/onnxruntime/tree/main/samples)
