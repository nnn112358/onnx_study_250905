# 第2章 ONNX Runtimeとアプリケーション開発

> 本章の概要: ONNX Runtimeの実行プロバイダー、ONNXの基本原理、Python連携、演算子属性など、開発に必要な要点を実践コードとともに解説します。

前章でONNX Runtimeの基本を学んだので、ここでは内部アーキテクチャと実践的な開発手法を掘り下げます。

## 2.1 実行プロバイダー（Execution Providers）

ONNX Runtimeの特徴のひとつが**実行プロバイダー**です。同一のONNXモデルを、さまざまなハードウェア上で最適化して実行するためのプラグイン機構です。

### 2.1.1 実行プロバイダーの概要

実行プロバイダーは、ONNX Runtimeのコアを構成する重要コンポーネントです。各プロバイダーは特定のハードウェア/ソフトウェアに最適化された推論エンジンを提供します。

**実行プロバイダーの主な役割:**
- ハードウェア固有の最適化の適用
- メモリ管理の効率化
- 並列処理の最適化
- 演算子の高速実装の提供

**プロバイダー選択の原理:**
ONNX Runtimeは、指定順に各演算子へ最適なプロバイダーを割り当てます。未対応の演算子は次点へ自動フォールバックします。

#### 主要な実行プロバイダー

システムに利用可能な実行プロバイダーを確認し、適切に設定することが高性能な推論の第一歩です。

```python
import onnxruntime as ort
import sys
import platform

def analyze_available_providers():
    """システムで利用可能なプロバイダーを詳細に分析"""
    
    print("=== ONNX Runtime 環境分析 ===")
    print(f"ONNX Runtime バージョン: {ort.__version__}")
    print(f"Pythonバージョン: {sys.version}")
    print(f"プラットフォーム: {platform.platform()}")
    
    providers = ort.get_available_providers()
    print(f"\n利用可能なプロバイダー数: {len(providers)}")
    
    # プロバイダーの詳細分析
    for i, provider in enumerate(providers, 1):
        print(f"\n{i}. {provider}")
        
        # プロバイダーの特性を解説
        if provider == "CPUExecutionProvider":
            print("   🖥️  CPU上での汎用実行（常に利用可能）")
            print("   💡 特徴: クロスプラットフォーム、安定性重視")
        elif provider == "CUDAExecutionProvider":
            print("   🚀 NVIDIA GPU上での高速実行")
            print("   💡 特徴: 深層学習に最適、高いスループット")
        elif provider == "DmlExecutionProvider":
            print("   🎮 DirectML（Windows GPU）での実行")
            print("   💡 特徴: WindowsのGPU抽象化レイヤー")
        elif provider == "TensorrtExecutionProvider":
            print("   ⚡ NVIDIA TensorRTでの最適化実行")
            print("   💡 特徴: 本番環境での最高性能")
        elif provider == "OpenVINOExecutionProvider":
            print("   🔧 Intel OpenVINOでの最適化実行")
            print("   💡 特徴: Intel CPU/GPU/VPUでの高速化")

# プロバイダー別のセッション作成関数（エラーハンドリング強化版）
def create_session_with_provider(model_path, provider_name, provider_options=None):
    """指定されたプロバイダーでセッションを作成"""
    try:
        if provider_options:
            # プロバイダー固有のオプション付きセッション作成
            providers = [(provider_name, provider_options)]
        else:
            providers = [provider_name]
            
        session = ort.InferenceSession(model_path, providers=providers)
        
        # セッション作成成功時の詳細情報
        print(f"✅ {provider_name}でセッション作成成功")
        print(f"   実際に使用されたプロバイダー: {session.get_providers()}")
        
        return session
        
    except Exception as e:
        print(f"❌ {provider_name}でのセッション作成失敗")
        print(f"   エラー詳細: {type(e).__name__}: {e}")
        return None

# 使用例とテスト
def test_providers(model_path="model.onnx"):
    """各プロバイダーでのセッション作成をテスト"""
    
    print("\n=== プロバイダーテスト ===")
    
    # 利用可能なプロバイダーを分析
    analyze_available_providers()
    
    available_providers = ort.get_available_providers()
    sessions = {}
    
    # 各プロバイダーでセッション作成を試行
    for provider in available_providers:
        print(f"\n🔬 {provider} をテスト中...")
        session = create_session_with_provider(model_path, provider)
        if session:
            sessions[provider] = session
    
    return sessions

# 実行
if __name__ == "__main__":
    # 注意: 実際のONNXモデルファイルが必要です
    # test_providers("your_model.onnx")
    analyze_available_providers()
```

**プロバイダー選択の実践的ガイドライン:**

1. **開発・テスト段階**: 
   - `CPUExecutionProvider`を使用（確実に動作）
   
2. **高性能が必要な場合**: 
   - GPUが利用可能なら`CUDAExecutionProvider`を最優先
   - フォールバック用に`CPUExecutionProvider`を併記

3. **本番環境での最適化**: 
   - `TensorrtExecutionProvider`（NVIDIA GPU）
   - `OpenVINOExecutionProvider`（Intel環境）

### 2.1.2 プロバイダー詳細

各実行プロバイダーには、それぞれ固有の特性と最適化オプションがあります。適切な設定により、推論性能を大幅に向上させることができます。

#### 1. CPUExecutionProvider

CPUExecutionProviderは、最も汎用性が高く、すべての環境で利用可能なプロバイダーです。マルチコアCPUの性能を最大限活用するための並列処理最適化を提供します。

**主な機能:**
- **Intel MKL-DNN統合**: Intel CPUでの高速化
- **OpenMP並列処理**: マルチスレッド実行の最適化
- **SIMD命令活用**: ベクトル演算の高速化
- **メモリレイアウト最適化**: キャッシュ効率の向上

```python
# CPU実行プロバイダーの設定例
cpu_options = {
    'intra_op_num_threads': 8,  # オペレーター内スレッド数
    'inter_op_num_threads': 4,  # オペレーター間スレッド数
}

session = ort.InferenceSession(
    "model.onnx",
    providers=[('CPUExecutionProvider', cpu_options)]
)
```

#### 2. CUDAExecutionProvider
NVIDIA GPU での高速処理：

```python
# CUDA実行プロバイダーの設定例
cuda_options = {
    'device_id': 0,                    # 使用するGPU ID
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # GPU メモリ制限（2GB）
    'cudnn_conv_algo_search': 'EXHAUSTIVE',    # 最適なconv算法を検索
}

session = ort.InferenceSession(
    "model.onnx",
    providers=[('CUDAExecutionProvider', cuda_options)]
)
```

#### 3. その他のプロバイダー

```python
# DirectML（DirectX 12）プロバイダー - Windows
dml_options = {'device_id': 0}
dml_session = ort.InferenceSession(
    "model.onnx",
    providers=[('DmlExecutionProvider', dml_options)]
)

# TensorRT プロバイダー - NVIDIA GPU最適化
trt_options = {
    'device_id': 0,
    'trt_max_workspace_size': 1073741824,  # 1GB
    'trt_fp16_enable': True,               # FP16精度を有効化
}
```

### 2.1.3 プロバイダーの追加

#### カスタムオペレーターライブラリの読み込み

```python
import onnxruntime as ort

# カスタムオペレーターライブラリの読み込み
session_options = ort.SessionOptions()
session_options.register_custom_ops_library("/path/to/custom_ops.so")

session = ort.InferenceSession("model_with_custom_ops.onnx", session_options)
```

#### プロバイダー優先順位の設定

```python
# プロバイダーの優先順位を設定
providers = [
    'TensorrtExecutionProvider',  # 最高優先度
    'CUDAExecutionProvider',      # 次の優先度
    'CPUExecutionProvider'        # フォールバック
]

session = ort.InferenceSession("model.onnx", providers=providers)
```

## 2.2　ONNX原理の紹介

### 2.2.1　ONNX基本概念

ONNXは、機械学習モデルの標準化された表現形式です。主要な概念を理解しましょう。

#### ONNXモデルの構造

```python
import onnx

# ONNXモデルの読み込み
model = onnx.load("model.onnx")

# モデルの基本情報
print(f"IR バージョン: {model.ir_version}")
print(f"プロデューサー名: {model.producer_name}")
print(f"プロデューサーバージョン: {model.producer_version}")

# グラフの情報
graph = model.graph
print(f"ノード数: {len(graph.node)}")
print(f"入力数: {len(graph.input)}")
print(f"出力数: {len(graph.output)}")
print(f"初期化子数: {len(graph.initializer)}")
```

#### モデルの可視化

```python
import onnx
from onnx import helper, numpy_helper

def print_model_structure(model_path):
    """ONNXモデルの構造を表示"""
    model = onnx.load(model_path)
    
    print("=== ONNX モデル構造 ===")
    
    # 入力情報
    print("\n--- 入力 ---")
    for input_tensor in model.graph.input:
        print(f"名前: {input_tensor.name}")
        print(f"データ型: {input_tensor.type.tensor_type.elem_type}")
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"形状: {shape}")
    
    # 出力情報
    print("\n--- 出力 ---")
    for output_tensor in model.graph.output:
        print(f"名前: {output_tensor.name}")
        print(f"データ型: {output_tensor.type.tensor_type.elem_type}")
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"形状: {shape}")
    
    # オペレーター情報
    print("\n--- オペレーター ---")
    op_types = {}
    for node in model.graph.node:
        op_type = node.op_type
        op_types[op_type] = op_types.get(op_type, 0) + 1
    
    for op_type, count in sorted(op_types.items()):
        print(f"{op_type}: {count}個")

# 使用例
print_model_structure("model.onnx")
```

### 2.2.2　ONNXの入力、出力、ノード、初期化子、属性

#### ノード（Node）の詳細分析

```python
import onnx

def analyze_nodes(model_path):
    """ノードの詳細を分析"""
    model = onnx.load(model_path)
    
    print("=== ノード分析 ===")
    for i, node in enumerate(model.graph.node):
        print(f"\nノード {i+1}:")
        print(f"  名前: {node.name}")
        print(f"  オペレータータイプ: {node.op_type}")
        print(f"  入力: {list(node.input)}")
        print(f"  出力: {list(node.output)}")
        
        # 属性の表示
        if node.attribute:
            print(f"  属性:")
            for attr in node.attribute:
                print(f"    {attr.name}: {helper.get_attribute_value(attr)}")

# 初期化子（Initializer）の分析
def analyze_initializers(model_path):
    """初期化子の詳細を分析"""
    model = onnx.load(model_path)
    
    print("=== 初期化子分析 ===")
    for initializer in model.graph.initializer:
        print(f"\n初期化子: {initializer.name}")
        print(f"  データ型: {initializer.data_type}")
        print(f"  形状: {list(initializer.dims)}")
        
        # 実際のデータを取得（小さなテンソルの場合のみ）
        if numpy_helper.to_array(initializer).size <= 10:
            data = numpy_helper.to_array(initializer)
            print(f"  データ: {data.flatten()}")

# 使用例
analyze_nodes("model.onnx")
analyze_initializers("model.onnx")
```

### 2.2.3　要素タイプ

ONNXでサポートされるデータタイプの詳細：

```python
from onnx import TensorProto

# ONNXデータタイプの一覧
data_types = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.STRING: "string",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
}

def check_model_data_types(model_path):
    """モデルで使用されているデータタイプを確認"""
    model = onnx.load(model_path)
    used_types = set()
    
    # 入力のデータタイプ
    for input_tensor in model.graph.input:
        elem_type = input_tensor.type.tensor_type.elem_type
        used_types.add(elem_type)
    
    # 初期化子のデータタイプ
    for initializer in model.graph.initializer:
        used_types.add(initializer.data_type)
    
    print("使用されているデータタイプ:")
    for type_id in used_types:
        type_name = data_types.get(type_id, f"Unknown({type_id})")
        print(f"  {type_name}")

check_model_data_types("model.onnx")
```

### 2.2.4　opsetバージョンとは？

ONNXのオペレータセット（opset）バージョン管理：

```python
import onnx

def check_opset_version(model_path):
    """モデルのopsetバージョンを確認"""
    model = onnx.load(model_path)
    
    print("=== Opsetバージョン情報 ===")
    for opset_import in model.opset_import:
        domain = opset_import.domain if opset_import.domain else "default"
        version = opset_import.version
        print(f"ドメイン: {domain}, バージョン: {version}")
    
    return model

# opsetバージョンの変換
def convert_opset_version(input_model_path, output_model_path, target_version):
    """opsetバージョンを変換"""
    try:
        from onnx import version_converter
        
        original_model = onnx.load(input_model_path)
        converted_model = version_converter.convert_version(original_model, target_version)
        onnx.save(converted_model, output_model_path)
        
        print(f"Opsetバージョンを{target_version}に変換しました")
    except Exception as e:
        print(f"変換エラー: {e}")

# 使用例
check_opset_version("model.onnx")
convert_opset_version("model.onnx", "model_v11.onnx", 11)
```

### 2.2.5　サブグラフ、テスト、ループ

#### 制御フロー演算子の理解

```python
import onnx
from onnx import helper

def analyze_control_flow(model_path):
    """制御フロー演算子を分析"""
    model = onnx.load(model_path)
    
    control_flow_ops = ['If', 'Loop', 'Scan']
    
    print("=== 制御フロー分析 ===")
    for node in model.graph.node:
        if node.op_type in control_flow_ops:
            print(f"\n制御フロー演算子発見: {node.op_type}")
            print(f"ノード名: {node.name}")
            
            # サブグラフの分析
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    print(f"サブグラフ属性: {attr.name}")
                    subgraph = attr.g
                    print(f"  サブグラフノード数: {len(subgraph.node)}")

# カスタムループ演算子の例
def create_loop_example():
    """ループ演算子の実装例"""
    # ループ条件のグラフ
    cond_graph = helper.make_graph(
        nodes=[],
        name="condition",
        inputs=[helper.make_tensor_value_info("i", TensorProto.INT64, [1])],
        outputs=[helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])]
    )
    
    # ループ本体のグラフ
    body_graph = helper.make_graph(
        nodes=[],
        name="body",
        inputs=[
            helper.make_tensor_value_info("i", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1])
        ],
        outputs=[
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, [1])
        ]
    )
    
    print("ループ演算子の構造を作成しました")

create_loop_example()
```

### 2.2.6　演算子スキャン

Scan演算子の詳細使用例：

```python
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_scan_example():
    """Scan演算子の使用例"""
    
    # RNNライクな操作をScanで実装
    # 各時間ステップで同じ変換を適用
    
    # スキャンボディのグラフを作成
    scan_body = helper.make_graph(
        nodes=[
            # 前の隠れ状態と入力を加算
            helper.make_node(
                "Add",
                inputs=["prev_h", "x_t"],
                outputs=["h_t"],
                name="add_node"
            )
        ],
        name="scan_body",
        inputs=[
            helper.make_tensor_value_info("prev_h", TensorProto.FLOAT, [4]),  # 前の状態
            helper.make_tensor_value_info("x_t", TensorProto.FLOAT, [4])      # 現在の入力
        ],
        outputs=[
            helper.make_tensor_value_info("h_t", TensorProto.FLOAT, [4])      # 新しい状態
        ]
    )
    
    print("Scan演算子のボディグラフを作成しました")
    return scan_body

scan_body = create_scan_example()
```

### 2.2.7　ツール

#### ONNX開発ツールセット

```python
import onnx
from onnx import checker, helper, shape_inference

def onnx_tools_demo(model_path):
    """ONNXツールの使用例"""
    
    # 1. モデルの読み込み
    model = onnx.load(model_path)
    
    # 2. モデルの検証
    try:
        checker.check_model(model)
        print("✓ モデルは有効です")
    except Exception as e:
        print(f"✗ モデルエラー: {e}")
    
    # 3. 形状推論
    try:
        inferred_model = shape_inference.infer_shapes(model)
        print("✓ 形状推論が完了しました")
        
        # 中間テンソルの形状を表示
        print("中間テンソルの形状:")
        for value_info in inferred_model.graph.value_info:
            print(f"  {value_info.name}: {[d.dim_value for d in value_info.type.tensor_type.shape.dim]}")
            
    except Exception as e:
        print(f"✗ 形状推論エラー: {e}")
    
    # 4. モデルの最適化（簡単な例）
    try:
        from onnx import optimizer
        
        # 利用可能な最適化パスを確認
        available_passes = optimizer.get_available_passes()
        print(f"利用可能な最適化パス: {available_passes}")
        
        # 基本的な最適化を適用
        optimized_model = optimizer.optimize(model, ['eliminate_identity'])
        print("✓ モデルの最適化が完了しました")
        
    except ImportError:
        print("最適化ツールが利用できません")
    except Exception as e:
        print(f"最適化エラー: {e}")

# 使用例
onnx_tools_demo("model.onnx")
```

## 2.3　ONNXとPython

### 2.3.1　線形回帰例

シンプルな線形回帰モデルをONNXで作成：

```python
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

def create_linear_regression_onnx():
    """線形回帰モデルをONNXで作成"""
    
    # モデルパラメータ（重みとバイアス）
    weight = np.array([[2.0, 3.0]], dtype=np.float32)  # 1x2の重み行列
    bias = np.array([1.0], dtype=np.float32)           # バイアス
    
    # 重みとバイアスの初期化子を作成
    weight_initializer = numpy_helper.from_array(weight, name="weight")
    bias_initializer = numpy_helper.from_array(bias, name="bias")
    
    # グラフのノードを作成
    # y = X * W^T + b
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["X", "weight"],
        outputs=["matmul_result"],
        name="matmul_node"
    )
    
    add_node = helper.make_node(
        "Add",
        inputs=["matmul_result", "bias"],
        outputs=["Y"],
        name="add_node"
    )
    
    # グラフを作成
    graph = helper.make_graph(
        nodes=[matmul_node, add_node],
        name="LinearRegression",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])],
        initializer=[weight_initializer, bias_initializer]
    )
    
    # モデルを作成
    model = helper.make_model(graph, producer_name="linear_regression_example")
    
    # モデルを保存
    onnx.save(model, "linear_regression.onnx")
    print("線形回帰モデルを作成しました: linear_regression.onnx")
    
    return model

def test_linear_regression():
    """作成した線形回帰モデルをテスト"""
    
    # モデルを作成
    model = create_linear_regression_onnx()
    
    # ONNX Runtimeで推論を実行
    session = ort.InferenceSession("linear_regression.onnx")
    
    # テストデータ
    X_test = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    
    # 推論実行
    outputs = session.run(None, {"X": X_test})
    predictions = outputs[0]
    
    print("入力データ:")
    print(X_test)
    print("予測結果:")
    print(predictions)
    
    # 手動計算での検証
    weight = np.array([[2.0, 3.0]], dtype=np.float32)
    bias = np.array([1.0], dtype=np.float32)
    manual_pred = np.dot(X_test, weight.T) + bias
    
    print("手動計算結果:")
    print(manual_pred)
    
    # 結果の比較
    if np.allclose(predictions, manual_pred):
        print("✓ 結果が一致しました")
    else:
        print("✗ 結果が一致しません")

# 実行
test_linear_regression()
```

### 2.3.2　初期化子、改良された線形計画法

より複雑な線形モデルと初期化子の活用：

```python
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

def create_advanced_linear_model():
    """改良された線形モデル（正規化、活性化関数付き）"""
    
    # パラメータの定義
    input_dim = 4
    hidden_dim = 8
    output_dim = 1
    
    # 重みとバイアスの初期化（Xavier初期化）
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim, dtype=np.float32)
    
    # バッチ正規化パラメータ
    scale = np.ones(hidden_dim, dtype=np.float32)
    bias_bn = np.zeros(hidden_dim, dtype=np.float32)
    mean = np.zeros(hidden_dim, dtype=np.float32)
    var = np.ones(hidden_dim, dtype=np.float32)
    
    # 初期化子の作成
    initializers = [
        numpy_helper.from_array(W1, name="W1"),
        numpy_helper.from_array(b1, name="b1"),
        numpy_helper.from_array(W2, name="W2"),
        numpy_helper.from_array(b2, name="b2"),
        numpy_helper.from_array(scale, name="scale"),
        numpy_helper.from_array(bias_bn, name="bias_bn"),
        numpy_helper.from_array(mean, name="mean"),
        numpy_helper.from_array(var, name="var"),
    ]
    
    # ノードの作成
    nodes = [
        # 第1層: X @ W1 + b1
        helper.make_node("MatMul", ["X", "W1"], ["hidden1"], name="matmul1"),
        helper.make_node("Add", ["hidden1", "b1"], ["hidden1_bias"], name="add1"),
        
        # バッチ正規化
        helper.make_node(
            "BatchNormalization",
            ["hidden1_bias", "scale", "bias_bn", "mean", "var"],
            ["bn_output"],
            epsilon=1e-5,
            name="batch_norm"
        ),
        
        # ReLU活性化
        helper.make_node("Relu", ["bn_output"], ["relu_output"], name="relu"),
        
        # 第2層: relu_output @ W2 + b2
        helper.make_node("MatMul", ["relu_output", "W2"], ["output_matmul"], name="matmul2"),
        helper.make_node("Add", ["output_matmul", "b2"], ["Y"], name="add2"),
    ]
    
    # グラフの作成
    graph = helper.make_graph(
        nodes=nodes,
        name="AdvancedLinearModel",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, input_dim])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, output_dim])],
        initializer=initializers
    )
    
    # モデルの作成と保存
    model = helper.make_model(graph, producer_name="advanced_linear_model")
    onnx.save(model, "advanced_linear_model.onnx")
    
    print("改良された線形モデルを作成しました: advanced_linear_model.onnx")
    return model

def test_advanced_model():
    """改良されたモデルのテスト"""
    
    # モデル作成
    model = create_advanced_linear_model()
    
    # モデル検証
    onnx.checker.check_model(model)
    print("✓ モデルの検証が完了しました")
    
    # 推論テスト
    session = ort.InferenceSession("advanced_linear_model.onnx")
    
    # テストデータ生成
    batch_size = 5
    input_dim = 4
    X_test = np.random.randn(batch_size, input_dim).astype(np.float32)
    
    # 推論実行
    outputs = session.run(None, {"X": X_test})
    predictions = outputs[0]
    
    print(f"入力形状: {X_test.shape}")
    print(f"出力形状: {predictions.shape}")
    print(f"出力サンプル: {predictions[:3].flatten()}")

# 実行
test_advanced_model()
```

### 2.3.3　ONNX構造の走査と初期化子のチェック

```python
import onnx
from onnx import numpy_helper
import numpy as np

def traverse_onnx_structure(model_path):
    """ONNX構造を詳細に走査"""
    
    model = onnx.load(model_path)
    
    print("=== ONNX構造の詳細走査 ===")
    
    # 1. 基本情報
    print(f"IRバージョン: {model.ir_version}")
    print(f"OpSetバージョン: {[f'{op.domain}:{op.version}' for op in model.opset_import]}")
    
    # 2. グラフ情報
    graph = model.graph
    print(f"\nグラフ名: {graph.name}")
    
    # 3. 入力の詳細分析
    print(f"\n--- 入力分析 ---")
    for i, input_tensor in enumerate(graph.input):
        print(f"入力{i+1}: {input_tensor.name}")
        tensor_type = input_tensor.type.tensor_type
        print(f"  データ型: {tensor_type.elem_type}")
        
        shape_info = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value:
                shape_info.append(str(dim.dim_value))
            elif dim.dim_param:
                shape_info.append(dim.dim_param)
            else:
                shape_info.append("?")
        print(f"  形状: [{', '.join(shape_info)}]")
    
    # 4. 初期化子の詳細分析
    print(f"\n--- 初期化子分析 ---")
    total_params = 0
    
    for i, initializer in enumerate(graph.initializer):
        print(f"\n初期化子{i+1}: {initializer.name}")
        print(f"  データ型: {initializer.data_type}")
        print(f"  形状: {list(initializer.dims)}")
        
        # パラメータ数の計算
        param_count = np.prod(initializer.dims)
        total_params += param_count
        print(f"  パラメータ数: {param_count}")
        
        # 統計情報（小さなテンソルの場合）
        if param_count <= 1000:
            tensor_data = numpy_helper.to_array(initializer)
            print(f"  最小値: {tensor_data.min():.6f}")
            print(f"  最大値: {tensor_data.max():.6f}")
            print(f"  平均値: {tensor_data.mean():.6f}")
            print(f"  標準偏差: {tensor_data.std():.6f}")
        else:
            print(f"  (統計情報スキップ: テンソルが大きすぎます)")
    
    print(f"\n総パラメータ数: {total_params}")
    
    # 5. ノードの詳細分析
    print(f"\n--- ノード分析 ---")
    op_count = {}
    
    for i, node in enumerate(graph.node):
        op_type = node.op_type
        op_count[op_type] = op_count.get(op_type, 0) + 1
        
        print(f"\nノード{i+1}: {node.name}")
        print(f"  演算子タイプ: {op_type}")
        print(f"  入力: {list(node.input)}")
        print(f"  出力: {list(node.output)}")
        
        # 属性の表示
        if node.attribute:
            print(f"  属性:")
            for attr in node.attribute:
                attr_value = onnx.helper.get_attribute_value(attr)
                if isinstance(attr_value, bytes):
                    print(f"    {attr.name}: <バイナリデータ>")
                elif isinstance(attr_value, onnx.GraphProto):
                    print(f"    {attr.name}: <サブグラフ>")
                else:
                    print(f"    {attr.name}: {attr_value}")
    
    # 6. 演算子の統計
    print(f"\n--- 演算子統計 ---")
    for op_type, count in sorted(op_count.items()):
        print(f"{op_type}: {count}個")
    
    # 7. 出力の詳細分析
    print(f"\n--- 出力分析 ---")
    for i, output_tensor in enumerate(graph.output):
        print(f"出力{i+1}: {output_tensor.name}")
        if output_tensor.type.tensor_type:
            tensor_type = output_tensor.type.tensor_type
            print(f"  データ型: {tensor_type.elem_type}")
            
            shape_info = []
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    shape_info.append(str(dim.dim_value))
                elif dim.dim_param:
                    shape_info.append(dim.dim_param)
                else:
                    shape_info.append("?")
            print(f"  形状: [{', '.join(shape_info)}]")

def check_initializer_health(model_path):
    """初期化子の健全性をチェック"""
    
    model = onnx.load(model_path)
    
    print("=== 初期化子健全性チェック ===")
    
    issues = []
    
    for initializer in model.graph.initializer:
        tensor_data = numpy_helper.to_array(initializer)
        
        # NaN値のチェック
        if np.isnan(tensor_data).any():
            issues.append(f"{initializer.name}: NaN値が含まれています")
        
        # 無限大値のチェック
        if np.isinf(tensor_data).any():
            issues.append(f"{initializer.name}: 無限大値が含まれています")
        
        # 異常に大きな値のチェック
        if np.abs(tensor_data).max() > 1000:
            issues.append(f"{initializer.name}: 異常に大きな値があります (max: {np.abs(tensor_data).max()})")
        
        # 全て0の重みのチェック
        if np.all(tensor_data == 0):
            issues.append(f"{initializer.name}: 全て0の重みです")
    
    if issues:
        print("発見された問題:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✓ 初期化子に問題は見つかりませんでした")

# 使用例
if __name__ == "__main__":
    # 先ほど作成したモデルを分析
    traverse_onnx_structure("advanced_linear_model.onnx")
    check_initializer_health("advanced_linear_model.onnx")
```

## 2.4　演算子属性

ONNXの演算子属性について詳しく学習します：

```python
import onnx
from onnx import helper, TensorProto, AttributeProto
import numpy as np

def demonstrate_operator_attributes():
    """さまざまな演算子属性のデモンストレーション"""
    
    print("=== 演算子属性のデモ ===")
    
    # 1. Conv演算子の属性
    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[3, 3],      # カーネルサイズ
        pads=[1, 1, 1, 1],       # パディング [上, 左, 下, 右]
        strides=[1, 1],          # ストライド
        dilations=[1, 1],        # 膨張
        group=1,                 # グループ畳み込み
        name="conv_example"
    )
    
    print("Convolution演算子の属性:")
    for attr in conv_node.attribute:
        print(f"  {attr.name}: {helper.get_attribute_value(attr)}")
    
    # 2. BatchNormalization演算子の属性
    bn_node = helper.make_node(
        'BatchNormalization',
        inputs=['X', 'scale', 'B', 'mean', 'var'],
        outputs=['Y'],
        epsilon=1e-5,            # 数値安定性のための小さな値
        momentum=0.9,            # 移動平均の重み
        name="batch_norm_example"
    )
    
    print("\nBatchNormalization演算子の属性:")
    for attr in bn_node.attribute:
        print(f"  {attr.name}: {helper.get_attribute_value(attr)}")
    
    # 3. Reshape演算子の属性
    reshape_node = helper.make_node(
        'Reshape',
        inputs=['X', 'shape'],
        outputs=['Y'],
        allowzero=0,             # 0による形状指定の許可
        name="reshape_example"
    )
    
    print("\nReshape演算子の属性:")
    for attr in reshape_node.attribute:
        print(f"  {attr.name}: {helper.get_attribute_value(attr)}")
    
    return [conv_node, bn_node, reshape_node]

def create_conv_model_with_attributes():
    """属性を詳細に設定した畳み込みモデル"""
    
    # 入力: [N, C, H, W] = [1, 3, 32, 32]
    # カーネル: [16, 3, 3, 3] (16個の3x3カーネル)
    
    # 重みの初期化
    kernel = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1
    bias = np.zeros(16, dtype=np.float32)
    
    # 初期化子の作成
    kernel_initializer = helper.make_tensor(
        name="conv_kernel",
        data_type=TensorProto.FLOAT,
        dims=[16, 3, 3, 3],
        vals=kernel.flatten()
    )
    
    bias_initializer = helper.make_tensor(
        name="conv_bias",
        data_type=TensorProto.FLOAT,
        dims=[16],
        vals=bias.flatten()
    )
    
    # ノードの作成（詳細な属性設定）
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'conv_kernel', 'conv_bias'],
        outputs=['conv_output'],
        kernel_shape=[3, 3],         # 3x3カーネル
        pads=[1, 1, 1, 1],          # 同じサイズ出力のためのパディング
        strides=[1, 1],             # ストライド1
        dilations=[1, 1],           # 通常の畳み込み
        group=1,                    # 通常の畳み込み（グループ化なし）
        name="detailed_conv"
    )
    
    # ReLU活性化
    relu_node = helper.make_node(
        'Relu',
        inputs=['conv_output'],
        outputs=['output'],
        name="relu_activation"
    )
    
    # グラフの作成
    graph = helper.make_graph(
        nodes=[conv_node, relu_node],
        name="ConvModelWithAttributes",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 32, 32])],
        initializer=[kernel_initializer, bias_initializer]
    )
    
    # モデルの作成
    model = helper.make_model(graph, producer_name="conv_attributes_demo")
    
    # モデルの保存
    onnx.save(model, "conv_model_with_attributes.onnx")
    print("属性詳細設定済み畳み込みモデルを作成しました")
    
    return model

def analyze_model_attributes(model_path):
    """モデルの全演算子属性を分析"""
    
    model = onnx.load(model_path)
    
    print(f"=== {model_path} の属性分析 ===")
    
    for i, node in enumerate(model.graph.node):
        print(f"\nノード {i+1}: {node.name} ({node.op_type})")
        
        if node.attribute:
            print("  属性:")
            for attr in node.attribute:
                attr_value = helper.get_attribute_value(attr)
                
                # 属性タイプごとの処理
                if attr.type == AttributeProto.INT:
                    print(f"    {attr.name} (INT): {attr_value}")
                elif attr.type == AttributeProto.INTS:
                    print(f"    {attr.name} (INTS): {list(attr_value)}")
                elif attr.type == AttributeProto.FLOAT:
                    print(f"    {attr.name} (FLOAT): {attr_value}")
                elif attr.type == AttributeProto.FLOATS:
                    print(f"    {attr.name} (FLOATS): {list(attr_value)}")
                elif attr.type == AttributeProto.STRING:
                    print(f"    {attr.name} (STRING): {attr_value.decode()}")
                elif attr.type == AttributeProto.TENSOR:
                    print(f"    {attr.name} (TENSOR): 形状{attr_value.dims}")
                elif attr.type == AttributeProto.GRAPH:
                    print(f"    {attr.name} (GRAPH): ノード数{len(attr_value.node)}")
                else:
                    print(f"    {attr.name}: {attr_value}")
        else:
            print("  属性: なし")

# デモの実行
if __name__ == "__main__":
    # 基本的な属性デモ
    demonstrate_operator_attributes()
    
    # 詳細な畳み込みモデルの作成
    create_conv_model_with_attributes()
    
    # 属性の分析
    analyze_model_attributes("conv_model_with_attributes.onnx")
```

## 2.5　実践的な開発パターンとベストプラクティス

### 2.5.1　エラーハンドリングとデバッグ

実際の開発では、さまざまなエラーに遭遇します。効果的なエラーハンドリング戦略を整理します。

```python
import onnxruntime as ort
import numpy as np
import logging
from typing import Dict, Any, Optional

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ONNXInferenceEngine:
    """堅牢なONNX推論エンジンクラス"""
    
    def __init__(self, model_path: str, providers: Optional[list] = None):
        """
        ONNXモデルの初期化
        
        Args:
            model_path: ONNXモデルファイルパス
            providers: 使用する実行プロバイダー
        """
        self.model_path = model_path
        self.session = None
        self.input_names = []
        self.output_names = []
        
        # プロバイダーのデフォルト設定
        if providers is None:
            providers = self._get_optimal_providers()
        
        try:
            self._initialize_session(providers)
            self._analyze_model_io()
            logger.info(f"モデル初期化成功: {model_path}")
        except Exception as e:
            logger.error(f"モデル初期化失敗: {e}")
            raise
    
    def _get_optimal_providers(self) -> list:
        """最適なプロバイダーを自動選択"""
        available = ort.get_available_providers()
        
        # 優先順位に基づいた選択
        preferred_order = [
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider', 
            'OpenVINOExecutionProvider',
            'DmlExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        selected = []
        for provider in preferred_order:
            if provider in available:
                selected.append(provider)
        
        logger.info(f"選択されたプロバイダー: {selected}")
        return selected
    
    def _initialize_session(self, providers: list):
        """セッション初期化"""
        session_options = ort.SessionOptions()
        
        # 最適化設定
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        
        # ログ設定
        session_options.log_severity_level = 3  # ERROR以上のみ
        
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                session_options,
                providers=providers
            )
        except Exception as e:
            # プロバイダー関連エラーの場合、CPUにフォールバック
            if any(keyword in str(e).lower() for keyword in ['provider', 'cuda', 'gpu']):
                logger.warning(f"プロバイダーエラー、CPUにフォールバック: {e}")
                self.session = ort.InferenceSession(
                    self.model_path,
                    session_options,
                    providers=['CPUExecutionProvider']
                )
            else:
                raise
    
    def _analyze_model_io(self):
        """モデルの入出力を分析"""
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 入力詳細の記録
        for input_meta in self.session.get_inputs():
            logger.info(f"入力 '{input_meta.name}': 形状={input_meta.shape}, 型={input_meta.type}")
        
        # 出力詳細の記録
        for output_meta in self.session.get_outputs():
            logger.info(f"出力 '{output_meta.name}': 形状={output_meta.shape}, 型={output_meta.type}")
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        予測実行（エラーハンドリング付き）
        
        Args:
            inputs: 入力データの辞書
            
        Returns:
            出力データの辞書
        """
        try:
            # 入力検証
            self._validate_inputs(inputs)
            
            # 推論実行
            outputs = self.session.run(self.output_names, inputs)
            
            # 出力を辞書形式で返す
            return {name: output for name, output in zip(self.output_names, outputs)}
            
        except Exception as e:
            logger.error(f"推論エラー: {e}")
            raise
    
    def _validate_inputs(self, inputs: Dict[str, np.ndarray]):
        """入力データの検証"""
        # 必要な入力がすべて提供されているか確認
        for required_input in self.input_names:
            if required_input not in inputs:
                raise ValueError(f"必要な入力が不足: {required_input}")
        
        # データ型と形状の検証
        for input_meta in self.session.get_inputs():
            input_name = input_meta.name
            expected_shape = input_meta.shape
            actual_data = inputs[input_name]
            
            # NaN値のチェック
            if np.isnan(actual_data).any():
                raise ValueError(f"入力'{input_name}'にNaN値が含まれています")
            
            # 無限大値のチェック
            if np.isinf(actual_data).any():
                raise ValueError(f"入力'{input_name}'に無限大値が含まれています")
            
            # 形状の動的チェック（-1は動的次元を表す）
            if len(expected_shape) != len(actual_data.shape):
                raise ValueError(
                    f"入力'{input_name}'の次元数が不一致: "
                    f"期待={len(expected_shape)}, 実際={len(actual_data.shape)}"
                )

# 使用例
def test_robust_inference():
    """堅牢な推論エンジンのテスト"""
    try:
        # エンジン初期化
        engine = ONNXInferenceEngine("model.onnx")
        
        # テスト用入力データ
        test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        inputs = {"input": test_input}
        
        # 推論実行
        results = engine.predict(inputs)
        
        logger.info(f"推論成功: 出力キー={list(results.keys())}")
        
    except FileNotFoundError:
        logger.error("モデルファイルが見つかりません")
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
```

### 2.5.2　パフォーマンス最適化

ONNXランタイムでのパフォーマンス最適化テクニック：

```python
import onnxruntime as ort
import numpy as np
import time
from contextlib import contextmanager

class PerformanceOptimizer:
    """ONNX推論のパフォーマンス最適化ツール"""
    
    @staticmethod
    def create_optimized_session(model_path: str, optimization_level: str = "all") -> ort.InferenceSession:
        """最適化されたセッションを作成"""
        
        session_options = ort.SessionOptions()
        
        # 最適化レベル設定
        optimization_map = {
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, 
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        }
        session_options.graph_optimization_level = optimization_map[optimization_level]
        
        # 並列実行設定
        session_options.intra_op_num_threads = 0  # システム最適値を使用
        session_options.inter_op_num_threads = 0  # システム最適値を使用
        
        # メモリ最適化
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        
        # 実行モード最適化
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        return ort.InferenceSession(model_path, session_options)
    
    @contextmanager
    def benchmark_context(self, name: str):
        """ベンチマーク用コンテキストマネージャー"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            print(f"{name}: {(end_time - start_time) * 1000:.2f} ms")
    
    def benchmark_inference(self, session: ort.InferenceSession, inputs: dict, num_runs: int = 100):
        """推論パフォーマンスのベンチマーク"""
        print(f"=== 推論ベンチマーク ({num_runs}回実行) ===")
        
        # ウォームアップ（JITコンパイル等のため）
        for _ in range(10):
            session.run(None, inputs)
        
        # ベンチマーク実行
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ミリ秒に変換
        
        # 統計情報の表示
        times = np.array(times)
        print(f"平均実行時間: {times.mean():.2f} ms")
        print(f"標準偏差: {times.std():.2f} ms")
        print(f"最小実行時間: {times.min():.2f} ms")
        print(f"最大実行時間: {times.max():.2f} ms")
        print(f"95パーセンタイル: {np.percentile(times, 95):.2f} ms")
        
        return times

# バッチ処理最適化の例
def optimize_batch_processing(model_path: str, data: np.ndarray, batch_size: int = 32):
    """バッチ処理による最適化"""
    
    session = PerformanceOptimizer.create_optimized_session(model_path)
    optimizer = PerformanceOptimizer()
    
    total_samples = data.shape[0]
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"バッチサイズ{batch_size}で{num_batches}バッチを処理")
    
    with optimizer.benchmark_context("バッチ処理"):
        results = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            batch_data = data[start_idx:end_idx]
            
            outputs = session.run(None, {"input": batch_data})
            results.append(outputs[0])
    
    return np.concatenate(results, axis=0)
```

### 2.5.3　モデル管理とバージョニング

```python
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any
import onnx

class ONNXModelManager:
    """ONNXモデルのバージョン管理とメタデータ管理"""
    
    def __init__(self, model_registry_path: str = "./model_registry"):
        self.registry_path = Path(model_registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """メタデータの読み込み"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """メタデータの保存"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def register_model(self, model_path: str, model_name: str, version: str, 
                      description: str = "", tags: list = None) -> str:
        """モデルの登録"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        # モデルファイルのハッシュ計算
        model_hash = self._calculate_file_hash(model_path)
        
        # モデル情報の取得
        model_info = self._extract_model_info(model_path)
        
        # 保存パス設定
        model_filename = f"{model_name}_{version}.onnx"
        destination_path = self.models_path / model_filename
        
        # モデルファイルのコピー
        import shutil
        shutil.copy2(model_path, destination_path)
        
        # メタデータの更新
        model_key = f"{model_name}:{version}"
        self.metadata[model_key] = {
            "name": model_name,
            "version": version,
            "description": description,
            "tags": tags or [],
            "file_path": str(destination_path),
            "file_hash": model_hash,
            "model_info": model_info,
            "registered_at": time.time()
        }
        
        self._save_metadata()
        print(f"モデル登録完了: {model_key}")
        return model_key
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """ファイルのSHA256ハッシュを計算"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_model_info(self, model_path: str) -> Dict[str, Any]:
        """モデルの基本情報を抽出"""
        model = onnx.load(model_path)
        
        return {
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "opset_versions": [{"domain": op.domain, "version": op.version} 
                             for op in model.opset_import],
            "inputs": [{"name": inp.name, "shape": str([d.dim_value for d in inp.type.tensor_type.shape.dim])} 
                      for inp in model.graph.input],
            "outputs": [{"name": out.name, "shape": str([d.dim_value for d in out.type.tensor_type.shape.dim])} 
                       for out in model.graph.output],
            "num_parameters": sum(np.prod(init.dims) for init in model.graph.initializer)
        }
    
    def list_models(self, name_filter: str = None) -> Dict[str, Any]:
        """登録されたモデルの一覧表示"""
        filtered_models = {}
        
        for key, info in self.metadata.items():
            if name_filter is None or name_filter in info["name"]:
                filtered_models[key] = info
        
        return filtered_models
    
    def load_model(self, model_name: str, version: str = "latest") -> str:
        """モデルの読み込み（パス取得）"""
        if version == "latest":
            # 最新バージョンを検索
            matching_keys = [k for k in self.metadata.keys() if k.startswith(f"{model_name}:")]
            if not matching_keys:
                raise ValueError(f"モデル '{model_name}' が見つかりません")
            
            # バージョン番号でソート（簡単な文字列ソート）
            latest_key = sorted(matching_keys)[-1]
        else:
            latest_key = f"{model_name}:{version}"
            if latest_key not in self.metadata:
                raise ValueError(f"モデル '{latest_key}' が見つかりません")
        
        model_info = self.metadata[latest_key]
        model_path = model_info["file_path"]
        
        # ファイル整合性の確認
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        current_hash = self._calculate_file_hash(model_path)
        if current_hash != model_info["file_hash"]:
            raise ValueError("モデルファイルの整合性チェックに失敗しました")
        
        print(f"モデル読み込み: {latest_key}")
        return model_path

# 使用例
def model_management_demo():
    """モデル管理デモ"""
    manager = ONNXModelManager("./my_model_registry")
    
    # モデル登録（実際のモデルファイルがある場合）
    try:
        manager.register_model(
            "my_model.onnx", 
            "image_classifier", 
            "v1.0",
            "画像分類用のResNetモデル",
            ["classification", "resnet", "imagenet"]
        )
    except FileNotFoundError:
        print("デモ用モデルファイルが見つかりません")
    
    # モデル一覧表示
    models = manager.list_models()
    print("登録済みモデル:")
    for key, info in models.items():
        print(f"  {key}: {info['description']}")
```

## まとめ

第2章では、ONNXランタイムの詳細な機能とアプリケーション開発技術について包括的に学習しました：

### 学習内容の要約

1. **実行プロバイダーの詳細理解**
   - CPU、CUDA、TensorRT、OpenVINO等の特性と選択方法
   - プロバイダー固有の最適化オプション
   - フォールバック機構の活用

2. **ONNX基本概念の実装**
   - モデル構造の詳細分析（ノード、初期化子、属性）
   - データタイプとOpsetバージョンの管理
   - 制御フロー演算子（Loop、Scan）の実装

3. **Python統合による実践的開発**
   - 線形回帰からCNNまでのモデル作成
   - 初期化子の詳細管理と健全性チェック
   - 形状推論と最適化パスの活用

4. **演算子属性の完全理解**
   - Convolution、BatchNormalization等の詳細設定
   - 属性タイプ別の分析と制御方法
   - カスタム属性の実装パターン

5. **実践的開発パターンの習得**
   - 堅牢なエラーハンドリング戦略
   - パフォーマンス最適化テクニック
   - モデル管理とバージョニングシステム

### 開発者への実践的ガイダンス

この章で学んだ技術により、以下の能力を獲得しました：

- **産業レベルの推論システム構築**: エラーハンドリングとログ機能を備えた堅牢なシステム
- **最適化戦略の実装**: ハードウェア特性に応じた最適なプロバイダー選択
- **効率的なモデル管理**: バージョン管理と整合性チェックを含む運用システム
- **詳細なデバッグ能力**: ONNX構造の完全な分析と問題特定

次章では、これらの基礎技術を発展させ、ONNXの高度な機能と性能分析手法について学習していきます。実際の商用環境で求められる、より複雑で高性能なシステム構築技術を習得します。
