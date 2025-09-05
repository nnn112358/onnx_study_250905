# 第3章 ONNXの機能と性能分析

> 本章の概要: Python APIの活用、外部データ、TensorProtoとNumPyの相互変換、補助関数によるモデル構築、モデル検査など、ONNXの実用機能を解説します。

前章の基礎を踏まえ、ONNXエコシステムの実用機能と性能最適化テクニックを掘り下げます。

ONNX（Open Neural Network Exchange）は、豊富な機能を備えた成熟したエコシステムです。これらを活用し、効率的で堅牢なアプリケーションを構築する方法を示します。

## 3.1　Python API概要

ONNXのPython APIは、機械学習開発者にとって重要なインターフェースのひとつです。直感的な設計で研究から本番まで幅広くサポートします。

### 3.1.1　ONNXモデルの読み込み

モデル読み込みはONNXワークフローの起点です。単純な読み込みから、大規模モデルや外部データを含むケースまで、代表的な方法を紹介します。

**モデル読み込みの基本原則:**
- **検証の重要性**: 読み込み時のモデル整合性チェック
- **メモリ効率**: 大規模モデルでのメモリ使用量最適化
- **エラーハンドリング**: 堅牢なエラー処理とデバッグ手法
- **パフォーマンス**: 読み込み速度の最適化手法

```python
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

def load_onnx_model_basic(model_path):
    """基本的なONNXモデル読み込み"""
    
    # ONNXモデルの読み込み（モデル構造の確認用）
    model = onnx.load(model_path)
    
    # 基本情報の表示
    print(f"モデルパス: {model_path}")
    print(f"IRバージョン: {model.ir_version}")
    print(f"プロデューサー: {model.producer_name}")
    print(f"モデルバージョン: {model.model_version}")
    
    return model

def load_onnx_for_inference(model_path, providers=None):
    """推論用のONNXセッション作成"""
    
    if providers is None:
        providers = ['CPUExecutionProvider']
    
    # セッションオプションの設定
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 0  # 自動設定
    
    # セッションの作成
    session = ort.InferenceSession(model_path, session_options, providers=providers)
    
    # セッション情報の表示
    print("=== セッション情報 ===")
    print(f"入力数: {len(session.get_inputs())}")
    print(f"出力数: {len(session.get_outputs())}")
    print(f"プロバイダー: {session.get_providers()}")
    
    # 入力・出力詳細
    for i, input_meta in enumerate(session.get_inputs()):
        print(f"入力{i+1}: {input_meta.name}, 形状: {input_meta.shape}, 型: {input_meta.type}")
    
    for i, output_meta in enumerate(session.get_outputs()):
        print(f"出力{i+1}: {output_meta.name}, 形状: {output_meta.shape}, 型: {output_meta.type}")
    
    return session

def load_with_validation(model_path):
    """検証付きモデル読み込み"""
    
    # ファイル存在チェック
    if not Path(model_path).exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    
    try:
        # モデル読み込み
        model = onnx.load(model_path)
        
        # モデル検証
        onnx.checker.check_model(model)
        print("✓ モデル検証が成功しました")
        
        # 形状推論
        inferred_model = onnx.shape_inference.infer_shapes(model)
        print("✓ 形状推論が成功しました")
        
        return inferred_model
        
    except Exception as e:
        print(f"✗ モデル読み込みエラー: {e}")
        raise

# 使用例
if __name__ == "__main__":
    model_path = "model.onnx"
    
    # 基本読み込み
    model = load_onnx_model_basic(model_path)
    
    # 推論セッション作成
    session = load_onnx_for_inference(model_path)
    
    # 検証付き読み込み
    validated_model = load_with_validation(model_path)
```

### 3.1.2 外部データを含むONNXモデルの読み込み

大規模モデルでは、重みが外部ファイルに分離されることがあります。

```python
import onnx
import os
from pathlib import Path

def load_model_with_external_data(model_path):
    """外部データを含むONNXモデルの読み込み"""
    
    model_dir = Path(model_path).parent
    
    # 外部データファイルの確認
    model = onnx.load(model_path)
    external_files = []
    
    for initializer in model.graph.initializer:
        if initializer.external_data:
            for data_info in initializer.external_data:
                if data_info.key == "location":
                    external_files.append(data_info.value)
    
    print(f"外部データファイル数: {len(external_files)}")
    for ext_file in set(external_files):
        ext_path = model_dir / ext_file
        if ext_path.exists():
            size_mb = ext_path.stat().st_size / (1024 * 1024)
            print(f"  {ext_file}: {size_mb:.2f} MB")
        else:
            print(f"  ⚠️ {ext_file}: ファイルが見つかりません")
    
    return model

def convert_to_external_data(model_path, output_path, size_threshold_mb=1):
    """モデルの重みを外部データに変換"""
    
    model = onnx.load(model_path)
    
    # 外部データに変換
    onnx.external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=size_threshold_mb * 1024 * 1024  # バイトに変換
    )
    
    # 変換されたモデルを保存
    onnx.save_model(model, output_path, save_as_external_data=True)
    
    print(f"外部データ形式で保存しました: {output_path}")
    
    return model

def load_external_data_info(model_path):
    """外部データの詳細情報を取得"""
    
    model = onnx.load(model_path, load_external_data=False)
    
    print("=== 外部データ情報 ===")
    
    for initializer in model.graph.initializer:
        if initializer.external_data:
            print(f"\n初期化子: {initializer.name}")
            print(f"形状: {list(initializer.dims)}")
            
            for data_info in initializer.external_data:
                if data_info.key == "location":
                    print(f"  ファイル: {data_info.value}")
                elif data_info.key == "offset":
                    print(f"  オフセット: {data_info.value}")
                elif data_info.key == "length":
                    print(f"  長さ: {data_info.value}")

# 使用例
def demo_external_data():
    """外部データ機能のデモ"""
    
    # 大きなモデルを想定したダミーモデル作成
    import numpy as np
    from onnx import helper, TensorProto, numpy_helper
    
    # 大きな重み行列（10MB程度）
    large_weight = np.random.randn(1000, 2500).astype(np.float32)
    
    # モデル作成
    weight_initializer = numpy_helper.from_array(large_weight, name="large_weight")
    
    node = helper.make_node(
        "MatMul",
        inputs=["input", "large_weight"],
        outputs=["output"],
        name="matmul"
    )
    
    graph = helper.make_graph(
        nodes=[node],
        name="LargeModel",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 1000])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2500])],
        initializer=[weight_initializer]
    )
    
    model = helper.make_model(graph, producer_name="external_data_demo")
    
    # 通常形式で保存
    onnx.save(model, "large_model.onnx")
    
    # 外部データ形式に変換
    convert_to_external_data("large_model.onnx", "large_model_external.onnx")
    
    # 外部データ情報の表示
    load_external_data_info("large_model_external.onnx")

demo_external_data()
```

### 3.1.3　TensorProtoとNumpy配列の操作

ONNXのTensorProtoとNumPy配列間の変換を学習：

```python
import onnx
import numpy as np
from onnx import TensorProto, numpy_helper

def numpy_to_tensorproto_demo():
    """NumPy配列からTensorProtoへの変換"""
    
    print("=== NumPy → TensorProto 変換 ===")
    
    # さまざまな形状・データ型の配列を作成
    arrays = {
        "float32_2d": np.random.randn(3, 4).astype(np.float32),
        "int64_1d": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "bool_3d": np.random.rand(2, 3, 4) > 0.5,
        "uint8_image": np.random.randint(0, 256, (1, 3, 224, 224), dtype=np.uint8)
    }
    
    tensor_protos = {}
    
    for name, array in arrays.items():
        # NumPy配列をTensorProtoに変換
        tensor_proto = numpy_helper.from_array(array, name=name)
        tensor_protos[name] = tensor_proto
        
        print(f"\n{name}:")
        print(f"  NumPy形状: {array.shape}")
        print(f"  NumPyデータ型: {array.dtype}")
        print(f"  TensorProto形状: {list(tensor_proto.dims)}")
        print(f"  TensorProtoデータ型: {tensor_proto.data_type}")
        print(f"  TensorProtoデータ型名: {TensorProto.DataType.Name(tensor_proto.data_type)}")
    
    return tensor_protos

def tensorproto_to_numpy_demo(tensor_protos):
    """TensorProtoからNumPy配列への変換"""
    
    print("\n=== TensorProto → NumPy 変換 ===")
    
    for name, tensor_proto in tensor_protos.items():
        # TensorProtoをNumPy配列に変換
        array = numpy_helper.to_array(tensor_proto)
        
        print(f"\n{name}:")
        print(f"  復元された形状: {array.shape}")
        print(f"  復元されたデータ型: {array.dtype}")
        print(f"  データサンプル: {array.flatten()[:5]}")  # 最初の5要素

def advanced_tensor_operations():
    """高度なテンソル操作"""
    
    print("\n=== 高度なテンソル操作 ===")
    
    # 大きなテンソルの効率的な作成
    large_array = np.random.randn(1000, 1000).astype(np.float32)
    
    # スパーステンソルのシミュレーション（ほとんどが0）
    sparse_array = np.zeros((100, 100), dtype=np.float32)
    sparse_array[np.random.randint(0, 100, 20), np.random.randint(0, 100, 20)] = np.random.randn(20)
    
    # 量子化シミュレーション（float32 → int8）
    float_array = np.random.randn(50, 50).astype(np.float32)
    
    # 正規化 [-1, 1] → [0, 255] → int8
    normalized = ((float_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
    quantized_tensor = numpy_helper.from_array(normalized, name="quantized")
    
    print("量子化例:")
    print(f"  元データ範囲: [{float_array.min():.3f}, {float_array.max():.3f}]")
    print(f"  量子化後範囲: [{normalized.min()}, {normalized.max()}]")
    
    # メモリ効率の計算
    original_size = float_array.nbytes
    quantized_size = normalized.nbytes
    compression_ratio = original_size / quantized_size
    
    print(f"  メモリ使用量: {original_size} → {quantized_size} bytes")
    print(f"  圧縮率: {compression_ratio:.1f}x")

def tensor_data_type_conversion():
    """データタイプ変換の詳細"""
    
    print("\n=== データタイプ変換 ===")
    
    # 元データ（float64）
    original = np.array([[1.7, 2.8, 3.9], [4.1, 5.2, 6.3]], dtype=np.float64)
    print(f"元データ (float64): \n{original}")
    
    # さまざまなデータタイプに変換
    conversions = [
        (np.float32, "float32への変換"),
        (np.float16, "float16への変換（半精度）"),
        (np.int32, "int32への変換"),
        (np.int8, "int8への変換"),
    ]
    
    for target_dtype, description in conversions:
        converted = original.astype(target_dtype)
        tensor_proto = numpy_helper.from_array(converted, name=f"data_{target_dtype.__name__}")
        
        print(f"\n{description}:")
        print(f"  変換後データ: \n{converted}")
        print(f"  TensorProtoタイプ: {TensorProto.DataType.Name(tensor_proto.data_type)}")
        print(f"  メモリ使用量: {original.nbytes} → {converted.nbytes} bytes")

# 実行例
def run_tensor_demos():
    """すべてのデモを実行"""
    
    # 基本変換
    tensor_protos = numpy_to_tensorproto_demo()
    tensorproto_to_numpy_demo(tensor_protos)
    
    # 高度な操作
    advanced_tensor_operations()
    
    # データタイプ変換
    tensor_data_type_conversion()

if __name__ == "__main__":
    run_tensor_demos()
```

### 3.1.4　補助関数を使用したONNXモデル作成

ONNXの補助関数（helper functions）を使った効率的なモデル作成：

```python
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

def create_simple_model_with_helpers():
    """補助関数を使用したシンプルなモデル作成"""
    
    print("=== 補助関数でのモデル作成 ===")
    
    # 1. テンソル情報の作成
    input_tensor = helper.make_tensor_value_info(
        name="input",
        elem_type=TensorProto.FLOAT,
        shape=[None, 784]  # バッチサイズは可変
    )
    
    output_tensor = helper.make_tensor_value_info(
        name="output",
        elem_type=TensorProto.FLOAT,
        shape=[None, 10]
    )
    
    # 2. 初期化子（重み）の作成
    W = np.random.randn(784, 10).astype(np.float32) * 0.01
    b = np.zeros(10, dtype=np.float32)
    
    W_initializer = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=[784, 10],
        vals=W.flatten()
    )
    
    b_initializer = helper.make_tensor(
        name="b",
        data_type=TensorProto.FLOAT,
        dims=[10],
        vals=b.flatten()
    )
    
    # 3. ノードの作成
    matmul_node = helper.make_node(
        op_type="MatMul",
        inputs=["input", "W"],
        outputs=["matmul_result"],
        name="dense_layer"
    )
    
    add_node = helper.make_node(
        op_type="Add",
        inputs=["matmul_result", "b"],
        outputs=["add_result"],
        name="bias_add"
    )
    
    softmax_node = helper.make_node(
        op_type="Softmax",
        inputs=["add_result"],
        outputs=["output"],
        axis=1,
        name="softmax_output"
    )
    
    # 4. グラフの作成
    graph = helper.make_graph(
        nodes=[matmul_node, add_node, softmax_node],
        name="SimpleClassifier",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[W_initializer, b_initializer]
    )
    
    # 5. モデルの作成
    model = helper.make_model(
        graph=graph,
        producer_name="onnx_tutorial",
        producer_version="1.0"
    )
    
    # 6. Opsetバージョンの設定
    model.opset_import[0].version = 13
    
    # 7. モデルの検証
    onnx.checker.check_model(model)
    
    print("✓ シンプルなモデルを作成しました")
    return model

def create_cnn_model_with_helpers():
    """CNNモデルの作成例"""
    
    print("\n=== CNNモデルの作成 ===")
    
    # 入力: [N, C, H, W] = [batch_size, 3, 224, 224]
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, 3, 224, 224]
    )
    
    # 出力: [N, num_classes] = [batch_size, 1000]
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 1000]
    )
    
    # 畳み込み層の重み
    conv_weight = np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.01
    conv_bias = np.zeros(64, dtype=np.float32)
    
    # 全結合層の重み（特徴マップサイズは計算で求める）
    # 224x224 -> (224-7+2*3)/2+1 = 112 -> 112x112 -> global avg pool -> 1x1
    fc_weight = np.random.randn(64, 1000).astype(np.float32) * 0.01
    fc_bias = np.zeros(1000, dtype=np.float32)
    
    # 初期化子
    initializers = [
        numpy_helper.from_array(conv_weight, name="conv_weight"),
        numpy_helper.from_array(conv_bias, name="conv_bias"),
        numpy_helper.from_array(fc_weight, name="fc_weight"),
        numpy_helper.from_array(fc_bias, name="fc_bias"),
    ]
    
    # ノード作成
    nodes = [
        # Convolution: input -> conv_output
        helper.make_node(
            "Conv",
            inputs=["input", "conv_weight", "conv_bias"],
            outputs=["conv_output"],
            kernel_shape=[7, 7],
            pads=[3, 3, 3, 3],
            strides=[2, 2],
            name="conv1"
        ),
        
        # ReLU: conv_output -> relu_output
        helper.make_node(
            "Relu",
            inputs=["conv_output"],
            outputs=["relu_output"],
            name="relu1"
        ),
        
        # Global Average Pooling: relu_output -> gap_output
        helper.make_node(
            "GlobalAveragePool",
            inputs=["relu_output"],
            outputs=["gap_output"],
            name="global_avg_pool"
        ),
        
        # Flatten: gap_output -> flatten_output
        helper.make_node(
            "Flatten",
            inputs=["gap_output"],
            outputs=["flatten_output"],
            axis=1,
            name="flatten"
        ),
        
        # Fully Connected: flatten_output -> fc_output
        helper.make_node(
            "MatMul",
            inputs=["flatten_output", "fc_weight"],
            outputs=["fc_output"],
            name="fc"
        ),
        
        # Add Bias: fc_output -> output
        helper.make_node(
            "Add",
            inputs=["fc_output", "fc_bias"],
            outputs=["output"],
            name="fc_bias"
        ),
    ]
    
    # グラフ作成
    graph = helper.make_graph(
        nodes=nodes,
        name="SimpleCNN",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers
    )
    
    # モデル作成
    model = helper.make_model(graph, producer_name="cnn_tutorial")
    model.opset_import[0].version = 13
    
    # 検証
    onnx.checker.check_model(model)
    
    print("✓ CNNモデルを作成しました")
    return model

def create_model_with_dynamic_shapes():
    """動的形状を持つモデルの作成"""
    
    print("\n=== 動的形状モデルの作成 ===")
    
    # 動的次元の定義
    batch_dim = "batch_size"
    seq_len_dim = "sequence_length"
    
    # 入力テンソル（バッチサイズとシーケンス長が可変）
    input_tensor = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [batch_dim, seq_len_dim, 128]  # [batch, seq_len, hidden_size]
    )
    
    # 出力テンソル
    output_tensor = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [batch_dim, seq_len_dim, 256]
    )
    
    # 重み（固定形状）
    weight = np.random.randn(128, 256).astype(np.float32) * 0.01
    bias = np.zeros(256, dtype=np.float32)
    
    initializers = [
        numpy_helper.from_array(weight, name="weight"),
        numpy_helper.from_array(bias, name="bias"),
    ]
    
    # ノード作成
    nodes = [
        helper.make_node(
            "MatMul",
            inputs=["input", "weight"],
            outputs=["matmul_output"],
            name="linear_transform"
        ),
        helper.make_node(
            "Add",
            inputs=["matmul_output", "bias"],
            outputs=["output"],
            name="add_bias"
        ),
    ]
    
    # グラフ作成
    graph = helper.make_graph(
        nodes=nodes,
        name="DynamicShapeModel",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers
    )
    
    # モデル作成
    model = helper.make_model(graph, producer_name="dynamic_shape_tutorial")
    model.opset_import[0].version = 13
    
    print("✓ 動的形状モデルを作成しました")
    print(f"  入力形状: [{batch_dim}, {seq_len_dim}, 128]")
    print(f"  出力形状: [{batch_dim}, {seq_len_dim}, 256]")
    
    return model

# モデル作成と保存のデモ
def model_creation_demo():
    """モデル作成デモの実行"""
    
    # 1. シンプルなモデル
    simple_model = create_simple_model_with_helpers()
    onnx.save(simple_model, "simple_classifier.onnx")
    
    # 2. CNNモデル
    cnn_model = create_cnn_model_with_helpers()
    onnx.save(cnn_model, "simple_cnn.onnx")
    
    # 3. 動的形状モデル
    dynamic_model = create_model_with_dynamic_shapes()
    onnx.save(dynamic_model, "dynamic_shape_model.onnx")
    
    print("\n全てのモデルを保存しました:")
    print("  - simple_classifier.onnx")
    print("  - simple_cnn.onnx")
    print("  - dynamic_shape_model.onnx")

if __name__ == "__main__":
    model_creation_demo()
```

### 3.1.5　ONNX IRの属性をマッピングするための変換ユーティリティ

ONNX中間表現（IR）の属性変換ユーティリティ：

```python
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto
import numpy as np
from typing import Dict, Any, List

class ONNXAttributeMapper:
    """ONNX属性のマッピングと変換ユーティリティ"""
    
    @staticmethod
    def extract_node_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
        """ノードから属性を抽出してPythonオブジェクトに変換"""
        
        attributes = {}
        
        for attr in node.attribute:
            attr_name = attr.name
            
            # 属性タイプに応じて値を抽出
            if attr.type == AttributeProto.INT:
                attributes[attr_name] = attr.i
            elif attr.type == AttributeProto.INTS:
                attributes[attr_name] = list(attr.ints)
            elif attr.type == AttributeProto.FLOAT:
                attributes[attr_name] = attr.f
            elif attr.type == AttributeProto.FLOATS:
                attributes[attr_name] = list(attr.floats)
            elif attr.type == AttributeProto.STRING:
                attributes[attr_name] = attr.s.decode('utf-8')
            elif attr.type == AttributeProto.STRINGS:
                attributes[attr_name] = [s.decode('utf-8') for s in attr.strings]
            elif attr.type == AttributeProto.TENSOR:
                attributes[attr_name] = onnx.numpy_helper.to_array(attr.t)
            elif attr.type == AttributeProto.GRAPH:
                attributes[attr_name] = attr.g
            elif attr.type == AttributeProto.SPARSE_TENSOR:
                attributes[attr_name] = attr.sparse_tensor
            else:
                # 未知の属性タイプ
                attributes[attr_name] = f"Unknown type: {attr.type}"
        
        return attributes
    
    @staticmethod
    def create_attributes_from_dict(attr_dict: Dict[str, Any]) -> List[onnx.AttributeProto]:
        """辞書からONNX属性リストを作成"""
        
        attributes = []
        
        for name, value in attr_dict.items():
            if isinstance(value, int):
                attr = helper.make_attribute(name, value)
            elif isinstance(value, float):
                attr = helper.make_attribute(name, value)
            elif isinstance(value, str):
                attr = helper.make_attribute(name, value)
            elif isinstance(value, list):
                if all(isinstance(x, int) for x in value):
                    attr = helper.make_attribute(name, value)
                elif all(isinstance(x, float) for x in value):
                    attr = helper.make_attribute(name, value)
                elif all(isinstance(x, str) for x in value):
                    attr = helper.make_attribute(name, value)
                else:
                    raise ValueError(f"Unsupported list type for attribute {name}")
            elif isinstance(value, np.ndarray):
                tensor = onnx.numpy_helper.from_array(value)
                attr = helper.make_attribute(name, tensor)
            elif isinstance(value, GraphProto):
                attr = helper.make_attribute(name, value)
            else:
                raise ValueError(f"Unsupported type for attribute {name}: {type(value)}")
            
            attributes.append(attr)
        
        return attributes
    
    @staticmethod
    def map_framework_to_onnx_attributes(framework_attrs: Dict[str, Any], 
                                       mapping_rules: Dict[str, str]) -> Dict[str, Any]:
        """フレームワーク固有の属性をONNX属性にマッピング"""
        
        onnx_attrs = {}
        
        for fw_name, fw_value in framework_attrs.items():
            # マッピングルールを適用
            if fw_name in mapping_rules:
                onnx_name = mapping_rules[fw_name]
                onnx_attrs[onnx_name] = fw_value
            else:
                # デフォルトでは名前をそのまま使用
                onnx_attrs[fw_name] = fw_value
        
        return onnx_attrs

def demonstrate_attribute_mapping():
    """属性マッピングのデモンストレーション"""
    
    print("=== ONNX属性マッピングデモ ===")
    
    # サンプルモデルの作成
    conv_attrs = {
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "strides": [1, 1],
        "dilations": [1, 1],
        "group": 1
    }
    
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        name="conv_layer",
        **conv_attrs
    )
    
    # 1. ノードから属性を抽出
    mapper = ONNXAttributeMapper()
    extracted_attrs = mapper.extract_node_attributes(conv_node)
    
    print("抽出された属性:")
    for name, value in extracted_attrs.items():
        print(f"  {name}: {value} (type: {type(value).__name__})")
    
    # 2. 属性の変更
    modified_attrs = extracted_attrs.copy()
    modified_attrs["strides"] = [2, 2]  # ストライドを変更
    modified_attrs["padding_mode"] = "SAME"  # 新しい属性を追加
    
    # 3. 変更された属性でノードを再作成
    new_attributes = mapper.create_attributes_from_dict(modified_attrs)
    
    new_conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        name="modified_conv_layer"
    )
    
    # 属性を手動で追加
    new_conv_node.attribute.extend(new_attributes)
    
    print("\n変更後の属性:")
    new_extracted_attrs = mapper.extract_node_attributes(new_conv_node)
    for name, value in new_extracted_attrs.items():
        print(f"  {name}: {value}")

def framework_to_onnx_mapping_demo():
    """フレームワーク→ONNX属性マッピングのデモ"""
    
    print("\n=== フレームワーク→ONNX マッピング ===")
    
    # PyTorch風の属性
    pytorch_conv_attrs = {
        "kernel_size": [3, 3],
        "padding": [1, 1, 1, 1],
        "stride": [2, 2],
        "dilation": [1, 1],
        "groups": 1,
        "bias": True
    }
    
    # PyTorch → ONNX マッピングルール
    pytorch_to_onnx_mapping = {
        "kernel_size": "kernel_shape",
        "padding": "pads",
        "stride": "strides",
        "dilation": "dilations",
        "groups": "group"
    }
    
    mapper = ONNXAttributeMapper()
    
    # マッピング適用
    onnx_attrs = mapper.map_framework_to_onnx_attributes(
        pytorch_conv_attrs, 
        pytorch_to_onnx_mapping
    )
    
    print("PyTorch属性:")
    for name, value in pytorch_conv_attrs.items():
        print(f"  {name}: {value}")
    
    print("\nONNX属性:")
    for name, value in onnx_attrs.items():
        print(f"  {name}: {value}")
    
    # ONNX ノードの作成
    try:
        # biasは属性ではなく入力として扱うため除外
        filtered_attrs = {k: v for k, v in onnx_attrs.items() if k != "bias"}
        
        onnx_node = helper.make_node(
            "Conv",
            inputs=["input", "weight", "bias"] if onnx_attrs.get("bias") else ["input", "weight"],
            outputs=["output"],
            name="mapped_conv",
            **filtered_attrs
        )
        
        print("\n✓ ONNX ノードの作成に成功しました")
        
        # 最終的な属性を確認
        final_attrs = mapper.extract_node_attributes(onnx_node)
        print("\n最終的なONNX属性:")
        for name, value in final_attrs.items():
            print(f"  {name}: {value}")
        
    except Exception as e:
        print(f"\n✗ ノード作成エラー: {e}")

def batch_attribute_conversion():
    """複数ノードの一括属性変換"""
    
    print("\n=== 一括属性変換 ===")
    
    # 複数のノードを含むグラフを作成
    nodes_info = [
        {
            "op_type": "Conv",
            "name": "conv1",
            "attributes": {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]}
        },
        {
            "op_type": "BatchNormalization", 
            "name": "bn1",
            "attributes": {"epsilon": 1e-5, "momentum": 0.9}
        },
        {
            "op_type": "Relu",
            "name": "relu1", 
            "attributes": {}
        }
    ]
    
    mapper = ONNXAttributeMapper()
    converted_nodes = []
    
    for i, node_info in enumerate(nodes_info):
        # ダミーの入出力
        inputs = [f"input_{i}", "weight"] if node_info["op_type"] == "Conv" else [f"input_{i}"]
        if node_info["op_type"] == "BatchNormalization":
            inputs.extend(["scale", "bias", "mean", "var"])
        
        outputs = [f"output_{i}"]
        
        # ノード作成
        node = helper.make_node(
            node_info["op_type"],
            inputs=inputs,
            outputs=outputs,
            name=node_info["name"],
            **node_info["attributes"]
        )
        
        converted_nodes.append(node)
        
        # 属性情報の表示
        attrs = mapper.extract_node_attributes(node)
        print(f"\n{node_info['name']} ({node_info['op_type']}):")
        if attrs:
            for attr_name, attr_value in attrs.items():
                print(f"  {attr_name}: {attr_value}")
        else:
            print("  属性なし")
    
    print(f"\n✓ {len(converted_nodes)}個のノードを変換しました")

# 実行例
if __name__ == "__main__":
    demonstrate_attribute_mapping()
    framework_to_onnx_mapping_demo()
    batch_attribute_conversion()
```

### 3.1.6　ONNXモデルの検査

包括的なモデル検査ツール：

```python
import onnx
from onnx import checker, shape_inference, version_converter
import numpy as np
from pathlib import Path
import warnings

class ONNXModelInspector:
    """ONNX モデルの包括的な検査クラス"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.issues = []
        
    def load_model(self) -> bool:
        """モデルを読み込み"""
        try:
            if not self.model_path.exists():
                self.issues.append(f"モデルファイルが見つかりません: {self.model_path}")
                return False
                
            self.model = onnx.load(str(self.model_path))
            return True
        except Exception as e:
            self.issues.append(f"モデル読み込みエラー: {e}")
            return False
    
    def basic_validation(self) -> bool:
        """基本的な検証"""
        print("=== 基本検証 ===")
        
        try:
            checker.check_model(self.model)
            print("✓ 基本検証: 成功")
            return True
        except checker.ValidationError as e:
            self.issues.append(f"検証エラー: {e}")
            print(f"✗ 基本検証: 失敗 - {e}")
            return False
        except Exception as e:
            self.issues.append(f"予期しないエラー: {e}")
            print(f"✗ 基本検証: 予期しないエラー - {e}")
            return False
    
    def check_model_structure(self) -> Dict[str, Any]:
        """モデル構造の詳細チェック"""
        print("\n=== モデル構造チェック ===")
        
        structure_info = {
            "ir_version": self.model.ir_version,
            "producer_name": self.model.producer_name,
            "producer_version": self.model.producer_version,
            "model_version": self.model.model_version,
            "doc_string": self.model.doc_string
        }
        
        # OpSet情報
        opset_info = {}
        for opset_import in self.model.opset_import:
            domain = opset_import.domain if opset_import.domain else "default"
            opset_info[domain] = opset_import.version
        structure_info["opset_versions"] = opset_info
        
        # グラフ統計
        graph = self.model.graph
        graph_stats = {
            "node_count": len(graph.node),
            "input_count": len(graph.input),
            "output_count": len(graph.output),
            "initializer_count": len(graph.initializer),
            "value_info_count": len(graph.value_info)
        }
        structure_info["graph_stats"] = graph_stats
        
        # 演算子統計
        op_count = {}
        for node in graph.node:
            op_type = node.op_type
            op_count[op_type] = op_count.get(op_type, 0) + 1
        structure_info["operator_stats"] = op_count
        
        # 結果表示
        print(f"IRバージョン: {structure_info['ir_version']}")
        print(f"プロデューサー: {structure_info['producer_name']} v{structure_info['producer_version']}")
        print(f"OpSet: {structure_info['opset_versions']}")
        print(f"ノード数: {graph_stats['node_count']}")
        print(f"演算子種類: {len(op_count)}種類")
        
        return structure_info
    
    def check_data_types_and_shapes(self) -> Dict[str, Any]:
        """データタイプと形状のチェック"""
        print("\n=== データタイプ・形状チェック ===")
        
        shape_info = {}
        
        # 入力の確認
        print("入力:")
        for i, input_tensor in enumerate(self.model.graph.input):
            shape = self._extract_shape(input_tensor.type.tensor_type.shape)
            dtype = input_tensor.type.tensor_type.elem_type
            dtype_name = TensorProto.DataType.Name(dtype)
            
            print(f"  {input_tensor.name}: {shape} ({dtype_name})")
            
            # 動的次元の検出
            dynamic_dims = [j for j, dim in enumerate(shape) if dim == "?" or isinstance(dim, str)]
            if dynamic_dims:
                print(f"    動的次元: {dynamic_dims}")
        
        # 出力の確認
        print("出力:")
        for i, output_tensor in enumerate(self.model.graph.output):
            shape = self._extract_shape(output_tensor.type.tensor_type.shape)
            dtype = output_tensor.type.tensor_type.elem_type
            dtype_name = TensorProto.DataType.Name(dtype)
            
            print(f"  {output_tensor.name}: {shape} ({dtype_name})")
        
        return shape_info
    
    def _extract_shape(self, tensor_shape):
        """テンソル形状を抽出"""
        shape = []
        for dim in tensor_shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        return shape
    
    def check_initializers(self) -> Dict[str, Any]:
        """初期化子（重み）のチェック"""
        print("\n=== 初期化子チェック ===")
        
        initializer_info = {
            "total_count": len(self.model.graph.initializer),
            "total_parameters": 0,
            "data_types": {},
            "suspicious_values": []
        }
        
        for initializer in self.model.graph.initializer:
            # パラメータ数の計算
            param_count = np.prod(initializer.dims) if initializer.dims else 1
            initializer_info["total_parameters"] += param_count
            
            # データタイプの統計
            dtype_name = TensorProto.DataType.Name(initializer.data_type)
            initializer_info["data_types"][dtype_name] = initializer_info["data_types"].get(dtype_name, 0) + 1
            
            # 値の健全性チェック（小さなテンソルのみ）
            if param_count <= 10000:  # 10K要素以下
                try:
                    tensor_data = onnx.numpy_helper.to_array(initializer)
                    
                    # NaN, Inf のチェック
                    if np.isnan(tensor_data).any():
                        initializer_info["suspicious_values"].append(f"{initializer.name}: NaN値を含む")
                    
                    if np.isinf(tensor_data).any():
                        initializer_info["suspicious_values"].append(f"{initializer.name}: 無限大値を含む")
                    
                    # 異常に大きな値
                    max_abs_val = np.abs(tensor_data).max()
                    if max_abs_val > 1000:
                        initializer_info["suspicious_values"].append(f"{initializer.name}: 大きな値 (max: {max_abs_val})")
                    
                    # 全て0の重み
                    if np.all(tensor_data == 0):
                        initializer_info["suspicious_values"].append(f"{initializer.name}: 全て0")
                        
                except Exception as e:
                    initializer_info["suspicious_values"].append(f"{initializer.name}: 読み込みエラー - {e}")
        
        # 結果表示
        print(f"初期化子数: {initializer_info['total_count']}")
        print(f"総パラメータ数: {initializer_info['total_parameters']:,}")
        print(f"データタイプ分布: {initializer_info['data_types']}")
        
        if initializer_info["suspicious_values"]:
            print("⚠️ 注意が必要な値:")
            for warning in initializer_info["suspicious_values"]:
                print(f"  {warning}")
        else:
            print("✓ 初期化子に問題は見つかりませんでした")
        
        return initializer_info
    
    def perform_shape_inference(self) -> bool:
        """形状推論の実行"""
        print("\n=== 形状推論 ===")
        
        try:
            inferred_model = shape_inference.infer_shapes(self.model)
            
            # 推論された形状情報を表示
            print("推論された中間テンソル形状:")
            for value_info in inferred_model.graph.value_info:
                if value_info.type.tensor_type:
                    shape = self._extract_shape(value_info.type.tensor_type.shape)
                    dtype = value_info.type.tensor_type.elem_type
                    dtype_name = TensorProto.DataType.Name(dtype)
                    print(f"  {value_info.name}: {shape} ({dtype_name})")
            
            print("✓ 形状推論: 成功")
            return True
            
        except Exception as e:
            self.issues.append(f"形状推論エラー: {e}")
            print(f"✗ 形状推論: 失敗 - {e}")
            return False
    
    def check_compatibility(self, target_opset_version: int = None) -> Dict[str, Any]:
        """互換性チェック"""
        print("\n=== 互換性チェック ===")
        
        compatibility_info = {
            "current_opsets": {},
            "conversion_possible": {},
            "warnings": []
        }
        
        # 現在のOpSetバージョンを取得
        for opset_import in self.model.opset_import:
            domain = opset_import.domain if opset_import.domain else "default"
            compatibility_info["current_opsets"][domain] = opset_import.version
        
        # ターゲットバージョンが指定されている場合の変換テスト
        if target_opset_version:
            try:
                converted_model = version_converter.convert_version(self.model, target_opset_version)
                compatibility_info["conversion_possible"][f"opset_{target_opset_version}"] = True
                print(f"✓ OpSet {target_opset_version} への変換: 可能")
            except Exception as e:
                compatibility_info["conversion_possible"][f"opset_{target_opset_version}"] = False
                compatibility_info["warnings"].append(f"OpSet {target_opset_version} 変換エラー: {e}")
                print(f"✗ OpSet {target_opset_version} への変換: 不可能 - {e}")
        
        # 古いOpSetの警告
        for domain, version in compatibility_info["current_opsets"].items():
            if domain == "default" and version < 11:
                compatibility_info["warnings"].append(f"古いOpSetバージョン: {version} (推奨: 11以上)")
        
        return compatibility_info
    
    def generate_report(self) -> str:
        """検査レポートの生成"""
        print("\n" + "="*50)
        print("ONNX モデル検査レポート")
        print("="*50)
        
        if not self.load_model():
            return "モデルの読み込みに失敗しました"
        
        # 各種チェックを実行
        basic_ok = self.basic_validation()
        structure_info = self.check_model_structure()
        self.check_data_types_and_shapes()
        initializer_info = self.check_initializers()
        shape_inference_ok = self.perform_shape_inference()
        compatibility_info = self.check_compatibility(target_opset_version=11)
        
        # 総合評価
        print("\n=== 総合評価 ===")
        if basic_ok and shape_inference_ok and not self.issues:
            print("✓ モデルは健全です")
            overall_status = "HEALTHY"
        elif self.issues:
            print("⚠️ 問題が見つかりました:")
            for issue in self.issues:
                print(f"  - {issue}")
            overall_status = "ISSUES_FOUND"
        else:
            print("? 一部のチェックで問題がありましたが、使用は可能かもしれません")
            overall_status = "PARTIAL_ISSUES"
        
        return overall_status

# 使用例
def inspect_model_demo(model_path: str):
    """モデル検査のデモ"""
    
    inspector = ONNXModelInspector(model_path)
    status = inspector.generate_report()
    
    print(f"\n最終ステータス: {status}")

# 実行例
if __name__ == "__main__":
    # 先ほど作成したモデルを検査
    inspect_model_demo("simple_classifier.onnx")
```

## 3.2　ONNX実用機能とツール

### 3.2.1　形状推論と最適化

ONNXの形状推論（Shape Inference）は、モデル内のテンソル形状を自動的に推論する重要な機能です。これにより、開発者はすべての中間テンソルの形状を明示的に指定する必要がなくなります。

```python
import onnx
from onnx import shape_inference, optimizer
import numpy as np
from typing import Dict, List, Tuple

class ONNXShapeInferenceEngine:
    """ONNX形状推論エンジン"""
    
    def __init__(self, model: onnx.ModelProto):
        self.original_model = model
        self.inferred_model = None
        self.shape_info = {}
    
    def perform_inference(self, verbose: bool = True) -> onnx.ModelProto:
        """形状推論を実行"""
        
        if verbose:
            print("=== 形状推論実行 ===")
        
        try:
            # 基本的な形状推論
            self.inferred_model = shape_inference.infer_shapes(self.original_model)
            
            if verbose:
                print("✓ 形状推論が成功しました")
                self._analyze_inferred_shapes()
            
            return self.inferred_model
            
        except Exception as e:
            if verbose:
                print(f"✗ 形状推論失敗: {e}")
            raise
    
    def _analyze_inferred_shapes(self):
        """推論された形状を分析"""
        
        print("\n--- 推論された形状情報 ---")
        
        # 入力形状
        print("入力テンソル:")
        for input_tensor in self.inferred_model.graph.input:
            shape = self._extract_shape_info(input_tensor.type.tensor_type.shape)
            print(f"  {input_tensor.name}: {shape}")
        
        # 中間テンソル形状
        print("\n中間テンソル:")
        for value_info in self.inferred_model.graph.value_info:
            if value_info.type.tensor_type:
                shape = self._extract_shape_info(value_info.type.tensor_type.shape)
                print(f"  {value_info.name}: {shape}")
        
        # 出力形状
        print("\n出力テンソル:")
        for output_tensor in self.inferred_model.graph.output:
            shape = self._extract_shape_info(output_tensor.type.tensor_type.shape)
            print(f"  {output_tensor.name}: {shape}")
    
    def _extract_shape_info(self, tensor_shape) -> List:
        """形状情報を抽出"""
        shape = []
        for dim in tensor_shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(f"'{dim.dim_param}'")
            else:
                shape.append("?")
        return shape
    
    def compare_before_after(self) -> Dict[str, Tuple[int, int]]:
        """推論前後の比較"""
        
        print("\n--- 推論前後の比較 ---")
        
        # 中間テンソル数の比較
        original_value_info_count = len(self.original_model.graph.value_info)
        inferred_value_info_count = len(self.inferred_model.graph.value_info)
        
        print(f"中間テンソル情報数: {original_value_info_count} → {inferred_value_info_count}")
        
        # 未定義だった形状の検出
        newly_defined_shapes = []
        for value_info in self.inferred_model.graph.value_info:
            # 元のモデルにこのテンソル情報があったかチェック
            original_found = False
            for orig_value_info in self.original_model.graph.value_info:
                if orig_value_info.name == value_info.name:
                    original_found = True
                    break
            
            if not original_found and value_info.type.tensor_type:
                shape = self._extract_shape_info(value_info.type.tensor_type.shape)
                newly_defined_shapes.append(f"{value_info.name}: {shape}")
        
        if newly_defined_shapes:
            print("新たに推論された形状:")
            for shape_info in newly_defined_shapes:
                print(f"  {shape_info}")
        
        return {
            "original_count": original_value_info_count,
            "inferred_count": inferred_value_info_count,
            "newly_defined": len(newly_defined_shapes)
        }

def demonstrate_shape_inference():
    """形状推論のデモンストレーション"""
    
    # サンプルモデルの作成（形状情報を一部省略）
    from onnx import helper, TensorProto, numpy_helper
    
    # 重みの初期化
    W1 = np.random.randn(256, 128).astype(np.float32)
    b1 = np.zeros(128, dtype=np.float32)
    W2 = np.random.randn(128, 64).astype(np.float32)
    b2 = np.zeros(64, dtype=np.float32)
    W3 = np.random.randn(64, 10).astype(np.float32)
    b3 = np.zeros(10, dtype=np.float32)
    
    # 初期化子
    initializers = [
        numpy_helper.from_array(W1, name="W1"),
        numpy_helper.from_array(b1, name="b1"),
        numpy_helper.from_array(W2, name="W2"),
        numpy_helper.from_array(b2, name="b2"),
        numpy_helper.from_array(W3, name="W3"),
        numpy_helper.from_array(b3, name="b3"),
    ]
    
    # ノードの作成（中間テンソルの形状情報は省略）
    nodes = [
        helper.make_node("MatMul", ["input", "W1"], ["hidden1_matmul"], name="layer1_matmul"),
        helper.make_node("Add", ["hidden1_matmul", "b1"], ["hidden1"], name="layer1_add"),
        helper.make_node("Relu", ["hidden1"], ["hidden1_relu"], name="layer1_relu"),
        
        helper.make_node("MatMul", ["hidden1_relu", "W2"], ["hidden2_matmul"], name="layer2_matmul"),
        helper.make_node("Add", ["hidden2_matmul", "b2"], ["hidden2"], name="layer2_add"),
        helper.make_node("Relu", ["hidden2"], ["hidden2_relu"], name="layer2_relu"),
        
        helper.make_node("MatMul", ["hidden2_relu", "W3"], ["output_matmul"], name="output_matmul"),
        helper.make_node("Add", ["output_matmul", "b3"], ["output"], name="output_add"),
    ]
    
    # グラフの作成
    graph = helper.make_graph(
        nodes=nodes,
        name="DeepNetwork",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 256])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 10])],
        initializer=initializers
    )
    
    # モデルの作成
    model = helper.make_model(graph, producer_name="shape_inference_demo")
    model.opset_import[0].version = 13
    
    print("元のモデル作成完了")
    
    # 形状推論エンジンの使用
    inference_engine = ONNXShapeInferenceEngine(model)
    inferred_model = inference_engine.perform_inference(verbose=True)
    
    # 比較結果
    comparison = inference_engine.compare_before_after()
    
    return model, inferred_model

def optimize_model_with_shape_inference():
    """形状推論と最適化の組み合わせ"""
    
    print("\n=== 形状推論 + 最適化 ===")
    
    # 先ほどのモデルを使用
    original_model, inferred_model = demonstrate_shape_inference()
    
    try:
        # 最適化の実行
        print("\n最適化実行中...")
        
        # 利用可能な最適化パスを確認
        available_passes = optimizer.get_available_passes()
        print(f"利用可能な最適化パス: {len(available_passes)}種類")
        
        # 基本的な最適化を適用
        basic_passes = [
            "eliminate_identity",
            "eliminate_dropout",
            "extract_constant_to_initializer",
            "fuse_consecutive_transposes"
        ]
        
        # 安全な最適化パスのみを選択
        safe_passes = [pass_name for pass_name in basic_passes if pass_name in available_passes]
        print(f"適用する最適化パス: {safe_passes}")
        
        optimized_model = optimizer.optimize(inferred_model, safe_passes)
        
        print("✓ 最適化完了")
        
        # 最適化前後の比較
        print(f"最適化前ノード数: {len(inferred_model.graph.node)}")
        print(f"最適化後ノード数: {len(optimized_model.graph.node)}")
        
        return optimized_model
        
    except ImportError:
        print("⚠️ optimizer モジュールが利用できません")
        return inferred_model
    except Exception as e:
        print(f"最適化エラー: {e}")
        return inferred_model

# 実行
if __name__ == "__main__":
    original, inferred = demonstrate_shape_inference()
    optimized = optimize_model_with_shape_inference()
```

### 3.2.2　バージョン変換と互換性管理

ONNXでは、OpSet（Operator Set）のバージョン管理により、異なるバージョン間の互換性を保持しています。

```python
import onnx
from onnx import version_converter, helper
import warnings
from typing import Dict, List, Optional

class ONNXVersionManager:
    """ONNXバージョン変換と互換性管理"""
    
    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.current_opsets = self._extract_opset_versions()
    
    def _extract_opset_versions(self) -> Dict[str, int]:
        """現在のOpSetバージョンを抽出"""
        opsets = {}
        for opset_import in self.model.opset_import:
            domain = opset_import.domain if opset_import.domain else "default"
            opsets[domain] = opset_import.version
        return opsets
    
    def check_compatibility(self, target_versions: Dict[str, int]) -> Dict[str, any]:
        """指定されたバージョンとの互換性をチェック"""
        
        print("=== 互換性チェック ===")
        
        compatibility_report = {
            "current_versions": self.current_opsets.copy(),
            "target_versions": target_versions,
            "compatibility_status": {},
            "conversion_required": {},
            "potential_issues": []
        }
        
        for domain, target_version in target_versions.items():
            current_version = self.current_opsets.get(domain, 0)
            
            if current_version == target_version:
                compatibility_report["compatibility_status"][domain] = "exact_match"
                print(f"✓ {domain}: v{current_version} (完全一致)")
            elif current_version < target_version:
                compatibility_report["compatibility_status"][domain] = "upgrade_needed"
                compatibility_report["conversion_required"][domain] = "upgrade"
                print(f"⬆️ {domain}: v{current_version} → v{target_version} (アップグレード必要)")
            else:
                compatibility_report["compatibility_status"][domain] = "downgrade_needed"
                compatibility_report["conversion_required"][domain] = "downgrade"
                print(f"⬇️ {domain}: v{current_version} → v{target_version} (ダウングレード必要)")
                compatibility_report["potential_issues"].append(
                    f"{domain}: ダウングレードにより機能が失われる可能性があります"
                )
        
        return compatibility_report
    
    def convert_version(self, target_version: int, domain: str = "") -> Optional[onnx.ModelProto]:
        """指定されたバージョンに変換"""
        
        print(f"\n=== バージョン変換: OpSet {target_version} ===")
        
        try:
            # バージョン変換の実行
            converted_model = version_converter.convert_version(self.model, target_version)
            
            # 変換結果の確認
            converted_opsets = {}
            for opset_import in converted_model.opset_import:
                domain_name = opset_import.domain if opset_import.domain else "default"
                converted_opsets[domain_name] = opset_import.version
            
            print(f"✓ 変換成功")
            print(f"  変換前: {self.current_opsets}")
            print(f"  変換後: {converted_opsets}")
            
            # 変換による変更点の検出
            self._analyze_conversion_changes(converted_model)
            
            return converted_model
            
        except Exception as e:
            print(f"✗ 変換失敗: {e}")
            return None
    
    def _analyze_conversion_changes(self, converted_model: onnx.ModelProto):
        """変換による変更点を分析"""
        
        print("\n--- 変換による変更点 ---")
        
        # ノード数の変化
        original_node_count = len(self.model.graph.node)
        converted_node_count = len(converted_model.graph.node)
        
        if original_node_count != converted_node_count:
            print(f"ノード数: {original_node_count} → {converted_node_count}")
        
        # 演算子タイプの変化
        original_ops = set(node.op_type for node in self.model.graph.node)
        converted_ops = set(node.op_type for node in converted_model.graph.node)
        
        new_ops = converted_ops - original_ops
        removed_ops = original_ops - converted_ops
        
        if new_ops:
            print(f"追加された演算子: {', '.join(new_ops)}")
        if removed_ops:
            print(f"削除された演算子: {', '.join(removed_ops)}")
        
        # 属性の変化（簡単なチェック）
        original_attr_count = sum(len(node.attribute) for node in self.model.graph.node)
        converted_attr_count = sum(len(node.attribute) for node in converted_model.graph.node)
        
        if original_attr_count != converted_attr_count:
            print(f"総属性数: {original_attr_count} → {converted_attr_count}")
    
    def batch_convert_multiple_versions(self, target_versions: List[int]) -> Dict[int, onnx.ModelProto]:
        """複数バージョンへの一括変換"""
        
        print(f"\n=== 複数バージョンへの一括変換 ===")
        print(f"変換対象バージョン: {target_versions}")
        
        converted_models = {}
        conversion_results = {}
        
        for version in target_versions:
            print(f"\nOpSet {version}への変換...")
            
            try:
                converted_model = self.convert_version(version)
                if converted_model:
                    converted_models[version] = converted_model
                    conversion_results[version] = "success"
                else:
                    conversion_results[version] = "failed"
            except Exception as e:
                conversion_results[version] = f"error: {e}"
                print(f"✗ OpSet {version} 変換エラー: {e}")
        
        # 結果サマリー
        print(f"\n--- 変換結果サマリー ---")
        successful_conversions = [v for v, r in conversion_results.items() if r == "success"]
        failed_conversions = [v for v, r in conversion_results.items() if r != "success"]
        
        print(f"成功: {len(successful_conversions)} / {len(target_versions)}")
        if successful_conversions:
            print(f"  成功したバージョン: {successful_conversions}")
        if failed_conversions:
            print(f"  失敗したバージョン: {failed_conversions}")
        
        return converted_models

def demonstrate_version_conversion():
    """バージョン変換のデモンストレーション"""
    
    # サンプルモデルの作成（OpSet 13）
    from onnx import helper, TensorProto, numpy_helper
    
    # 重みの初期化
    weight = np.random.randn(10, 5).astype(np.float32)
    bias = np.zeros(5, dtype=np.float32)
    
    # ノードの作成
    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["matmul_output"], name="matmul"),
        helper.make_node("Add", ["matmul_output", "bias"], ["add_output"], name="add"),
        helper.make_node("Softmax", ["add_output"], ["output"], axis=1, name="softmax")
    ]
    
    # グラフの作成
    graph = helper.make_graph(
        nodes=nodes,
        name="VersionConversionDemo",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 10])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 5])],
        initializer=[
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias")
        ]
    )
    
    # モデルの作成（OpSet 13）
    model = helper.make_model(graph, producer_name="version_demo")
    model.opset_import[0].version = 13
    
    print("OpSet 13モデルを作成しました")
    
    # バージョンマネージャーの使用
    version_manager = ONNXVersionManager(model)
    
    # 互換性チェック
    target_versions = {"default": 11}
    compatibility_report = version_manager.check_compatibility(target_versions)
    
    # バージョン変換
    converted_model_v11 = version_manager.convert_version(11)
    
    # 複数バージョンへの一括変換
    target_list = [9, 10, 11, 12, 14]
    converted_models = version_manager.batch_convert_multiple_versions(target_list)
    
    return model, converted_models

# 実行例
if __name__ == "__main__":
    demonstrate_version_conversion()
```

### 3.2.3　モデル最適化とグラフ変換

ONNXモデルの性能向上のための最適化技術について学習します。

```python
import onnx
from onnx import helper, optimizer
import numpy as np
from typing import Dict, List, Optional, Set

class ONNXModelOptimizer:
    """ONNX モデル最適化エンジン"""
    
    def __init__(self, model: onnx.ModelProto):
        self.original_model = model
        self.optimized_model = None
        self.optimization_stats = {}
    
    def analyze_optimization_potential(self) -> Dict[str, any]:
        """最適化ポテンシャルの分析"""
        
        print("=== 最適化ポテンシャル分析 ===")
        
        analysis = {
            "node_count": len(self.original_model.graph.node),
            "operator_distribution": {},
            "potential_optimizations": [],
            "graph_complexity": {}
        }
        
        # 演算子分布の分析
        for node in self.original_model.graph.node:
            op_type = node.op_type
            analysis["operator_distribution"][op_type] = analysis["operator_distribution"].get(op_type, 0) + 1
        
        # 最適化の可能性を検出
        self._detect_optimization_opportunities(analysis)
        
        # グラフ複雑度の計算
        analysis["graph_complexity"] = self._calculate_graph_complexity()
        
        # 結果表示
        print(f"総ノード数: {analysis['node_count']}")
        print(f"演算子種類: {len(analysis['operator_distribution'])}種類")
        print("演算子分布:")
        for op_type, count in sorted(analysis["operator_distribution"].items()):
            print(f"  {op_type}: {count}")
        
        if analysis["potential_optimizations"]:
            print("\n検出された最適化機会:")
            for optimization in analysis["potential_optimizations"]:
                print(f"  • {optimization}")
        
        return analysis
    
    def _detect_optimization_opportunities(self, analysis: Dict):
        """最適化機会の検出"""
        
        op_dist = analysis["operator_distribution"]
        nodes = self.original_model.graph.node
        
        # Identity ノードの検出
        if "Identity" in op_dist:
            analysis["potential_optimizations"].append(
                f"Identity演算子の削除 ({op_dist['Identity']}個)"
            )
        
        # Dropout ノードの検出（推論時には不要）
        if "Dropout" in op_dist:
            analysis["potential_optimizations"].append(
                f"Dropout演算子の削除 ({op_dist['Dropout']}個)"
            )
        
        # 連続するTranspose操作の検出
        consecutive_transposes = self._find_consecutive_transposes(nodes)
        if consecutive_transposes:
            analysis["potential_optimizations"].append(
                f"連続するTranspose操作の統合 ({consecutive_transposes}組)"
            )
        
        # 定数の統合機会の検出
        constant_nodes = [node for node in nodes if node.op_type == "Constant"]
        if constant_nodes:
            analysis["potential_optimizations"].append(
                f"定数の初期化子への統合 ({len(constant_nodes)}個)"
            )
        
        # BatchNormalization + Relu パターン
        bn_relu_pairs = self._find_bn_relu_patterns(nodes)
        if bn_relu_pairs:
            analysis["potential_optimizations"].append(
                f"BatchNorm+ReLU融合の可能性 ({bn_relu_pairs}組)"
            )
    
    def _find_consecutive_transposes(self, nodes: List[onnx.NodeProto]) -> int:
        """連続するTranspose操作を検出"""
        count = 0
        node_outputs = {node.name: node.output for node in nodes}
        
        for i, node in enumerate(nodes):
            if node.op_type == "Transpose" and i < len(nodes) - 1:
                # 次のノードがTransposeかチェック
                for output in node.output:
                    for j, next_node in enumerate(nodes[i+1:], i+1):
                        if output in next_node.input and next_node.op_type == "Transpose":
                            count += 1
                            break
        return count
    
    def _find_bn_relu_patterns(self, nodes: List[onnx.NodeProto]) -> int:
        """BatchNorm + ReLU パターンを検出"""
        count = 0
        
        for i, node in enumerate(nodes):
            if node.op_type == "BatchNormalization" and i < len(nodes) - 1:
                for output in node.output:
                    for next_node in nodes[i+1:]:
                        if output in next_node.input and next_node.op_type == "Relu":
                            count += 1
                            break
        return count
    
    def _calculate_graph_complexity(self) -> Dict[str, int]:
        """グラフ複雑度の計算"""
        
        nodes = self.original_model.graph.node
        
        # 入力・出力接続の分析
        all_inputs = set()
        all_outputs = set()
        
        for node in nodes:
            all_inputs.update(node.input)
            all_outputs.update(node.output)
        
        # 分岐・合流の数
        output_usage = {}
        for node in nodes:
            for input_name in node.input:
                output_usage[input_name] = output_usage.get(input_name, 0) + 1
        
        branch_count = sum(1 for count in output_usage.values() if count > 1)
        
        return {
            "total_edges": len(all_inputs),
            "branch_points": branch_count,
            "max_fan_out": max(output_usage.values()) if output_usage else 0
        }
    
    def apply_standard_optimizations(self) -> onnx.ModelProto:
        """標準的な最適化を適用"""
        
        print("\n=== 標準最適化の適用 ===")
        
        try:
            # 利用可能な最適化パスを取得
            available_passes = optimizer.get_available_passes()
            print(f"利用可能な最適化パス: {len(available_passes)}")
            
            # 安全で一般的な最適化パスを選択
            standard_passes = [
                "eliminate_identity",
                "eliminate_dropout", 
                "extract_constant_to_initializer",
                "eliminate_unused_initializer",
                "fuse_consecutive_transposes",
                "fuse_add_bias_into_conv",
                "fuse_bn_into_conv",
                "fuse_transpose_into_gemm"
            ]
            
            # 利用可能なパスのみを選択
            applicable_passes = [p for p in standard_passes if p in available_passes]
            print(f"適用する最適化パス: {applicable_passes}")
            
            # 最適化の実行
            self.optimized_model = optimizer.optimize(self.original_model, applicable_passes)
            
            # 最適化効果の分析
            self._analyze_optimization_effect()
            
            print("✓ 標準最適化が完了しました")
            return self.optimized_model
            
        except ImportError:
            print("⚠️ optimizer モジュールが利用できません")
            self.optimized_model = self.original_model
            return self.optimized_model
        except Exception as e:
            print(f"最適化エラー: {e}")
            self.optimized_model = self.original_model
            return self.optimized_model
    
    def _analyze_optimization_effect(self):
        """最適化効果の分析"""
        
        if not self.optimized_model:
            return
        
        print("\n--- 最適化効果 ---")
        
        # ノード数の変化
        original_nodes = len(self.original_model.graph.node)
        optimized_nodes = len(self.optimized_model.graph.node)
        node_reduction = original_nodes - optimized_nodes
        node_reduction_pct = (node_reduction / original_nodes * 100) if original_nodes > 0 else 0
        
        print(f"ノード数: {original_nodes} → {optimized_nodes} ({node_reduction:+d}, {node_reduction_pct:+.1f}%)")
        
        # 初期化子数の変化
        original_init = len(self.original_model.graph.initializer)
        optimized_init = len(self.optimized_model.graph.initializer)
        init_reduction = original_init - optimized_init
        
        print(f"初期化子数: {original_init} → {optimized_init} ({init_reduction:+d})")
        
        # 演算子タイプの変化
        original_ops = set(node.op_type for node in self.original_model.graph.node)
        optimized_ops = set(node.op_type for node in self.optimized_model.graph.node)
        
        removed_ops = original_ops - optimized_ops
        added_ops = optimized_ops - original_ops
        
        if removed_ops:
            print(f"削除された演算子タイプ: {', '.join(removed_ops)}")
        if added_ops:
            print(f"追加された演算子タイプ: {', '.join(added_ops)}")
        
        # 統計情報の保存
        self.optimization_stats = {
            "node_reduction": node_reduction,
            "node_reduction_percentage": node_reduction_pct,
            "initializer_reduction": init_reduction,
            "removed_operators": list(removed_ops),
            "added_operators": list(added_ops)
        }
    
    def custom_optimization_pass(self) -> onnx.ModelProto:
        """カスタム最適化パスの実装例"""
        
        print("\n=== カスタム最適化パス ===")
        
        if not self.optimized_model:
            working_model = self.original_model
        else:
            working_model = self.optimized_model
        
        # カスタム最適化の実装（例: 不要な Cast 操作の削除）
        optimized_nodes = []
        removed_casts = 0
        
        for node in working_model.graph.node:
            # Cast演算子で入力と出力の型が同じ場合は削除
            if node.op_type == "Cast":
                # 実際の実装では、入力テンソルの型を確認する必要がある
                # ここでは簡単な例として示す
                cast_attr = next((attr for attr in node.attribute if attr.name == "to"), None)
                if cast_attr:
                    # この例では削除条件を簡略化
                    print(f"Cast操作を検討: {node.name}")
                    optimized_nodes.append(node)  # 実際の判定ロジックが必要
                else:
                    optimized_nodes.append(node)
            else:
                optimized_nodes.append(node)
        
        # 新しいグラフを作成
        new_graph = helper.make_graph(
            nodes=optimized_nodes,
            name=working_model.graph.name,
            inputs=working_model.graph.input,
            outputs=working_model.graph.output,
            initializer=working_model.graph.initializer,
            value_info=working_model.graph.value_info
        )
        
        # 新しいモデルを作成
        custom_optimized_model = helper.make_model(new_graph)
        custom_optimized_model.ir_version = working_model.ir_version
        custom_optimized_model.opset_import.extend(working_model.opset_import)
        
        print(f"カスタム最適化完了: {removed_casts}個のCast操作を確認")
        
        return custom_optimized_model

def demonstrate_model_optimization():
    """モデル最適化のデモンストレーション"""
    
    # より複雑なサンプルモデルの作成
    from onnx import helper, TensorProto, numpy_helper
    
    # 重みとバイアス
    conv_weight = np.random.randn(32, 3, 3, 3).astype(np.float32)
    conv_bias = np.zeros(32, dtype=np.float32)
    bn_scale = np.ones(32, dtype=np.float32)
    bn_bias = np.zeros(32, dtype=np.float32)
    bn_mean = np.zeros(32, dtype=np.float32)
    bn_var = np.ones(32, dtype=np.float32)
    
    # 初期化子
    initializers = [
        numpy_helper.from_array(conv_weight, name="conv_weight"),
        numpy_helper.from_array(conv_bias, name="conv_bias"),
        numpy_helper.from_array(bn_scale, name="bn_scale"),
        numpy_helper.from_array(bn_bias, name="bn_bias"),
        numpy_helper.from_array(bn_mean, name="bn_mean"),
        numpy_helper.from_array(bn_var, name="bn_var"),
    ]
    
    # 最適化されやすいパターンを含むノード
    nodes = [
        # Convolution
        helper.make_node(
            "Conv", 
            ["input", "conv_weight", "conv_bias"], 
            ["conv_output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            name="conv"
        ),
        
        # Identity (不要な操作)
        helper.make_node("Identity", ["conv_output"], ["identity_output"], name="identity"),
        
        # BatchNormalization
        helper.make_node(
            "BatchNormalization",
            ["identity_output", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            ["bn_output"],
            epsilon=1e-5,
            name="batch_norm"
        ),
        
        # ReLU
        helper.make_node("Relu", ["bn_output"], ["relu_output"], name="relu"),
        
        # Dropout (推論時不要)
        helper.make_node(
            "Dropout", 
            ["relu_output"], 
            ["dropout_output"],
            ratio=0.5,
            name="dropout"
        ),
        
        # 別のIdentity (不要)
        helper.make_node("Identity", ["dropout_output"], ["output"], name="identity2"),
    ]
    
    # グラフ作成
    graph = helper.make_graph(
        nodes=nodes,
        name="OptimizationDemo",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 224, 224])],
        initializer=initializers
    )
    
    # モデル作成
    model = helper.make_model(graph, producer_name="optimization_demo")
    model.opset_import[0].version = 13
    
    print("最適化対象モデルを作成しました")
    
    # 最適化エンジンの使用
    optimizer_engine = ONNXModelOptimizer(model)
    
    # 最適化ポテンシャルの分析
    analysis = optimizer_engine.analyze_optimization_potential()
    
    # 標準最適化の適用
    optimized_model = optimizer_engine.apply_standard_optimizations()
    
    # カスタム最適化の適用
    custom_optimized = optimizer_engine.custom_optimization_pass()
    
    return model, optimized_model, custom_optimized

# 実行例
if __name__ == "__main__":
    original, standard_opt, custom_opt = demonstrate_model_optimization()
```

## まとめ

第3章では、ONNXの高度な機能と性能分析について包括的に学習しました：

### 学習内容の詳細総括

1. **Python API の完全習得**
   - **モデル読み込み技術**: 基本読み込み、外部データ対応、堅牢な検証付き読み込み
   - **TensorProto操作**: NumPy配列との双方向変換、データタイプ最適化、量子化技術
   - **補助関数による効率的モデル作成**: シンプルな分類器からCNN、動的形状モデルまで
   - **属性マッピングシステム**: フレームワーク間の属性変換、一括変換処理

2. **実用的ONNX機能の活用**
   - **形状推論エンジン**: 自動的なテンソル形状推論、推論前後の詳細比較
   - **バージョン管理**: OpSet互換性チェック、安全なバージョン変換、一括変換処理
   - **モデル最適化**: 標準最適化パス適用、カスタム最適化実装、最適化効果の定量分析

3. **高度な分析・診断ツール**
   - **包括的モデル検査**: 構造分析、データタイプ検証、初期化子健全性チェック
   - **最適化ポテンシャル分析**: 最適化機会の自動検出、グラフ複雑度評価
   - **互換性評価**: 複数バージョンでの互換性テスト、変換リスク評価

### 実践的な開発能力の向上

この章で習得した技術により、以下の高度な開発能力を身につけました：

**モデル品質保証**:
- 堅牢な検証機能を持つモデル読み込み
- 包括的な健全性チェックとエラー検出
- バージョン互換性の事前評価

**性能最適化**:
- 自動的な最適化機会の発見
- 標準的およびカスタムの最適化技術
- 最適化効果の定量的測定

**開発効率化**:
- 補助関数による高速なプロトタイピング
- フレームワーク間での属性マッピング
- バッチ処理による一括変換作業

**本格的な商用システム構築**:
- 外部データを含む大規模モデル対応
- 形状推論による自動的な構造補完
- 複数バージョン対応による長期保守性

次章では、これらの高度な技術を基盤として、データ最適化とモデル変換の深い技術について学習し、さらに高度なONNXエコシステムの活用方法を探求していきます。
