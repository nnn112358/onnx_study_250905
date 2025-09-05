# 第4章 ONNXのデータとオペランド最適化

> 本章の概要: データ最適化とオペランド管理を扱います。廃止演算子の移行、新しい浮動小数点形式（E4M3FNUZ/E5M2FNUZ）の活用など、実務に必要な要点を解説します。

前章ではONNXの高度機能と性能分析を扱いました。本章ではデータ最適化に焦点を当て、商用システムで求められる効率的なデータ処理とオペランド管理を解説します。

現代の機械学習では、計算効率とメモリ効率の最適化が重要課題です。ONNXは、これに対してさまざまなデータタイプと最適化機能を提供しており、適切に活用することで高性能かつ省メモリなモデルを構築できます。

## 4.1 実験演算子と画像カテゴリ定義の管理

### 4.1.1　廃止された実験演算子

ONNXの演算子は時間とともに進化し、一部の実験的演算子は廃止されています。これらを適切に管理する方法を学習します：

```python
import onnx
from onnx import helper, checker
import warnings

class DeprecatedOperatorHandler:
    """廃止された演算子の処理クラス"""
    
    # 廃止された演算子のマッピング
    DEPRECATED_OPS = {
        "Upsample": {
            "replacement": "Resize",
            "migration_guide": "scales属性をsizes属性に変更し、mode='nearest'または'linear'を指定"
        },
        "Scatter": {
            "replacement": "ScatterND",
            "migration_guide": "インデックスの形式とデータレイアウトを調整"
        },
        "ConstantFill": {
            "replacement": "ConstantOfShape",
            "migration_guide": "値指定方法を調整"
        }
    }
    
    def __init__(self, model_path):
        self.model = onnx.load(model_path)
        self.deprecated_nodes = []
        
    def scan_deprecated_operators(self):
        """廃止された演算子をスキャン"""
        print("=== 廃止された演算子のスキャン ===")
        
        self.deprecated_nodes = []
        
        for node in self.model.graph.node:
            if node.op_type in self.DEPRECATED_OPS:
                self.deprecated_nodes.append({
                    'node': node,
                    'op_type': node.op_type,
                    'name': node.name,
                    'replacement': self.DEPRECATED_OPS[node.op_type]
                })
        
        if self.deprecated_nodes:
            print(f"⚠️ {len(self.deprecated_nodes)}個の廃止された演算子が見つかりました:")
            for dep_node in self.deprecated_nodes:
                print(f"  - {dep_node['name']}: {dep_node['op_type']}")
                print(f"    → 推奨: {dep_node['replacement']['replacement']}")
                print(f"    移行ガイド: {dep_node['replacement']['migration_guide']}")
        else:
            print("✓ 廃止された演算子は見つかりませんでした")
        
        return self.deprecated_nodes
    
    def migrate_upsample_to_resize(self, upsample_node):
        """Upsample → Resize の移行"""
        
        # 元の属性を取得
        scales = None
        mode = "nearest"  # デフォルト
        
        for attr in upsample_node.attribute:
            if attr.name == "scales":
                scales = list(attr.floats)
            elif attr.name == "mode":
                mode = attr.s.decode('utf-8')
        
        # Resizeノードを作成
        resize_node = helper.make_node(
            "Resize",
            inputs=upsample_node.input + ["", "scales"],  # ROIは空、scalesを追加
            outputs=upsample_node.output,
            name=upsample_node.name + "_migrated",
            mode=mode,
            coordinate_transformation_mode="asymmetric"  # Upsampleの動作に合わせる
        )
        
        # スケール値の初期化子を作成
        if scales:
            scales_initializer = helper.make_tensor(
                name="scales",
                data_type=onnx.TensorProto.FLOAT,
                dims=[len(scales)],
                vals=scales
            )
            
            # モデルに初期化子を追加
            self.model.graph.initializer.append(scales_initializer)
        
        return resize_node
    
    def auto_migrate_model(self):
        """自動移行の実行"""
        print("\n=== 自動移行の実行 ===")
        
        if not self.deprecated_nodes:
            print("移行が必要なノードはありません")
            return self.model
        
        new_nodes = []
        migrated_count = 0
        
        for node in self.model.graph.node:
            if node.op_type == "Upsample":
                # Upsampleを移行
                new_node = self.migrate_upsample_to_resize(node)
                new_nodes.append(new_node)
                migrated_count += 1
                print(f"✓ {node.name}: Upsample → Resize に移行")
            else:
                new_nodes.append(node)
        
        # グラフのノードを更新
        del self.model.graph.node[:]
        self.model.graph.node.extend(new_nodes)
        
        print(f"✓ {migrated_count}個のノードを移行しました")
        
        return self.model

def create_model_with_deprecated_ops():
    """廃止された演算子を含むモデルの作成（テスト用）"""
    
    # Upsample演算子を含むモデルを作成
    upsample_node = helper.make_node(
        "Upsample",
        inputs=["input"],
        outputs=["upsampled"],
        name="upsample_deprecated",
        scales=[1.0, 1.0, 2.0, 2.0],  # NCHW format
        mode="nearest"
    )
    
    # グラフ作成
    graph = helper.make_graph(
        nodes=[upsample_node],
        name="DeprecatedOpsModel",
        inputs=[helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info("upsampled", onnx.TensorProto.FLOAT, [1, 3, 64, 64])]
    )
    
    # モデル作成
    model = helper.make_model(graph, producer_name="deprecated_ops_demo")
    
    # OpSetバージョンを古いものに設定（Upsampleが存在した時代）
    model.opset_import[0].version = 9
    
    onnx.save(model, "model_with_deprecated_ops.onnx")
    return model

# 使用例
def demo_deprecated_ops_handling():
    """廃止演算子処理のデモ"""
    
    # 廃止演算子を含むモデルを作成
    create_model_with_deprecated_ops()
    
    # 処理クラスのインスタンス化
    handler = DeprecatedOperatorHandler("model_with_deprecated_ops.onnx")
    
    # 廃止演算子のスキャン
    deprecated_ops = handler.scan_deprecated_operators()
    
    # 自動移行の実行
    if deprecated_ops:
        migrated_model = handler.auto_migrate_model()
        
        # 移行後のモデルを保存
        onnx.save(migrated_model, "model_migrated.onnx")
        print("\n✓ 移行後のモデルを保存しました: model_migrated.onnx")
        
        # 移行後のモデルを検証
        try:
            checker.check_model(migrated_model)
            print("✓ 移行後のモデル検証に成功しました")
        except Exception as e:
            print(f"✗ 移行後のモデル検証でエラー: {e}")

demo_deprecated_ops_handling()
```

### 4.1.2　画像カテゴリ定義

画像分類タスクでのカテゴリ定義と管理：

```python
import onnx
from onnx import helper, TensorProto
import numpy as np
import json

class ImageCategoryManager:
    """画像カテゴリ定義管理クラス"""
    
    def __init__(self):
        self.categories = {}
        self.category_mappings = {}
    
    def load_imagenet_categories(self):
        """ImageNetカテゴリの読み込み（サンプル）"""
        
        # ImageNet1000クラスのサンプル（実際は1000クラス）
        imagenet_sample = {
            0: {"id": "n01440764", "name": "tench", "description": "Tinca tinca"},
            1: {"id": "n01443537", "name": "goldfish", "description": "Carassius auratus"},
            2: {"id": "n01484850", "name": "great_white_shark", "description": "Carcharodon carcharias"},
            3: {"id": "n01491361", "name": "tiger_shark", "description": "Galeocerdo cuvier"},
            4: {"id": "n01494475", "name": "hammerhead", "description": "hammerhead shark"},
            # ... 実際は996個続く
            999: {"id": "n15075141", "name": "toilet_tissue", "description": "bathroom tissue"}
        }
        
        self.categories["imagenet"] = imagenet_sample
        print(f"ImageNetカテゴリを読み込みました: {len(imagenet_sample)}クラス")
        
        return imagenet_sample
    
    def create_custom_categories(self, category_dict):
        """カスタムカテゴリの作成"""
        
        self.categories["custom"] = category_dict
        print(f"カスタムカテゴリを作成しました: {len(category_dict)}クラス")
        
        # カテゴリ情報の表示
        for idx, (class_id, info) in enumerate(category_dict.items()):
            if idx < 5:  # 最初の5つのみ表示
                print(f"  {class_id}: {info}")
        
        return category_dict
    
    def create_category_mapping(self, source_categories, target_categories):
        """カテゴリ間のマッピング作成"""
        
        mapping = {}
        
        # 名前ベースのマッピング（簡単な例）
        for src_id, src_info in source_categories.items():
            src_name = src_info.get("name", "").lower()
            
            for tgt_id, tgt_info in target_categories.items():
                tgt_name = tgt_info.get("name", "").lower()
                
                if src_name == tgt_name or src_name in tgt_name or tgt_name in src_name:
                    mapping[src_id] = tgt_id
                    break
        
        print(f"カテゴリマッピングを作成しました: {len(mapping)}個のマッピング")
        
        return mapping
    
    def add_category_metadata_to_model(self, model, category_type="imagenet"):
        """モデルにカテゴリメタデータを追加"""
        
        if category_type not in self.categories:
            raise ValueError(f"カテゴリタイプ {category_type} が見つかりません")
        
        categories = self.categories[category_type]
        
        # カテゴリ情報をJSONとしてシリアライズ
        category_json = json.dumps(categories, ensure_ascii=False, indent=2)
        
        # モデルのメタデータに追加
        metadata_props = model.metadata_props
        
        # 既存のメタデータをチェック
        category_key = f"categories_{category_type}"
        
        # メタデータプロパティを追加
        category_prop = onnx.StringStringEntryProto()
        category_prop.key = category_key
        category_prop.value = category_json
        
        metadata_props.append(category_prop)
        
        print(f"✓ モデルに{category_type}カテゴリメタデータを追加しました")
        
        return model

def create_classification_model_with_categories():
    """カテゴリ情報付き分類モデルの作成"""
    
    print("=== カテゴリ情報付き分類モデルの作成 ===")
    
    # シンプルな分類モデル
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, 3, 224, 224]
    )
    
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 10]  # 10クラス分類
    )
    
    # ダミーの重み
    weight = np.random.randn(150528, 10).astype(np.float32) * 0.01  # 適当なサイズ
    bias = np.zeros(10, dtype=np.float32)
    
    weight_initializer = helper.make_tensor(
        "weight", TensorProto.FLOAT, [150528, 10], weight.flatten()
    )
    bias_initializer = helper.make_tensor(
        "bias", TensorProto.FLOAT, [10], bias
    )
    
    # ノード作成（簡略化）
    flatten_node = helper.make_node(
        "Flatten", ["input"], ["flattened"], axis=1, name="flatten"
    )
    
    matmul_node = helper.make_node(
        "MatMul", ["flattened", "weight"], ["matmul_out"], name="fc"
    )
    
    add_node = helper.make_node(
        "Add", ["matmul_out", "bias"], ["output"], name="add_bias"
    )
    
    # グラフ作成
    graph = helper.make_graph(
        nodes=[flatten_node, matmul_node, add_node],
        name="ClassificationWithCategories",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_initializer, bias_initializer]
    )
    
    # モデル作成
    model = helper.make_model(graph, producer_name="category_demo")
    
    # カテゴリマネージャーでカテゴリ情報を追加
    category_manager = ImageCategoryManager()
    
    # カスタムカテゴリの作成（10クラス）
    custom_categories = {
        0: {"name": "cat", "description": "domestic cat"},
        1: {"name": "dog", "description": "domestic dog"},
        2: {"name": "bird", "description": "flying bird"},
        3: {"name": "fish", "description": "swimming fish"},
        4: {"name": "car", "description": "automobile"},
        5: {"name": "bicycle", "description": "two-wheel bicycle"},
        6: {"name": "airplane", "description": "flying aircraft"},
        7: {"name": "ship", "description": "water vessel"},
        8: {"name": "tree", "description": "woody plant"},
        9: {"name": "flower", "description": "blooming plant"}
    }
    
    category_manager.create_custom_categories(custom_categories)
    
    # モデルにカテゴリ情報を追加
    model_with_categories = category_manager.add_category_metadata_to_model(
        model, "custom"
    )
    
    # モデルを保存
    onnx.save(model_with_categories, "classification_with_categories.onnx")
    
    return model_with_categories, category_manager

def extract_categories_from_model(model_path):
    """モデルからカテゴリ情報を抽出"""
    
    print("=== モデルからカテゴリ情報を抽出 ===")
    
    model = onnx.load(model_path)
    
    # メタデータからカテゴリ情報を抽出
    categories_found = {}
    
    for metadata_prop in model.metadata_props:
        if metadata_prop.key.startswith("categories_"):
            category_type = metadata_prop.key.replace("categories_", "")
            try:
                categories_data = json.loads(metadata_prop.value)
                categories_found[category_type] = categories_data
                print(f"✓ {category_type}カテゴリを抽出: {len(categories_data)}クラス")
            except json.JSONDecodeError as e:
                print(f"✗ {category_type}カテゴリの解析エラー: {e}")
    
    if not categories_found:
        print("カテゴリ情報が見つかりませんでした")
    
    return categories_found

# 使用例とテスト
def demo_category_management():
    """カテゴリ管理のデモ"""
    
    # カテゴリ付きモデルの作成
    model, category_manager = create_classification_model_with_categories()
    
    # モデルからカテゴリ情報を抽出してテスト
    extracted_categories = extract_categories_from_model("classification_with_categories.onnx")
    
    # 抽出されたカテゴリの表示
    if extracted_categories:
        for category_type, categories in extracted_categories.items():
            print(f"\n{category_type}カテゴリの詳細:")
            for class_id, info in list(categories.items())[:5]:  # 最初の5つのみ
                print(f"  {class_id}: {info}")

demo_category_management()
```

## 4.2　ONNXタイプ

### 4.2.1　PyTorchでの例

PyTorchからONNXへの型変換とデータタイプの対応：

```python
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

class TypeMappingDemo:
    """PyTorch-ONNX間のタイプマッピングデモ"""
    
    def __init__(self):
        self.torch_to_onnx_types = {
            torch.float32: onnx.TensorProto.FLOAT,
            torch.float64: onnx.TensorProto.DOUBLE,
            torch.float16: onnx.TensorProto.FLOAT16,
            torch.int8: onnx.TensorProto.INT8,
            torch.int16: onnx.TensorProto.INT16,
            torch.int32: onnx.TensorProto.INT32,
            torch.int64: onnx.TensorProto.INT64,
            torch.uint8: onnx.TensorProto.UINT8,
            torch.bool: onnx.TensorProto.BOOL,
        }
    
    def create_multi_type_model(self):
        """複数のデータタイプを使用するモデル"""
        
        class MultiTypeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_float32 = nn.Conv2d(3, 16, 3, padding=1)
                self.linear_float32 = nn.Linear(16 * 32 * 32, 10)
                
            def forward(self, x_float32):
                # float32での処理
                x = self.conv_float32(x_float32)
                x = torch.relu(x)
                x = x.view(x.size(0), -1)
                
                # float16への変換（混合精度の例）
                x_float16 = x.half()
                
                # 再びfloat32に戻してlinear層へ
                x_float32_back = x_float16.float()
                output = self.linear_float32(x_float32_back)
                
                return output
        
        return MultiTypeModel()
    
    def export_with_different_types(self):
        """異なるデータタイプでのエクスポート"""
        
        print("=== 異なるデータタイプでのエクスポート ===")
        
        model = self.create_multi_type_model()
        model.eval()
        
        # ダミー入力
        dummy_input = torch.randn(1, 3, 32, 32)
        
        # 1. float32でのエクスポート（デフォルト）
        torch.onnx.export(
            model,
            dummy_input,
            "model_float32.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        # 2. エクスポートされたモデルのタイプ情報を確認
        self.analyze_model_types("model_float32.onnx")
        
        # 3. float16モデルの作成とエクスポート
        model_fp16 = model.half()
        dummy_input_fp16 = dummy_input.half()
        
        try:
            torch.onnx.export(
                model_fp16,
                dummy_input_fp16,
                "model_float16.onnx",
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
            
            self.analyze_model_types("model_float16.onnx")
            
        except Exception as e:
            print(f"float16エクスポートエラー: {e}")
    
    def analyze_model_types(self, model_path):
        """モデルのデータタイプを分析"""
        
        print(f"\n--- {model_path} のタイプ分析 ---")
        
        model = onnx.load(model_path)
        
        # 入力タイプ
        print("入力:")
        for input_tensor in model.graph.input:
            if input_tensor.type.tensor_type:
                dtype = input_tensor.type.tensor_type.elem_type
                dtype_name = onnx.TensorProto.DataType.Name(dtype)
                print(f"  {input_tensor.name}: {dtype_name}")
        
        # 初期化子タイプ
        type_counts = {}
        for initializer in model.graph.initializer:
            dtype_name = onnx.TensorProto.DataType.Name(initializer.data_type)
            type_counts[dtype_name] = type_counts.get(dtype_name, 0) + 1
        
        print("初期化子（重み）:")
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count}個")
        
        # 出力タイプ
        print("出力:")
        for output_tensor in model.graph.output:
            if output_tensor.type.tensor_type:
                dtype = output_tensor.type.tensor_type.elem_type
                dtype_name = onnx.TensorProto.DataType.Name(dtype)
                print(f"  {output_tensor.name}: {dtype_name}")

def demonstrate_quantization_types():
    """量子化タイプのデモ"""
    
    print("\n=== 量子化タイプのデモ ===")
    
    # int8量子化の例
    def create_quantized_model():
        """量子化モデルの作成"""
        
        from onnx import helper, TensorProto, numpy_helper
        
        # float32の重み
        weight_fp32 = np.random.randn(10, 5).astype(np.float32)
        
        # int8に量子化（簡単な例）
        scale = np.abs(weight_fp32).max() / 127.0
        weight_int8 = np.round(weight_fp32 / scale).clip(-128, 127).astype(np.int8)
        
        print(f"量子化スケール: {scale}")
        print(f"元の重み範囲: [{weight_fp32.min():.3f}, {weight_fp32.max():.3f}]")
        print(f"量子化後範囲: [{weight_int8.min()}, {weight_int8.max()}]")
        
        # 初期化子の作成
        weight_initializer = numpy_helper.from_array(weight_int8, name="quantized_weight")
        scale_initializer = numpy_helper.from_array(np.array([scale], dtype=np.float32), name="scale")
        zero_point_initializer = numpy_helper.from_array(np.array([0], dtype=np.int8), name="zero_point")
        
        # DequantizeLinearノード（int8 → float32）
        dequant_node = helper.make_node(
            "DequantizeLinear",
            inputs=["quantized_weight", "scale", "zero_point"],
            outputs=["dequantized_weight"],
            name="dequantize"
        )
        
        # MatMulノード
        matmul_node = helper.make_node(
            "MatMul",
            inputs=["input", "dequantized_weight"],
            outputs=["output"],
            name="matmul"
        )
        
        # グラフ作成
        graph = helper.make_graph(
            nodes=[dequant_node, matmul_node],
            name="QuantizedModel",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 10])],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 5])],
            initializer=[weight_initializer, scale_initializer, zero_point_initializer]
        )
        
        # モデル作成
        model = helper.make_model(graph, producer_name="quantization_demo")
        model.opset_import[0].version = 13  # DequantizeLinearをサポート
        
        return model
    
    # 量子化モデルの作成と保存
    quantized_model = create_quantized_model()
    onnx.save(quantized_model, "quantized_model.onnx")
    
    # タイプ分析
    demo = TypeMappingDemo()
    demo.analyze_model_types("quantized_model.onnx")

def test_type_conversion_runtime():
    """ランタイムでのタイプ変換テスト"""
    
    print("\n=== ランタイム タイプ変換テスト ===")
    
    # float16モデルでの推論テスト
    try:
        session_fp32 = ort.InferenceSession("model_float32.onnx")
        
        # float32入力での推論
        input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
        
        outputs_fp32 = session_fp32.run(None, {"input": input_data})
        
        print("✓ float32モデルでの推論成功")
        print(f"  出力形状: {outputs_fp32[0].shape}")
        print(f"  出力データタイプ: {outputs_fp32[0].dtype}")
        
    except Exception as e:
        print(f"✗ float32推論エラー: {e}")
    
    # float16モデルでの推論テスト（存在する場合）
    try:
        session_fp16 = ort.InferenceSession("model_float16.onnx")
        
        # float16入力での推論
        input_data_fp16 = input_data.astype(np.float16)
        
        outputs_fp16 = session_fp16.run(None, {"input": input_data_fp16})
        
        print("✓ float16モデルでの推論成功")
        print(f"  出力形状: {outputs_fp16[0].shape}")
        print(f"  出力データタイプ: {outputs_fp16[0].dtype}")
        
    except FileNotFoundError:
        print("float16モデルが見つかりません（エクスポート時にエラーが発生した可能性）")
    except Exception as e:
        print(f"✗ float16推論エラー: {e}")

# 実行例
def run_type_demos():
    """タイプデモの実行"""
    
    demo = TypeMappingDemo()
    
    # PyTorchからのエクスポート
    demo.export_with_different_types()
    
    # 量子化タイプのデモ
    demonstrate_quantization_types()
    
    # ランタイムテスト
    test_type_conversion_runtime()

if __name__ == "__main__":
    run_type_demos()
```

### 4.2.2　演算子慣例

ONNX演算子の命名規則とベストプラクティス：

```python
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

class OperatorConventions:
    """ONNX演算子の慣例とベストプラクティス"""
    
    def __init__(self):
        self.naming_conventions = {
            "input_prefix": "input_",
            "output_prefix": "output_",
            "weight_prefix": "weight_",
            "bias_prefix": "bias_",
            "intermediate_prefix": "intermediate_"
        }
    
    def demonstrate_naming_conventions(self):
        """命名規則のデモンストレーション"""
        
        print("=== ONNX演算子命名規則 ===")
        
        # 1. 適切な命名例
        good_names = {
            "inputs": ["input_image", "input_sequence", "input_features"],
            "outputs": ["output_logits", "output_probabilities", "output_embeddings"],
            "weights": ["weight_conv1", "weight_fc_classifier", "weight_embedding"],
            "biases": ["bias_conv1", "bias_fc_classifier"],
            "intermediates": ["conv1_output", "relu1_output", "pool1_output"]
        }
        
        print("✓ 適切な命名例:")
        for category, names in good_names.items():
            print(f"  {category}: {names}")
        
        # 2. 避けるべき命名例
        bad_names = {
            "ambiguous": ["x", "y", "z", "temp", "data"],
            "non_descriptive": ["layer1", "node2", "tensor3"],
            "inconsistent": ["InputImage", "output_logits", "Weight_1"]
        }
        
        print("\n✗ 避けるべき命名例:")
        for category, names in bad_names.items():
            print(f"  {category}: {names}")
    
    def create_well_named_model(self):
        """適切な命名規則に従ったモデル"""
        
        print("\n=== 適切な命名のモデル作成 ===")
        
        # 入力・出力テンソル（わかりやすい名前）
        input_tensor = helper.make_tensor_value_info(
            "input_rgb_image", TensorProto.FLOAT, [None, 3, 224, 224]
        )
        
        output_tensor = helper.make_tensor_value_info(
            "output_class_probabilities", TensorProto.FLOAT, [None, 1000]
        )
        
        # 重みとバイアス（層ごとに明確な名前）
        conv1_weight = np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.01
        conv1_bias = np.zeros(64, dtype=np.float32)
        
        classifier_weight = np.random.randn(25088, 1000).astype(np.float32) * 0.01
        classifier_bias = np.zeros(1000, dtype=np.float32)
        
        # 初期化子（説明的な名前）
        initializers = [
            numpy_helper.from_array(conv1_weight, name="weight_conv1_feature_extractor"),
            numpy_helper.from_array(conv1_bias, name="bias_conv1_feature_extractor"),
            numpy_helper.from_array(classifier_weight, name="weight_classifier_head"),
            numpy_helper.from_array(classifier_bias, name="bias_classifier_head"),
        ]
        
        # ノード（処理内容が明確な名前）
        nodes = [
            helper.make_node(
                "Conv",
                inputs=["input_rgb_image", "weight_conv1_feature_extractor", "bias_conv1_feature_extractor"],
                outputs=["conv1_feature_maps"],
                kernel_shape=[7, 7],
                pads=[3, 3, 3, 3],
                strides=[2, 2],
                name="conv1_feature_extraction"
            ),
            
            helper.make_node(
                "Relu",
                inputs=["conv1_feature_maps"],
                outputs=["conv1_activated_features"],
                name="conv1_activation"
            ),
            
            helper.make_node(
                "GlobalAveragePool",
                inputs=["conv1_activated_features"],
                outputs=["global_pooled_features"],
                name="global_average_pooling"
            ),
            
            helper.make_node(
                "Flatten",
                inputs=["global_pooled_features"],
                outputs=["flattened_features"],
                axis=1,
                name="feature_flattening"
            ),
            
            helper.make_node(
                "MatMul",
                inputs=["flattened_features", "weight_classifier_head"],
                outputs=["classification_logits"],
                name="classification_projection"
            ),
            
            helper.make_node(
                "Add",
                inputs=["classification_logits", "bias_classifier_head"],
                outputs=["biased_logits"],
                name="bias_addition"
            ),
            
            helper.make_node(
                "Softmax",
                inputs=["biased_logits"],
                outputs=["output_class_probabilities"],
                axis=1,
                name="probability_normalization"
            ),
        ]
        
        # グラフ作成
        graph = helper.make_graph(
            nodes=nodes,
            name="WellNamedImageClassifier",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=initializers
        )
        
        # モデル作成（メタデータも充実）
        model = helper.make_model(
            graph,
            producer_name="onnx_naming_convention_demo",
            producer_version="1.0",
            doc_string="Properly named image classification model for educational purposes"
        )
        
        # モデル情報の追加
        model.model_version = 1
        
        # メタデータの追加
        metadata = [
            ("description", "Image classification model with proper naming conventions"),
            ("input_description", "RGB image tensor in NCHW format"),
            ("output_description", "Class probabilities for 1000 ImageNet categories"),
            ("model_type", "image_classification"),
            ("framework", "onnx_tutorial")
        ]
        
        for key, value in metadata:
            meta_prop = onnx.StringStringEntryProto()
            meta_prop.key = key
            meta_prop.value = value
            model.metadata_props.append(meta_prop)
        
        return model
    
    def validate_naming_conventions(self, model):
        """命名規則の検証"""
        
        print("\n=== 命名規則検証 ===")
        
        issues = []
        
        # 1. 入力名の検証
        for input_tensor in model.graph.input:
            name = input_tensor.name
            if len(name) < 3:
                issues.append(f"入力名が短すぎます: {name}")
            if name.lower() in ['x', 'input', 'data']:
                issues.append(f"入力名が汎用的すぎます: {name}")
        
        # 2. 出力名の検証
        for output_tensor in model.graph.output:
            name = output_tensor.name
            if len(name) < 3:
                issues.append(f"出力名が短すぎます: {name}")
            if name.lower() in ['y', 'output', 'result']:
                issues.append(f"出力名が汎用的すぎます: {name}")
        
        # 3. ノード名の検証
        for node in model.graph.node:
            name = node.name
            if not name:
                issues.append(f"ノード名が空です: {node.op_type}")
            elif len(name) < 3:
                issues.append(f"ノード名が短すぎます: {name}")
        
        # 4. 初期化子名の検証
        for initializer in model.graph.initializer:
            name = initializer.name
            if len(name) < 3:
                issues.append(f"初期化子名が短すぎます: {name}")
            if not any(prefix in name.lower() for prefix in ['weight', 'bias', 'scale', 'mean', 'var']):
                issues.append(f"初期化子の目的が不明: {name}")
        
        # 結果報告
        if issues:
            print(f"✗ {len(issues)}個の命名規則違反が見つかりました:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ 命名規則に問題はありません")
        
        return len(issues) == 0

def demonstrate_operator_best_practices():
    """演算子のベストプラクティス"""
    
    print("\n=== 演算子ベストプラクティス ===")
    
    best_practices = {
        "attribute_naming": {
            "description": "属性名は演算子の仕様に厳密に従う",
            "good": "kernel_shape=[3, 3]",
            "bad": "kernel_size=[3, 3]"
        },
        "input_output_order": {
            "description": "入力・出力の順序は仕様書通りに",
            "good": "Conv(input, weight, bias)",
            "bad": "Conv(weight, input, bias)"
        },
        "data_layout": {
            "description": "データレイアウトを明示",
            "good": "input shape: [N, C, H, W] (NCHW format)",
            "bad": "input shape: [1, 224, 224, 3] (format unclear)"
        },
        "version_compatibility": {
            "description": "OpSetバージョンを考慮した演算子選択",
            "good": "Resize (OpSet 11+) for upsampling",
            "bad": "Upsample (deprecated in OpSet 11)"
        }
    }
    
    for practice, info in best_practices.items():
        print(f"\n{practice.upper()}:")
        print(f"  説明: {info['description']}")
        print(f"  ✓ 良い例: {info['good']}")
        print(f"  ✗ 悪い例: {info['bad']}")

# 実行例
def demo_operator_conventions():
    """演算子慣例のデモ実行"""
    
    conventions = OperatorConventions()
    
    # 命名規則の説明
    conventions.demonstrate_naming_conventions()
    
    # 適切に命名されたモデルの作成
    well_named_model = conventions.create_well_named_model()
    onnx.save(well_named_model, "well_named_model.onnx")
    
    # 命名規則の検証
    is_valid = conventions.validate_naming_conventions(well_named_model)
    
    # ベストプラクティスの説明
    demonstrate_operator_best_practices()
    
    print(f"\n✓ 適切に命名されたモデルを保存しました: well_named_model.onnx")
    print(f"命名規則適合性: {'✓ 合格' if is_valid else '✗ 要改善'}")

if __name__ == "__main__":
    demo_operator_conventions()
```

## 4.3　E4M3FNUZとE5M2FNUZ

新しい浮動小数点形式（E4M3FNUZ、E5M2FNUZ）の説明と使用法：

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import struct

class FloatFormatHandler:
    """新しい浮動小数点形式のハンドラー"""
    
    def __init__(self):
        self.formats = {
            "E4M3FNUZ": {
                "exponent_bits": 4,
                "mantissa_bits": 3,
                "finite": True,
                "no_negative_zero": True,
                "description": "4-bit exponent, 3-bit mantissa, finite values, no negative zero"
            },
            "E5M2FNUZ": {
                "exponent_bits": 5, 
                "mantissa_bits": 2,
                "finite": True,
                "no_negative_zero": True,
                "description": "5-bit exponent, 2-bit mantissa, finite values, no negative zero"
            }
        }
    
    def explain_format_characteristics(self):
        """浮動小数点形式の特性を説明"""
        
        print("=== 新しい浮動小数点形式 ===")
        
        for format_name, props in self.formats.items():
            print(f"\n{format_name}:")
            print(f"  指数部: {props['exponent_bits']} bits")
            print(f"  仮数部: {props['mantissa_bits']} bits")
            print(f"  有限値のみ: {props['finite']}")
            print(f"  負のゼロなし: {props['no_negative_zero']}")
            print(f"  説明: {props['description']}")
            
            # 理論的な値の範囲を計算
            total_bits = 1 + props['exponent_bits'] + props['mantissa_bits']
            max_exponent = (2 ** props['exponent_bits']) - 1
            
            print(f"  総ビット数: {total_bits}")
            print(f"  最大指数: {max_exponent}")
    
    def simulate_precision_loss(self):
        """精度損失のシミュレーション"""
        
        print("\n=== 精度損失シミュレーション ===")
        
        # オリジナルのfloat32値
        original_values = [
            0.1234567,
            1.0,
            100.0,
            0.001,
            -0.5,
            3.14159,
            0.0
        ]
        
        print("元の値 (float32) → シミュレートされた低精度値:")
        
        for value in original_values:
            # E4M3形式のシミュレート（簡易版）
            e4m3_sim = self._simulate_e4m3(value)
            
            # E5M2形式のシミュレート（簡易版）
            e5m2_sim = self._simulate_e5m2(value)
            
            print(f"  {value:10.7f} → E4M3: {e4m3_sim:10.7f}, E5M2: {e5m2_sim:10.7f}")
    
    def _simulate_e4m3(self, value):
        """E4M3形式の簡易シミュレーション"""
        if value == 0.0:
            return 0.0
        
        # 符号、指数、仮数の分離（簡略化）
        sign = -1 if value < 0 else 1
        abs_value = abs(value)
        
        if abs_value == 0:
            return 0.0
        
        # log2での指数計算
        log_val = np.log2(abs_value)
        exp = int(np.floor(log_val))
        
        # 指数の制限（E4M3の範囲内）
        exp = max(-7, min(8, exp))  # 4-bitの制限
        
        # 仮数の量子化（3-bitの制限）
        mantissa_scale = 2 ** (3)  # 3-bit mantissa
        normalized_mantissa = abs_value / (2 ** exp)
        quantized_mantissa = np.round((normalized_mantissa - 1.0) * mantissa_scale) / mantissa_scale + 1.0
        
        result = sign * quantized_mantissa * (2 ** exp)
        return result
    
    def _simulate_e5m2(self, value):
        """E5M2形式の簡易シミュレーション"""
        if value == 0.0:
            return 0.0
        
        sign = -1 if value < 0 else 1
        abs_value = abs(value)
        
        if abs_value == 0:
            return 0.0
        
        log_val = np.log2(abs_value)
        exp = int(np.floor(log_val))
        
        # 指数の制限（E5M2の範囲内）
        exp = max(-15, min(16, exp))  # 5-bitの制限
        
        # 仮数の量子化（2-bitの制限）
        mantissa_scale = 2 ** (2)  # 2-bit mantissa
        normalized_mantissa = abs_value / (2 ** exp)
        quantized_mantissa = np.round((normalized_mantissa - 1.0) * mantissa_scale) / mantissa_scale + 1.0
        
        result = sign * quantized_mantissa * (2 ** exp)
        return result

### 4.3.1　指数バイアス問題

def explain_exponent_bias():
    """指数バイアスの説明"""
    
    print("\n=== 指数バイアス問題 ===")
    
    bias_examples = {
        "IEEE 754 float32": {
            "exponent_bits": 8,
            "bias": 127,
            "range": "[-126, 127]"
        },
        "IEEE 754 float16": {
            "exponent_bits": 5,
            "bias": 15,
            "range": "[-14, 15]"
        },
        "E4M3FNUZ": {
            "exponent_bits": 4,
            "bias": 7,
            "range": "[-7, 8]"
        },
        "E5M2FNUZ": {
            "exponent_bits": 5,
            "bias": 15,
            "range": "[-15, 16]"
        }
    }
    
    print("各形式の指数バイアス:")
    for format_name, props in bias_examples.items():
        print(f"  {format_name}:")
        print(f"    指数ビット数: {props['exponent_bits']}")
        print(f"    バイアス: {props['bias']}")
        print(f"    実効指数範囲: {props['range']}")
    
    print("\n指数バイアスの目的:")
    print("  - 指数を符号なし整数として格納可能にする")
    print("  - ハードウェア実装を簡素化")
    print("  - 浮動小数点数の比較を整数比較で実現")

### 4.3.2　Castノードでのデータタイプ変換

def demonstrate_cast_operations():
    """Castノードでのデータタイプ変換"""
    
    print("\n=== Castノードでのデータタイプ変換 ===")
    
    # さまざまなタイプ変換のデモ
    conversions = [
        (TensorProto.FLOAT, TensorProto.FLOAT16, "float32 → float16"),
        (TensorProto.FLOAT16, TensorProto.FLOAT, "float16 → float32"),
        (TensorProto.FLOAT, TensorProto.INT8, "float32 → int8"),
        (TensorProto.INT8, TensorProto.FLOAT, "int8 → float32"),
        (TensorProto.DOUBLE, TensorProto.FLOAT, "double → float32")
    ]
    
    nodes = []
    initializers = []
    
    # 元データの作成
    input_data = np.array([1.5, -2.3, 0.1, 100.7, -0.001], dtype=np.float32)
    input_initializer = helper.make_tensor(
        "input_data", TensorProto.FLOAT, [5], input_data.flatten()
    )
    initializers.append(input_initializer)
    
    # 各変換のノードを作成
    for i, (from_type, to_type, desc) in enumerate(conversions):
        input_name = "input_data" if i == 0 else f"cast_output_{i-1}"
        output_name = f"cast_output_{i}"
        
        cast_node = helper.make_node(
            "Cast",
            inputs=[input_name],
            outputs=[output_name],
            to=to_type,
            name=f"cast_{desc.replace(' ', '_').replace('→', 'to')}"
        )
        nodes.append(cast_node)
        
        print(f"ノード {i+1}: {desc}")
        print(f"  入力: {TensorProto.DataType.Name(from_type)}")
        print(f"  出力: {TensorProto.DataType.Name(to_type)}")
    
    # 最終出力として元のfloat32に戻す
    final_cast = helper.make_node(
        "Cast",
        inputs=[f"cast_output_{len(conversions)-1}"],
        outputs=["final_output"],
        to=TensorProto.FLOAT,
        name="final_cast_to_float32"
    )
    nodes.append(final_cast)
    
    # グラフ作成
    graph = helper.make_graph(
        nodes=nodes,
        name="CastDemonstration",
        inputs=[],  # 初期化子から開始
        outputs=[helper.make_tensor_value_info("final_output", TensorProto.FLOAT, [5])],
        initializer=initializers
    )
    
    # モデル作成
    model = helper.make_model(graph, producer_name="cast_demo")
    model.opset_import[0].version = 13
    
    return model

def test_cast_precision_loss():
    """Cast操作での精度損失のテスト"""
    
    print("\n=== Cast精度損失テスト ===")
    
    # 精度損失を起こしやすい値
    test_values = [
        3.141592653589793,  # π（高精度）
        1.0000001,          # 1に近い値
        0.0000001,          # 非常に小さな値
        1000000.5,          # 大きな値の小数部
        -0.123456789        # 負の小数
    ]
    
    print("精度損失テスト値:")
    for i, value in enumerate(test_values):
        # 各形式での表現
        float16_val = np.float16(value)
        
        # int8への変換（スケーリング必要）
        scaled_int8 = int(np.clip(value * 100, -128, 127))
        recovered_from_int8 = scaled_int8 / 100.0
        
        print(f"  値{i+1}: {value}")
        print(f"    float32: {np.float32(value)}")
        print(f"    float16: {float16_val} (誤差: {abs(float16_val - value):.2e})")
        print(f"    int8→float: {recovered_from_int8} (誤差: {abs(recovered_from_int8 - value):.2e})")

# 実行例
def demo_new_float_formats():
    """新しい浮動小数点形式のデモ"""
    
    handler = FloatFormatHandler()
    
    # 形式の特性説明
    handler.explain_format_characteristics()
    
    # 精度損失シミュレーション
    handler.simulate_precision_loss()
    
    # 指数バイアス説明
    explain_exponent_bias()
    
    # Cast操作のデモ
    cast_model = demonstrate_cast_operations()
    onnx.save(cast_model, "cast_demo_model.onnx")
    
    # 精度損失テスト
    test_cast_precision_loss()
    
    print("\n✓ Castデモモデルを保存しました: cast_demo_model.onnx")

if __name__ == "__main__":
    demo_new_float_formats()
```

## 4.4　高度なデータタイプ最適化

### 4.4.1　整数タイプの詳細活用

ONNXでサポートされる整数タイプの特性と最適化技術について学習します：

```python
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

class IntegerTypeOptimizer:
    """整数タイプ最適化クラス"""
    
    def __init__(self):
        self.integer_types = {
            TensorProto.INT8: {"range": (-128, 127), "size": 1, "signed": True},
            TensorProto.UINT8: {"range": (0, 255), "size": 1, "signed": False},
            TensorProto.INT16: {"range": (-32768, 32767), "size": 2, "signed": True},
            TensorProto.UINT16: {"range": (0, 65535), "size": 2, "signed": False},
            TensorProto.INT32: {"range": (-2147483648, 2147483647), "size": 4, "signed": True},
            TensorProto.UINT32: {"range": (0, 4294967295), "size": 4, "signed": False},
            TensorProto.INT64: {"range": (-9223372036854775808, 9223372036854775807), "size": 8, "signed": True},
            TensorProto.UINT64: {"range": (0, 18446744073709551615), "size": 8, "signed": False},
        }
    
    def analyze_data_range(self, data: np.ndarray):
        """データ範囲の分析と最適な整数タイプの推奨"""
        
        print("=== データ範囲分析 ===")
        
        min_val = np.min(data)
        max_val = np.max(data)
        
        print(f"データ範囲: [{min_val}, {max_val}]")
        print(f"データ型: {data.dtype}")
        print(f"要素数: {data.size}")
        print(f"メモリ使用量: {data.nbytes} bytes")
        
        # 最適な整数タイプを推奨
        suitable_types = []
        
        for tensor_type, props in self.integer_types.items():
            type_min, type_max = props["range"]
            if min_val >= type_min and max_val <= type_max:
                suitable_types.append({
                    "type": tensor_type,
                    "name": TensorProto.DataType.Name(tensor_type),
                    "size": props["size"],
                    "memory_usage": data.size * props["size"]
                })
        
        if suitable_types:
            print("\n適用可能な整数タイプ:")
            for type_info in suitable_types:
                memory_reduction = (data.nbytes - type_info["memory_usage"]) / data.nbytes * 100
                print(f"  {type_info['name']}: {type_info['memory_usage']} bytes "
                      f"({memory_reduction:+.1f}% メモリ削減)")
        else:
            print("\n⚠️ 適用可能な整数タイプがありません（範囲外）")
        
        return suitable_types
    
    def create_integer_optimized_model(self):
        """整数最適化されたモデルの作成"""
        
        print("\n=== 整数最適化モデルの作成 ===")
        
        # カテゴリIDやインデックスなど、整数値のみを扱うデータ
        category_lookup = np.array([0, 1, 2, 3, 4], dtype=np.int64)  # 本来はINT8で十分
        embedding_indices = np.random.randint(0, 1000, size=50, dtype=np.int64)  # UINT16で十分
        
        # 最適化前の分析
        print("最適化前:")
        self.analyze_data_range(category_lookup)
        self.analyze_data_range(embedding_indices)
        
        # 最適な型に変換
        optimized_category_lookup = category_lookup.astype(np.int8)
        optimized_embedding_indices = embedding_indices.astype(np.uint16)
        
        print("\n最適化後:")
        self.analyze_data_range(optimized_category_lookup)
        self.analyze_data_range(optimized_embedding_indices)
        
        # ONNXモデルの作成
        nodes = [
            # カテゴリルックアップ（int8使用）
            helper.make_node(
                "Gather",
                inputs=["embedding_table", "category_ids"],
                outputs=["category_embeddings"],
                name="category_lookup"
            ),
            
            # インデックスベースの操作（uint16使用）
            helper.make_node(
                "Gather", 
                inputs=["large_embedding_table", "embedding_indices"],
                outputs=["embeddings"],
                name="embedding_lookup"
            ),
            
            # 結果の結合
            helper.make_node(
                "Add",
                inputs=["category_embeddings", "embeddings"],
                outputs=["combined_embeddings"],
                name="combine_embeddings"
            )
        ]
        
        # 初期化子（最適化された型で）
        embedding_table = np.random.randn(5, 128).astype(np.float32)
        large_embedding_table = np.random.randn(1000, 128).astype(np.float32)
        
        initializers = [
            numpy_helper.from_array(embedding_table, name="embedding_table"),
            numpy_helper.from_array(large_embedding_table, name="large_embedding_table"),
        ]
        
        # グラフ作成
        graph = helper.make_graph(
            nodes=nodes,
            name="IntegerOptimizedModel",
            inputs=[
                helper.make_tensor_value_info("category_ids", TensorProto.INT8, [None]),
                helper.make_tensor_value_info("embedding_indices", TensorProto.UINT16, [None])
            ],
            outputs=[
                helper.make_tensor_value_info("combined_embeddings", TensorProto.FLOAT, [None, 128])
            ],
            initializer=initializers
        )
        
        # モデル作成
        model = helper.make_model(graph, producer_name="integer_optimization_demo")
        model.opset_import[0].version = 13
        
        return model

def demonstrate_boolean_operations():
    """ブール型演算のデモ"""
    
    print("\n=== ブール型演算のデモ ===")
    
    # ブール型での論理演算
    bool_data1 = np.array([True, False, True, False], dtype=bool)
    bool_data2 = np.array([True, True, False, False], dtype=bool)
    
    print("ブール配列1:", bool_data1)
    print("ブール配列2:", bool_data2)
    
    # 論理演算の結果
    and_result = bool_data1 & bool_data2
    or_result = bool_data1 | bool_data2
    xor_result = bool_data1 ^ bool_data2
    not_result = ~bool_data1
    
    print("AND結果:", and_result)
    print("OR結果:", or_result)
    print("XOR結果:", xor_result)
    print("NOT結果:", not_result)
    
    # ONNXでのブール演算モデル
    nodes = [
        helper.make_node("And", ["bool_input1", "bool_input2"], ["and_output"], name="logical_and"),
        helper.make_node("Or", ["bool_input1", "bool_input2"], ["or_output"], name="logical_or"),
        helper.make_node("Xor", ["bool_input1", "bool_input2"], ["xor_output"], name="logical_xor"),
        helper.make_node("Not", ["bool_input1"], ["not_output"], name="logical_not"),
    ]
    
    graph = helper.make_graph(
        nodes=nodes,
        name="BooleanOperationsModel",
        inputs=[
            helper.make_tensor_value_info("bool_input1", TensorProto.BOOL, [None]),
            helper.make_tensor_value_info("bool_input2", TensorProto.BOOL, [None])
        ],
        outputs=[
            helper.make_tensor_value_info("and_output", TensorProto.BOOL, [None]),
            helper.make_tensor_value_info("or_output", TensorProto.BOOL, [None]),
            helper.make_tensor_value_info("xor_output", TensorProto.BOOL, [None]),
            helper.make_tensor_value_info("not_output", TensorProto.BOOL, [None])
        ]
    )
    
    model = helper.make_model(graph, producer_name="boolean_ops_demo")
    return model

# 実行例
if __name__ == "__main__":
    optimizer = IntegerTypeOptimizer()
    
    # ランダムデータでの最適化分析
    test_data = np.random.randint(-100, 100, size=1000, dtype=np.int32)
    optimizer.analyze_data_range(test_data)
    
    # 整数最適化モデルの作成
    int_model = optimizer.create_integer_optimized_model()
    onnx.save(int_model, "integer_optimized_model.onnx")
    
    # ブール演算モデルの作成
    bool_model = demonstrate_boolean_operations()
    onnx.save(bool_model, "boolean_operations_model.onnx")
```

### 4.4.2　浮動小数点タイプの詳細比較

異なる浮動小数点形式の特性と適用場面について詳しく学習します：

```python
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import struct

class FloatingPointAnalyzer:
    """浮動小数点タイプ分析クラス"""
    
    def __init__(self):
        self.float_types = {
            "float64": {"tensor_type": TensorProto.DOUBLE, "numpy_type": np.float64, "bytes": 8, "precision": "double"},
            "float32": {"tensor_type": TensorProto.FLOAT, "numpy_type": np.float32, "bytes": 4, "precision": "single"},
            "float16": {"tensor_type": TensorProto.FLOAT16, "numpy_type": np.float16, "bytes": 2, "precision": "half"},
            "bfloat16": {"tensor_type": TensorProto.BFLOAT16, "numpy_type": None, "bytes": 2, "precision": "brain_float"}
        }
    
    def compare_precision_and_range(self):
        """各浮動小数点形式の精度と範囲を比較"""
        
        print("=== 浮動小数点形式の比較 ===")
        
        test_values = [
            1.0,
            0.1,
            0.01,
            0.001,
            np.pi,
            np.e,
            1e-7,
            1e7,
            0.123456789
        ]
        
        print("テスト値での精度比較:")
        print(f"{'値':<15} {'float64':<20} {'float32':<15} {'float16':<15}")
        print("-" * 70)
        
        for value in test_values:
            f64_val = np.float64(value)
            f32_val = np.float32(value)
            f16_val = np.float16(value)
            
            print(f"{value:<15} {f64_val:<20} {f32_val:<15} {f16_val:<15}")
        
        # 精度損失の分析
        print("\n精度損失分析:")
        for value in test_values:
            f32_error = abs(np.float32(value) - value) / value if value != 0 else 0
            f16_error = abs(np.float16(value) - value) / value if value != 0 else 0
            
            print(f"値 {value}: float32誤差={f32_error:.2e}, float16誤差={f16_error:.2e}")
    
    def analyze_memory_usage(self, array_shape):
        """メモリ使用量の分析"""
        
        print(f"\n=== メモリ使用量分析（形状: {array_shape}） ===")
        
        total_elements = np.prod(array_shape)
        
        for name, props in self.float_types.items():
            if props["numpy_type"] is not None:
                memory_usage = total_elements * props["bytes"]
                memory_mb = memory_usage / (1024 * 1024)
                
                print(f"{name:<10}: {memory_usage:>10,} bytes ({memory_mb:>6.2f} MB)")
        
        # メモリ削減効果
        float32_memory = total_elements * 4
        float16_memory = total_elements * 2
        
        reduction_pct = (float32_memory - float16_memory) / float32_memory * 100
        print(f"\nfloat32→float16変換による削減: {reduction_pct:.1f}%")
    
    def create_mixed_precision_model(self):
        """混合精度モデルの作成"""
        
        print("\n=== 混合精度モデルの作成 ===")
        
        # 高精度が必要な重み（float32維持）
        critical_weight = np.random.randn(1000, 1000).astype(np.float32) * 0.01
        
        # 低精度でも問題ない特徴量（float16で格納）
        feature_weight = np.random.randn(128, 64).astype(np.float16)  
        
        # 初期化子
        initializers = [
            numpy_helper.from_array(critical_weight, name="critical_weight_fp32"),
            numpy_helper.from_array(feature_weight.astype(np.float32), name="feature_weight_fp16_stored_as_fp32"),  # ONNXではfloat32として格納
        ]
        
        # ノード作成（精度を意識した処理）
        nodes = [
            # 入力をfloat16にキャスト（メモリ効率化）
            helper.make_node(
                "Cast",
                inputs=["input_fp32"],
                outputs=["input_fp16"],
                to=TensorProto.FLOAT16,
                name="cast_input_to_fp16"
            ),
            
            # float16での軽い処理
            helper.make_node(
                "MatMul",
                inputs=["input_fp16", "feature_weight_fp16_stored_as_fp32"],
                outputs=["feature_output_fp16"],
                name="feature_processing"
            ),
            
            # 重要な計算の前にfloat32に戻す
            helper.make_node(
                "Cast",
                inputs=["feature_output_fp16"], 
                outputs=["feature_output_fp32"],
                to=TensorProto.FLOAT,
                name="cast_to_fp32_for_critical"
            ),
            
            # 高精度が必要な処理（float32）
            helper.make_node(
                "MatMul",
                inputs=["feature_output_fp32", "critical_weight_fp32"],
                outputs=["final_output"],
                name="critical_processing"
            )
        ]
        
        # グラフ作成
        graph = helper.make_graph(
            nodes=nodes,
            name="MixedPrecisionModel",
            inputs=[helper.make_tensor_value_info("input_fp32", TensorProto.FLOAT, [None, 1000])],
            outputs=[helper.make_tensor_value_info("final_output", TensorProto.FLOAT, [None, 1000])],
            initializer=initializers
        )
        
        model = helper.make_model(graph, producer_name="mixed_precision_demo")
        model.opset_import[0].version = 13
        
        print("✓ 混合精度モデルを作成しました")
        print("  - 軽い処理: float16使用")
        print("  - 重要な処理: float32使用")
        print("  - 自動的な型変換を含む")
        
        return model

def demonstrate_overflow_underflow():
    """オーバーフロー・アンダーフローのデモ"""
    
    print("\n=== オーバーフロー・アンダーフロー分析 ===")
    
    # 各型での限界値テスト
    test_scenarios = [
        ("大きな値", 1e20),
        ("小さな値", 1e-20), 
        ("非常に大きな値", 1e40),
        ("非常に小さな値", 1e-40)
    ]
    
    for scenario_name, value in test_scenarios:
        print(f"\n{scenario_name} ({value}):")
        
        # float32での表現
        f32_val = np.float32(value)
        f32_finite = np.isfinite(f32_val)
        
        # float16での表現
        f16_val = np.float16(value)  
        f16_finite = np.isfinite(f16_val)
        
        print(f"  float32: {f32_val} ({'有限' if f32_finite else '無限/NaN'})")
        print(f"  float16: {f16_val} ({'有限' if f16_finite else '無限/NaN'})")
        
        # オーバーフロー/アンダーフローの検出
        if not f32_finite:
            print("  ⚠️ float32でオーバーフロー/アンダーフロー発生")
        if not f16_finite:
            print("  ⚠️ float16でオーバーフロー/アンダーフロー発生")

def create_numerical_stability_model():
    """数値安定性を考慮したモデル"""
    
    print("\n=== 数値安定性考慮モデル ===")
    
    # 数値的に不安定な操作の例
    nodes = [
        # 1. 大きな値の減算（精度損失リスク）
        helper.make_node("Sub", ["large_val1", "large_val2"], ["subtle_diff"], name="risky_subtraction"),
        
        # 2. 安全な正規化（数値安定性考慮）
        helper.make_node("ReduceMean", ["input_data"], ["mean"], axes=[-1], keepdims=1, name="compute_mean"),
        helper.make_node("Sub", ["input_data", "mean"], ["centered"], name="center_data"),
        helper.make_node("Mul", ["centered", "centered"], ["squared"], name="square_centered"),
        helper.make_node("ReduceMean", ["squared"], ["variance"], axes=[-1], keepdims=1, name="compute_variance"),
        
        # 小さな値を加えて数値安定性を保つ
        helper.make_node("Add", ["variance", "epsilon"], ["stable_variance"], name="add_epsilon"),
        helper.make_node("Sqrt", ["stable_variance"], ["std"], name="compute_std"),
        helper.make_node("Div", ["centered", "std"], ["normalized"], name="normalize"),
        
        # 3. ログ計算での数値安定性
        helper.make_node("Max", ["logits"], ["max_logit"], axes=[-1], keepdims=1, name="find_max"),
        helper.make_node("Sub", ["logits", "max_logit"], ["shifted_logits"], name="shift_logits"),
        helper.make_node("Exp", ["shifted_logits"], ["exp_logits"], name="compute_exp"),
        helper.make_node("ReduceSum", ["exp_logits"], ["sum_exp"], axes=[-1], keepdims=1, name="sum_exp"),
        helper.make_node("Log", ["sum_exp"], ["log_sum_exp"], name="compute_log_sum_exp"),
        helper.make_node("Add", ["log_sum_exp", "max_logit"], ["stable_log_softmax"], name="stable_log_softmax")
    ]
    
    # 定数の初期化子
    epsilon_val = np.array([1e-8], dtype=np.float32)
    epsilon_init = numpy_helper.from_array(epsilon_val, name="epsilon")
    
    graph = helper.make_graph(
        nodes=nodes,
        name="NumericalStabilityModel", 
        inputs=[
            helper.make_tensor_value_info("large_val1", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("large_val2", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("input_data", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("subtle_diff", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("normalized", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("stable_log_softmax", TensorProto.FLOAT, [None, None])
        ],
        initializer=[epsilon_init]
    )
    
    model = helper.make_model(graph, producer_name="numerical_stability_demo")
    return model

# 実行例
if __name__ == "__main__":
    analyzer = FloatingPointAnalyzer()
    
    # 精度と範囲の比較
    analyzer.compare_precision_and_range()
    
    # メモリ使用量分析（典型的なCNNの重み形状）
    analyzer.analyze_memory_usage((512, 512, 3, 3))
    
    # 混合精度モデルの作成
    mixed_model = analyzer.create_mixed_precision_model()
    onnx.save(mixed_model, "mixed_precision_model.onnx")
    
    # オーバーフロー/アンダーフロー分析
    demonstrate_overflow_underflow()
    
    # 数値安定性モデルの作成
    stability_model = create_numerical_stability_model()
    onnx.save(stability_model, "numerical_stability_model.onnx")
    
    print("\n✓ 浮動小数点最適化モデルを保存しました")
```

## まとめ

第4章では、ONNXにおけるデータとオペランド最適化について包括的に学習しました：

### 学習内容の詳細総括

1. **実験演算子と進化管理**
   - **廃止演算子の自動検出**: モデル内の非推奨演算子を効率的に識別
   - **自動移行システム**: Upsample→Resize等の安全な変換手法
   - **バージョン互換性管理**: OpSet進化に対応する堅牢な移行戦略

2. **画像分類システムの高度化**
   - **カテゴリメタデータ管理**: モデルへの構造化カテゴリ情報埋め込み
   - **カテゴリマッピング**: 異なる分類体系間の自動対応
   - **メタデータ抽出**: モデルからの分類情報の効率的な取得

3. **データタイプシステムの完全理解**
   - **PyTorch-ONNX型対応**: フレームワーク間での型変換の詳細
   - **量子化技術**: int8量子化による効率的なモデル圧縮
   - **ランタイム型変換**: 推論時の動的型変換とパフォーマンス影響

4. **演算子設計のベストプラクティス**
   - **命名規則の体系化**: 保守性の高いモデル構造設計
   - **属性管理の標準化**: ONNX仕様準拠の確実な実装
   - **品質保証システム**: 自動検証による品質担保

5. **次世代浮動小数点形式の活用**
   - **E4M3FNUZ/E5M2FNUZ**: 新形式の特性理解と実装
   - **指数バイアス最適化**: ハードウェア効率を考慮した設計
   - **精度とメモリのトレードオフ**: 最適なバランス点の発見

6. **高度なデータタイプ最適化**
   - **整数型の効率活用**: メモリ削減と計算効率化
   - **混合精度戦略**: float16/float32の適切な使い分け
   - **数値安定性保証**: オーバーフロー/アンダーフロー対策

### 実践的な開発能力の向上

この章で習得した技術により、以下の高度な開発能力を獲得しました：

**データ効率化**:
- 適切なデータタイプ選択による大幅なメモリ削減
- 精度を保ちながらの計算効率向上
- ハードウェア特性を活用した最適化

**品質保証**:
- 自動的な演算子互換性チェック
- 数値安定性を考慮した堅牢な設計
- 包括的な型システム検証

**保守性向上**:
- 標準化された命名規則の適用
- 構造化されたメタデータ管理
- 長期保守に対応した設計パターン

**パフォーマンス最適化**:
- 混合精度による計算効率化
- 整数型活用による軽量化
- 新しい浮動小数点形式の戦略的活用

### 商用システムへの応用

これらの技術は、実際の商用システムにおいて以下のような価値を提供します：

- **エッジデバイス対応**: メモリ制約の厳しい環境でのモデル展開
- **クラウドコスト削減**: 効率的なデータ処理による計算リソース節約
- **リアルタイム処理**: 低レイテンシが要求されるアプリケーションの実現
- **長期運用**: 技術進化に対応する持続可能なシステム設計

次章では、これらのデータ最適化技術を基盤として、モデル全体の性能分析と最適化手法について学習し、さらに実践的なONNXシステム構築技術を探求していきます。
