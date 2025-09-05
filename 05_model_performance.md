# 第5章 ONNXモデルの性能と応用

> 本章の概要: グラフ最適化、ORT形式による効率的デプロイ、BERTモデルの検証など、性能最適化と応用を実践的に解説します。

前章では、ONNXにおけるデータとオペランド最適化について学習しました。この章では、より実践的な観点から、モデル全体の性能最適化と実際のアプリケーションでの活用方法について深く探求していきます。

現代の機械学習では、精度だけでなく推論速度・メモリ効率・デプロイ容易性も重要です。ONNX Runtimeは多層的な最適化機能を提供し、適切に活用することで商用レベルの高性能システムを構築できます。

## 5.1 ONNX Runtimeのグラフ最適化

### 5.1.1 概要

ONNX Runtimeは、モデルグラフを分析して自動最適化を行い、推論速度の向上とメモリ使用量の削減を実現します。

```python
import onnxruntime as ort
import onnx
import numpy as np
import time
from pathlib import Path

class GraphOptimizationDemo:
    """ONNX Runtimeのグラフ最適化デモ"""
    
    def __init__(self):
        self.optimization_levels = {
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL: "最適化なし",
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC: "基本最適化",
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED: "拡張最適化", 
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL: "全最適化"
        }
    
    def create_sample_model(self):
        """最適化効果を確認できるサンプルモデルの作成"""
        
        from onnx import helper, TensorProto, numpy_helper
        
        print("=== サンプルモデルの作成 ===")
        
        # 最適化の恩恵を受けやすいパターンを含むモデル
        # 1. Identity操作（削除可能）
        # 2. 連続するReshape操作（統合可能）
        # 3. 不要なTranspose操作
        
        # 初期化子
        weight1 = np.random.randn(256, 128).astype(np.float32) * 0.01
        bias1 = np.zeros(256, dtype=np.float32)
        weight2 = np.random.randn(128, 256).astype(np.float32) * 0.01
        bias2 = np.zeros(128, dtype=np.float32)
        
        initializers = [
            numpy_helper.from_array(weight1, name="weight1"),
            numpy_helper.from_array(bias1, name="bias1"),
            numpy_helper.from_array(weight2, name="weight2"),
            numpy_helper.from_array(bias2, name="bias2"),
        ]
        
        # 最適化可能な操作を含むノード群
        nodes = [
            # 入力をreshape（不要になる可能性）
            helper.make_node(
                "Reshape",
                inputs=["input", "shape1"],
                outputs=["reshaped1"],
                name="reshape1"
            ),
            
            # Identity操作（削除可能）
            helper.make_node(
                "Identity",
                inputs=["reshaped1"],
                outputs=["identity_out"],
                name="unnecessary_identity"
            ),
            
            # さらにreshape
            helper.make_node(
                "Reshape", 
                inputs=["identity_out", "shape2"],
                outputs=["reshaped2"],
                name="reshape2"
            ),
            
            # 線形層1
            helper.make_node(
                "MatMul",
                inputs=["reshaped2", "weight1"],
                outputs=["matmul1_out"],
                name="linear1"
            ),
            
            helper.make_node(
                "Add",
                inputs=["matmul1_out", "bias1"],
                outputs=["linear1_out"],
                name="bias_add1"
            ),
            
            # ReLU活性化
            helper.make_node(
                "Relu",
                inputs=["linear1_out"],
                outputs=["relu1_out"],
                name="relu1"
            ),
            
            # 別のIdentity操作（削除可能）
            helper.make_node(
                "Identity",
                inputs=["relu1_out"],
                outputs=["identity2_out"],
                name="another_identity"
            ),
            
            # 線形層2
            helper.make_node(
                "MatMul",
                inputs=["identity2_out", "weight2"],
                outputs=["matmul2_out"],
                name="linear2"
            ),
            
            helper.make_node(
                "Add",
                inputs=["matmul2_out", "bias2"],
                outputs=["output"],
                name="bias_add2"
            ),
        ]
        
        # reshape用の定数を追加
        shape1_data = np.array([1, 128], dtype=np.int64)
        shape2_data = np.array([1, 128], dtype=np.int64)
        
        initializers.extend([
            numpy_helper.from_array(shape1_data, name="shape1"),
            numpy_helper.from_array(shape2_data, name="shape2"),
        ])
        
        # グラフ作成
        graph = helper.make_graph(
            nodes=nodes,
            name="OptimizationDemo",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128])],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 128])],
            initializer=initializers
        )
        
        # モデル作成
        model = helper.make_model(graph, producer_name="optimization_demo")
        model.opset_import[0].version = 13
        
        onnx.save(model, "optimization_demo_model.onnx")
        print("✓ 最適化デモ用モデルを作成しました")
        
        return model
    
    def compare_optimization_levels(self, model_path):
        """各最適化レベルでの性能比較"""
        
        print("\\n=== 最適化レベル別性能比較 ===")
        
        results = {}
        
        # テスト用入力データ
        input_data = np.random.randn(1, 128).astype(np.float32)
        
        for opt_level, description in self.optimization_levels.items():
            print(f"\\n{description} ({opt_level}):")
            
            # セッションオプションの設定
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = opt_level
            
            # プロファイリングを有効化
            session_options.enable_profiling = True
            
            try:
                # セッション作成
                session = ort.InferenceSession(model_path, session_options)
                
                # ウォームアップ
                for _ in range(5):
                    session.run(None, {"input": input_data})
                
                # 推論時間測定
                start_time = time.time()
                num_iterations = 100
                
                for _ in range(num_iterations):
                    outputs = session.run(None, {"input": input_data})
                
                end_time = time.time()
                avg_time = (end_time - start_time) / num_iterations * 1000  # ms
                
                # プロファイルの取得
                prof_file = session.end_profiling()
                
                results[opt_level] = {
                    "avg_time_ms": avg_time,
                    "profile_file": prof_file,
                    "output_shape": outputs[0].shape
                }
                
                print(f"  平均推論時間: {avg_time:.2f} ms")
                print(f"  プロファイルファイル: {prof_file}")
                
                # プロファイルファイルのサイズを確認
                if Path(prof_file).exists():
                    file_size = Path(prof_file).stat().st_size
                    print(f"  プロファイルサイズ: {file_size} bytes")
                
            except Exception as e:
                print(f"  ✗ エラー: {e}")
                results[opt_level] = {"error": str(e)}
        
        # 結果の比較
        self._analyze_optimization_results(results)
        
        return results
    
    def _analyze_optimization_results(self, results):
        """最適化結果の分析"""
        
        print("\\n=== 最適化効果分析 ===")
        
        # エラーがなかった結果のみを分析
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if len(valid_results) < 2:
            print("比較に十分な結果がありません")
            return
        
        # 基準（最適化なし）との比較
        baseline_key = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        if baseline_key not in valid_results:
            baseline_key = list(valid_results.keys())[0]
        
        baseline_time = valid_results[baseline_key]["avg_time_ms"]
        
        print(f"基準時間（{self.optimization_levels[baseline_key]}): {baseline_time:.2f} ms")
        print("\\n最適化レベル別の改善:")
        
        for opt_level, result in valid_results.items():
            if opt_level == baseline_key:
                continue
                
            time_ms = result["avg_time_ms"]
            improvement = ((baseline_time - time_ms) / baseline_time) * 100
            
            print(f"  {self.optimization_levels[opt_level]}: {time_ms:.2f} ms ({improvement:+.1f}%)")
    
    def analyze_optimized_graph(self, model_path):
        """最適化されたグラフの分析"""
        
        print("\\n=== 最適化グラフ分析 ===")
        
        # 元のモデル分析
        original_model = onnx.load(model_path)
        original_nodes = len(original_model.graph.node)
        
        # 各最適化レベルでセッションを作成し、最適化後の情報を取得
        for opt_level, description in self.optimization_levels.items():
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = opt_level
            session_options.optimized_model_filepath = f"optimized_model_{opt_level}.onnx"
            
            try:
                session = ort.InferenceSession(model_path, session_options)
                
                # 最適化されたモデルが保存された場合
                optimized_path = f"optimized_model_{opt_level}.onnx"
                if Path(optimized_path).exists():
                    optimized_model = onnx.load(optimized_path)
                    optimized_nodes = len(optimized_model.graph.node)
                    
                    print(f"\\n{description}:")
                    print(f"  元のノード数: {original_nodes}")
                    print(f"  最適化後ノード数: {optimized_nodes}")
                    print(f"  削減されたノード数: {original_nodes - optimized_nodes}")
                    
                    # 削除された演算子タイプの分析
                    self._analyze_removed_operations(original_model, optimized_model)
                
            except Exception as e:
                print(f"{description}: 分析エラー - {e}")
    
    def _analyze_removed_operations(self, original_model, optimized_model):
        """削除された演算子の分析"""
        
        # 元のモデルの演算子統計
        original_ops = {}
        for node in original_model.graph.node:
            op_type = node.op_type
            original_ops[op_type] = original_ops.get(op_type, 0) + 1
        
        # 最適化後のモデルの演算子統計
        optimized_ops = {}
        for node in optimized_model.graph.node:
            op_type = node.op_type
            optimized_ops[op_type] = optimized_ops.get(op_type, 0) + 1
        
        # 削除された演算子を特定
        removed_ops = {}
        for op_type, count in original_ops.items():
            optimized_count = optimized_ops.get(op_type, 0)
            if optimized_count < count:
                removed_ops[op_type] = count - optimized_count
        
        if removed_ops:
            print(f"  削除された演算子:")
            for op_type, count in removed_ops.items():
                print(f"    {op_type}: {count}個")
        else:
            print(f"  削除された演算子はありません")

def demonstrate_specific_optimizations():
    """特定の最適化パターンのデモ"""
    
    print("\\n=== 特定の最適化パターン ===")
    
    optimizations = {
        "constant_folding": {
            "description": "定数畳み込み - コンパイル時に計算可能な演算を事前実行",
            "example": "Add(Constant(1), Constant(2)) → Constant(3)"
        },
        "identity_elimination": {
            "description": "Identity演算子の除去 - 何もしない演算子を削除",
            "example": "Identity(x) → x"
        },
        "reshape_fusion": {
            "description": "連続するReshape操作の統合",
            "example": "Reshape(Reshape(x, [a]), [b]) → Reshape(x, [b])"
        },
        "conv_batch_norm_fusion": {
            "description": "Conv + BatchNormの融合",
            "example": "BatchNorm(Conv(x, W)) → Conv(x, W', b')"
        },
        "gemm_optimization": {
            "description": "行列演算の最適化",
            "example": "Add(MatMul(A, B), C) → Gemm(A, B, C)"
        }
    }
    
    for opt_name, info in optimizations.items():
        print(f"\\n{opt_name.upper()}:")
        print(f"  説明: {info['description']}")
        print(f"  例: {info['example']}")

# 実行例
def demo_graph_optimization():
    """グラフ最適化デモの実行"""
    
    demo = GraphOptimizationDemo()
    
    # サンプルモデルの作成
    model = demo.create_sample_model()
    
    # 最適化レベル別比較
    results = demo.compare_optimization_levels("optimization_demo_model.onnx")
    
    # 最適化されたグラフの分析
    demo.analyze_optimized_graph("optimization_demo_model.onnx")
    
    # 特定の最適化パターンの説明
    demonstrate_specific_optimizations()

if __name__ == "__main__":
    demo_graph_optimization()
```

### 5.1.2　ONNXランタイムグラフ最適化使用方法

実践的な最適化設定と使用方法：

```python
import onnxruntime as ort
import onnx
import numpy as np
from typing import Dict, Any, List
import json

class OptimizationConfigurator:
    """最適化設定の詳細管理"""
    
    def __init__(self):
        self.provider_specific_options = {
            "CPUExecutionProvider": {
                "intra_op_num_threads": 0,  # 自動設定
                "inter_op_num_threads": 0,  # 自動設定
                "kmp_affinity": "granularity=fine,verbose,compact,1,0"
            },
            "CUDAExecutionProvider": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                "cudnn_conv_algo_search": "EXHAUSTIVE"
            }
        }
    
    def create_optimized_session(self, model_path: str, 
                               optimization_level: str = "all",
                               providers: List[str] = None,
                               custom_options: Dict[str, Any] = None) -> ort.InferenceSession:
        """最適化されたセッションの作成"""
        
        print(f"=== 最適化セッション作成: {optimization_level} ===")
        
        # デフォルトプロバイダー
        if providers is None:
            providers = ["CPUExecutionProvider"]
        
        # セッションオプションの設定
        session_options = ort.SessionOptions()
        
        # 最適化レベルの設定
        opt_level_map = {
            "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        }
        
        session_options.graph_optimization_level = opt_level_map.get(
            optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        
        # 並列処理の設定
        if custom_options:
            if "intra_op_num_threads" in custom_options:
                session_options.intra_op_num_threads = custom_options["intra_op_num_threads"]
            if "inter_op_num_threads" in custom_options:
                session_options.inter_op_num_threads = custom_options["inter_op_num_threads"]
        
        # プロファイリング設定
        session_options.enable_profiling = True
        
        # 最適化されたモデルの保存設定
        session_options.optimized_model_filepath = f"optimized_{optimization_level}_model.onnx"
        
        # プロバイダー固有オプション
        provider_options = []
        for provider in providers:
            if provider in self.provider_specific_options:
                provider_options.append((provider, self.provider_specific_options[provider]))
            else:
                provider_options.append(provider)
        
        # セッション作成
        try:
            session = ort.InferenceSession(
                model_path, 
                session_options, 
                providers=provider_options
            )
            
            print(f"✓ セッション作成成功")
            print(f"  使用プロバイダー: {session.get_providers()}")
            print(f"  最適化レベル: {optimization_level}")
            
            return session
            
        except Exception as e:
            print(f"✗ セッション作成失敗: {e}")
            raise
    
    def benchmark_different_configurations(self, model_path: str):
        """異なる設定での性能ベンチマーク"""
        
        print("\\n=== 設定別ベンチマーク ===")
        
        # テストする設定
        configurations = [
            {
                "name": "デフォルト",
                "optimization_level": "all",
                "providers": ["CPUExecutionProvider"],
                "custom_options": {}
            },
            {
                "name": "基本最適化のみ",
                "optimization_level": "basic", 
                "providers": ["CPUExecutionProvider"],
                "custom_options": {}
            },
            {
                "name": "最適化なし",
                "optimization_level": "none",
                "providers": ["CPUExecutionProvider"], 
                "custom_options": {}
            },
            {
                "name": "マルチスレッド最適化",
                "optimization_level": "all",
                "providers": ["CPUExecutionProvider"],
                "custom_options": {
                    "intra_op_num_threads": 4,
                    "inter_op_num_threads": 2
                }
            }
        ]
        
        # GPU設定（利用可能な場合）
        if "CUDAExecutionProvider" in ort.get_available_providers():
            configurations.append({
                "name": "GPU最適化",
                "optimization_level": "all",
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "custom_options": {}
            })
        
        results = {}
        
        # 各設定でベンチマーク実行
        for config in configurations:
            try:
                session = self.create_optimized_session(
                    model_path,
                    config["optimization_level"],
                    config["providers"],
                    config["custom_options"]
                )
                
                # 性能測定
                benchmark_result = self._run_benchmark(session, config["name"])
                results[config["name"]] = benchmark_result
                
                # プロファイリング終了
                prof_file = session.end_profiling()
                results[config["name"]]["profile_file"] = prof_file
                
            except Exception as e:
                print(f"設定 '{config['name']}' でエラー: {e}")
                results[config["name"]] = {"error": str(e)}
        
        # 結果の比較
        self._compare_benchmark_results(results)
        
        return results
    
    def _run_benchmark(self, session: ort.InferenceSession, config_name: str) -> Dict[str, float]:
        """ベンチマーク実行"""
        
        print(f"\\nベンチマーク実行: {config_name}")
        
        # 入力データの準備
        input_meta = session.get_inputs()[0]
        input_shape = input_meta.shape
        
        # 動的次元を固定値に置換
        fixed_shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim is None:
                fixed_shape.append(1)  # バッチサイズを1に
            else:
                fixed_shape.append(dim)
        
        input_data = np.random.randn(*fixed_shape).astype(np.float32)
        
        # ウォームアップ
        for _ in range(10):
            session.run(None, {input_meta.name: input_data})
        
        # 実際の測定
        import time
        
        times = []
        num_runs = 50
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            outputs = session.run(None, {input_meta.name: input_data})
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # ms
        
        # 統計計算
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"  平均時間: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  最小時間: {min_time:.2f} ms")
        print(f"  最大時間: {max_time:.2f} ms")
        
        return {
            "avg_time_ms": avg_time,
            "std_time_ms": std_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "num_runs": num_runs
        }
    
    def _compare_benchmark_results(self, results: Dict[str, Dict[str, Any]]):
        """ベンチマーク結果の比較"""
        
        print("\\n=== ベンチマーク結果比較 ===")
        
        # エラーなしの結果のみ抽出
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not valid_results:
            print("有効な結果がありません")
            return
        
        # 基準となる設定（通常は最も遅い設定）
        baseline_name = max(valid_results.keys(), 
                          key=lambda k: valid_results[k]["avg_time_ms"])
        baseline_time = valid_results[baseline_name]["avg_time_ms"]
        
        print(f"基準設定: {baseline_name} ({baseline_time:.2f} ms)")
        print("\\n相対性能:")
        
        # 速い順にソート
        sorted_results = sorted(valid_results.items(), 
                              key=lambda x: x[1]["avg_time_ms"])
        
        for name, result in sorted_results:
            avg_time = result["avg_time_ms"]
            speedup = baseline_time / avg_time
            improvement = (1 - avg_time / baseline_time) * 100
            
            print(f"  {name:20s}: {avg_time:6.2f} ms ({speedup:.2f}x, {improvement:+5.1f}%)")

def create_optimization_guide():
    """最適化ガイドの作成"""
    
    print("\\n=== ONNX最適化実践ガイド ===")
    
    guide = {
        "基本原則": [
            "まず最適化レベルを 'all' に設定して効果を確認",
            "プロファイリングを有効にしてボトルネックを特定",
            "ハードウェアに応じたプロバイダーを選択",
            "並列処理設定をシステムに合わせて調整"
        ],
        "CPU最適化": [
            "intra_op_num_threads を CPU コア数に設定",
            "inter_op_num_threads を 1-2 に設定（通常）",
            "KMP アフィニティ設定でCPU利用効率向上",
            "NUMA対応システムでは適切なメモリ配置を考慮"
        ],
        "GPU最適化": [
            "CUDAExecutionProvider を最優先に設定",
            "GPU メモリ制限を適切に設定",
            "cuDNN の算法検索を有効化",
            "メモリプールの拡張戦略を調整"
        ],
        "モデル別最適化": [
            "CNNモデル: Conv+BatchNorm融合が効果的",
            "Transformerモデル: AttentionとLinearの最適化重要",
            "RNNモデル: シーケンス長に応じたメモリ管理",
            "量子化モデル: INT8演算子の最適化パスを有効化"
        ],
        "注意点": [
            "最適化レベルが高いほど良いとは限らない",
            "プロファイリングによる実測が重要",
            "メモリ使用量と速度のトレードオフを考慮",
            "デプロイ環境での最終確認を必ず実施"
        ]
    }
    
    for category, items in guide.items():
        print(f"\\n{category}:")
        for item in items:
            print(f"  • {item}")

# 使用例とデモ
def demo_optimization_configuration():
    """最適化設定デモの実行"""
    
    configurator = OptimizationConfigurator()
    
    # まずサンプルモデルが必要
    # （前のセクションで作成されたものを使用）
    model_path = "optimization_demo_model.onnx"
    
    if not Path(model_path).exists():
        print("サンプルモデルが見つかりません。先にサンプルモデルを作成してください。")
        return
    
    # 各種設定でのベンチマーク
    results = configurator.benchmark_different_configurations(model_path)
    
    # 最適化ガイドの表示
    create_optimization_guide()
    
    # 結果をJSONで保存
    with open("optimization_benchmark_results.json", "w") as f:
        # NumPy配列などをシリアライズ可能な形式に変換
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict) and "error" not in value:
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    print("\\n✓ ベンチマーク結果を保存しました: optimization_benchmark_results.json")

if __name__ == "__main__":
    demo_optimization_configuration()
```

## 5.2　ORTモデル形式

### 5.2.1　ORTモデル形式とは？

ONNX Runtime（ORT）形式は、ONNXモデルを最適化済みの独自バイナリ形式で保存する仕組みです：

```python
import onnxruntime as ort
import onnx
import numpy as np
from pathlib import Path
import subprocess
import json

class ORTFormatHandler:
    """ORT形式の処理クラス"""
    
    def __init__(self):
        self.format_info = {
            "ORT": {
                "description": "ONNX Runtime最適化済みバイナリ形式",
                "advantages": [
                    "読み込み速度の高速化",
                    "メモリ使用量の削減",
                    "セキュリティの向上（難読化）",
                    "最適化の事前適用"
                ],
                "disadvantages": [
                    "可読性の低下",
                    "デバッグの困難性",
                    "バージョン依存性"
                ]
            }
        }
    
    def explain_ort_format(self):
        """ORT形式の説明"""
        
        print("=== ORT（ONNX Runtime）形式の概要 ===")
        
        info = self.format_info["ORT"]
        
        print(f"説明: {info['description']}")
        
        print("\\n利点:")
        for advantage in info["advantages"]:
            print(f"  ✓ {advantage}")
        
        print("\\n欠点:")
        for disadvantage in info["disadvantages"]:
            print(f"  ⚠ {disadvantage}")
        
        print("\\n適用場面:")
        use_cases = [
            "本番環境でのデプロイ",
            "エッジデバイスでの推論",
            "モデルの知的財産保護",
            "最大限の推論性能が必要な場合"
        ]
        
        for use_case in use_cases:
            print(f"  • {use_case}")
    
    def demonstrate_format_differences(self):
        """ONNX形式とORT形式の違いのデモ"""
        
        print("\\n=== ONNX vs ORT 形式の違い ===")
        
        # サンプルモデルの作成
        model = self._create_sample_model()
        
        # ONNX形式で保存
        onnx_path = "sample_model.onnx"
        onnx.save(model, onnx_path)
        
        # ORT形式に変換（後で実装）
        ort_path = "sample_model.ort"
        
        # ファイルサイズの比較
        onnx_size = Path(onnx_path).stat().st_size
        print(f"ONNX形式サイズ: {onnx_size:,} bytes")
        
        # 可読性の比較
        print("\\n可読性:")
        print("  ONNX: テキストエディタで構造確認可能")
        print("  ORT:  バイナリ形式のため直接確認不可")
        
        # セキュリティの比較
        print("\\nセキュリティ:")
        print("  ONNX: モデル構造とパラメータが平文で格納")
        print("  ORT:  バイナリ形式によりリバースエンジニアリングが困難")
        
        return model
    
    def _create_sample_model(self):
        """サンプルモデルの作成"""
        
        from onnx import helper, TensorProto, numpy_helper
        
        # シンプルな線形モデル
        weight = np.random.randn(10, 5).astype(np.float32) * 0.1
        bias = np.zeros(5, dtype=np.float32)
        
        weight_initializer = numpy_helper.from_array(weight, name="weight")
        bias_initializer = numpy_helper.from_array(bias, name="bias")
        
        nodes = [
            helper.make_node("MatMul", ["input", "weight"], ["matmul_out"], name="linear"),
            helper.make_node("Add", ["matmul_out", "bias"], ["output"], name="bias_add"),
            helper.make_node("Relu", ["output"], ["final_output"], name="activation")
        ]
        
        graph = helper.make_graph(
            nodes=nodes,
            name="SampleModel",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 10])],
            outputs=[helper.make_tensor_value_info("final_output", TensorProto.FLOAT, [None, 5])],
            initializer=[weight_initializer, bias_initializer]
        )
        
        model = helper.make_model(graph, producer_name="ort_format_demo")
        model.opset_import[0].version = 13
        
        return model

def analyze_ort_benefits():
    """ORT形式の利点を定量的に分析"""
    
    print("\\n=== ORT形式の定量的利点 ===")
    
    benefits = {
        "読み込み速度": {
            "onnx": "プロトバッファのパース + グラフ構築",
            "ort": "事前最適化済みバイナリの直接読み込み",
            "improvement": "2-5倍高速"
        },
        "メモリ使用量": {
            "onnx": "テキスト形式 + 実行時最適化メモリ",
            "ort": "最適化済みバイナリのみ",
            "improvement": "10-30%削減"
        },
        "初期化時間": {
            "onnx": "グラフ最適化 + プロバイダー初期化",
            "ort": "プロバイダー初期化のみ",
            "improvement": "3-10倍高速"
        },
        "ファイルサイズ": {
            "onnx": "プロトバッファ形式（圧縮可能）",
            "ort": "カスタムバイナリ形式",
            "improvement": "5-20%削減（場合による）"
        }
    }
    
    for metric, info in benefits.items():
        print(f"\\n{metric}:")
        print(f"  ONNX: {info['onnx']}")
        print(f"  ORT:  {info['ort']}")
        print(f"  改善: {info['improvement']}")

### 5.2.2　ONNXモデルのORT形式への変換

def demonstrate_ort_conversion():
    """ORT形式への変換デモ"""
    
    print("\\n=== ORT形式変換デモ ===")
    
    # 変換方法の説明
    conversion_methods = {
        "Python API": {
            "description": "onnxruntime API を使用した変換",
            "code_example": """
# セッション作成時に最適化モデルを出力
session_options = ort.SessionOptions()
session_options.optimized_model_filepath = "model.ort"
session = ort.InferenceSession("model.onnx", session_options)
            """,
            "pros": ["プログラム内で直接実行", "カスタム設定が可能"],
            "cons": ["セッション作成が必要", "ランタイム依存"]
        },
        "コマンドラインツール": {
            "description": "onnxruntime_perf_test を使用",
            "code_example": """
# コマンドライン例
onnxruntime_perf_test -m model.onnx -o model.ort -O 99
            """,
            "pros": ["シンプルな実行", "バッチ処理に適している"],
            "cons": ["ツールの利用可能性に依存", "設定の制限"]
        },
        "C++ API": {
            "description": "C++ APIを使用した変換",
            "code_example": """
// C++ 例
Ort::SessionOptions session_options;
session_options.SetOptimizedModelFilePath("model.ort");
Ort::Session session(env, "model.onnx", session_options);
            """,
            "pros": ["高性能", "組み込み用途に適している"],
            "cons": ["実装の複雑性", "C++知識が必要"]
        }
    }
    
    for method, info in conversion_methods.items():
        print(f"\\n{method}:")
        print(f"  説明: {info['description']}")
        print(f"  利点: {', '.join(info['pros'])}")
        print(f"  欠点: {', '.join(info['cons'])}")
        print(f"  例:{info['code_example']}")

def create_ort_conversion_script():
    """ORT変換スクリプトの作成"""
    
    script_content = '''
import onnxruntime as ort
import argparse
import sys
from pathlib import Path

def convert_to_ort(onnx_path, ort_path, optimization_level="all", providers=None):
    """ONNXモデルをORT形式に変換"""
    
    print(f"変換開始: {onnx_path} → {ort_path}")
    
    # デフォルトプロバイダー
    if providers is None:
        providers = ["CPUExecutionProvider"]
    
    # セッションオプション
    session_options = ort.SessionOptions()
    
    # 最適化レベル設定
    opt_levels = {
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    }
    
    session_options.graph_optimization_level = opt_levels.get(
        optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    
    # 最適化済みモデルの出力パス設定
    session_options.optimized_model_filepath = str(ort_path)
    
    try:
        # セッション作成（これによりORT形式が生成される）
        session = ort.InferenceSession(str(onnx_path), session_options, providers=providers)
        
        # 変換成功の確認
        if Path(ort_path).exists():
            original_size = Path(onnx_path).stat().st_size
            converted_size = Path(ort_path).stat().st_size
            
            print(f"✓ 変換完了")
            print(f"  元サイズ: {original_size:,} bytes")
            print(f"  変換後: {converted_size:,} bytes")
            print(f"  圧縮率: {converted_size/original_size:.2f}")
        else:
            print("✗ 変換ファイルが見つかりません")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 変換エラー: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX to ORT converter")
    parser.add_argument("input", help="入力ONNXファイル")
    parser.add_argument("output", help="出力ORTファイル")
    parser.add_argument("--optimization", choices=["none", "basic", "extended", "all"], 
                       default="all", help="最適化レベル")
    parser.add_argument("--gpu", action="store_true", help="GPU最適化を有効化")
    
    args = parser.parse_args()
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
    
    success = convert_to_ort(args.input, args.output, args.optimization, providers)
    sys.exit(0 if success else 1)
    '''
    
    # スクリプトファイルに保存
    script_path = "onnx_to_ort_converter.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content.strip())
    
    print(f"\\n✓ ORT変換スクリプトを作成しました: {script_path}")
    print("\\n使用例:")
    print(f"  python {script_path} model.onnx model.ort")
    print(f"  python {script_path} model.onnx model.ort --optimization extended")
    print(f"  python {script_path} model.onnx model.ort --gpu")

# 実際の変換デモ
def demo_actual_conversion():
    """実際のORT変換デモ"""
    
    print("\\n=== 実際のORT変換デモ ===")
    
    # サンプルモデル作成
    handler = ORTFormatHandler()
    model = handler.demonstrate_format_differences()
    
    onnx_path = "sample_model.onnx"
    ort_path = "sample_model.ort"
    
    # Python APIでのORT変換
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = ort_path
    
    try:
        print("\\nORT形式への変換中...")
        session = ort.InferenceSession(onnx_path, session_options)
        
        if Path(ort_path).exists():
            original_size = Path(onnx_path).stat().st_size
            ort_size = Path(ort_path).stat().st_size
            
            print(f"✓ 変換完了")
            print(f"  ONNX サイズ: {original_size:,} bytes")
            print(f"  ORT サイズ: {ort_size:,} bytes") 
            print(f"  サイズ比: {ort_size/original_size:.2f}")
            
            # 両形式での推論速度比較
            compare_inference_speed(onnx_path, ort_path)
            
        else:
            print("✗ ORT形式の生成に失敗しました")
            
    except Exception as e:
        print(f"✗ 変換エラー: {e}")

def compare_inference_speed(onnx_path, ort_path):
    """ONNX形式とORT形式の推論速度比較"""
    
    print("\\n--- 推論速度比較 ---")
    
    import time
    
    # テスト用入力データ
    input_data = np.random.randn(1, 10).astype(np.float32)
    num_iterations = 100
    
    # ONNX形式での推論
    try:
        session_onnx = ort.InferenceSession(onnx_path)
        
        # ウォームアップ
        for _ in range(10):
            session_onnx.run(None, {"input": input_data})
        
        # 測定
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            outputs_onnx = session_onnx.run(None, {"input": input_data})
        onnx_time = (time.perf_counter() - start_time) / num_iterations * 1000
        
        print(f"ONNX形式: {onnx_time:.3f} ms/inference")
        
    except Exception as e:
        print(f"ONNX形式テストエラー: {e}")
        return
    
    # ORT形式での推論
    try:
        session_ort = ort.InferenceSession(ort_path)
        
        # ウォームアップ
        for _ in range(10):
            session_ort.run(None, {"input": input_data})
        
        # 測定
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            outputs_ort = session_ort.run(None, {"input": input_data})
        ort_time = (time.perf_counter() - start_time) / num_iterations * 1000
        
        print(f"ORT形式:  {ort_time:.3f} ms/inference")
        
        # 速度向上の計算
        speedup = onnx_time / ort_time
        improvement = (1 - ort_time / onnx_time) * 100
        
        print(f"速度向上: {speedup:.2f}x ({improvement:+.1f}%)")
        
        # 結果の一致確認
        if np.allclose(outputs_onnx[0], outputs_ort[0], rtol=1e-5):
            print("✓ 出力結果は一致しています")
        else:
            print("⚠ 出力結果に差異があります")
            
    except Exception as e:
        print(f"ORT形式テストエラー: {e}")

# メイン実行関数
def demo_ort_format():
    """ORT形式デモの実行"""
    
    handler = ORTFormatHandler()
    
    # ORT形式の説明
    handler.explain_ort_format()
    
    # 定量的利点の分析
    analyze_ort_benefits()
    
    # 変換方法の説明
    demonstrate_ort_conversion()
    
    # 変換スクリプトの作成
    create_ort_conversion_script()
    
    # 実際の変換デモ
    demo_actual_conversion()

if __name__ == "__main__":
    demo_ort_format()
```

## 5.3　実世界での性能最適化実践

### 5.3.1　BERTモデル検証と最適化

大規模なTransformerモデルであるBERTを例に、実世界での性能最適化を実践的に学習します：

```python
import onnxruntime as ort
import onnx
import numpy as np
import time
from typing import Dict, List, Tuple
import json
from pathlib import Path

class BERTOptimizationValidator:
    """BERTモデルの最適化検証クラス"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.optimization_results = {}
        self.test_configurations = self._get_test_configurations()
    
    def _get_test_configurations(self) -> List[Dict]:
        """テスト設定の取得"""
        
        configurations = [
            {
                "name": "baseline_cpu",
                "providers": ["CPUExecutionProvider"],
                "optimization": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                "options": {}
            },
            {
                "name": "optimized_cpu", 
                "providers": ["CPUExecutionProvider"],
                "optimization": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                "options": {
                    "intra_op_num_threads": 4,
                    "inter_op_num_threads": 1
                }
            },
            {
                "name": "optimized_cpu_parallel",
                "providers": ["CPUExecutionProvider"],
                "optimization": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                "options": {
                    "intra_op_num_threads": 0,  # 自動設定
                    "inter_op_num_threads": 0   # 自動設定
                }
            }
        ]
        
        # GPU設定（利用可能な場合）
        if "CUDAExecutionProvider" in ort.get_available_providers():
            configurations.extend([
                {
                    "name": "gpu_basic",
                    "providers": [
                        ("CUDAExecutionProvider", {
                            "device_id": 0,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "gpu_mem_limit": 2 * 1024 * 1024 * 1024
                        }),
                        "CPUExecutionProvider"
                    ],
                    "optimization": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                    "options": {}
                },
                {
                    "name": "gpu_optimized",
                    "providers": [
                        ("CUDAExecutionProvider", {
                            "device_id": 0,
                            "arena_extend_strategy": "kSameAsRequested",
                            "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": True
                        }),
                        "CPUExecutionProvider"
                    ],
                    "optimization": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                    "options": {}
                }
            ])
        
        return configurations
    
    def validate_bert_optimizations(self):
        """BERT最適化の検証"""
        
        print("=== BERT最適化検証 ===")
        
        # テスト用入力の生成
        test_inputs = self._generate_bert_test_inputs()
        
        results = {}
        
        for config in self.test_configurations:
            print(f"\n設定テスト: {config['name']}")
            
            try:
                # セッション作成
                session = self._create_session(config)
                
                # 推論テスト
                latency_results = self._benchmark_latency(session, test_inputs, config['name'])
                
                # メモリ使用量測定
                memory_results = self._measure_memory_usage(session, test_inputs)
                
                # 精度検証
                accuracy_results = self._validate_accuracy(session, test_inputs)
                
                results[config['name']] = {
                    "latency": latency_results,
                    "memory": memory_results,
                    "accuracy": accuracy_results,
                    "configuration": config
                }
                
                print(f"  平均レイテンシ: {latency_results['avg_ms']:.2f} ms")
                print(f"  メモリ使用量: {memory_results['peak_mb']:.1f} MB")
                print(f"  精度保持: {'✓' if accuracy_results['valid'] else '✗'}")
                
            except Exception as e:
                print(f"  エラー: {e}")
                results[config['name']] = {"error": str(e)}
        
        # 結果の比較分析
        self._analyze_optimization_results(results)
        
        return results
    
    def _generate_bert_test_inputs(self) -> Dict[str, np.ndarray]:
        """BERTテスト用入力の生成"""
        
        # BERT-baseの典型的な入力形状
        batch_size = 1
        sequence_length = 128  # 一般的なシーケンス長
        
        test_inputs = {
            "input_ids": np.random.randint(
                0, 30522, size=(batch_size, sequence_length), dtype=np.int64
            ),
            "attention_mask": np.ones(
                (batch_size, sequence_length), dtype=np.int64
            ),
            "token_type_ids": np.zeros(
                (batch_size, sequence_length), dtype=np.int64
            )
        }
        
        return test_inputs
    
    def _create_session(self, config: Dict) -> ort.InferenceSession:
        """設定に基づくセッション作成"""
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = config["optimization"]
        
        # カスタムオプションの適用
        for key, value in config["options"].items():
            setattr(session_options, key, value)
        
        # プロファイリング有効化
        session_options.enable_profiling = True
        
        return ort.InferenceSession(self.model_path, session_options, providers=config["providers"])
    
    def _benchmark_latency(self, session: ort.InferenceSession, 
                          test_inputs: Dict[str, np.ndarray], 
                          config_name: str) -> Dict[str, float]:
        """レイテンシベンチマーク"""
        
        # ウォームアップ
        for _ in range(10):
            session.run(None, test_inputs)
        
        # 実際の測定
        times = []
        num_runs = 50
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            outputs = session.run(None, test_inputs)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # ms
        
        times = np.array(times)
        
        return {
            "avg_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99))
        }
    
    def _measure_memory_usage(self, session: ort.InferenceSession, 
                            test_inputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """メモリ使用量測定"""
        
        import psutil
        import gc
        
        # ベースラインメモリ
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 推論実行
        for _ in range(10):
            outputs = session.run(None, test_inputs)
        
        # ピークメモリ測定
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            "baseline_mb": baseline_memory,
            "peak_mb": peak_memory,
            "delta_mb": peak_memory - baseline_memory
        }
    
    def _validate_accuracy(self, session: ort.InferenceSession, 
                          test_inputs: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """精度検証"""
        
        try:
            # 推論実行
            outputs = session.run(None, test_inputs)
            
            # 出力の健全性チェック
            valid = True
            for output in outputs:
                if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                    valid = False
                    break
            
            return {
                "valid": valid,
                "output_shapes": [output.shape for output in outputs],
                "output_dtypes": [str(output.dtype) for output in outputs]
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _analyze_optimization_results(self, results: Dict):
        """最適化結果の分析"""
        
        print("\n=== 最適化結果分析 ===")
        
        # エラーなしの結果のみ
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if len(valid_results) < 2:
            print("比較に十分な結果がありません")
            return
        
        # ベースライン設定
        baseline_key = "baseline_cpu"
        if baseline_key not in valid_results:
            baseline_key = list(valid_results.keys())[0]
        
        baseline = valid_results[baseline_key]
        baseline_latency = baseline["latency"]["avg_ms"]
        baseline_memory = baseline["memory"]["peak_mb"]
        
        print(f"ベースライン ({baseline_key}):")
        print(f"  レイテンシ: {baseline_latency:.2f} ms")
        print(f"  メモリ: {baseline_memory:.1f} MB")
        
        print("\n最適化効果:")
        for name, result in valid_results.items():
            if name == baseline_key:
                continue
            
            latency = result["latency"]["avg_ms"]
            memory = result["memory"]["peak_mb"]
            
            latency_improvement = (baseline_latency - latency) / baseline_latency * 100
            memory_change = (memory - baseline_memory) / baseline_memory * 100
            
            print(f"  {name}:")
            print(f"    レイテンシ: {latency:.2f} ms ({latency_improvement:+.1f}%)")
            print(f"    メモリ: {memory:.1f} MB ({memory_change:+.1f}%)")

def demonstrate_production_optimization():
    """本番環境での最適化デモ"""
    
    print("\n=== 本番環境最適化戦略 ===")
    
    optimization_strategies = {
        "モデルレベル最適化": {
            "techniques": [
                "量子化（INT8/FP16）",
                "プルーニング（重み削減）", 
                "知識蒸留（モデル圧縮）",
                "動的形状最適化"
            ],
            "benefits": [
                "メモリ使用量削減",
                "推論速度向上",
                "エネルギー効率改善"
            ]
        },
        "ランタイム最適化": {
            "techniques": [
                "グラフレベル最適化",
                "演算子融合",
                "メモリレイアウト最適化",
                "並列実行最適化"
            ],
            "benefits": [
                "計算効率向上",
                "キャッシュヒット率改善",
                "スループット向上"
            ]
        },
        "デプロイメント最適化": {
            "techniques": [
                "バッチング戦略",
                "負荷分散",
                "キャッシュ戦略",
                "リソース管理"
            ],
            "benefits": [
                "スケーラビリティ向上",
                "レスポンス時間安定化",
                "リソース利用効率化"
            ]
        },
        "ハードウェア最適化": {
            "techniques": [
                "GPU最適化",
                "専用ハードウェア活用",
                "NUMA最適化",
                "ベクトル化活用"
            ],
            "benefits": [
                "計算性能最大化",
                "電力効率改善",
                "コスト最適化"
            ]
        }
    }
    
    for strategy, details in optimization_strategies.items():
        print(f"\n{strategy}:")
        print("  技術:")
        for technique in details["techniques"]:
            print(f"    • {technique}")
        print("  効果:")
        for benefit in details["benefits"]:
            print(f"    ✓ {benefit}")

### 5.3.2　本番環境での性能監視

def create_performance_monitoring_system():
    """性能監視システムの実装"""
    
    print("\n=== 性能監視システム ===")
    
    monitoring_code = '''
import onnxruntime as ort
import time
import logging
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional
import threading
import json

class PerformanceMonitor:
    """ONNXモデルの性能監視クラス"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = threading.Lock()
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """ロガーの設定"""
        logger = logging.getLogger("ONNXPerformanceMonitor")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def record_inference(self, model_name: str, latency_ms: float, 
                        input_size: int, memory_usage_mb: float = None):
        """推論メトリクスの記録"""
        
        with self.lock:
            self.metrics[f"{model_name}_latency"].append(latency_ms)
            self.metrics[f"{model_name}_input_size"].append(input_size)
            if memory_usage_mb is not None:
                self.metrics[f"{model_name}_memory"].append(memory_usage_mb)
            
            # 異常値の検出
            self._detect_anomalies(model_name, latency_ms)
    
    def _detect_anomalies(self, model_name: str, current_latency: float):
        """異常値検出"""
        
        latency_key = f"{model_name}_latency"
        latencies = list(self.metrics[latency_key])
        
        if len(latencies) < 10:  # 十分なデータがない場合はスキップ
            return
        
        # 簡単な異常検出（平均 + 3σ）
        mean_latency = np.mean(latencies[:-1])  # 現在の値を除く
        std_latency = np.std(latencies[:-1])
        
        if current_latency > mean_latency + 3 * std_latency:
            self.logger.warning(
                f"異常なレイテンシを検出: {model_name} - "
                f"{current_latency:.2f}ms (平均: {mean_latency:.2f}ms)"
            )
    
    def get_statistics(self, model_name: str) -> Dict:
        """統計情報の取得"""
        
        with self.lock:
            latency_key = f"{model_name}_latency"
            memory_key = f"{model_name}_memory"
            
            if latency_key not in self.metrics:
                return {}
            
            latencies = list(self.metrics[latency_key])
            
            stats = {
                "latency": {
                    "avg": np.mean(latencies),
                    "std": np.std(latencies),
                    "min": np.min(latencies),
                    "max": np.max(latencies),
                    "p95": np.percentile(latencies, 95),
                    "p99": np.percentile(latencies, 99)
                },
                "throughput": len(latencies) / (max(latencies) / 1000) if latencies else 0
            }
            
            if memory_key in self.metrics:
                memory_values = list(self.metrics[memory_key])
                stats["memory"] = {
                    "avg": np.mean(memory_values),
                    "peak": np.max(memory_values)
                }
            
            return stats
    
    def export_metrics(self, filename: str):
        """メトリクスのエクスポート"""
        
        with self.lock:
            export_data = {}
            for key, values in self.metrics.items():
                export_data[key] = list(values)
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        self.logger.info(f"メトリクスをエクスポートしました: {filename}")

class MonitoredInferenceSession:
    """監視付き推論セッション"""
    
    def __init__(self, model_path: str, model_name: str, 
                 monitor: PerformanceMonitor, **kwargs):
        self.session = ort.InferenceSession(model_path, **kwargs)
        self.model_name = model_name
        self.monitor = monitor
    
    def run(self, output_names, input_feed, run_options=None):
        """監視付き推論実行"""
        
        # 入力サイズの計算
        input_size = sum(
            np.prod(arr.shape) for arr in input_feed.values() 
            if isinstance(arr, np.ndarray)
        )
        
        # メモリ使用量測定（オプション）
        import psutil
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 推論実行
        start_time = time.perf_counter()
        outputs = self.session.run(output_names, input_feed, run_options)
        end_time = time.perf_counter()
        
        # メトリクス記録
        latency_ms = (end_time - start_time) * 1000
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_usage = current_memory - initial_memory
        
        self.monitor.record_inference(
            self.model_name, latency_ms, input_size, memory_usage
        )
        
        return outputs

# 使用例
def demo_performance_monitoring():
    """性能監視デモ"""
    
    monitor = PerformanceMonitor()
    
    # 監視付きセッションの作成（仮想的な例）
    # session = MonitoredInferenceSession("model.onnx", "bert_base", monitor)
    
    # シミュレートされた推論データ
    for i in range(100):
        # 正常なレイテンシ
        normal_latency = np.random.normal(50, 5)  # 50ms ± 5ms
        monitor.record_inference("bert_base", normal_latency, 1000)
        
        # 時々異常な値を挿入
        if i == 50:
            monitor.record_inference("bert_base", 200, 1000)  # 異常値
    
    # 統計情報の取得
    stats = monitor.get_statistics("bert_base")
    print("統計情報:", json.dumps(stats, indent=2))
    
    # メトリクスのエクスポート
    monitor.export_metrics("performance_metrics.json")

if __name__ == "__main__":
    demo_performance_monitoring()
    '''
    
    with open("performance_monitoring.py", "w", encoding="utf-8") as f:
        f.write(monitoring_code.strip())
    
    print("✓ 性能監視システムを作成しました: performance_monitoring.py")

## まとめ

第5章では、ONNXモデルの性能最適化と実践的な応用について包括的に学習しました：

### 学習内容の詳細総括

1. **グラフレベル最適化の完全理解**
   - **自動最適化機能**: ONNX Runtimeの4段階最適化レベル
   - **最適化パターン**: 定数畳み込み、演算子融合、Identity除去等の具体的手法
   - **効果測定**: 定量的な性能改善の測定と分析手法

2. **実践的最適化設定**
   - **プロバイダー固有最適化**: CPU/GPU別の詳細な設定方法
   - **並列処理最適化**: スレッド数とアフィニティの適切な設定
   - **メモリ管理最適化**: アリーナ戦略とメモリ制限の調整

3. **ORT形式による高速化**
   - **バイナリ形式の利点**: 読み込み速度とメモリ効率の改善
   - **変換手法**: Python API、CLI、C++ APIでの変換方法
   - **デプロイメント活用**: 本番環境での効率的な運用

4. **実世界での性能検証**
   - **BERTモデル最適化**: 大規模モデルでの実践的最適化
   - **多次元評価**: レイテンシ、メモリ、精度の総合評価
   - **設定比較**: 異なる構成での定量的性能比較

5. **本番環境対応**
   - **性能監視システム**: リアルタイムメトリクス収集と異常検出
   - **多層最適化戦略**: モデル、ランタイム、デプロイメント、ハードウェア各レベル
   - **スケーラビリティ**: 高負荷環境での安定運用

### 実践的な開発能力の向上

この章で習得した技術により、以下の高度な開発能力を獲得しました：

**性能最適化**:
- グラフレベルでの自動最適化活用
- ハードウェア特性に応じた設定調整
- 多次元でのパフォーマンス評価

**本番運用**:
- 大規模モデルの効率的なデプロイメント
- リアルタイム性能監視システム構築
- 異常検出とアラート機能実装

**品質保証**:
- 最適化前後での精度検証
- 包括的なベンチマーク実施
- エラー処理と例外対応

**スケーラビリティ**:
- 負荷変動に対応する動的最適化
- メモリ効率とスループットのバランス最適化
- マルチ環境対応（CPU/GPU/エッジ）

### 商用システムでの価値

これらの技術は、実際の商用システムにおいて以下のような具体的価値を提供します：

**コスト削減**:
- 計算リソース使用量の最適化によるクラウドコスト削減
- エネルギー効率改善による運用コスト低減
- ハードウェア要件の最適化による初期投資削減

**ユーザー体験向上**:
- レスポンス時間短縮による快適性向上
- 高負荷時での安定性確保
- エッジデバイスでの軽快な動作実現

**運用効率化**:
- 自動監視による障害早期発見
- 性能データに基づく予防保守
- デプロイメント自動化による運用負荷軽減

**技術的競争力**:
- 最新最適化技術の効果的活用
- ハードウェア進化への迅速な対応
- スケーラブルなシステム設計による成長対応

次章では、これらの性能最適化技術を活用した具体的な事例研究として、FedAS（連合学習）、スナップショット圧縮イメージング、スペクトル再構成など、最先端の応用例について詳しく学習していきます。
