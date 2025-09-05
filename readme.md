# ONNX マスターガイド

ONNX/ONNX Runtime を「導入→実装→最適化→応用」まで実務目線で体系化した、ハンズオン中心のドキュメントセットです。最低限のコマンドと再利用しやすいコード断片に絞り、現場投入までを一直線に繋ぎます。

## 収録内容
- 第1章: インストールと基本使用 — [01_install_and_usage.md](01_install_and_usage.md)
- 第2章: Runtimeと開発 — [02_runtime_and_development.md](02_runtime_and_development.md)
- 第3章: 機能と性能分析 — [03_features_and_performance.md](03_features_and_performance.md)
- 第4章: データ最適化 — [04_data_optimization.md](04_data_optimization.md)
- 第5章: モデル性能と応用 — [05_model_performance.md](05_model_performance.md)
- 第6章: 事例分析（ケーススタディ） — [06_case_studies.md](06_case_studies.md)

各章は独立して読めます。まずは第1章から動かし、第5章で最適化し、第6章で応用の全体像を掴む流れがおすすめです。

## こんな人におすすめ
- ONNX/ORT を初めて本番導入するエンジニア
- PyTorch/TF からのエクスポート〜推論最適化の実務フローを固めたい方
- CPU/GPU 切替やグラフ最適化、モデル軽量化の勘所を掴みたい方

## すぐに始める
1) Python 環境で ORT を導入
```bash
pip install onnxruntime  # CPU 版
# GPU 版: pip install onnxruntime-gpu  （CUDA/cuDNN 要件に注意）
```
2) [01_install_and_usage.md](01_install_and_usage.md) の「PythonでのONNXランタイム使用」から動かす
3) 速度/メモリを詰めるときは [05_model_performance.md](05_model_performance.md) のグラフ最適化へ

## 推奨環境（目安）
- OS: Windows 10 / macOS 10.15+ / Linux
- Python: 3.8+
- メモリ: 8GB 以上推奨
- GPU 利用時: 対応 CUDA / cuDNN（NVIDIA GPU, CC 6.0+）

## ファイル一覧
- [01_install_and_usage.md](01_install_and_usage.md): ORT の導入・言語別サンプル・API 概観
- [02_runtime_and_development.md](02_runtime_and_development.md): 実行プロバイダー/ONNX 原理/演算子属性/開発例
- [03_features_and_performance.md](03_features_and_performance.md): Python API・ブロードキャスト・検査/変換
- [04_data_optimization.md](04_data_optimization.md): データ型/実験演算子/画像カテゴリ・最適化ノウハウ
- [05_model_performance.md](05_model_performance.md): グラフ最適化・ORT 形式・BERT など性能検証
- [06_case_studies.md](06_case_studies.md): 連合学習など応用事例の実装スケッチ
- [title.txt](title.txt): 章構成の目次（ページ見出し）

## ナビゲーション
- 初学〜導入: [01_install_and_usage.md](01_install_and_usage.md)
- 開発の勘所: [02_runtime_and_development.md](02_runtime_and_development.md)
- 仕様理解の補強: [03_features_and_performance.md](03_features_and_performance.md)
- 省メモリ/効率化: [04_data_optimization.md](04_data_optimization.md)
- 本番最適化: [05_model_performance.md](05_model_performance.md)
- 応用/ケース: [06_case_studies.md](06_case_studies.md)

## よくあるつまずき
- CUDA 不一致: ドライバ/CUDA/cuDNN の整合をまず確認
- モデル互換: opset と演算子サポートの差異に注意（`onnx.checker` で検査）
- 速度が出ない: 実行プロバイダー設定と最適化レベル（ORT_ENABLE_ALL）を確認

不明点や追加要望があれば、改善方針に反映します。
