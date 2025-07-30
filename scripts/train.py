import torch
from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class ConnectorYOLOTrainer:
    def __init__(self, data_yaml_path, model_size='n'):
        """
        YOLOv8によるコネクタ検出モデルの学習クラス

        Args:
            data_yaml_path: データセット設定ファイルのパス
            model_size: モデルサイズ ('n', 's', 'm', 'l', 'x')
        """
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.model = None
        self.results = None

        # GPUの確認
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

    def initialize_model(self):
        """モデルを初期化"""
        model_name = f'yolov8{self.model_size}.pt'
        self.model = YOLO(model_name)
        print(f"Initialized {model_name}")

    def train(self, epochs=100, imgsz=640, batch_size=16):
        """
        モデルを学習

        Args:
            epochs: エポック数
            imgsz: 入力画像サイズ
            batch_size: バッチサイズ
        """
        if self.model is None:
            self.initialize_model()

        # 学習パラメータ
        train_args = {
            'data': self.data_yaml_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': self.device,
            'project': 'connector_detection',
            'name': f'yolov8{self.model_size}_connector',
            'save_period': 10,  # 10エポックごとにチェックポイント保存
            'patience': 50,  # Early stopping patience
            'save': True,
            'plots': True,
            'cache': True,  # データキャッシュでロード高速化
        }

        # クラス不均衡対策
        # class_weights = [1.4, 1.0]  # Blurred, Occluded

        print("Starting training...")
        self.results = self.model.train(**train_args)

        print("Training completed!")
        return self.results

    def evaluate(self):
        """モデルの評価"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # バリデーションデータで評価
        val_results = self.model.val()

        # テストデータで評価（存在する場合）
        test_results = self.model.val(split='test')

        return val_results, test_results

    def export_model(self, format='onnx'):
        """モデルのエクスポート"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.export(format=format)
        print(f"Model exported in {format} format")

    def analyze_training_results(self):
        """学習結果の分析"""
        if self.results is None:
            raise ValueError("No training results available!")

        # 学習履歴の可視化
        results_df = pd.read_csv(self.results.save_dir / 'results.csv')

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(results_df['epoch'], results_df['train/box_loss'], label='Train Box Loss')
        axes[0, 0].plot(results_df['epoch'], results_df['val/box_loss'], label='Val Box Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(results_df['epoch'], results_df['train/cls_loss'], label='Train Cls Loss')
        axes[0, 1].plot(results_df['epoch'], results_df['val/cls_loss'], label='Val Cls Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].legend()

        # Metrics
        axes[1, 0].plot(results_df['epoch'], results_df['metrics/precision(B)'], label='Precision')
        axes[1, 0].plot(results_df['epoch'], results_df['metrics/recall(B)'], label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()

        axes[1, 1].plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP@0.5')
        axes[1, 1].plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[1, 1].set_title('mAP Scores')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.results.save_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# 使用例
if __name__ == "__main__":
    # 学習の実行
    trainer = ConnectorYOLOTrainer('../connector_dataset/data.yaml', model_size='n')

    # 学習実行
    results = trainer.train(
        epochs=100,
        imgsz=640,
        batch_size=32  # GPU性能に応じて調整
    )

    # 評価
    val_results, test_results = trainer.evaluate()

    # 結果分析
    trainer.analyze_training_results()

    # モデルエクスポート
    trainer.export_model('onnx')
