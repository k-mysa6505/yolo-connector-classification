import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os

class ConnectorYOLOInference:
    def __init__(self, model_path):
        """
        推論クラスの初期化
        
        Args:
            model_path: 学習済みモデルのパス
        """
        self.model = YOLO(model_path)
        self.class_names = ['Blurred', 'Occluded']
        
    def predict_single_image(self, image_path, conf_threshold=0.5, save_result=True):
        """
        単一画像の推論
        
        Args:
            image_path: 画像パス
            conf_threshold: 信頼度閾値
            save_result: 結果画像を保存するか
        """
        results = self.model(image_path, conf=conf_threshold)
        
        if save_result:
            # 結果を保存
            output_path = f"inference_results/{Path(image_path).stem}_result.jpg"
            os.makedirs("inference_results", exist_ok=True)
            results[0].save(output_path)
        
        return results[0]
    
    def predict_batch(self, image_dir, conf_threshold=0.5):
        """
        バッチ推論
        
        Args:
            image_dir: 画像ディレクトリ
            conf_threshold: 信頼度閾値
        """
        image_paths = list(Path(image_dir).glob("*.jpg")) + \
                     list(Path(image_dir).glob("*.png")) + \
                     list(Path(image_dir).glob("*.jpeg"))
        
        results = []
        for img_path in image_paths:
            result = self.model(str(img_path), conf=conf_threshold)
            results.append({
                'image_path': str(img_path),
                'result': result[0]
            })
        
        return results
    
    def evaluate_test_set(self, test_image_dir, test_label_dir, conf_threshold=0.5):
        """
        テストセットでの詳細評価
        
        Args:
            test_image_dir: テスト画像ディレクトリ
            test_label_dir: テストラベルディレクトリ
            conf_threshold: 信頼度閾値
        """
        image_paths = list(Path(test_image_dir).glob("*.jpg")) + \
                     list(Path(test_image_dir).glob("*.png"))
        
        predictions = []
        ground_truths = []
        
        for img_path in image_paths:
            # 推論実行
            result = self.model(str(img_path), conf=conf_threshold)[0]
            
            # ラベルファイルを読み込み
            label_path = Path(test_label_dir) / (img_path.stem + '.txt')
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    gt_lines = f.readlines()
                
                # 予測結果の処理
                if result.boxes is not None:
                    pred_classes = result.boxes.cls.cpu().numpy()
                    pred_confs = result.boxes.conf.cpu().numpy()
                    
                    for cls, conf in zip(pred_classes, pred_confs):
                        predictions.append(int(cls))
                
                # Ground Truthの処理
                for line in gt_lines:
                    if line.strip():
                        gt_class = int(line.split()[0])
                        ground_truths.append(gt_class)
        
        return predictions, ground_truths
    
    def generate_evaluation_report(self, predictions, ground_truths, save_path="evaluation_report"):
        """
        評価レポートの生成
        
        Args:
            predictions: 予測結果リスト
            ground_truths: 正解ラベルリスト
            save_path: 保存パス
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 分類レポート
        report = classification_report(
            ground_truths, 
            predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # データフレームに変換
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{save_path}/classification_report.csv")
        
        print("Classification Report:")
        print(classification_report(ground_truths, predictions, target_names=self.class_names))
        
        # 混同行列
        cm = confusion_matrix(ground_truths, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 性能メトリクスの可視化
        metrics_df = report_df.iloc[:-3, :-1]  # マクロ平均等を除く
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics_df['precision'].plot(kind='bar', ax=axes[0], title='Precision by Class')
        axes[0].set_xticklabels(self.class_names, rotation=0)
        
        metrics_df['recall'].plot(kind='bar', ax=axes[1], title='Recall by Class')
        axes[1].set_xticklabels(self.class_names, rotation=0)
        
        metrics_df['f1-score'].plot(kind='bar', ax=axes[2], title='F1-Score by Class')
        axes[2].set_xticklabels(self.class_names, rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/metrics_by_class.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return report_df
    
    def interactive_inference(self):
        """
        インタラクティブ推論（デモ用）
        """
        print("Interactive Inference Mode")
        print("Enter image path (or 'quit' to exit):")
        
        while True:
            image_path = input("Image path: ").strip()
            
            if image_path.lower() == 'quit':
                break
            
            if not os.path.exists(image_path):
                print("File not found!")
                continue
            
            try:
                result = self.predict_single_image(image_path)
                
                if result.boxes is not None:
                    for i, (box, cls, conf) in enumerate(zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.cls.cpu().numpy(),
                        result.boxes.conf.cpu().numpy()
                    )):
                        class_name = self.class_names[int(cls)]
                        print(f"Detection {i+1}: {class_name} (confidence: {conf:.3f})")
                else:
                    print("No objects detected")
                    
            except Exception as e:
                print(f"Error: {e}")

# 使用例
if __name__ == "__main__":
    # 学習済みモデルのパス（適切に設定してください）
    model_path = "connector_detection/yolov8n_connector/weights/best.pt"
    
    # 推論クラスの初期化
    inference = ConnectorYOLOInference(model_path)
    
    # テストセットでの評価
    predictions, ground_truths = inference.evaluate_test_set(
        "connector_dataset/test/images",
        "connector_dataset/test/labels"
    )
    
    # 評価レポート生成
    report_df = inference.generate_evaluation_report(predictions, ground_truths)
    
    # インタラクティブ推論の開始
    # inference.interactive_inference()