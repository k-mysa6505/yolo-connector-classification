#!/usr/bin/env python3
"""
Connector Detection System using YOLOv8
課題提出用統合スクリプト

Usage:
    python connector_detection.py --image_path <path_to_image>
    python connector_detection.py --batch_dir <path_to_directory>
"""

import argparse
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
import os
import sys

class ConnectorDetectionSystem:
    def __init__(self, model_path="best_model.pt"):
        """
        コネクタ検出システムの初期化

        Args:
            model_path: 学習済みモデルのパス
        """
        self.model_path = model_path
        self.class_names = {0: 'Blurred', 1: 'Occluded'}

        # モデルの読み込み
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        # デバイス確認
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def detect_connectors(self, image_path, conf_threshold=0.5, save_visualization=True):
        """
        画像からコネクタを検出

        Args:
            image_path: 入力画像のパス
            conf_threshold: 信頼度閾値
            save_visualization: 検出結果の可視化を保存するか

        Returns:
            dict: 検出結果の辞書
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 推論実行
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        result = results[0]

        # 結果の整理
        detections = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                detection = {
                    'id': i + 1,
                    'class': self.class_names[int(cls)],
                    'class_id': int(cls),
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3])
                    }
                }
                detections.append(detection)

        # 結果の可視化・保存
        if save_visualization and detections:
            output_dir = "detection_results"
            os.makedirs(output_dir, exist_ok=True)

            # 元画像の読み込み
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # バウンディングボックスの描画
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])

                # クラスに応じて色を変更
                color = (0, 255, 0) if det['class'] == 'Blurred' else (255, 0, 0)  # Green for Blurred, Red for Occluded

                # バウンディングボックス描画
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # ラベル描画
                label = f"{det['class']}: {det['confidence']:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 結果画像の保存
            output_path = os.path.join(output_dir, f"{Path(image_path).stem}_detected.jpg")
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to: {output_path}")

        # 統計情報
        stats = {
            'total_detections': len(detections),
            'blurred_count': sum(1 for d in detections if d['class'] == 'Blurred'),
            'occluded_count': sum(1 for d in detections if d['class'] == 'Occluded'),
            'average_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
        }

        return {
            'image_path': image_path,
            'detections': detections,
            'statistics': stats
        }

    def batch_detection(self, image_directory, output_file="batch_results.json"):
        """
        ディレクトリ内の全画像に対してバッチ検出

        Args:
            image_directory: 画像ディレクトリのパス
            output_file: 結果保存ファイル名

        Returns:
            list: 全画像の検出結果リスト
        """
        if not os.path.exists(image_directory):
            raise FileNotFoundError(f"Directory not found: {image_directory}")

        # 対応画像拡張子
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(Path(image_directory).glob(f"*{ext}"))
            image_paths.extend(Path(image_directory).glob(f"*{ext.upper()}"))

        if not image_paths:
            print(f"No images found in {image_directory}")
            return []

        print(f"Processing {len(image_paths)} images...")

        all_results = []

        for i, img_path in enumerate(image_paths):
            try:
                print(f"Processing {i+1}/{len(image_paths)}: {img_path.name}")
                result = self.detect_connectors(str(img_path), save_visualization=False)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # 結果をJSONファイルに保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 統計サマリー
        total_detections = sum(r['statistics']['total_detections'] for r in all_results)
        total_blurred = sum(r['statistics']['blurred_count'] for r in all_results)
        total_occluded = sum(r['statistics']['occluded_count'] for r in all_results)

        print(f"\n=== Batch Detection Summary ===")
        print(f"Total images processed: {len(all_results)}")
        print(f"Total detections: {total_detections}")
        print(f"Blurred connectors: {total_blurred}")
        print(f"Occluded connectors: {total_occluded}")
        print(f"Results saved to: {output_file}")

        return all_results

    def print_detection_summary(self, result):
        """
        検出結果のサマリーを表示

        Args:
            result: detect_connectorsの返り値
        """
        print(f"\n=== Detection Results for {Path(result['image_path']).name} ===")
        print(f"Total detections: {result['statistics']['total_detections']}")
        print(f"Blurred: {result['statistics']['blurred_count']}")
        print(f"Occluded: {result['statistics']['occluded_count']}")
        print(f"Average confidence: {result['statistics']['average_confidence']:.3f}")

        if result['detections']:
            print("\nDetailed results:")
            for det in result['detections']:
                print(f"  ID {det['id']}: {det['class']} (confidence: {det['confidence']:.3f})")
        else:
            print("No connectors detected in this image.")


def main():
    """
    メイン関数 - コマンドライン引数の処理
    """
    parser = argparse.ArgumentParser(
        description='Connector Detection System using YOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 単一画像の検出
    python connector_detection.py --image_path sample_image.jpg

    # ディレクトリ内全画像の検出
    python connector_detection.py --batch_dir ./test_images

    # 信頼度閾値を指定
    python connector_detection.py --image_path sample.jpg --confidence 0.3
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, help='Path to single image file')
    group.add_argument('--batch_dir', type=str, help='Path to directory containing images')

    parser.add_argument('--model_path', type=str, default='best_model.pt',
                       help='Path to trained YOLO model (default: best_model.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detection (default: 0.5)')
    parser.add_argument('--output', type=str, default='detection_results.json',
                       help='Output file for batch results (default: detection_results.json)')

    args = parser.parse_args()

    # システムの初期化
    try:
        detector = ConnectorDetectionSystem(args.model_path)
    except Exception as e:
        print(f"Failed to initialize detection system: {e}")
        sys.exit(1)

    # 検出実行
    if args.image_path:
        # 単一画像の検出
        try:
            result = detector.detect_connectors(
                args.image_path,
                conf_threshold=args.confidence
            )
            detector.print_detection_summary(result)

            # 結果をJSONファイルにも保存
            with open('single_detection_result.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to: single_detection_result.json")

        except Exception as e:
            print(f"Error during detection: {e}")
            sys.exit(1)

    elif args.batch_dir:
        # バッチ検出
        try:
            results = detector.batch_detection(args.batch_dir, args.output)
            if not results:
                print("No images were processed successfully.")
                sys.exit(1)
        except Exception as e:
            print(f"Error during batch detection: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()