import os
import shutil
import random
from pathlib import Path
import yaml

def reorganize_project_structure():
    """
    現在のプロジェクト構造をYOLOv8用に最適化
    """

    # 推奨ファイル構造
    recommended_structure = """
    推奨ファイル構造:

    /
    ├─ dataset/                          # 元データ（保持）
    │  ├─ images/                        # 元画像
    │  ├─ xmls/                          # 元アノテーション
    │  └─ data.yaml                      # 既存設定ファイル
    │
    ├─ connector_dataset/                # YOLOv8用データセット（新規作成）
    │  ├─ train/
    │  │  ├─ images/
    │  │  └─ labels/
    │  ├─ val/
    │  │  ├─ images/
    │  │  └─ labels/
    │  ├─ test/
    │  │  ├─ images/
    │  │  └─ labels/
    │  └─ data.yaml                      # YOLOv8用設定
    │
    ├─ scripts/                          # 学習・推論スクリプト
    │  ├─ train_model.py                 # 学習スクリプト
    │  ├─ evaluate_model.py              # 評価スクリプト
    │  ├─ inference.py                   # 推論スクリプト
    │  └─ analysis_report.py             # 分析レポート
    │
    ├─ models/                           # 学習済みモデル保存
    │  └─ (学習後に生成)
    │
    ├─ results/                          # 実験結果
    │  ├─ training_logs/
    │  ├─ evaluation_results/
    │  └─ inference_outputs/
    │
    ├─ submission/                       # 提出用ファイル
    │  ├─ connector_detection.py         # 提出スクリプト
    │  ├─ best_model.pt                  # 最終モデル
    │  └─ method_explanation.md          # 手法説明
    │
    ├─ labelImg/                         # 既存（保持）
    ├─ .gitignore
    ├─ Pipfile
    ├─ Pipfile.lock
    ├─ README.md
    ├─ requirements.txt
    └─ setup.py                          # このスクリプト
    """

    print(recommended_structure)
    return recommended_structure

def create_directory_structure():
    """
    必要なディレクトリ構造を作成
    """
    directories = [
        'connector_dataset/train/images',
        'connector_dataset/train/labels',
        'connector_dataset/val/images',
        'connector_dataset/val/labels',
        'connector_dataset/test/images',
        'connector_dataset/test/labels',
        'scripts',
        'models',
        'results/training_logs',
        'results/evaluation_results',
        'results/inference_outputs',
        'submission'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def convert_annotations_to_yolo(source_annotations_dir, source_images_dir):
    """
    XMLまたはtxt形式のアノテーションをYOLO形式に変換

    Note: 現在のアノテーションが既にYOLO形式（.txt）の場合は、
    単純にコピーするだけで済む可能性があります
    """

    print("Checking annotation format...")

    # xmlsディレクトリ内のファイルを確認
    xml_dir = Path(source_annotations_dir)
    if not xml_dir.exists():
        print(f"Annotations directory not found: {xml_dir}")
        return False

    # ファイル拡張子を確認
    txt_files = list(xml_dir.glob('*.txt'))
    xml_files = list(xml_dir.glob('*.xml'))

    if txt_files:
        print(f"Found {len(txt_files)} .txt annotation files")
        print("Assuming YOLO format. Will copy directly.")
        return 'yolo'
    elif xml_files:
        print(f"Found {len(xml_files)} .xml annotation files")
        print("Need to convert from XML to YOLO format.")
        return 'xml'
    else:
        print("No annotation files found!")
        return None

def split_and_organize_dataset(source_images_dir, source_labels_dir,
                              train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    データセットを分割してYOLO形式で整理
    """

    print("Organizing dataset...")

    # 画像ファイルのリストを取得
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(Path(source_images_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(source_images_dir).glob(f'*{ext.upper()}')))

    print(f"Found {len(image_files)} images")

    # ファイルをシャッフル
    random.seed(42)  # 再現性のため
    random.shuffle(image_files)

    # データ分割
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    print(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # ファイルをコピー
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, files in splits.items():
        print(f"Processing {split_name} split...")

        for img_file in files:
            # 画像ファイルをコピー
            dst_img = Path(f'connector_dataset/{split_name}/images/{img_file.name}')
            shutil.copy2(img_file, dst_img)

            # 対応するラベルファイルをコピー
            label_file = Path(source_labels_dir) / (img_file.stem + '.txt')
            dst_label = Path(f'connector_dataset/{split_name}/labels/{img_file.stem}.txt')

            if label_file.exists():
                shutil.copy2(label_file, dst_label)
            else:
                # 空のラベルファイルを作成
                dst_label.touch()
                print(f"Warning: No label file found for {img_file.name}")

def create_yolo_config():
    """
    YOLOv8用のdata.yamlファイルを作成
    """

    config = {
        'path': str(Path.cwd() / 'connector_dataset'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 2,
        'names': {
            0: 'Blurred',
            1: 'Occluded'
        }
    }

    config_path = Path('connector_dataset/data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created YOLO config: {config_path}")

def copy_scripts():
    """
    必要なスクリプトファイルを作成
    """

    scripts_info = {
        'train_model.py': 'YOLOv8学習スクリプト',
        'evaluate_model.py': '推論・評価スクリプト',
        'connector_detection.py': '提出用統合スクリプト',
        'analysis_report.py': 'レポート用分析スクリプト'
    }

    print("\n以下のスクリプトを手動で作成してください：")
    for script, description in scripts_info.items():
        if script == 'connector_detection.py':
            target_path = f'submission/{script}'
        else:
            target_path = f'scripts/{script}'
        print(f"  {target_path} - {description}")

def update_gitignore():
    """
    .gitignoreファイルを更新
    """

    gitignore_additions = """
# YOLOv8 specific
models/
results/
runs/
*.pt
*.onnx

# Dataset (optional - comment out if you want to track dataset)
connector_dataset/

# Inference outputs
inference_results/
detection_results/
evaluation_report/
analysis_report/

# Temporary files
*.tmp
*.cache
"""

    with open('.gitignore', 'a') as f:
        f.write(gitignore_additions)

    print("Updated .gitignore")

def main():
    """
    メイン実行関数
    """
    print("=== プロジェクト構造再編成 ===\n")

    # 現在の構造を確認
    if not Path('dataset/images').exists():
        print("Error: dataset/images directory not found!")
        return

    if not Path('dataset/xmls').exists():
        print("Error: dataset/xmls directory not found!")
        return

    # ディレクトリ構造を作成
    create_directory_structure()

    # アノテーション形式を確認
    annotation_format = convert_annotations_to_yolo('dataset/xmls', 'dataset/images')

    if annotation_format is None:
        print("Cannot proceed without annotation files!")
        return

    # データセットを分割・整理
    split_and_organize_dataset('dataset/images', 'dataset/xmls')

    # YOLO設定ファイルを作成
    create_yolo_config()

    # .gitignoreを更新
    update_gitignore()

    # スクリプト情報を表示
    copy_scripts()

    print("\n=== 完了 ===")
    print("プロジェクト構造の再編成が完了しました。")
    print("\n次のステップ:")
    print("1. 前回提供したスクリプトを適切なディレクトリに配置")
    print("2. requirements.txtを更新してYOLOv8関連パッケージを追加")
    print("3. scripts/train_model.py を実行して学習開始")

    # 推奨構造を再表示
    reorganize_project_structure()

if __name__ == "__main__":
    main()
