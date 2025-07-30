import os
from collections import Counter

def count_classes(labels_dir):
    class_counts = Counter()
    for txt_file in os.listdir(labels_dir):
        if txt_file.endswith('.txt'):
            with open(os.path.join(labels_dir, txt_file), 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if class_id == 2:
                            print(f"Warning: Class ID 2 found in {txt_file}, which is not expected.")
                        class_counts[class_id] += 1
    return class_counts

# 各分割での分布確認
for split in ['train', 'val', 'test']:
    counts = count_classes(f'connector_dataset/{split}/labels')
    print(f"{split}: {dict(counts)}")
