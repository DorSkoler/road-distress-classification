import os
import random
import json

RAW_DIR = '../raw'  # or wherever your images are
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
SEED = 42  # for reproducibility

def get_all_roads(raw_dir):
    roads = set()
    for fname in os.listdir(raw_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            parts = fname.split('_')
            if len(parts) >= 2:
                roads.add(f'road_{parts[1]}')
    return sorted(list(roads))

def split_roads(roads, ratios, seed=42):
    random.seed(seed)
    random.shuffle(roads)
    n = len(roads)
    n_train = int(ratios['train'] * n)
    n_val = int(ratios['val'] * n)
    train = roads[:n_train]
    val = roads[n_train:n_train+n_val]
    test = roads[n_train+n_val:]
    return {'train': train, 'val': val, 'test': test}

if __name__ == '__main__':
    roads = get_all_roads(RAW_DIR)
    print(f'Found {len(roads)} unique roads.')
    splits = split_roads(roads, SPLIT_RATIOS, SEED)
    print(json.dumps(splits, indent=2))