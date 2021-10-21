from maskrcnn_benchmark.data.datasets.chinese_collect import ChineseCollectDataset
from tqdm import tqdm
dataset = ChineseCollectDataset('./datasets/chinese_collect',is_train=False)
for sample in tqdm(dataset):
    continue
