import pathlib
import re

path = pathlib.Path(__file__).parent

cache = {

}
for file in path.iterdir():
    if not file.name.startswith("Movies"): continue
    try:
        with open(file, "r") as f:
            print(file.name)
            cache[file.name] = []
            for i, line in enumerate(f):
                if line.startswith("train_loss:"):
                    print(i, line)
                    cache[file.name].append((i, line))
    except Exception as e:
        print(e)

...
data = list(cache.items())
metrics_regex = re.compile(r'Recall@5:(\d+\.\d+)\tRecall@10:(\d+\.\d+)\tRecall@20:(\d+\.\d+)'
                           r'\tNDGG@5:(\d+\.\d+)\tNDGG10@10:(\d+\.\d+)\tNDGG@20:(\d+\.\d+)')

best_metrics = {}
# 遍历数据集
for file_name, results in data:
    # 初始化所有指标的最佳值和相应的迭代索引
    best_recall5, best_recall10, best_recall20 = 0, 0, 0
    best_ndgg5, best_ndgg10, best_ndgg20 = 0, 0, 0
    best_epoch_recall5, best_epoch_recall10, best_epoch_recall20 = None, None, None
    best_epoch_ndgg5, best_epoch_ndgg10, best_epoch_ndgg20 = None, None, None

    for idx, line in results:
        match = metrics_regex.search(line)
        if match:
            recall5, recall10, recall20 = map(float, match.groups()[:3])
            ndgg5, ndgg10, ndgg20 = map(float, match.groups()[3:])

            # 找到最佳 Recall 对应的迭代
            if recall5 > best_recall5:
                best_recall5 = recall5
                best_epoch_recall5 = idx
            if recall10 > best_recall10:
                best_recall10 = recall10
                best_epoch_recall10 = idx
            if recall20 > best_recall20:
                best_recall20 = recall20
                best_epoch_recall20 = idx

            # 找到最佳 NDGG 对应的迭代
            if ndgg5 > best_ndgg5:
                best_ndgg5 = ndgg5
                best_epoch_ndgg5 = idx
            if ndgg10 > best_ndgg10:
                best_ndgg10 = ndgg10
                best_epoch_ndgg10 = idx
            if ndgg20 > best_ndgg20:
                best_ndgg20 = ndgg20
                best_epoch_ndgg20 = idx

    # 保存每个指标的最佳值和相应的迭代索引
    best_metrics[file_name] = {
        "best_recall@5": best_recall5, "best_epoch_recall@5": best_epoch_recall5,
        "best_recall@10": best_recall10, "best_epoch_recall@10": best_epoch_recall10,
        "best_recall@20": best_recall20, "best_epoch_recall@20": best_epoch_recall20,
        "best_ndgg@5": best_ndgg5, "best_epoch_ndgg@5": best_epoch_ndgg5,
        "best_ndgg@10": best_ndgg10, "best_epoch_ndgg@10": best_epoch_ndgg10,
        "best_ndgg@20": best_ndgg20, "best_epoch_ndgg@20": best_epoch_ndgg20
    }

# 打印结果
# for file_name, metrics in best_metrics.items():
#     print(f"File: {file_name}")
#     for key, value in metrics.items():
#         print(f"{key}: {value}")
#     print()
# 初始化每个指标的全局最大值和对应的文件名与迭代索引
global_best = {
    "best_recall@5": (0, None, None),
    "best_recall@10": (0, None, None),
    "best_recall@20": (0, None, None),
    "best_ndgg@5": (0, None, None),
    "best_ndgg@10": (0, None, None),
    "best_ndgg@20": (0, None, None)
}

# 遍历每个文件的最佳指标
for file_name, metrics in best_metrics.items():
    # 比较每个指标，找到全局最大值
    for key in global_best:
        if metrics[key] > global_best[key][0]:
            global_best[key] = (metrics[key], file_name, metrics[f"best_epoch_{key.split('_')[1]}"])

to_save = {}
# 打印每个指标的全局最大值、对应的文件名和迭代索引
for key, (value, file_name, epoch) in global_best.items():
    print(f"Global best {key}: {value} from file {file_name} at epoch {epoch}")
    to_save[key] = value

import json
json.dump(to_save, open("results.json", "w"))