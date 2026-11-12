# -*- coding: utf-8 -*-
"""embed_cluster_probe.py —— 用 BERT 句向量做「表示空间 / 聚簇」诊断

【它回答什么】
  ① 表示空间里 18 个类分得开吗？ → 该不该合并近义类、该合并哪几对
  ② 哪些训练样本像「标错了」？   → 产出待人工复核清单 CSV

【它不做什么】
  **不训练、不改模型**，只加载 saved/best_model.pt 做一次前向。
  所以成本极低（几分钟），可反复跑。

【用法】必须 cd 到 src（config.py 用相对路径 '../data'）
    cd bert_project_11/src
    python embed_cluster_probe.py                  # 全量
    python embed_cluster_probe.py --limit 5000     # 抽样，先看趋势（快）
    python embed_cluster_probe.py --k 14,18        # 指定聚簇数
    python embed_cluster_probe.py --no-knn         # 跳过近邻复核（最省时）

【输出】控制台报告 + ../embed_out/ 下的 CSV
"""
import argparse
import csv
import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert_classifier import BertClassifier
from config import Config
from utils import TextDataset, collate_fn, load_raw_data

conf = Config()
CLASSES = conf.class_list
NC = conf.num_classes
OUT_DIR = '../embed_out'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=0,
                   help='>0 时只用前 N 条（抽样加速）')
    p.add_argument('--k', type=str, default='18',
                   help='聚簇数，可给多个用逗号分隔，如 14,18')
    p.add_argument('--top-frac', type=float, default=0.10,
                   help='按质心间隔筛出的“疑似错标”比例（默认 10%%）')
    p.add_argument('--knn-k', type=int, default=20, help='近邻复核的 k')
    p.add_argument('--no-knn', action='store_true', help='跳过近邻复核')
    p.add_argument('--ckpt', type=str, default='best_model.pt',
                   help='saved/ 下的权重文件名')
    return p.parse_args()


@torch.no_grad()
def extract_embeddings(model, data, tag):
    """返回 L2 归一化后的 [CLS] 句向量 (N,768) 与标签 (N,)"""
    loader = DataLoader(TextDataset(data), batch_size=conf.batch_size,
                        shuffle=False, collate_fn=collate_fn)
    embs, labs = [], []
    for ids, am, y in tqdm(loader, desc=f'embed {tag}'):
        out = model.bert(input_ids=ids.to(conf.device),
                         attention_mask=am.to(conf.device))
        embs.append(out.pooler_output.float().cpu().numpy())
        labs.append(y.numpy())
    E = np.concatenate(embs)
    L = np.concatenate(labs)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)   # 归一化后内积=余弦
    return E, L


def class_centroids(E, L):
    C = np.stack([E[L == c].mean(0) if (L == c).any() else np.zeros(E.shape[1])
                  for c in range(NC)])
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    return C


def centroid_confusion(E, L, C, topn=15):
    """类级质心混淆：真实类 c 的样本里，最近质心落在别类 c' 的比例"""
    sim = E @ C.T
    nearest = sim.argmax(1)
    acc = balanced_accuracy_score(L, nearest)
    pairs = []
    for c in range(NC):
        m = L == c
        if m.sum() == 0:
            continue
        cnt = np.bincount(nearest[m], minlength=NC)
        for t in range(NC):
            if t != c and cnt[t] > 0:
                pairs.append((cnt[t] / m.sum(), c, t, int(cnt[t]), int(m.sum())))
    pairs.sort(reverse=True)
    return acc, pairs[:topn]


def cluster_report(E, L, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    pred = km.fit_predict(E)
    correct, rows = 0, []
    for cl in range(k):
        m = pred == cl
        if m.sum() == 0:
            continue
        cnt = np.bincount(L[m], minlength=NC)
        top = int(cnt.argmax())
        correct += cnt[top]
        top3 = [CLASSES[i] for i in np.argsort(-cnt)[:3] if cnt[i] > 0]
        rows.append((int(m.sum()), CLASSES[top], cnt[top] / m.sum(), top3))
    purity = correct / len(L)
    ari = adjusted_rand_score(L, pred)
    nmi = normalized_mutual_info_score(L, pred)
    sil = float('nan')
    if len(E) > k:
        sil = silhouette_score(E, pred, sample_size=min(6000, len(E)),
                               random_state=42)
    return purity, ari, nmi, sil, rows


def mislabel_candidates(E, L, C, data, top_frac, knn_k, use_knn):
    """先用「到本类质心 vs 到最近他类质心」的间隔排序，再用近邻复核"""
    sim = E @ C.T
    own = sim[np.arange(len(L)), L]
    other = sim.copy()
    other[np.arange(len(L)), L] = -9.0
    best_other = other.max(1)
    best_other_cls = other.argmax(1)
    margin = own - best_other
    nearest_cls = sim.argmax(1)          # 注意：这里含自身类，等价于 argmax(own, best_other)

    n = max(200, int(len(L) * top_frac))
    idx = np.argsort(margin)[:n]         # 间隔最小的最可疑

    if use_knn:
        nn = NearestNeighbors(n_neighbors=min(knn_k + 1, len(E)),
                              algorithm='brute', metric='cosine', n_jobs=-1)
        nn.fit(E)
        _, ind = nn.kneighbors(E[idx])
        nb = ind[:, 1:]                                   # 去掉自己
        vote = np.array([np.bincount(L[r], minlength=NC).argmax() for r in nb])
        keep = vote != L[idx]                             # 近邻多数类 ≠ 自己标签
        idx, vote = idx[keep], vote[keep]
    else:
        vote = nearest_cls[idx]
        keep = vote != L[idx]
        idx, vote = idx[keep], vote[keep]

    rows = [(int(i), data[i][0], CLASSES[L[i]], CLASSES[vote[j]],
             float(margin[i]), CLASSES[nearest_cls[i]]) for j, i in enumerate(idx)]
    return rows


def main():
    args = parse_args()

    print('=' * 72)
    print(f'device={conf.device}  classes={NC}  pad_size={conf.pad_size}')
    print('=' * 72)

    ckpt = os.path.join(conf.model_save_path, args.ckpt)
    if not os.path.exists(ckpt):
        raise SystemExit(f'找不到权重 {ckpt}，先跑完 train.py 再执行本脚本')
    model = BertClassifier().to(conf.device)
    model.load_state_dict(torch.load(ckpt, map_location=conf.device))
    model.eval()
    print(f'已加载权重：{ckpt}\n')

    train_data = load_raw_data(conf.train_path)
    test_data = load_raw_data(conf.test_path)
    if args.limit:
        train_data = train_data[:args.limit]
        test_data = test_data[:args.limit]
    print(f'train {len(train_data)} 条 / test {len(test_data)} 条\n')

    Etr, Ltr = extract_embeddings(model, train_data, 'train')
    Ete, Lte = extract_embeddings(model, test_data, 'test')
    print(f'\n句向量 shape: train {Etr.shape}  test {Ete.shape}\n')

    Ctr = class_centroids(Etr, Ltr)

    print('-' * 72)
    print('【1】类级质心混淆（train）—— 谁和谁在表示空间里撞了')
    print('-' * 72)
    acc_tr, pairs_tr = centroid_confusion(Etr, Ltr, Ctr)
    print(f'用「最近类质心」直接判决的 balanced accuracy = {acc_tr:.4f}\n')
    print(f'{"真实类":<12}{"被认成":<12}{"占比":>8}{"条数":>8}{"该类总数":>10}')
    for frac, c, t, n, tot in pairs_tr:
        print(f'{CLASSES[c]:<12}{CLASSES[t]:<12}{frac:>8.1%}{n:>8}{tot:>10}')

    print()
    print('-' * 72)
    print('【2】训练样本里的「疑似错标」')
    print('-' * 72)
    print(f'筛选口径：到本类质心的相似度 ≪ 到最近他类质心（取最可疑的 {args.top_frac:.0%}）'
          f'{"，再用 k=%d 近邻复核" % args.knn_k if not args.no_knn else ""}')
    rows = mislabel_candidates(Etr, Ltr, Ctr, train_data, args.top_frac,
                               args.knn_k, not args.no_knn)
    print(f'→ 命中 {len(rows)} 条（占训练集 {len(rows) / len(train_data):.2%}）\n')
    from collections import Counter
    pair_cnt = Counter((r[2], r[3]) for r in rows)
    print('Top 错标组合（原标签 → 更像的类）：')
    for (a, b), c in pair_cnt.most_common(12):
        print(f'  {a:<12} → {b:<12} {c:>5} 条')

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'suspect_labels.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['行号', '文本', '原标签', '更像的标签', '质心间隔', '最近质心类',
                    '处理', '新标签'])
        for r in rows:
            w.writerow(list(r) + ['', ''])
    print(f'\n已导出待复核清单：{csv_path}')
    print('  ↑ 「行号」= 该条在 data/train.csv 里的行序（含表头时已对齐为数据行号）')
    print('  ↑ 最后两列「处理 / 新标签」留给你手工填：处理填 删 或 改（留空=保留），'
          '改 时把正确类名写进「新标签」')

    print()
    print('-' * 72)
    print('【3】聚簇纯度（train）—— 无监督能恢复到什么程度')
    print('-' * 72)
    for k in [int(x) for x in args.k.split(',') if x.strip()]:
        purity, ari, nmi, sil, crows = cluster_report(Etr, Ltr, k)
        print(f'k={k:<3} purity={purity:.4f}  ARI={ari:.4f}  NMI={nmi:.4f}  '
              f'silhouette={sil:.4f}')
        crows.sort(reverse=True)
        print(f'  最大的 5 个簇：')
        for size, top, share, top3 in crows[:5]:
            print(f'    {size:>6} 条 | 主类 {top}({share:.0%}) | 前三 {"/".join(top3)}')
        print()

    print('=' * 72)
    print('结果怎么读：')
    print('  · ARI / NMI 越高 → 表示空间越贴合现有标签，聚簇「再分类」越没用')
    print('  · 质心混淆里集中的类对 → 强烈提示「类定义重叠」，优先考虑合并')
    print('  · suspect_labels.csv 打开肉眼扫一遍，判断是标签错还是真难')
    print('=' * 72)


if __name__ == '__main__':
    main()
