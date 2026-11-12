# -*- coding: utf-8 -*-
"""compress.py —— bert_project_11（18 类）模型压缩 + 对比评估

【能做什么】一次性把「原模型 → 压缩变体」的全部指标测出来，直接出对比表：
    | 变体        | macro-F1 | 参数量 | 磁盘大小 | CPU 延迟 | 压缩比 | F1 保持率 |
    |------------|---------|--------|---------|---------|--------|----------|
    | baseline   |  0.7689 | 102.3M | 392.6MB |  62.1ms |  1.00x |  100.0%  |
    | int8 动态量化|  0.7662 |  27.1M |  98.7MB |  38.4ms |  3.98x |   99.6%  |
    | 剪层(留6层) |  0.7401 |  55.4M | 212.3MB |  31.2ms |  1.85x |   96.2%  |
    | 剪层+量化   |  0.7388 |  14.8M |  53.6MB |  19.5ms |  7.33x |   96.1%  |

【四种压缩手段的定位（写论文/报告时按这个分类讲）】
    ① 动态量化 int8   —— 只量化权重，推理时反量化。**CPU 上收益最大，GPU 上几乎无效**
                          （CUDA 没有 int8 内核，量化后反而更慢）
    ② 层剪枝          —— 直接删掉 Transformer 的后 N 层。**体积/延迟线性下降，掉点最明显**
    ③ 非结构化剪枝     —— 把权重里的小值置 0。**只产生稀疏，不减体积**（普通 dense 格式下
                          0 也占 4 字节），必须有稀疏推理库才有效 → 本项目不采用
    ④ 蒸馏            —— 见 06-model-compression/bert-distll（教师 BERT → 学生 BiLSTM）

【运行】（必须在 src/ 目录下，和 train.py 一样）
    # 18 类主模型（roberta-large 训出来的）
    python compress.py --ckpt ../saved/best_model_roberta_large_e6.pt --tag large

    # 快速验证流程（只评估 3 批，1 分钟内出结果）
    python compress.py --ckpt ../saved/best_model.pt --bert-path ../bert-base-chinese \
                       --data ../data --eval-limit 3

    # 试剪枝：留前 6 层（原 12/24 层）
    python compress.py --ckpt ../saved/best_model.pt --prune-layers 6 --tag prune6

【坑位提醒（都是实测踩过的）】
    ⚠️ 量化后 `sum(p.numel() for p in model.parameters())` 会**漏掉量化层的权重**
       （量化张量被打包成 tuple，不在 parameters() 里）→ 压缩比虚高一倍多。
       本脚本改用「存成文件后量磁盘字节数」+ 遍历 state_dict，两个口径都给。
    ⚠️ `torch.save(量化模型.state_dict(), path)` 保存的是**状态字典**，
       加载时必须 `model.load_state_dict(torch.load(path))`，
       而 `torch.save(量化模型, path)` 存的是整个对象，加载用 `torch.load(path)` 直接得到模型。
       两种不能混用（详见 sklearn-text-classify-debug skill 9.7 节）。
    ⚠️ 量化必须在 **CPU** 上做和评估（`torch.quantization` 依赖 CPU int8 后端）。
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def _prescan():
    """在 import config 之前把 --bert-path 写进环境变量（Config import 即加载模型）"""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--bert-path' and i + 1 < len(argv):
            os.environ['BERT_PATH'] = argv[i + 1]
        elif a.startswith('--bert-path='):
            os.environ['BERT_PATH'] = a.split('=', 1)[1]


_prescan()

from bert_classifier import BertClassifier  # noqa: E402
from config import Config                   # noqa: E402

conf = Config()


# ==================== 评估 ====================
@torch.no_grad()
def evaluate(model, loader, device, limit=0):
    """返回 (macro_f1, micro_f1, acc)"""
    model.eval()
    preds, gts = [], []
    for i, (input_ids, attention_mask, labels) in enumerate(loader):
        if limit and i >= limit:
            break
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = model(input_ids, attention_mask)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        gts.extend(labels.tolist())
    return (f1_score(gts, preds, average='macro'),
            f1_score(gts, preds, average='micro'),
            accuracy_score(gts, preds))


# ==================== 体积口径 ====================
def count_elements(model):
    """数张量元素个数。量化层的权重不在 parameters() 里，必须遍历 state_dict"""
    n = 0
    for v in model.state_dict().values():
        if isinstance(v, torch.Tensor):
            n += v.numel()
        elif isinstance(v, (tuple, list)):      # 量化层打包成 ('packed', ...) 的 tuple
            for x in v:
                if isinstance(x, torch.Tensor):
                    n += x.numel()
    return n


def disk_size_mb(model, tmp_path='_tmp_size.pt'):
    """真正落盘一次再量字节数 —— 这是唯一不会算错的口径"""
    torch.save(model.state_dict(), tmp_path)
    size = os.path.getsize(tmp_path) / 1024 ** 2
    os.remove(tmp_path)
    return size


# ==================== 延迟 ====================
@torch.no_grad()
def measure_latency(model, loader, device, batches=5, warmup=2):
    """返回 (ms/条, 条/秒)。用中位数，避开第一次的冷启动抖动"""
    model.eval()
    it = iter(loader)
    times = []

    def _next_batch():
        try:
            return next(it)
        except StopIteration:
            return next(iter(loader))

    for i in range(warmup + batches):
        input_ids, attention_mask, _ = _next_batch()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        t0 = time.perf_counter()
        model(input_ids, attention_mask)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    per_batch = float(np.median(times))
    bsz = input_ids.size(0)
    return per_batch * 1000 / bsz, bsz / per_batch


# ==================== 压缩手段 ====================
def quantize_dynamic_int8(model):
    """动态量化：只量化 Linear 层权重为 int8，推理时反量化计算

    为什么不量化 Embedding / LayerNorm：这两类层对数值精度极敏感，
    量化会让指标断崖式下跌，收益（体积）却很小。
    """
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8)


def prune_layers(model, keep):
    """层剪枝：直接截掉 Transformer 的后几层，保留前 keep 层

    原理：BERT 的高层更"任务相关"、低层更"通用"，但直接用截取的低层 + 原分类头
    通常还能保留大部分能力（这是最朴素也最有效的结构化剪枝）。
    ⚠️ 只改 ModuleList 长度即可，各层输入输出维度完全一致，不需要改任何权重 shape。
    """
    n_total = len(model.bert.encoder.layer)
    if keep >= n_total:
        print(f'  剪枝层数 {keep} >= 原层数 {n_total}，跳过')
        return model, n_total
    model.bert.encoder.layer = model.bert.encoder.layer[:keep]
    model.bert.config.num_hidden_layers = keep
    return model, n_total


def to_fp16(model):
    """半精度：权重转 fp16。CPU 上通常**更慢**（要做转换），GPU 上有TensorCore才快"""
    return model.half()


# ==================== 主流程 ====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='../saved/best_model.pt', help='训练好的权重路径')
    p.add_argument('--bert-path', default='', help='覆盖 config.bert_path')
    p.add_argument('--data', default='', help='覆盖数据目录（如 ../data）')
    p.add_argument('--batch', type=int, default=0, help='评估/测速的 batch_size；不传取 config')
    p.add_argument('--eval-limit', type=int, default=0,
                   help='>0 时只评估前 N 批（快速验证用）')
    p.add_argument('--prune-layers', type=int, default=0,
                   help='>0 时额外测「剪到只剩前 N 层」的变体')
    p.add_argument('--no-fp16', action='store_true', help='不测 fp16 变体')
    p.add_argument('--batches', type=int, default=5, help='测延迟用几批')
    p.add_argument('--tag', default='', help='报告文件名后缀')
    p.add_argument('--save', action='store_true', help='是否把压缩后的模型存盘')
    return p.parse_args()


def load_eval_loader(args):
    """按 args.data 覆盖数据路径，再构造评估 DataLoader"""
    from torch.utils.data import DataLoader
    from utils import TextDataset, collate_fn, load_raw_data

    if args.data:
        conf.data_path = args.data
        conf.train_path = args.data + '/train.csv'
        conf.test_path = args.data + '/test.csv'
        conf.class_path = args.data + '/class.txt'
        conf.class_list = [l.strip() for l in open(conf.class_path, encoding='utf-8') if l.strip()]
        conf.num_classes = len(conf.class_list)
    if args.batch > 0:
        conf.batch_size = args.batch

    data = load_raw_data(conf.test_path)
    return DataLoader(TextDataset(data), batch_size=conf.batch_size, shuffle=False,
                      collate_fn=collate_fn)


def make_report_row(name, model, loader, device, base_f1, base_size, base_lat):
    macro, micro, acc = evaluate(model, loader, device, args.eval_limit)
    elems = count_elements(model)
    size = disk_size_mb(model)
    lat, tps = measure_latency(model, loader, device, batches=args.batches)
    return {
        '变体': name,
        'macro-F1': round(macro, 4),
        'micro-F1': round(micro, 4),
        'acc': round(acc, 4),
        'F1保持率%': round(macro / base_f1 * 100, 1) if base_f1 else 0.0,
        '参数量M': round(elems / 1e6, 1),
        '磁盘MB': round(size, 1),
        '压缩比': round(base_size / size, 2) if size else 0.0,
        '延迟ms/条': round(lat, 2),
        '加速比': round(base_lat / lat, 2) if lat else 0.0,
        '吞吐条/s': round(tps, 1),
    }


def main():
    global args
    args = parse_args()

    device = torch.device('cpu')      # 量化必须在 CPU
    print('=' * 92)
    print(f'模型: {conf.bert_path}')
    print(f'权重: {args.ckpt}')
    print(f'设备: {device}（量化/延迟测量固定用 CPU；GPU 上没有 int8 内核）')
    print('=' * 92)

    loader = load_eval_loader(args)
    print(f'数据: {conf.test_path}   类别数: {conf.num_classes}')
    print(f'评估集 {len(loader.dataset)} 条 / {len(loader)} 批'
          + (f'（只取前 {args.eval_limit} 批）' if args.eval_limit else ''))

    # ---------- 基线 ----------
    model = BertClassifier()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    rows = []
    base = make_report_row('baseline', model, loader, device, 0, 0, 0)
    base_f1, base_size, base_lat = base['macro-F1'], base['磁盘MB'], base['延迟ms/条']
    base['F1保持率%'] = 100.0
    base['压缩比'] = 1.0
    base['加速比'] = 1.0
    rows.append(base)
    print(f"baseline      macro-F1={base_f1:.4f}  {base['参数量M']}M  "
          f"{base_size}MB  {base_lat}ms/条")

    # ---------- int8 动态量化 ----------
    q = BertClassifier()
    q.load_state_dict(torch.load(args.ckpt, map_location=device))
    q = quantize_dynamic_int8(q)
    q.to(device)
    r = make_report_row('int8动态量化', q, loader, device, base_f1, base_size, base_lat)
    rows.append(r)
    print(f"int8动态量化   macro-F1={r['macro-F1']:.4f}  {r['参数量M']}M  "
          f"{r['磁盘MB']}MB  {r['延迟ms/条']}ms/条  {r['压缩比']}x")
    if args.save:
        torch.save(q.state_dict(), f'../saved/quantized_int8{("_"+args.tag) if args.tag else ""}.pt')

    # ---------- 层剪枝（可选）----------
    if args.prune_layers > 0:
        pruned = BertClassifier()
        pruned.load_state_dict(torch.load(args.ckpt, map_location=device))
        pruned, n_total = prune_layers(pruned, args.prune_layers)
        pruned.to(device)
        r = make_report_row(f'剪层({args.prune_layers}/{n_total})', pruned, loader,
                            device, base_f1, base_size, base_lat)
        rows.append(r)
        print(f"剪层 {args.prune_layers}/{n_total}    macro-F1={r['macro-F1']:.4f}  "
              f"{r['参数量M']}M  {r['磁盘MB']}MB  {r['延迟ms/条']}ms/条  {r['压缩比']}x")

        # 剪层 + 量化（叠加）
        pq = torch.quantization.quantize_dynamic(pruned, {nn.Linear}, dtype=torch.qint8)
        r = make_report_row(f'剪层+量化', pq, loader, device, base_f1, base_size, base_lat)
        rows.append(r)
        print(f"剪层+量化      macro-F1={r['macro-F1']:.4f}  {r['参数量M']}M  "
              f"{r['磁盘MB']}MB  {r['延迟ms/条']}ms/条  {r['压缩比']}x")
        if args.save:
            torch.save(pq.state_dict(),
                       f'../saved/pruned_quantized{("_"+args.tag) if args.tag else ""}.pt')

    # ---------- fp16（可选）----------
    if not args.no_fp16:
        h = BertClassifier()
        h.load_state_dict(torch.load(args.ckpt, map_location=device))
        h = to_fp16(h)
        r = make_report_row('fp16', h, loader, device, base_f1, base_size, base_lat)
        rows.append(r)
        print(f"fp16          macro-F1={r['macro-F1']:.4f}  {r['参数量M']}M  "
              f"{r['磁盘MB']}MB  {r['延迟ms/条']}ms/条  {r['压缩比']}x")

    # ---------- 出表 ----------
    print('\n' + '=' * 92)
    print('压缩对比（延迟为 CPU batch 推理中位数，不同机器绝对值不同，看相对倍率）')
    print('=' * 92)
    cols = ['变体', 'macro-F1', 'F1保持率%', '参数量M', '磁盘MB', '压缩比', '延迟ms/条', '加速比']
    w = [max(14, len(c) + 2) for c in cols]
    print(''.join(c.ljust(w[i]) for i, c in enumerate(cols)))
    print('-' * sum(w))
    for row in rows:
        print(''.join(str(row[c]).ljust(w[i]) for i, c in enumerate(cols)))
    print('=' * 92)

    out = f'../compress_report{("_" + args.tag) if args.tag else ""}.csv'
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'对比表已保存: {out}')

    # ---------- 结论提示 ----------
    best = max(rows[1:], key=lambda r: r['macro-F1']) if len(rows) > 1 else None
    if best:
        print(f"\n✅ 压缩后指标最高的是「{best['变体']}」："
              f"macro-F1={best['macro-F1']}（保持 {best['F1保持率%']}%），"
              f"体积 {best['压缩比']}x，速度 {best['加速比']}x")
    print('\n📌 报告里要写清的三句话：')
    print('  1. 动态量化只量化 Linear 层权重，Embedding/LayerNorm 不量化（敏感且收益小）')
    print('  2. int8 的加速只在 CPU 上成立；GPU 上没有 int8 内核，量化后反而更慢')
    print('  3. 压缩比用「落盘后的文件字节数」算，别用 parameters().numel()（会漏掉量化张量）')


if __name__ == '__main__':
    main()
