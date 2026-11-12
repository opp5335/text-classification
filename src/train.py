# -*- coding: utf-8 -*-
"""train.py —— BERT 文本分类训练（bert_project_11，18 类领域分类）

【验证集说明】
    本项目只有 train.csv / test.csv 两份（8:2 切分），没有独立 dev 集。
    这里按需求**直接拿 test 当验证集**，用于每轮评估、选最优权重。
    ⚠️ 代价：最终 test 指标是「反复挑过的最优轮」的分数，偏乐观约 0.5~1 点。

【数据】config.data_path 指向 ../data_synth
    train 50153 条 / test 12537 条，18 类，表头 text,label（无 BOM）。
    已做近重复感知切分：同一「改写组」不会跨 train/test，跨集近重复 0 条。

【运行】必须 cd 到 src（config.py 用相对路径 '../data_synth'）
    # 默认 = chinese-roberta-wwm-ext-large（需先下载权重）
    python train.py --tag roberta_large

    # 退回 wwm-base（对比用，原 0.7689 那条线）
    python train.py --bert-path ../chinese-bert-wwm-ext --batch 64 --lr 2e-5 --tag wwm_v2

    # 显存不够：小 batch + 梯度累积（有效 batch 仍是 32）
    python train.py --batch 16 --accum 2 --tag large_accum2

    # 本机快速验证流程（只跑 3 步 + 只评估 1 批）
    python train.py --bert-path ../bert-base-chinese --epochs 1 --max-steps 3 --eval-limit 1

【耗时参考】train 50153 条
    base  @batch64 → 783 步/epoch，4090 约 40 s/epoch
    large @batch32 → 1566 步/epoch，4090 约 2~3 分钟/epoch（开 AMP 更快）
"""
import argparse
import itertools
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from tqdm import tqdm


def _prescan_bert_path():
    """在 import config 之前把 --bert-path 写进环境变量（Config 构造时就要读模型目录）"""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--bert-path' and i + 1 < len(argv):
            os.environ['BERT_PATH'] = argv[i + 1]
        elif a.startswith('--bert-path='):
            os.environ['BERT_PATH'] = a.split('=', 1)[1]


_prescan_bert_path()

from bert_classifier import BertClassifier  # noqa: E402
from config import Config                   # noqa: E402
from utils import build_dataloader          # noqa: E402

conf = Config()

# ---------------- 训练策略 ----------------
WARMUP_RATIO = 0.1        # 前 10% 步数线性升温
LABEL_SMOOTH = 0.1        # 本数据标签噪声约 5%~10%，这一项正对症


def amp_autocast(enabled):
    """混合精度 autocast（兼容 torch 新旧写法）"""
    try:                                   # torch >= 2.4
        from torch.amp import autocast
        return autocast('cuda', enabled=enabled)
    except (ImportError, TypeError):       # 老版本
        return torch.cuda.amp.autocast(enabled=enabled)


def amp_scaler(enabled):
    """混合精度 GradScaler（兼容 torch 新旧写法）"""
    try:                                   # torch >= 2.4
        from torch.amp import GradScaler
        return GradScaler('cuda', enabled=enabled)
    except (ImportError, TypeError):       # 老版本
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ==================== ② Focal Loss ====================
class FocalLoss(nn.Module):
    """Focal Loss（支持与 label_smoothing 叠加）

    标准 CE 对所有样本一视同仁；Focal 用 (1-p_t)^gamma 压低"已经分对"的样本权重，
    让梯度集中在难样本上 —— 正对弱类（政府组织 / 房地产 / 文体娱乐）。
    """

    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=-1)                     # (B, C)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,) 真实类的 log p
        pt = logpt.exp().clamp(min=1e-7, max=1.0)
        if self.ls > 0:
            ce = -(1.0 - self.ls) * logpt - self.ls * logp.mean(dim=-1)
        else:
            ce = -logpt
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


# ==================== ② 分层学习率 LLRD ====================
def make_param_groups(model, base_lr, weight_decay, layer_decay=0.9):
    """Layer-wise Learning Rate Decay

    越底层 lr 越小（乘 layer_decay 的幂），分类头用完整 lr；
    bias / LayerNorm 排除在 weight_decay 之外（BERT 微调标准实践）。
    """
    n_layers = model.bert.config.num_hidden_layers          # base 12 / large 24
    L = n_layers + 1                                        # +1 表示 embeddings 层

    def layer_id(name):
        if name.startswith('bert.embeddings'):
            return 0
        m = re.match(r'bert\.encoder\.layer\.(\d+)\.', name)
        if m:
            return int(m.group(1)) + 1
        return L + 1                                        # pooler / fc 等头部

    groups, no_decay_kw = {}, ('bias', 'LayerNorm.weight')
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if any(k in name for k in no_decay_kw) else weight_decay
        lid = layer_id(name)
        lr = base_lr if lid > L else base_lr * (layer_decay ** (L - lid))
        key = (lid, wd)
        groups.setdefault(key, {'params': [], 'lr': lr, 'weight_decay': wd})
        groups[key]['params'].append(p)

    out = sorted(groups.values(), key=lambda g: g['lr'])
    for g in out:
        g['lr'] = round(g['lr'], 12)
    return out


# ==================== ② EMA ====================
class EMA:
    """权重指数滑动平均：训练中维护一份"平滑版"权重，评估/保存时用它，泛化更稳"""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self):
        self.backup = [p.detach().clone() for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.copy_(s)

    @torch.no_grad()
    def restore(self):
        if self.backup is None:
            return
        for b, p in zip(self.backup, self.params):
            p.copy_(b)
        self.backup = None


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=0,
                   help='训练轮数；不传时取 config.num_epochs（当前 4）')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=0,
                   help='>0 时每轮最多跑这么多步（本机验证流程用，正式训练保持 0）')
    p.add_argument('--eval-limit', type=int, default=0,
                   help='>0 时评估只用前 N 批（调试用，正式训练保持 0 = 全量）')
    p.add_argument('--tag', type=str, default='', help='权重文件名后缀，多组实验区分用')

    # ---- 模型 ----
    p.add_argument('--bert-path', type=str, default='',
                   help='覆盖 config.bert_path，如 ../chinese-bert-wwm-ext')

    # ---- ③ 超参覆盖（不传则取 config.py 的值）----
    p.add_argument('--lr', type=float, default=0, help='学习率；不传取 config.learning_rate')
    p.add_argument('--batch', type=int, default=0, help='batch_size；不传取 config.batch_size')
    p.add_argument('--wd', type=float, default=-1, help='weight_decay；不传取 config.weight_decay')

    # ---- 显存相关 ----
    p.add_argument('--accum', type=int, default=1,
                   help='梯度累积步数；有效 batch = batch * accum（显存不够时用）')
    p.add_argument('--no-fp16', action='store_true',
                   help='关闭 AMP 混合精度（默认在 GPU 上开启；若 loss 出 nan 用它回退）')

    # ---- ② 三个技巧的开关（消融实验用，默认跟随 config）----
    p.add_argument('--no-focal', action='store_true', help='关掉 Focal Loss，退回普通 CE')
    p.add_argument('--no-llrd', action='store_true', help='关掉分层学习率，全模型同一 lr')
    p.add_argument('--no-ema', action='store_true', help='关掉 EMA（显存紧张时可关）')
    p.add_argument('--gamma', type=float, default=conf.focal_gamma,
                   help='Focal Loss 的 gamma（默认 2.0；0 即退化为普通 CE）')
    p.add_argument('--ema-decay', type=float, default=conf.ema_decay, help='EMA 衰减率')
    p.add_argument('--layer-decay', type=float, default=conf.layer_decay,
                   help='分层学习率衰减系数（默认 0.9；1.0 即不衰减）')
    return p.parse_args()


def evaluate(model, loader, limit=0, use_amp=False):
    """在验证集上评估，返回 (micro_f1, macro_f1, acc, y_true, y_pred)"""
    model.eval()
    preds, gts = [], []
    it = loader if not limit else itertools.islice(loader, limit)
    with torch.no_grad():
        for input_ids, attention_mask, labels in it:
            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            with amp_autocast(use_amp):
                logits = model(input_ids, attention_mask)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            gts.extend(labels.tolist())
    return (f1_score(gts, preds, average='micro'),
            f1_score(gts, preds, average='macro'),
            accuracy_score(gts, preds),
            gts, preds)


def model2train():
    args = parse_args()
    if args.epochs <= 0:
        args.epochs = conf.num_epochs

    # ③ 超参覆盖：命令行 > config.py
    if args.lr > 0:
        conf.learning_rate = args.lr
    if args.batch > 0:
        conf.batch_size = args.batch
    if args.wd >= 0:
        conf.weight_decay = args.wd

    # ② 三个技巧开关
    use_focal = conf.use_focal and not args.no_focal
    use_llrd = conf.use_llrd and not args.no_llrd
    use_ema = conf.use_ema and not args.no_ema

    # 混合精度：仅 CUDA 下生效；CPU 自动关闭
    use_amp = torch.cuda.is_available() and not args.no_fp16

    set_seed(args.seed)

    print('=' * 72)
    print(f'device={conf.device}  seed={args.seed}  epochs={args.epochs}  '
          f'lr={conf.learning_rate}  batch={conf.batch_size} x accum={args.accum}  '
          f'max_len={conf.pad_size}  num_classes={conf.num_classes}')
    print(f'技巧: Focal(gamma={args.gamma})={use_focal}  '
          f'LLRD(decay={args.layer_decay})={use_llrd}  EMA({args.ema_decay})={use_ema}  '
          f'AMP={use_amp}')
    if conf.device.type == 'cuda':
        print('GPU：', torch.cuda.get_device_name(0),
              f'  显存 {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print('⚠️ 没有检测到 GPU，本机训练会非常慢，建议只跑 --max-steps 验证流程')
    print('=' * 72)

    # ★ 只有 train / test 两份：拿 test 当验证集
    train_loader, dev_loader = build_dataloader()
    print(f'train : {len(train_loader.dataset)} 条 / {len(train_loader)} 批')
    print(f'dev(用 test): {len(dev_loader.dataset)} 条 / {len(dev_loader)} 批')

    model = BertClassifier().to(conf.device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'模型: {conf.bert_path}')
    print(f'      参数量 {n_param / 1e6:.1f}M   层数 {conf.bert_config.num_hidden_layers}   '
          f'hidden {conf.hidden_size}   vocab {conf.bert_config.vocab_size}')

    # ② 分层学习率 or 统一学习率
    if use_llrd:
        param_groups = make_param_groups(
            model, conf.learning_rate, conf.weight_decay, args.layer_decay)
        optimizer = AdamW(param_groups, lr=conf.learning_rate)
        print(f'分层学习率: {len(param_groups)} 组, '
              f'底层 lr={param_groups[0]["lr"]:.2e} → 头部 lr={param_groups[-1]["lr"]:.2e}')
    else:
        optimizer = AdamW(model.parameters(), lr=conf.learning_rate,
                          weight_decay=conf.weight_decay)

    # ② Focal Loss or 普通 CE（两者都保留 label_smoothing）
    if use_focal:
        loss_fn = FocalLoss(gamma=args.gamma, label_smoothing=LABEL_SMOOTH)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # ② EMA（在 optimizer 建好、训练开始前初始化）
    ema = EMA(model, decay=args.ema_decay) if use_ema else None

    steps_per_epoch = len(train_loader)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    # 梯度累积后，参数更新次数会变少，warmup / 总步数按「更新次数」算
    updates_per_epoch = max(1, math.ceil(steps_per_epoch / args.accum))
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    def lr_lambda(step):
        if step < warmup_steps:                     # 线性升温
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = amp_scaler(use_amp)
    print(f'total_updates={total_steps}  warmup={warmup_steps}  '
          f'有效 batch={conf.batch_size * args.accum}')

    os.makedirs(conf.model_save_path, exist_ok=True)
    best_f1 = 0.0
    name = f'best_model{("_" + args.tag) if args.tag else ""}.pt'
    best_path = os.path.join(conf.model_save_path, name)

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t_epoch = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for idx, (input_ids, attention_mask, labels) in enumerate(pbar):
            if args.max_steps and idx >= args.max_steps:
                break

            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)

            with amp_autocast(use_amp):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)

            scaler.scale(loss / args.accum).backward()
            # 累积够 accum 步（或本轮最后一批）才真正更新一次参数
            if (idx + 1) % args.accum == 0 or (idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if ema is not None:
                    ema.update()      # ② 每次参数更新后刷新滑动平均

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss / (idx + 1):.4f}',
                'acc': f'{correct / total:.4f}',
                # 分层学习率时各组 lr 不同，显示最大的那个（= base_lr × 调度系数）
                'lr': f"{max(g['lr'] for g in optimizer.param_groups):.2e}",
            })

        used = min(time.time() - t_epoch, 1e9)
        if ema is not None:
            ema.apply()           # ② 用平滑权重评估
        micro, macro, acc, gts, preds = evaluate(
            model, dev_loader, args.eval_limit, use_amp)
        if ema is not None:
            ema.restore()         # ② 还原训练权重，下一轮继续训练
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
                if conf.device.type == 'cuda' else 0.0)
        suffix = f'（仅前 {args.eval_limit} 批）' if args.eval_limit else ''
        print(f'\n[Epoch {epoch + 1}/{args.epochs}] 用时 {used / 60:.1f} 分钟  '
              f'dev micro-F1={micro:.4f}  macro-F1={macro:.4f}  acc={acc:.4f}  '
              f'峰值显存={peak:.1f}GB{suffix}')

        # ★ 用 macro-F1 选最优（18 类均衡时 micro-F1 恒等于 accuracy，看不出类间差异）
        if macro > best_f1:
            best_f1 = macro
            if ema is not None:
                ema.apply()
            torch.save(model.state_dict(), best_path)
            if ema is not None:
                ema.restore()
            print(f'  >>> 刷新最优（macro-F1={best_f1:.4f}），已保存 {best_path}')
        else:
            print(f'  --- 未刷新（历史最优 macro-F1={best_f1:.4f}）')

    # ---- 收尾：加载最优权重，打印逐类报告 ----
    print('\n' + '=' * 72)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=conf.device))
        _, _, _, gts, preds = evaluate(model, dev_loader, args.eval_limit, use_amp)
        print(f'最优权重 {best_path}  最优 macro-F1 = {best_f1:.4f}')
        print('\n逐类指标（仅验证集；没有真 dev 时它就是 test，偏乐观）：')
        # 显式给 labels，避免调试模式（只评估前几批）时某些类没出现导致类别数对不上
        print(classification_report(gts, preds, target_names=conf.class_list,
                                    labels=list(range(conf.num_classes)),
                                    digits=4, zero_division=0))
    print('=' * 72)


if __name__ == '__main__':
    model2train()
