# -*- coding: utf-8 -*-
"""predict.py —— BERT 文本分类推理（bert_project_11，18 类领域分类）

【为什么要有它】
    train.py 训完只存 state_dict，没有推理入口。本脚本把「加载 → 单例复用 → 预测」
    这条链路固化下来，避免每次写一坨重复代码。

【和课堂参考代码的三处关键差异（都是会出错/会误判的地方）】
    ① tokenizer 必须带 truncation + max_length=conf.pad_size
       参考代码只写了 return_tensors='pt'。文本一旦超过 512 字，BERT 的位置编码
       直接越界报错；而训练时用的是
           padding='max_length', truncation=True, max_length=128
       推理必须同口径，否则输入分布和训练不一致。
    ② 补了 softmax 置信度 + top-k
       只输出 argmax 类名，无法判断「模型有多确定」，也没法人工复核边界样本。
    ③ 权重路径可配 + 自动探测
       参考代码硬编码 '../saved/best_model.pt'。本项目那个文件其实是**烟测留下的
       未训练权重**（macro-F1 0.119），真模型是 best_model_rawtwo_e15.pt（0.7896）。

【运行】必须先 cd 到 src（config.py 用相对路径 '../data_synth'）
    python predict.py --text "同步A股首秀：港股缩量回调"
    python predict.py --text "..." --text "..."          # 多条
    python predict.py --file texts.txt                   # 每行一条
    python predict.py --input-csv in.csv --output-csv out.csv
    python predict.py --interactive                      # 交互式
    python predict.py                                    # 跑内置示例

    # 指定权重 / 换模型 / 强制 CPU
    python predict.py --ckpt ../saved/best_model_rawtwo_e15.pt --text "..."
    python predict.py --bert-path ../bert-base-chinese --device cpu --text "..."

【本机注意】
    本机无独立显卡，1.3GB 的 large 版在 CPU 上约几百 ms/条，能用但慢。
    要快就先在云端跑 compress.py 出 int8 版（约 350MB，CPU 提速约 3x）。
"""
import argparse
import csv
import glob
import os
import sys

import torch

if hasattr(sys.stdout, 'reconfigure'):
    # Windows 控制台默认 GBK，遇到个别不可编码字符会直接抛 UnicodeEncodeError
    sys.stdout.reconfigure(errors='replace')


def _prescan_bert_path():
    """在 import config 之前把 --bert-path 写进环境变量

    Config() 构造时就要读模型目录，import 之后再来改就来不及了。
    """
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--bert-path' and i + 1 < len(argv):
            os.environ['BERT_PATH'] = argv[i + 1]
        elif a.startswith('--bert-path='):
            os.environ['BERT_PATH'] = a.split('=', 1)[1]


_prescan_bert_path()

from bert_classifier import BertClassifier  # noqa: E402
from config import Config                   # noqa: E402

conf = Config()
CLASS_LIST = conf.class_list

# 权重自动探测顺序（第一个存在的胜出）
CKPT_PREFERENCE = [
    '../saved/best_model_rawtwo_e15.pt',   # 当前最好：0.7896
    '../saved/best_model.pt',
]

# ---------------- 单例模式：模型只加载一次 ----------------
_model = None
_loaded_ckpt = None


def _load_state_dict(path, device):
    """读权重，并兼容 DataParallel 保存的 'module.' 前缀"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'找不到权重文件：{os.path.abspath(path)}\n'
            f'  可用权重：{_list_ckpts()}\n'
            f'  用 --ckpt <路径> 指定'
        )
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and any(str(k).startswith('module.') for k in sd):
        sd = {str(k).replace('module.', '', 1): v for k, v in sd.items()}
    return sd


def _check_shape_compat(model, sd, ckpt):
    """权重与模型维度不匹配时给出人话提示

    否则 torch 会抛几百行 'size mismatch for bert.encoder.layer.N...'，
    根本看不出真正的原因只是「large 权重配了 base 模型」。
    """
    m_sd = model.state_dict()
    key = 'bert.embeddings.word_embeddings.weight'
    if key in sd and key in m_sd and tuple(sd[key].shape) != tuple(m_sd[key].shape):
        ck_dim = sd[key].shape[-1]
        m_dim = m_sd[key].shape[-1]
        raise RuntimeError(
            f'\n❌ 权重与模型维度不匹配\n'
            f'   权重 {os.path.basename(ckpt)} 的隐层维度 = {ck_dim}'
            f'，当前模型 = {m_dim}\n'
            f'   （1024 = roberta-large / 768 = base 或 wwm-ext）\n'
            f'   --ckpt 与 --bert-path 必须配套，例如：\n'
            f'     权重 1024 维 → --bert-path ../chinese-roberta-wwm-ext-large\n'
            f'     权重  768 维 → --bert-path ../chinese-bert-wwm-ext\n'
        )


def _list_ckpts():
    cands = sorted(glob.glob('../saved/*.pt'))
    return [os.path.basename(p) for p in cands] or ['(saved 目录为空)']


def resolve_ckpt():
    """按 CKPT_PREFERENCE 挑一个存在的权重；都没有就取 saved 下最新的"""
    for p in CKPT_PREFERENCE:
        if os.path.exists(p):
            if os.path.basename(p) == 'best_model.pt':
                print('⚠️  正在使用 ../saved/best_model.pt —— 该文件是早期烟测留下的'
                      '未训练权重（macro-F1 仅 0.119），如非本意请用 --ckpt 指定真模型。')
            return p
    cands = sorted(glob.glob('../saved/*.pt'), key=os.path.getmtime, reverse=True)
    if cands:
        print(f'⚠️  未找到默认权重，自动选用 saved 下最新的：{cands[0]}')
        return cands[0]
    raise FileNotFoundError(
        f'../saved 下没有任何 .pt 权重。\n'
        f'  本地：{os.path.abspath("../saved")}\n'
        f'  云端训练产物在 /root/autodl-tmp/bert_project_11/saved/，需先下载回来。'
    )


def load_model(ckpt_path=None, device=None):
    """加载模型（单例）。第一次调用真正加载，之后直接返回缓存。

    global _model 是单例的关键：模块级变量，跨函数调用保持。
    """
    global _model, _loaded_ckpt

    if _model is None:
        ckpt = ckpt_path or resolve_ckpt()
        dev = torch.device(device) if device else conf.device
        print(f'正在加载模型...\n  权重: {ckpt}\n  设备: {dev}')
        model = BertClassifier()
        sd = _load_state_dict(ckpt, dev)
        _check_shape_compat(model, sd, ckpt)
        model.load_state_dict(sd)
        model.to(dev)
        model.eval()
        _model = model
        _loaded_ckpt = ckpt
        print(f'模型就绪：层数 {conf.bert_config.num_hidden_layers} / '
              f'维度 {conf.hidden_size} / {conf.num_classes} 类')
    return _model


def _encode(texts):
    """tokenize，口径与训练时的 collate_fn 完全一致"""
    enc = conf.tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=conf.pad_size,
        return_tensors='pt',
    )
    return enc['input_ids'], enc['attention_mask']


@torch.no_grad()
def predict_batch(texts, topk=3, batch_size=32, ckpt=None, device=None):
    """批量预测

    Args:
        texts: List[str] 文本列表
    Returns:
        List[dict]，每项 {text, pred_class, pred_id, confidence, topk:[{class,prob}]}
    """
    if isinstance(texts, str):
        texts = [texts]
    model = load_model(ckpt, device)
    dev = next(model.parameters()).device
    k = max(1, min(topk, len(CLASS_LIST)))

    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        input_ids, attention_mask = _encode(chunk)
        logits = model(input_ids.to(dev), attention_mask.to(dev))
        probs = torch.softmax(logits.float(), dim=-1)
        top_p, top_i = probs.topk(k, dim=-1)

        for j, text in enumerate(chunk):
            idxs = top_i[j].tolist()
            probs_j = top_p[j].tolist()
            results.append({
                'text': text,
                'pred_class': CLASS_LIST[idxs[0]],
                'pred_id': idxs[0],
                'confidence': probs_j[0],
                'topk': [{'class': CLASS_LIST[c], 'prob': p}
                         for c, p in zip(idxs, probs_j)],
            })
    return results


def predict(text, topk=3, ckpt=None, device=None):
    """单条预测，返回 dict"""
    return predict_batch([text], topk=topk, ckpt=ckpt, device=device)[0]


def predict_data(data, topk=3):
    """兼容课堂参考代码的接口：入参 {'text': ...}，就地把结果写回并返回"""
    r = predict(data['text'], topk=topk)
    data['text'] = r['text']
    data['pred_class'] = r['pred_class']
    data['confidence'] = r['confidence']
    data['topk'] = r['topk']
    return data


def _print_result(r, topk):
    print(f"\n文本: {r['text']}")
    print(f"预测: {r['pred_class']}   (置信度 {r['confidence']:.4f})")
    if topk > 1:
        tail = '  '.join(f"{t['class']} {t['prob']:.4f}" for t in r['topk'])
        print(f"Top{topk}: {tail}")


def run_csv(in_path, out_path, topk=3, batch_size=32, ckpt=None, device=None):
    """读 CSV（认 text 列，没有就用第一列），输出带预测结果的 CSV"""
    with open(in_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        text_col = 'text' if 'text' in fieldnames else (fieldnames[0] if fieldnames else None)
        if text_col is None:
            raise ValueError(f'{in_path} 没有可用的列')
        rows = list(reader)

    texts = [(row.get(text_col) or '').strip() for row in rows]
    print(f'共 {len(texts)} 条，文本列 = {text_col}')
    results = predict_batch(texts, topk=topk, batch_size=batch_size,
                            ckpt=ckpt, device=device)

    out_fields = list(fieldnames) + ['pred_class', 'confidence']
    out_path = out_path or 'predict_out.csv'
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row, r in zip(rows, results):
            row = dict(row)
            row['pred_class'] = r['pred_class']
            row['confidence'] = round(r['confidence'], 4)
            writer.writerow(row)
    print(f'已写出: {os.path.abspath(out_path)}')


DEMO = [
    '同步A股首秀：港股缩量回调',
    '国家统计局发布前三季度国民经济运行情况',
    '某房地产企业发布上半年财报，销售额同比下降',
    '新款新能源汽车上市，续航里程提升至七百公里',
]


def main():
    p = argparse.ArgumentParser(description='BERT 文本分类推理')
    p.add_argument('--ckpt', default='', help='权重路径（默认自动探测 saved 下的最好版本）')
    p.add_argument('--bert-path', default='', help='覆盖 config.bert_path（需与权重匹配）')
    p.add_argument('--device', default='', help='cpu / cuda，默认自动')
    p.add_argument('--text', action='append', default=[], help='待预测文本，可重复多次')
    p.add_argument('--file', default='', help='文本文件，每行一条')
    p.add_argument('--input-csv', default='', help='输入 CSV（认 text 列）')
    p.add_argument('--output-csv', default='', help='输出 CSV 路径')
    p.add_argument('--interactive', action='store_true', help='交互式逐条输入')
    p.add_argument('--topk', type=int, default=3, help='输出前 k 个候选（默认 3）')
    p.add_argument('--batch', type=int, default=32, help='批量推理的 batch（默认 32）')
    args = p.parse_args()

    dev = args.device or None
    ckpt = args.ckpt or None

    # ---- 模式 1：CSV 批量 ----
    if args.input_csv:
        run_csv(args.input_csv, args.output_csv, topk=args.topk,
                batch_size=args.batch, ckpt=ckpt, device=dev)
        return

    # ---- 模式 2：文件 / 命令行 / 内置示例 ----
    texts = list(args.text)
    if args.file:
        with open(args.file, 'r', encoding='utf-8-sig') as f:
            texts += [ln.strip() for ln in f if ln.strip()]
    if not texts and not args.interactive:
        texts = DEMO
        print('（未指定输入，跑内置示例）')

    if texts:
        results = predict_batch(texts, topk=args.topk, batch_size=args.batch,
                                ckpt=ckpt, device=dev)
        for r in results:
            _print_result(r, args.topk)

    # ---- 模式 3：交互式 ----
    if args.interactive:
        print('\n交互式模式（输入文本回车预测，空行或 Ctrl+C 退出）')
        while True:
            try:
                s = input('\n>>> ').strip()
            except (EOFError, KeyboardInterrupt):
                print('\n退出。')
                break
            if not s:
                break
            _print_result(predict(s, topk=args.topk, ckpt=ckpt, device=dev), args.topk)


if __name__ == '__main__':
    main()
