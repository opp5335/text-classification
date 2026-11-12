# -*- coding: utf-8 -*-
"""merge_classes.py —— 按映射表把近义类合并，重建 train/test/class.txt

【为什么需要它】
  embed_cluster_probe.py 的「类级质心混淆」会告诉我们哪几对类在表示空间里撞车。
  如果撞得厉害，说明是「类定义重叠」而不是模型不行 —— **合并**才是有效动作。

【用法】必须 cd 到 src
    cd bert_project_11/src
    python merge_classes.py --example-map                      # 生成示例映射文件
    python merge_classes.py --map merge_map.json --dry-run     # 先看报告，不写文件
    python merge_classes.py --map merge_map.json --out ../data_merged

【产出】--out 目录下：class.txt / train.csv / test.csv / merge_report.txt
  ⚠️ **不覆盖原 data/**。确认满意后手动替换（可回退）：
      mv ../data ../data_18cls_backup && mv ../data_merged ../data

【格式说明】新 CSV 写 **不带 BOM** 的 utf-8。
  原 train.csv/test.csv 带 BOM，会让首行多一个 \\ufeff（utils.py 用 encoding='utf-8' 读不掉），
  这里顺手修掉。
"""
import argparse
import csv
import json
import os
from collections import Counter

EXAMPLE_MAP = {
    "_说明": "键 = 原类名（必须是 class.txt 里有的），值 = 新类名。未列出的类保持原名。",
    "_示例": "下面三行只是示范，请按 embed_cluster_probe.py 的质心混淆结果改成你自己的方案",
    "政府组织": "公共管理与社会组织",
    "国际组织": "公共管理与社会组织",
    "教育": "公共管理与社会组织",
}


def read_pairs(path):
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.reader(f):
            if len(r) < 2:
                continue
            t, l = r[0].strip(), r[1].strip()
            if t and l:
                rows.append((t, l))
    return rows


def write_pairs(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        for t, l in rows:
            w.writerow([t, l])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', dest='map_path', default='merge_map.json')
    ap.add_argument('--data-dir', default='../data')
    ap.add_argument('--out', default='../data_merged')
    ap.add_argument('--dry-run', action='store_true', help='只打印报告，不写文件')
    ap.add_argument('--example-map', action='store_true', help='生成示例映射文件后退出')
    a = ap.parse_args()

    if a.example_map:
        if os.path.exists(a.map_path):
            raise SystemExit(f'{a.map_path} 已存在，请先改名或删掉再生成')
        with open(a.map_path, 'w', encoding='utf-8') as f:
            json.dump(EXAMPLE_MAP, f, ensure_ascii=False, indent=2)
        print(f'已写出示例映射：{a.map_path}')
        print('  → 键 = 原类名，值 = 新类名；不需要合并的类不用写。')
        print('  → 按 embed_cluster_probe.py 的质心混淆结果改成你自己的方案。')
        return

    class_path = os.path.join(a.data_dir, 'class.txt')
    old_classes = [x.strip() for x in open(class_path, encoding='utf-8') if x.strip()]
    mapping = json.load(open(a.map_path, encoding='utf-8'))
    mapping = {k: v for k, v in mapping.items() if not k.startswith('_')}

    bad = [k for k in mapping if k not in old_classes]
    if bad:
        raise SystemExit(f'映射里出现 class.txt 中不存在的类名：{bad}\n'
                         f'class.txt 实际内容：{old_classes}')

    new_classes = []
    for c in old_classes:
        nc = mapping.get(c, c)
        if nc not in new_classes:
            new_classes.append(nc)

    def conv(rows):
        out, dropped = [], 0
        for t, l in rows:
            if l not in old_classes:
                dropped += 1
                continue
            out.append((t, mapping.get(l, l)))
        return out, dropped

    tr, tr_drop = conv(read_pairs(os.path.join(a.data_dir, 'train.csv')))
    te, te_drop = conv(read_pairs(os.path.join(a.data_dir, 'test.csv')))

    L = [f'原类数 {len(old_classes)}  →  新类数 {len(new_classes)}', '']
    L.append('合并映射：')
    if mapping:
        for k, v in mapping.items():
            L.append(f'  {k:<16} →  {v}')
    else:
        L.append('  （空 —— 没有合并任何类）')
    L.append('')

    ratio_warn = []
    for tag, rows, dropped in [('train', tr, tr_drop), ('test', te, te_drop)]:
        cnt = Counter(l for _, l in rows)
        sizes = [cnt.get(c, 0) for c in new_classes]
        L.append(f'[{tag}] {len(rows)} 条（丢弃无法识别标签 {dropped} 条），覆盖 {len(cnt)} 类')
        for c, n in zip(new_classes, sizes):
            L.append(f'  {c:<16}{n:>7}')
        mn, mx = min(sizes), max(sizes)
        L.append(f'  最大/最小 = {mx}/{mn} = {mx / max(mn, 1):.2f}×')
        if mx / max(mn, 1) > 3:
            ratio_warn.append(tag)
        L.append('')

    L.append('⚠️ 注意：')
    L.append('  1) 合并后类分布不再均衡，macro-F1 与合并前**不可直接比较**'
             '（分母的类数变了）。')
    L.append('     要跨口径比较，请同时看 micro-F1 / accuracy。')
    L.append('  2) 类定义变更属于「数据契约」，需要与需求方确认。')

    report = '\n'.join(L)
    print(report)

    if a.dry_run:
        print('\n[dry-run] 未写任何文件。')
        return

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'class.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(new_classes) + '\n')
    write_pairs(os.path.join(a.out, 'train.csv'), tr)
    write_pairs(os.path.join(a.out, 'test.csv'), te)
    with open(os.path.join(a.out, 'merge_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    print(f'\n已写出到 {a.out}/：class.txt, train.csv, test.csv, merge_report.txt')
    print('确认无误后替换（可回退）：')
    print(f'  mv {a.data_dir} {a.data_dir}_backup && mv {a.out} {a.data_dir}')
    if ratio_warn:
        print(f'⚠️ {"/".join(ratio_warn)} 集类间不均衡已超过 3×，训练时可考虑加权或分层采样。')


if __name__ == '__main__':
    main()
