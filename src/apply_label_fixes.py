# -*- coding: utf-8 -*-
"""apply_label_fixes.py —— 把人工复核结果应用到训练集，产出干净的训练数据

【配合谁用】
  embed_cluster_probe.py 导出的 ../embed_out/suspect_labels.csv。
  那张表有 8 列：行号 / 文本 / 原标签 / 更像的标签 / 质心间隔 / 最近质心类 / 处理 / 新标签
  你在 Excel 里逐条判断，只填**最后两列**：
    · 「处理」列：留空 = 保留原样；填「删」= 丢弃这条；填「改」= 改成「新标签」列的值
    · 「新标签」列：仅当处理填「改」时需要，必须是 class.txt 里存在的类名

【用法】必须 cd 到 src
    cd bert_project_11/src
    python apply_label_fixes.py --review ../embed_out/suspect_labels.csv \
                                --data-dir ../data --out ../data_clean
    python apply_label_fixes.py --review ... --dry-run     # 只看统计不改文件

【产出】--out 下：train.csv（清洗后）、class.txt（原样复制）、fix_report.txt
  ⚠️ **只改 train.csv，绝不碰 test.csv** —— test 是评估基准，动它等于作弊。
  ⚠️ 复核时请**先只看「文本」再决定**，别被「更像的标签」那列牵着走（那是模型的猜测，
     模型已经过拟合了训练噪声，它的判断本身可能就被噪声带偏）。
"""
import argparse
import csv
import os
from collections import Counter

DEL_KW = {'删', '删除', '去', '丢弃', 'del', 'delete', 'd', 'x'}
FIX_KW = {'改', '修改', '更正', 'fix', 'c', 'edit'}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', default='../embed_out/suspect_labels.csv')
    ap.add_argument('--data-dir', default='../data')
    ap.add_argument('--out', default='../data_clean')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    classes = [x.strip() for x in open(os.path.join(a.data_dir, 'class.txt'),
                                       encoding='utf-8') if x.strip()]
    cset = set(classes)

    train = read_pairs(os.path.join(a.data_dir, 'train.csv'))
    print(f'原训练集 {len(train)} 条')

    fixes, dels, blank, bad_rows = {}, set(), 0, []
    with open(a.review, encoding='utf-8-sig', newline='') as f:
        for i, row in enumerate(csv.DictReader(f)):
            idx_raw = (row.get('行号') or '').strip()
            if not idx_raw:
                continue
            try:
                idx = int(float(idx_raw))
            except ValueError:
                bad_rows.append((i + 2, f'行号不是数字: {idx_raw!r}'))
                continue
            if not (0 <= idx < len(train)):
                bad_rows.append((i + 2, f'行号越界: {idx}'))
                continue

            act = (row.get('处理') or '').strip().lower()
            newl = (row.get('新标签') or '').strip()

            if act in DEL_KW:
                dels.add(idx)
            elif act in FIX_KW:
                if newl not in cset:
                    bad_rows.append((i + 2, f'新标签 {newl!r} 不在 class.txt 里'))
                    continue
                fixes[idx] = newl
            elif act == '':
                blank += 1
            else:
                bad_rows.append((i + 2, f'「处理」取值无法识别: {act!r}（用 删/改 或留空）'))

    if bad_rows:
        print(f'\n⚠️ 有 {len(bad_rows)} 行有问题，已跳过：')
        for ln, msg in bad_rows[:20]:
            print(f'  第 {ln} 行：{msg}')
        if len(bad_rows) > 20:
            print(f'  ... 还有 {len(bad_rows) - 20} 行')

    conflict = sorted(set(fixes) & dels)
    if conflict:
        print(f'\n⚠️ {len(conflict)} 行同时被标了 删 和 改，按「删」处理：{conflict[:10]}')

    before = Counter(l for _, l in train)
    out = []
    for i, (t, l) in enumerate(train):
        if i in dels:
            continue
        out.append((t, fixes.get(i, l)))
    after = Counter(l for _, l in out)

    L = []
    L.append(f'复核清单共处理：留空(保留) {blank} 条，改标签 {len(fixes)} 条，删除 {len(dels)} 条')
    L.append(f'训练集 {len(train)} → {len(out)} 条（净减 {len(train) - len(out)}）')
    L.append('')
    L.append('各类变化（原 → 新）：')
    for c in classes:
        b, af = before.get(c, 0), after.get(c, 0)
        L.append(f'  {c:<16}{b:>7} → {af:>7}   {af - b:+d}')
    report = '\n'.join(L)
    print('\n' + report)

    if a.dry_run:
        print('\n[dry-run] 未写任何文件。')
        return

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'train.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        for t, l in out:
            w.writerow([t, l])
    with open(os.path.join(a.out, 'class.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(classes) + '\n')
    with open(os.path.join(a.out, 'fix_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    print(f'\n已写出到 {a.out}/：train.csv, class.txt, fix_report.txt')
    print('⚠️ test.csv 未被改动（评估基准不能动）。替换时记得把它一起带过去：')
    print(f'  mkdir -p {a.out} && cp {a.data_dir}/test.csv {a.out}/test.csv')
    print(f'  mv {a.data_dir} {a.data_dir}_backup && mv {a.out} {a.data_dir}')


if __name__ == '__main__':
    main()
