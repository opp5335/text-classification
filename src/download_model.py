# -*- coding: utf-8 -*-
"""下载 HF 上的中文预训练模型到本地目录（本机 / AutoDL 云端都能用）

【为什么不用 from_pretrained 直接在线拉】
  1) AutoDL 上 huggingface.co 经常超时 → 走 hf-mirror.com 镜像
  2) hf-mirror 不支持 Xet 存储 → 必须 HF_HUB_DISABLE_XET=1，否则报 404
     （之前下 wwm-ext 时踩过：显示 100% 但文件 0 字节 / 404）
  3) 多线程下载大权重容易中断 → max_workers=1 更稳（可断点续传）

【用法】在 src 目录下执行（相对路径 ../xxx 才对得上 config.py）
    python download_model.py
    python download_model.py --repo hfl/chinese-roberta-wwm-ext-large \
                             --dir ../chinese-roberta-wwm-ext-large
    python download_model.py --repo hfl/chinese-bert-wwm-ext --dir ../chinese-bert-wwm-ext

【推荐中文模型】
    hfl/chinese-roberta-wwm-ext-large   24 层 / 1024 维 / 325M  ← 当前默认
    hfl/chinese-bert-wwm-ext            12 层 / 768 维 / 102M   ← 0.7689 那条线
    hfl/chinese-macbert-large           24 层 / 1024 维（纠错式预训练，也可试）
    bert-base-chinese                   12 层 / 768 维（基线）
"""
import argparse
import os
import sys

# ★ 必须在 import huggingface_hub 之前设置，否则不生效
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')

import huggingface_hub  # noqa: E402

print(f'huggingface_hub 版本: {huggingface_hub.__version__}')
print(f'HF_ENDPOINT = {os.environ["HF_ENDPOINT"]}')
print(f'HF_HUB_DISABLE_XET = {os.environ["HF_HUB_DISABLE_XET"]}')

# 至少要有的文件（缺权重=训练必炸）
MUST_HAVE = ['config.json', 'vocab.txt']


def snapshot(repo_id, local_dir):
    """兼容新旧版 huggingface_hub 的参数签名"""
    from huggingface_hub import snapshot_download
    kwargs = dict(
        repo_id=repo_id,
        max_workers=1,              # 单线程，断线重传更稳
        resume_download=True,
    )
    try:
        # 新版（>=0.23）：local_dir 直接下真文件，不再有 use_symlinks 参数
        return snapshot_download(local_dir=local_dir, **kwargs)
    except TypeError:
        # 旧版：需要显式关掉软链接，否则下到 ~/.cache 只留一堆链接
        return snapshot_download(local_dir=local_dir,
                                 local_dir_use_symlinks=False, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='hfl/chinese-roberta-wwm-ext-large',
                   help='HF 仓库 id')
    p.add_argument('--dir', default='../chinese-roberta-wwm-ext-large',
                   help='本地保存目录（相对 src ../ 即项目根，和 config.py 对齐）')
    p.add_argument('--force', action='store_true', help='已存在也重新下载')
    args = p.parse_args()

    target = os.path.abspath(args.dir)
    os.makedirs(target, exist_ok=True)

    if not args.force:
        got = os.listdir(target)
        # 已有 config.json + vocab.txt 且存在权重文件，就跳过
        weight = [f for f in got if f.startswith('pytorch_model') or f.startswith('model.safetensors')]
        if all(f in got for f in MUST_HAVE) and weight:
            print(f'✅ 已存在，跳过下载：{target}')
            print('   如需重下加 --force')
            return 0

    print(f'⬇️  {args.repo}  ->  {target}')
    path = snapshot(args.repo, target)

    # 校验
    files = sorted(os.listdir(path))
    print('\n下载完成，目录内容：')
    for f in files:
        fp = os.path.join(path, f)
        size = os.path.getsize(fp) / 1024 ** 2 if os.path.isfile(fp) else 0
        print(f'  {f:<32} {size:>9.1f} MB')

    missing = [f for f in MUST_HAVE if f not in files]
    weight = [f for f in files if f.startswith('pytorch_model')
              or f.startswith('model.safetensors')]
    if missing:
        print(f'\n❌ 缺少关键文件 {missing}，请检查网络后重跑（可加 --force）')
        return 1
    if not weight:
        print('\n❌ 没有权重文件（pytorch_model.bin / model.safetensors）。')
        print('   hf-mirror 的 Xet 404 会表现为"下完但没权重"，'
              '确认 HF_HUB_DISABLE_XET=1 后重跑')
        return 1

    # 轻量自检：只读 config.json，不 import torch/transformers
    # （无卡模式只有 2GB 内存，import torch 有 OOM 风险，这一步刻意保持零依赖）
    try:
        import json
        cfg = json.load(open(os.path.join(path, 'config.json'), encoding='utf-8'))
        print(f"\n模型结构：architectures={cfg.get('architectures')}  "
              f"hidden={cfg.get('hidden_size')}  "
              f"层数={cfg.get('num_hidden_layers')}  "
              f"vocab={cfg.get('vocab_size')}")
        if cfg.get('num_hidden_layers') == 12:
            print('⚠️ 层数是 12 → 下到的是 base 不是 large，检查 --repo 参数')
    except Exception as e:
        print(f'（读 config.json 失败：{e}）')

    print(f'\n✅ 模型就绪：{path}')
    print('   下一步：cd src && python train.py --tag roberta_large')
    return 0


if __name__ == '__main__':
    sys.exit(main())
