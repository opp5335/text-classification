# -*- coding: utf-8 -*-
"""verify.py —— 上云后 30 秒链路验证（在 src/ 目录下运行）

确认：GPU 可用 → 类别数 → 数据条数/批次数 → 模型前向 shape。
任何一步报错，先停下来看报错，别硬跑训练。
"""
import time

t0 = time.time()

import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU', torch.cuda.get_device_name(0),
          '%.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))

from config import Config
conf = Config()
print('classes', len(conf.class_list), '| device', conf.device,
      '| pad_size', conf.pad_size, '| batch', conf.batch_size)

from utils import build_dataloader
tr, dv = build_dataloader()
print('train', len(tr.dataset), '| dev', len(dv.dataset), '| batches/epoch', len(tr))

from bert_classifier import BertClassifier
m = BertClassifier().to(conf.device)
ids, mask, lab = next(iter(tr))
logits = m(ids.to(conf.device), mask.to(conf.device))
print('logits', tuple(logits.shape), '| expect', (conf.batch_size, conf.num_classes))

print('OK  %.1fs' % (time.time() - t0))
