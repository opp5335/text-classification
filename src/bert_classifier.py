# -*- coding: utf-8 -*-
"""BERT + FC 文本分类模型（18 类领域分类）

配套：config.py（超参/路径）、utils.py（数据加载）
"""
import torch.nn as nn
from transformers import BertModel

from config import Config

conf = Config()


class BertClassifier(nn.Module):
    """BERT 编码 + [CLS] 池化向量接一层线性分类头"""

    def __init__(self):
        super().__init__()
        # ① 模型由 conf.bert_path 决定（现为 chinese-bert-wwm-ext）
        # ⚠️ 必须显式传入 conf.bert_config，否则 config.py 里改的
        #    hidden_dropout_prob / attention_probs_dropout_prob **不会生效**
        #    （from_pretrained 默认用模型自带的 config 覆盖）
        self.bert = BertModel.from_pretrained(conf.bert_path, config=conf.bert_config)
        # 768 -> 18，输出 logits（不过 softmax，交给 CrossEntropyLoss）
        self.fc = nn.Linear(conf.hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        # input_ids / attention_mask: (batch, 128)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output          # (batch, 768)
        return self.fc(pooled)              # (batch, 18)


if __name__ == '__main__':
    from utils import build_dataloader

    train_loader, _ = build_dataloader()
    inputs, mask, labels = next(iter(train_loader))
    model = BertClassifier()
    logits = model(inputs, mask)
    print('logits shape:', tuple(logits.shape))     # (64, 18)
    print('labels[:8]  :', labels[:8].tolist())
