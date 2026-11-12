# -*- coding: utf-8 -*-
"""配置（bert_project_11，18 类领域分类）

【模型演进】
    bert-base-chinese (0.6121)
      → chinese-bert-wwm-ext        + Focal/LLRD/EMA → 0.7689
      → chinese-roberta-wwm-ext-large  （当前默认，24 层 / hidden 1024 / 325M）

【为什么换 large 而不是别的】
    chinese-roberta-wwm-ext-large 与 bert-base-chinese **架构一致（BertModel）、词表一致（21128）**，
    所以 tokenizer 不用换、BertConfig/BertModel 不用换，只改这一行路径即可，风险最低。
    large 版在中文分类任务上通常比 base 高 2~4 个 macro-F1 点。

【怎么临时换回 / 换别的模型】（无需改本文件）
    1) 命令行：python train.py --bert-path ../chinese-bert-wwm-ext --tag wwm_base
    2) 环境变量：export BERT_PATH=../chinese-bert-wwm-ext
    3) 改下面的 DEFAULT_BERT_PATH

【③ 超参随模型一起调整】
    lr 2e-5 → 1e-5      large 模型更容易被大 lr 带崩，标准做法是减半
    batch 64 → 32       large 参数量 3 倍（325M），显存占用约 3 倍
    epochs 4 保持       base 版第 4 轮仍在涨（0.7649→0.7689），large 收敛更快，4 轮够
    dropout 0.2 保持    （要生效必须把 bert_config 传进 from_pretrained，bert_classifier 已改）
"""
import os

import torch
from transformers import BertConfig, BertTokenizer

# 默认模型路径（可被环境变量 BERT_PATH / 命令行 --bert-path 覆盖）
DEFAULT_BERT_PATH = os.environ.get('BERT_PATH', '../chinese-roberta-wwm-ext-large')

# 几种常用中文模型，方便命令行直接引用（仅作记录，代码不强制）
MODEL_PRESETS = {
    'base': '../bert-base-chinese',
    'wwm': '../chinese-bert-wwm-ext',
    'large': '../chinese-roberta-wwm-ext-large',
}


class Config():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---------------- 数据路径 ----------------
        self.data_path = '../data_synth'
        self.train_path = self.data_path + '/train.csv'
        self.test_path = self.data_path + '/test.csv'
        self.class_path = self.data_path + '/class.txt'
        self.class_list = [line.strip() for line in open(self.class_path, encoding='utf-8')]

        # ---------------- 模型 ----------------
        self.model_name = 'bert'
        self.bert_path = DEFAULT_BERT_PATH
        self._check_model_dir()
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path)

        # 加大 dropout，抑制过拟合（需 bert_classifier 传入 config 才生效）
        self.bert_config.hidden_dropout_prob = 0.2
        self.bert_config.attention_probs_dropout_prob = 0.2

        self.model_save_path = '../saved'
        self.hidden_size = self.bert_config.hidden_size      # large = 1024，base = 768

        # ---------------- 训练超参 ----------------
        self.num_epochs = 4
        self.batch_size = 32              # large 用 32；显存富余可 --batch 48/64
        self.learning_rate = 1e-5         # large 用 1e-5（base 是 2e-5）
        self.weight_decay = 0.05
        self.pad_size = 128
        self.num_classes = len(self.class_list)

        # ---------------- ② Focal Loss / LLRD / EMA 的默认开关 ----------------
        self.use_focal = True
        self.focal_gamma = 2.0
        self.use_llrd = True
        self.layer_decay = 0.9
        self.use_ema = True
        self.ema_decay = 0.999

        # ---------------- 模型压缩相关 ----------------
        self.quantized_bert_sd_save_path = '../saved/bert_quantized_state_dict.pt'
        self.quantized_bert_save_path = '../saved/bert_quantized.pt'

    def _check_model_dir(self):
        """模型目录缺失时给出可直接复制的下载命令，而不是抛一堆 transformers 栈"""
        need = ['config.json', 'vocab.txt']
        missing = [f for f in need if not os.path.exists(os.path.join(self.bert_path, f))]
        if not missing:
            return
        msg = [
            '',
            '=' * 72,
            f'❌ 模型目录 {os.path.abspath(self.bert_path)} 缺少文件: {missing}',
            '',
            '请先下载权重（云端执行）：',
            '  export HF_ENDPOINT=https://hf-mirror.com',
            '  export HF_HUB_DISABLE_XET=1',
            f'  python download_model.py --repo hfl/chinese-roberta-wwm-ext-large '
            f'--dir {self.bert_path}',
            '',
            '或先退回已有的 base/wwm 模型：',
            '  python train.py --bert-path ../chinese-bert-wwm-ext --batch 64 --lr 2e-5',
            '=' * 72,
            '',
        ]
        raise FileNotFoundError('\n'.join(msg))


if __name__ == '__main__':
    conf = Config()
    print(f'任务分类列表为:\n{conf.class_list}')
    print(f'bert_path      :\t{conf.bert_path}')
    print(f'模型的词嵌入维度:\t{conf.hidden_size}')
    print(f'层数           :\t{conf.bert_config.num_hidden_layers}')
    print(f'dropout        :\t{conf.bert_config.hidden_dropout_prob}')
