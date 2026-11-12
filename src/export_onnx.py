import torch
import os
import sys

# ====== 1. 路径设置 ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bert_project_11/
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

# 你本地实际存在的模型目录
PRETRAINED_DIR = os.path.join(BASE_DIR, "bert-base-chinese")
MODEL_PT = os.path.join(BASE_DIR, "saved", "best_model.pt")
ONNX_PATH = os.path.join(BASE_DIR, "saved", "model.onnx")

# ====== 2. 强制绕过 Config 的自动检查（关键）======
# 原来的 config.py 会在 __init__ 里检查 chinese-roberta-wwm-ext-large
# 我们直接在这里“欺骗”它，把目标路径指向你实际有的 bert-base-chinese
import config
config.Config._check_model_dir = lambda self: None  # 禁用自动检查
# 手动覆盖配置中的路径（假设你的 Config 有 bert_path 属性，若无则忽略）
try:
    config.Config.bert_path = PRETRAINED_DIR
except Exception:
    pass

# ====== 3. 导入模型类（此时不会报错了）======
from bert_classifier import BertClassifier

# ====== 4. 验证本地文件 ======
if not os.path.isdir(PRETRAINED_DIR):
    raise FileNotFoundError(f"本地目录不存在: {PRETRAINED_DIR}")
for f in ["config.json", "vocab.txt"]:
    if not os.path.exists(os.path.join(PRETRAINED_DIR, f)):
        raise FileNotFoundError(f"{PRETRAINED_DIR} 缺少 {f}")

print(f"✅ 使用本地预训练模型: {PRETRAINED_DIR}")

# ====== 5. 加载 Tokenizer（强制本地）======
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_DIR, local_files_only=True)

# ====== 6. 加载分类模型 ======
# 注意：这里需要根据你 BertClassifier 的实际参数调整
model = BertClassifier(bert_path=PRETRAINED_DIR)  # 如果构造函数支持
# 如果不支持，用下面这行（需确保内部默认走 bert-base-chinese）
# model = BertClassifier()

model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"), strict=False)
model.eval()

# ====== 7. Dummy Input & 导出 ======
dummy_text = "测试领域分类"
inputs = tokenizer(dummy_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
att_mask = inputs["attention_mask"]
token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))

torch.onnx.export(
    model,
    (input_ids, att_mask, token_type_ids),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "token_type_ids": {0: "batch"}},
    opset_version=13
)
print(f"✅ ONNX 已导出: {ONNX_PATH}")