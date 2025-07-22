import torch

# 模型路径
MODEL_FILE = "llama3.2-1B-instruct.pth"
TOKENIZER_FILE = "tokenizer.model"

# 设备设置
DEVICE = torch.device("cpu")  # 你也可以改成手动控制 "cuda" / "mps"

# 最大上下文长度
MODEL_CONTEXT_LENGTH = 8192

# 生成设置
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.0
TOP_K = 1
