
import torch
import torch.nn as nn

from attention import(
    GroupedQueryAttention,
    compute_rope_params,
)


LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "emb_dim": 2048,                 # Embedding dimension
    "n_heads": 32,                   # Number of attention heads
    "n_layers": 16,                  # Number of layers
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "dtype": torch.float32,

    "use_norm": True,
    "dropout": 0.1,
    "return_attn": True,

    # "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

LLAMA32_CONFIG_3B = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "emb_dim": 3072,                 # Embedding dimension
    "n_heads": 24,                   # Number of attention heads
    "n_layers": 28,                  # Number of layers
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}


class QuirkBox(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusuable utilities
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    # 一个完整的 Transformer 模型的前向传播 forward() 函数,包含输入嵌入、位置编码、注意力 mask、多层 TransformerBlock 的堆叠，以及最终输出 logits
    # 输入 token id → 词嵌入 → 加位置编码 → 通过多层 Transformer Block → 输出预测 logits（+ 可选注意力矩阵）
    def forward(self, in_idx, return_attn=False):    # in_idx: 输入的 token 索引（形状可能是 [batch_size, seq_len]）
        tok_embeds = self.tok_emb(in_idx)   # 把 token id 转换成 词向量嵌入，形状变为 [batch, seq_len, emb_dim], self.tok_emb 是嵌入层（nn.Embedding）
        x = tok_embeds

        # 构建一个 上三角的注意力 mask，用于 因果注意力（causal attention）：防止每个 token 看到未来的 token
        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        attn_maps = []

        for block in self.trf_blocks:   # self.trf_blocks 是多个堆叠的 TransformerBlock（通常是 nn.ModuleList）。对输入依次应用多个 Block，进行深层特征提取
            if return_attn:
                x, attn = block(x, mask, self.cos, self.sin, return_attn = True)
                attn_maps.append(attn)
            else:
                x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)  # 对最后输出做一次归一化（RMSNorm 或 LayerNorm）
        logits = self.out_head(x.to(self.cfg["dtype"])) # 把 transformer 的输出投影到词表大小的维度，用于计算概率；out_head 是一个线性层（类似 nn.Linear(emb_dim, vocab_size)）；.to(self.cfg["dtype"]) 可能是为了统一精度（比如 float16 或 float32）
        return (logits, attn_maps) if return_attn else logits


class TransformerBlock(nn.Module):  # 继承自 nn.Module，表明它是一个可训练的神经网络模块
    def __init__(self, cfg):
        super().__init__()  # 接收一个 cfg 配置字典，控制超参数和功能启用
        self.use_norm = cfg.get("use_norm", True)   # 默认启用归一化
        self.dropout = nn.Dropout(cfg.get("dropout", 0.1))  # 防止过拟合的 dropout 层，默认 0.1
        # 实例化一个 分组查询注意力层（GroupedQueryAttention）
        self.att = GroupedQueryAttention(   
            d_in=cfg["emb_dim"],    # 参数如嵌入维度、头数、键值分组数、数据类型都来自 cfg
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)  # 一个标准的两层前馈神经网络模块
        if self.use_norm:   # 如果启用归一化，则使用 RMSNorm 替代常规的 LayerNorm
            self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
            self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin, return_attn=False):   # x: 输入嵌入 (形状通常是 [batch_size, seq_len, emb_dim])；mask: 注意力 mask，用于屏蔽不相关位置；cos, sin: 旋转位置编码参数（可能用于 RoPE）；return_attn: 是否返回注意力权重
        print("🥰💕💖"*10)#("🧪 GQA forward(), return_attn =", return_attn)
        # 🔷 第一段：多头注意力机制（Self-Attention）（处理 token 间依赖）
        # Pre-Norm Transformer: Shortcut connection for attention block
        # 归一化 → 子模块（Attention 或 FF）→ Dropout → 残差连接
        shortcut = x    # 保存当前的输入 x，这是为了 稍后用于残差连接（Residual Connection）
        if self.use_norm:
            x = self.norm1(x)   
        x, attn_weights = self.att(x, mask, cos, sin, return_attn=return_attn)   # 传入注意力层，获得新的表示 x 和（可选的）注意力权重
        #x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = self.dropout(x)     # 在输出后应用 Dropout，用于防止过拟合。Dropout 是训练时“随机丢弃神经元”的技术，用于防止过拟合
        x = x + shortcut  # Add the original input back # 残差连接（Residual Connection）。将前馈层的输出加上原始输入 x（保存在 shortcut 中）。这样做可以让梯度更容易反向传播，避免梯度消失问题。

        # 🔷 第二段：前馈神经网络 FeedForward（FFN）（处理每个 token 的表示）
        # Shortcut connection for feed-forward block
        shortcut = x
        if self.use_norm:
            x = self.norm2(x) # 对输入 x 做第二次归一化（这里使用的是 RMSNorm）。这是标准 Transformer 的前馈层前使用归一化的方式（也就是 Pre-Norm）。
        x = self.ff(x)  # 通过前馈网络处理输入
        x=self.dropout(x)
        x = x + shortcut  # Add the original input back

        return x if not return_attn else (x, attn_weights)  # 如果 return_attn=True，就返回输出和注意力权重


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)



# def text_to_token_ids(text, tokenizer):
#     encoded = tokenizer.encode(text)
#     encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
#     return encoded_tensor
def text_to_token_ids(text, tokenizer, allowed_special=set()):
    if hasattr(tokenizer, 'encode'):
        return torch.tensor(tokenizer.encode(text, allowed_special=allowed_special), dtype=torch.long)[None, :]
    else:
        raise ValueError("Tokenizer does not support .encode()")


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None, return_attn=False):

    attn_maps = []
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        print("🧪 Step A: start generate()")
        idx_cond = idx[:, -context_size:]
        print("🧪 Step B: shape", idx_cond.shape)

        with torch.no_grad():
            if return_attn:
                logits, attn_maps = model(idx_cond, return_attn=True)
            else:
                logits = model(idx_cond)
        print("🧪 Step C: logits done", logits.shape)

        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    if return_attn:
        print("🔍 Attention maps collected:", len(attn_maps))
        print("🔍 Layer 0 shape:", attn_maps[0].shape)  # (batch, heads, tokens, tokens)
        return (idx, attn_maps)
    else:
        return idx