"""从超参数核算 Decoder-only 模型的总参数量与每 token 激活参数量。

只计入矩阵参数（嵌入、注意力投影、FFN/专家、路由器），忽略 RMSNorm 增益、偏置等
数量级小于 0.1% 的项。所有超参数取自各模型的论文或官方 config.json，见第四部分第一章。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int  # 稠密 FFN 的隐藏宽度（MoE 模型中稠密层使用）
    gated_ffn: bool = True  # SwiGLU 等门控 FFN 有 3 个矩阵，经典 FFN 有 2 个
    tie_embeddings: bool = False
    # MoE：n_experts == 0 表示稠密模型
    n_experts: int = 0
    top_k: int = 0
    n_shared_experts: int = 0
    d_expert: int = 0
    n_dense_layers: int = 0  # 开头若干层保持稠密 FFN（DeepSeek-V3 为 3）
    moe_every: int = 1  # 每隔几层放一个 MoE 层（LLaMA 4 Maverick 为 2）
    # MLA：kv_lora_rank > 0 时启用（DeepSeek-V2/V3、Kimi K2）
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    rope_head_dim: int = 0
    v_head_dim: int = 0

    def attention_params(self) -> int:
        d, h = self.d_model, self.n_heads
        if self.kv_lora_rank:
            qk_dim = self.head_dim + self.rope_head_dim  # 内容维 + 解耦 RoPE 维
            q = d * self.q_lora_rank + self.q_lora_rank * h * qk_dim if self.q_lora_rank else d * h * qk_dim
            kv_down = d * (self.kv_lora_rank + self.rope_head_dim)
            kv_up = self.kv_lora_rank * h * (self.head_dim + self.v_head_dim)
            return q + kv_down + kv_up + h * self.v_head_dim * d
        q_o = 2 * d * h * self.head_dim
        k_v = 2 * d * self.n_kv_heads * self.head_dim
        return q_o + k_v

    def ffn_params(self, width: int) -> int:
        return (3 if self.gated_ffn else 2) * self.d_model * width

    def is_moe_layer(self, layer: int) -> bool:
        if not self.n_experts or layer < self.n_dense_layers:
            return False
        return (layer - self.n_dense_layers) % self.moe_every == self.moe_every - 1

    def layer_params(self, layer: int) -> tuple[int, int]:
        """返回第 layer 层的 (总参数, 每 token 激活参数)。"""
        attn = self.attention_params()
        if not self.is_moe_layer(layer):
            ffn = self.ffn_params(self.d_ff)
            return attn + ffn, attn + ffn
        expert = self.ffn_params(self.d_expert)
        router = self.d_model * self.n_experts
        shared = self.n_shared_experts * expert
        total = attn + router + shared + self.n_experts * expert
        active = attn + router + shared + self.top_k * expert
        return total, active

    def count(self) -> tuple[int, int]:
        """返回 (总参数, 激活参数)。激活参数计入输入嵌入与输出头各一份。"""
        embed = self.vocab_size * self.d_model * (1 if self.tie_embeddings else 2)
        layers = [self.layer_params(i) for i in range(self.n_layers)]
        return embed + sum(t for t, _ in layers), embed + sum(a for _, a in layers)


# region specs
LLAMA2_7B = ModelSpec("LLaMA-2-7B", 32000, 4096, 32, 32, 32, 128, 11008)
LLAMA3_8B = ModelSpec("LLaMA-3-8B", 128256, 4096, 32, 32, 8, 128, 14336)
MIXTRAL_8X7B = ModelSpec("Mixtral-8x7B", 32000, 4096, 32, 32, 8, 128, 14336,
                         n_experts=8, top_k=2, d_expert=14336)
QWEN3_235B = ModelSpec("Qwen3-235B-A22B", 151936, 4096, 94, 64, 4, 128, 12288,
                       n_experts=128, top_k=8, d_expert=1536)
GPT_OSS_120B = ModelSpec("gpt-oss-120b", 201088, 2880, 36, 64, 8, 64, 2880,
                         n_experts=128, top_k=4, d_expert=2880)
DEEPSEEK_V3 = ModelSpec("DeepSeek-V3", 129280, 7168, 61, 128, 128, 128, 18432,
                        n_experts=256, top_k=8, n_shared_experts=1, d_expert=2048,
                        n_dense_layers=3, q_lora_rank=1536, kv_lora_rank=512,
                        rope_head_dim=64, v_head_dim=128)
# endregion


def kv_cache_bytes_per_token(spec: ModelSpec, bytes_per_elem: int = 2) -> int:
    """每个 token 在全部层中占用的 KV Cache 字节数（MLA 只存潜向量与 RoPE 键）。"""
    if spec.kv_lora_rank:
        per_layer = spec.kv_lora_rank + spec.rope_head_dim
    else:
        per_layer = 2 * spec.n_kv_heads * spec.head_dim
    return spec.n_layers * per_layer * bytes_per_elem
