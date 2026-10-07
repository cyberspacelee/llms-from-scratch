import torch
from torch.utils.checkpoint import checkpoint

from llms_from_scratch.distributed.planner import (
    GB,
    LLAMA3_70B,
    DeepSeekV3Config,
    DensePlan,
    achieved_flops_per_gpu,
    activation_bytes_per_layer,
    ib_dispatch_bytes,
    job_mtbf,
    moe_state_bytes_per_gpu,
    optimal_checkpoint_interval,
    pipeline_activation_factor,
    saved_activation_bytes,
    training_days,
    training_flops_per_token,
)
from llms_from_scratch.transformer.model import Block, GPTConfig, rope_frequencies


def test_korthikanti_table():
    s, b, h, a, t = 2048, 1, 12288, 96, 8  # GPT-3 175B：5as/h = 80
    sbh = s * b * h
    assert activation_bytes_per_layer(s, b, h, a) == sbh * (34 + 80)
    assert activation_bytes_per_layer(s, b, h, a, t) == sbh * (10 + 24 / t + 80 / t)
    assert activation_bytes_per_layer(s, b, h, a, t, sequence_parallel=True) == sbh * (34 + 80) / t
    assert activation_bytes_per_layer(s, b, h, a, t, True, "selective") == sbh * 34 / t
    assert activation_bytes_per_layer(s, b, h, a, t, False, "selective") == sbh * (10 + 24 / t)
    assert activation_bytes_per_layer(s, b, h, a, t, False, "full") == 2 * sbh


def test_pipeline_activation_factor():
    assert pipeline_activation_factor(8, 32) == 1.0  # 第一个 stage 存 p 个微批 × L/p 层 = L 层
    assert pipeline_activation_factor(8, 4) == 0.5  # 微批不够 p 个
    assert abs(pipeline_activation_factor(4, 16, interleave=2) - (1 + 3 / 8)) < 1e-12


def test_model_sizes():
    assert round(LLAMA3_70B.params() / 1e9, 1) == 70.6
    v3 = DeepSeekV3Config()
    assert round(v3.total_params() / 1e9) == 671  # 技术报告：671B 总参数
    assert round(v3.active_params() / 1e9) in (37, 38)  # 技术报告：每 token 激活 37B


def test_dense_plan_70b():
    plan = DensePlan(LLAMA3_70B, gpus=4096, tp=8, pp=4, seq=8192, micro_batch=1,
                     global_batch_tokens=16 * 2**20)
    summary = plan.summary()
    assert summary["dp"] == 128 and summary["m"] == 16
    assert abs(summary["bubble"] - 3 / 19) < 1e-12
    # 每卡 70.55B/32 ≈ 2.2B 参数：BF16 权重与梯度 4 字节/参数，加 12 字节/参数的优化器状态 /128
    local = LLAMA3_70B.params() / 32
    assert abs(plan.state_bytes() - (4 * local + 12 * local / 128)) < 1
    # TP + SP + 选择性重计算：每层 34·sbh/t，第一个 stage 相当于存了全部 80 层
    assert abs(plan.activation_bytes() - 80 * 34 * 8192 * 8192 / 8) < 1
    assert summary["total_gb"] < 40
    # 不做流水线：状态占 35 GB，加激活约 58 GB，仍放得进 80 GB，但余量小
    no_pp = DensePlan(LLAMA3_70B, gpus=4096, tp=8, pp=1, seq=8192, micro_batch=1,
                      global_batch_tokens=16 * 2**20)
    assert 55 < no_pp.summary()["total_gb"] < 60


def test_throughput_estimates():
    flops = training_flops_per_token(LLAMA3_70B.params(), 80, 8192, 8192)
    assert abs(flops / (6 * LLAMA3_70B.params()) - 1.076) < 0.01  # 注意力约多 7.6%
    days = training_days(15e12, flops, gpus=4096, peak_flops=989e12, mfu=0.4)
    assert 45 < days < 52
    # DeepSeek-V3：14.8T token，2.664M H800 GPU 小时 → 只算 6N 时每卡约 348 TFLOP/s
    v3 = DeepSeekV3Config()
    per_gpu = achieved_flops_per_gpu(14.8e12, 6 * v3.active_params(), 2.664e6)
    assert 340e12 < per_gpu < 355e12


def test_moe_plan_deepseek_v3():
    v3 = DeepSeekV3Config()
    state = moe_state_bytes_per_gpu(v3, pp=16, ep=64, dp=128)
    assert 0.6e9 < state["expert_params"] < 0.7e9  # 每卡约 4 个专家 × 3.6 层
    assert state["total"] / GB < 14
    # 8 个专家副本若都跨节点要发 8 次；节点受限后最多发 4 次
    assert ib_dispatch_bytes(4096, 4, 7168, 1) == ib_dispatch_bytes(4096, 8, 7168, 1) / 2


def test_checkpoint_saves_only_inputs():
    torch.manual_seed(0)
    cfg = GPTConfig(d_model=64, n_heads=4, d_ff=256, context_length=128)
    block = Block(cfg, 0)
    x = torch.randn(2, 128, 64, requires_grad=True)
    freqs = rope_frequencies(cfg.head_dim, 128)
    plain = saved_activation_bytes(lambda: block(x, freqs))
    grad_plain = x.grad.clone()
    x.grad = None
    ckpt = saved_activation_bytes(lambda: checkpoint(block, x, freqs, use_reentrant=False))
    # 重计算只保存块的输入（x 与旋转因子），反向时重跑一遍前向
    assert ckpt == x.numel() * 4 + freqs.numel() * 8
    assert plain > 20 * ckpt
    torch.testing.assert_close(x.grad, grad_plain)


def test_checkpoint_interval():
    # Llama 3：54 天 419 次意外中断 → 平均约 3.1 小时一次
    mtbf = 54 * 24 / 419
    assert 3.0 < mtbf < 3.2
    tau = optimal_checkpoint_interval(60, mtbf * 3600)
    assert 1100 < tau < 1200  # 写检查点停顿 1 分钟时，约每 19 分钟存一次
    # 1.6 万张卡、每张卡平均 5 万小时无故障 → 作业约 3 小时出一次故障
    assert round(job_mtbf(50_000, 16_384), 1) == 3.1
