import pytest

from llms_from_scratch.gpu.hardware import A100, B200, H100, V100, occupancy


def test_ridge_points():
    # 312 TFLOPS / 2.039 TB/s ≈ 153 FLOP/byte；989.4 / 3.35 ≈ 295
    assert A100.ridge_point("bf16") == pytest.approx(153.0, rel=0.01)
    assert H100.ridge_point("bf16") == pytest.approx(295.3, rel=0.01)
    assert B200.ridge_point("bf16") == pytest.approx(281.25, rel=0.01)


def test_compute_grows_faster_than_bandwidth():
    compute = H100.tflops["fp16"] / V100.tflops["fp16"]
    bandwidth = H100.hbm_tb_s / V100.hbm_tb_s
    assert compute > 2 * bandwidth


def test_occupancy_full():
    occ = occupancy(threads_per_block=256, regs_per_thread=32, smem_per_block=0, spec=H100)
    assert occ.active_warps == 64 and occ.occupancy == 1.0


def test_occupancy_register_limited():
    # 128 寄存器/线程 → 每 warp 4096 个，65536 / 4096 = 16 个 warp
    occ = occupancy(threads_per_block=128, regs_per_thread=128, smem_per_block=0, spec=H100)
    assert occ.limiter == "registers" and occ.active_warps == 16 and occ.occupancy == 0.25


def test_occupancy_smem_limited():
    # 每 block 48 KB + 1 KB 保留，228 KB 只放得下 4 个 block
    occ = occupancy(threads_per_block=128, regs_per_thread=32, smem_per_block=48 * 1024, spec=H100)
    assert occ.limiter == "shared memory" and occ.blocks_per_sm == 4 and occ.active_warps == 16
