export type TrackId = 'math' | 'frameworks' | 'principles' | 'training' | 'gpu' | 'systems' | 'advanced'

export type Track = {
  id: TrackId
  /** One letter that prefixes chapter codes: M3, P6, S4. */
  letter: string
  label: string
  summary: string
}

/** Reading order of the whole site. */
export const tracks: Track[] = [
  {
    id: 'math',
    letter: 'M',
    label: '数学基础',
    summary: '从随机事件、贝叶斯与信息量走到预测损失，再用矩阵、梯度和谱分解理解训练与数值稳定性。',
  },
  {
    id: 'frameworks', letter: 'F', label: '数组与框架',
    summary: '用 NumPy 与 PyTorch 对齐数组的轴、存储、梯度、模块，以及数据、更新与训练状态。',
  },
  {
    id: 'principles',
    letter: 'P',
    label: '模型原理',
    summary: '从文本搭起基础 Decoder，再组装 RoPE、GQA、RMSNorm 与 SwiGLU 的完整现代模型。',
  },
  {
    id: 'training', letter: 'T', label: '训练与评估',
    summary: '完成文本训练、磁盘恢复与适配，再研究缩放规律、数据工程与分布式训练。',
  },
  {
    id: 'gpu', letter: 'G', label: 'GPU 编程',
    summary: '把数组工作映射到线程、warp 与 block，理解访存、同步、分块和可信的性能测量。',
  },
  {
    id: 'systems',
    letter: 'S',
    label: '推理系统',
    summary: '给一次生成记账，看清单卡上限，再沿 nano-vLLM 源码走到调度、分页缓存与集群。',
  },
  {
    id: 'advanced', letter: 'A', label: '进阶模型',
    summary: '沿不同设计轴理解稀疏专家、压缩状态、偏好与推理训练，以及多模态输入。',
  },
]

export function getTrack(id: string): Track {
  const track = tracks.find((item) => item.id === id)
  if (!track) throw new Error(`Unknown track "${id}"`)
  return track
}
