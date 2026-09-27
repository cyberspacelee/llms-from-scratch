export type TrackId = 'math' | 'principles' | 'training' | 'gpu' | 'systems' | 'advanced'

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
    summary: '从统计、分布、矩阵和微积分出发，手算并验证一次完整的神经网络训练。',
  },
  {
    id: 'principles',
    letter: 'P',
    label: '模型原理',
    summary: '从文本与预测目标搭起完整 Decoder，推导位置、现代结构、生成与缓存。',
  },
  {
    id: 'training', letter: 'T', label: '训练与评估',
    summary: '准备数据，完整训练与恢复小语言模型，再做评估、指令微调和低秩适配。',
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
