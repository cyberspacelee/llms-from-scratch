export type TrackId = 'math' | 'principles' | 'systems'

export type Track = {
  id: TrackId
  /** One letter that prefixes chapter codes: M3, P2, S4. */
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
    summary: '注意力为什么需要位置，sin/cos 与 RoPE 怎样写进点积，KV cache 为什么与整段计算相等。',
  },
  {
    id: 'systems',
    letter: 'S',
    label: '推理系统',
    summary: '给一次生成记账，看清单卡上限，再沿 nano-vLLM 源码走到调度、分页缓存与集群。',
  },
]

export function getTrack(id: string): Track {
  const track = tracks.find((item) => item.id === id)
  if (!track) throw new Error(`Unknown track "${id}"`)
  return track
}
