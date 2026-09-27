import { useState } from 'react'
import { headCost, locate, misplacedRows, stageLayout, type HeadStage } from '../../lib/multihead-model'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

const stages: { id: HeadStage; label: string; code: string }[] = [
  { id: 'linear', label: '1 · 线性投影输出', code: 'q = self.q(x)' },
  { id: 'split', label: '2 · 拆出头轴', code: 'q.reshape(B, T, n_q, d_h)' },
  { id: 'heads', label: '3 · 头轴换到前面', code: '.transpose(1, 2)' },
  { id: 'merged', label: '4 · 注意力后合并', code: 'y.transpose(1, 2).contiguous().reshape(B, T, d)' },
  { id: 'wrong', label: '错误 · 直接 reshape', code: 'q.reshape(B, n_q, T, d_h)' },
]

const headFill = [
  'fill-accent-soft stroke-accent',
  'fill-accent2-soft stroke-accent2',
  'fill-info-soft stroke-info',
  'fill-sunken stroke-rule-strong',
]

export default function MultiHeadLab() {
  const [length, setLength] = useState(4)
  const [heads, setHeads] = useState(2)
  const [headWidth, setHeadWidth] = useState(2)
  const [stage, setStage] = useState<HeadStage>('heads')
  const [token, setToken] = useState(1)
  const [feature, setFeature] = useState(2)
  const config = { length, heads, headWidth }
  const d = heads * headWidth
  const t = Math.min(token, length - 1), c = Math.min(feature, d - 1)
  const layout = stageLayout(stage, config)
  const place = locate(stage, config, t, c)
  const cost = headCost(config)
  const perHead = stage === 'heads' || stage === 'wrong'
  const cell = Math.min(40, Math.floor(380 / (perHead ? d + (heads - 1) * 0.8 : d + (stage === 'split' ? (heads - 1) * 0.4 : 0))))
  const gap = perHead ? cell * 0.8 : stage === 'split' ? cell * 0.4 : 0
  const top = 44
  const contentWidth = perHead ? heads * headWidth * cell + (heads - 1) * gap : d * cell + (stage === 'split' ? (heads - 1) * gap : 0)
  const width = Math.max(contentWidth + 40, 430)
  const height = top + length * cell + 58

  // Draw every source element (tt, cc) at its coordinate in the current stage.
  const cells = []
  for (let tt = 0; tt < length; tt++) for (let cc = 0; cc < d; cc++) {
    const p = locate(stage, config, tt, cc)
    let x: number, y: number
    if (perHead) {
      x = 20 + p.coord[0] * (headWidth * cell + gap) + p.coord[2] * cell
      y = top + p.coord[1] * cell
    } else {
      const h = Math.floor(cc / headWidth)
      x = 20 + cc * cell + (stage === 'split' ? h * gap : 0)
      y = top + tt * cell
    }
    const selected = tt === t && cc === c
    cells.push(<g key={`${tt}-${cc}`}>
      <rect x={x} y={y} width={cell - 2} height={cell - 2} rx="3"
        className={`${headFill[Math.floor(cc / headWidth)]} ${selected ? 'stroke-3' : ''}`} />
      <text x={x + cell / 2 - 1} y={y + cell / 2 + 3} textAnchor="middle" className={`${pen.mono} text-[10px]`}>{tt}·{cc}</text>
    </g>)
  }
  const labels = perHead
    ? Array.from({ length: heads }, (_, h) => <text key={h} x={20 + h * (headWidth * cell + gap) + (headWidth * cell) / 2} y={top - 10} textAnchor="middle" className={stage === 'wrong' ? pen.textB : pen.textA}>头 {h}</text>)
    : Array.from({ length: heads }, (_, h) => <text key={h} x={20 + (h * headWidth + headWidth / 2) * cell + (stage === 'split' ? h * gap : 0)} y={top - 10} textAnchor="middle" className={pen.muted}>头 {h} 的列</text>)
  const misplaced = perHead ? misplacedRows(stage, config) : 0
  const shape = `(B, ${layout.shape.join(', ')})`

  return <LabFrame title="多头注意力里的轴变换" hint="每格写“来源 token·特征列”，颜色是它原本属于的头">
    <Controls>
      <Range label="序列长 T" value={length} min={2} max={6} onChange={setLength} />
      <Range label="头数 n_q" value={heads} min={1} max={4} onChange={setHeads} />
      <Range label="头维度 d_h" value={headWidth} min={1} max={3} onChange={setHeadWidth} />
      <label className="text-sm">步骤
        <select value={stage} onChange={event => setStage(event.target.value as HeadStage)} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">
          {stages.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
      </label>
      <Range label="跟踪 token t" value={t} min={0} max={length - 1} onChange={setToken} />
      <Range label="跟踪特征列 c" value={c} min={0} max={d - 1} onChange={setFeature} />
    </Controls>
    <svg viewBox={`0 0 ${width} ${height}`} className={`${pen.canvas} max-w-150`} role="img" aria-label={`${stages.find(item => item.id === stage)!.label}：形状 ${shape}，元素 ${t}·${c} 位于坐标 ${place.coord.join(',')}`}>
      <text x="20" y="16" className={pen.mono}>{stages.find(item => item.id === stage)!.code}</text>
      {labels}
      {cells}
      <text x="20" y={top + length * cell + 22}>{perHead ? '每个面板是一头：行 = 查询/键位置，列 = 头内特征' : '行 = token 位置，列 = 隐藏特征'}</text>
      <text x="20" y={top + length * cell + 44} className={stage === 'wrong' && misplaced ? pen.textB : pen.muted}>
        {stage === 'wrong' ? (misplaced ? `${misplaced}/${heads * length} 行的内容不是“该行 token 的该头特征”：形状对了，含义错了` : '只有一头时两种写法恰好相同') : `形状 ${shape} · 样本内 stride ${layout.stride.join(', ')} · ${layout.contiguous ? '连续' : '非连续：只换了 stride，没搬数据'}`}
      </text>
    </svg>
    <Readout>
      d = n_q·d_h = {d} · 元素 {t}·{c} → 坐标 ({place.coord.join(', ')})，存储偏移 {place.offset}，读作 token {place.token} / 头 {place.head}
      {' · '}每条样本分数元素 n_q·T² = {cost.scoreElements} · Q/K/V/O 权重 4d² = {cost.projectionWeights}，与头数无关
    </Readout>
  </LabFrame>
}
