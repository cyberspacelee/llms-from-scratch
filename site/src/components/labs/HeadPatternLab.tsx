import { useState } from 'react'
import { firstHead, headWeights, previousHead, tokens, type MaskMode } from '../../lib/head-pattern-model'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

const heads = [
  { name: '头 0 · 读前一个位置', head: previousHead },
  { name: '头 1 · 读开头 BOS', head: firstHead },
]

export default function HeadPatternLab() {
  const [mode, setMode] = useState<MaskMode>('causal')
  const [scale, setScale] = useState(3)
  const [query, setQuery] = useState(2)
  const weights = heads.map(item => headWeights(item.head, mode, scale))
  const cell = 44
  return <LabFrame title="两个头，同一段输入，两种读取方式" hint="行是查询位置，列是键位置；颜色越深权重越大">
    <Controls>
      <label className="text-sm">可见关系
        <select value={mode} onChange={event => setMode(event.target.value as MaskMode)} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">
          <option value="causal">因果：只看 j ≤ i</option>
          <option value="bidirectional">双向：看全部位置</option>
        </select>
      </label>
      <Range label="q、k 长度倍数" value={scale} min={0.5} max={6} step={0.5} onChange={setScale} format={v => v.toFixed(1)} />
      <Range label="查看查询位置 i" value={query} min={0} max={3} onChange={setQuery} />
    </Controls>
    <svg viewBox="0 0 560 300" className={pen.canvas} role="img" aria-label={`查询 ${tokens[query]} 在两个头上的权重：${weights.map(w => w[query].map(v => v.toFixed(2)).join('、')).join('；')}`}>
      {heads.map((item, h) => {
        const x0 = 60 + h * 270
        return <g key={h}>
          <text x={x0 + 2 * cell} y="18" textAnchor="middle" className={h ? pen.textB : pen.textA}>{item.name}</text>
          {tokens.map((token, j) => <text key={j} x={x0 + j * cell + cell / 2} y="40" textAnchor="middle" className={pen.muted}>{token}</text>)}
          {tokens.map((token, i) => <text key={i} x={x0 - 8} y={52 + i * cell + cell / 2 + 4} textAnchor="end" className={i === query ? pen.textA : pen.muted}>{token}</text>)}
          {weights[h].map((row, i) => row.map((w, j) => {
            const hidden = mode === 'causal' && j > i
            return <g key={`${i}-${j}`}>
              <rect x={x0 + j * cell} y={52 + i * cell} width={cell - 3} height={cell - 3} rx="3"
                className={hidden ? 'fill-sunken stroke-rule' : h ? 'fill-accent2 stroke-accent2' : 'fill-accent stroke-accent'}
                fillOpacity={hidden ? 1 : Math.round((0.08 + 0.92 * w) * 100) / 100} strokeWidth={i === query ? 2.5 : 1} />
              <text x={x0 + j * cell + cell / 2 - 1} y={52 + i * cell + cell / 2 + 3} textAnchor="middle" className={`${pen.mono} text-[11px]`}>{hidden ? '—' : w.toFixed(2)}</text>
            </g>
          }))}
        </g>
      })}
      <text x="20" y="250">同一个输入，两套 W_Q、W_K 得到不同的 T×T 权重；各头结果拼接后由 W_O 混合。</text>
      <text x="20" y="274" className={pen.muted}>{mode === 'causal' ? '灰格被遮罩：它们不进入分母，权重恒为 0。' : '去掉遮罩后，早期位置也能读到未来 token。'}</text>
    </svg>
    <Readout>
      查询 {tokens[query]}（i={query}）· 头 0 权重 [{weights[0][query].map(v => v.toFixed(3)).join(', ')}] · 头 1 权重 [{weights[1][query].map(v => v.toFixed(3)).join(', ')}] · 长度倍数越大，softmax 越接近硬选择
    </Readout>
  </LabFrame>
}
