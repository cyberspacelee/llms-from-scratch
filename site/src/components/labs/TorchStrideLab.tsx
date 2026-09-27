import { useState } from 'react'
import { tensorLayout } from '../../lib/framework-tensor-model'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'
import Formula from './Formula'

export default function TorchStrideLab() {
  const [transposed, setTransposed] = useState(false)
  const [index, setIndex] = useState(0)
  const cols = transposed ? 2 : 3
  const row = Math.floor(index / cols), col = index % cols
  const state = tensorLayout(transposed, row, col)
  return <LabFrame title="张量坐标怎样走到同一块存储" hint="arange(6) · 不含复制">
    <Controls>
      <label className="flex h-8 items-center gap-2 text-sm"><input type="checkbox" checked={transposed} onChange={event => setTransposed(event.target.checked)} className="accent-accent" />转置两个轴</label>
      <Range label="逻辑展开位置" value={index} min={0} max={5} onChange={setIndex} />
    </Controls>
    <svg viewBox="0 0 320 285" className={`${pen.canvas} max-w-96`} role="img" aria-label={`形状 ${state.rows}乘${state.cols}，stride ${state.stride}，选中坐标 ${row},${col} 指向存储 ${state.offset}`}>
      <text x="12" y="20">逻辑张量：({state.rows}, {state.cols})</text>
      {state.traversal.map((offset, i) => <g key={i}>
        <rect x={12 + i % cols * 92} y={35 + Math.floor(i / cols) * 36} width="82" height="30" rx="3" className={i === index ? 'fill-accent2-soft stroke-accent2 stroke-2' : 'fill-accent-soft stroke-accent'} />
        <text x={53 + i % cols * 92} y={55 + Math.floor(i / cols) * 36} textAnchor="middle">{offset}</text>
      </g>)}
      <path d={`M${53 + col * 92} ${65 + row * 36}V174H${36 + state.offset * 48}V207m-4-5 4 5 4-5`} className={pen.guide} />
      <text x="12" y="195">底层存储：元素偏移 0–5</text>
      {Array.from({ length: 6 }, (_, offset) => <g key={offset}><rect x={12 + offset * 48} y="210" width="43" height="32" rx="3" className={offset === state.offset ? 'fill-accent2-soft stroke-accent2 stroke-2' : 'fill-sunken stroke-rule'} /><text x={33.5 + offset * 48} y="231" textAnchor="middle">{offset}</text></g>)}
      <text x="12" y="272">存储顺序不变：0,1,2,3,4,5</text>
    </svg>
    <Readout><Formula>{String.raw`p=${row}\times${state.stride[0]}+${col}\times${state.stride[1]}=${state.offset}`}</Formula> · stride ({state.stride.join(', ')}) · {transposed ? '非连续；按逻辑顺序 reshape(6) 需复制' : '连续；view(6) 可共享存储'} · 逻辑遍历 [{state.traversal.join(', ')}]</Readout>
  </LabFrame>
}
