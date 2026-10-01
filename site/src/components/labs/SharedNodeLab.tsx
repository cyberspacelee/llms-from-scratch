import { useId, useState } from 'react'
import { sharedGraph } from '../../lib/math-labs-model'
import { Arrow, Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function SharedNodeLab() {
  const [x, setX] = useState(2)
  const [direct, setDirect] = useState(true)
  const [throughB, setThroughB] = useState(true)
  const [stage, setStage] = useState(3)
  const arrow = useId()
  const g = sharedGraph(x, direct, throughB)
  const node = (cxp: number, cyp: number, name: string, value: number, grad: string) => <g>
    <circle cx={cxp} cy={cyp} r="26" className="fill-raised stroke-ink stroke-[1.5]" />
    <text x={cxp} y={cyp + 5} textAnchor="middle" className="font-semibold">{name}</text>
    <text x={cxp} y={cyp - 36} textAnchor="middle" className={pen.textA}>{value}</text>
    <text x={cxp} y={cyp + 48} textAnchor="middle" className={pen.textB}>{grad}</text>
  </g>
  return <LabFrame title="共享节点：两条路径的梯度要相加" hint="a = x²，b = 3a，ℓ = a + b · 绿色是前向值，橙色是反向梯度">
    <Controls>
      <Range label="输入 x" value={x} min={-3} max={3} step={0.5} onChange={setX} format={v => v.toFixed(1)} />
      <Range label="反向阶段" value={stage} min={0} max={3} onChange={setStage} />
      <label className="flex h-8 items-center gap-2 self-end text-sm"><input type="checkbox" checked={direct} onChange={event => setDirect(event.target.checked)} className="accent-accent" />保留 a → ℓ 的直接路径</label>
      <label className="flex h-8 items-center gap-2 self-end text-sm"><input type="checkbox" checked={throughB} onChange={event => setThroughB(event.target.checked)} className="accent-accent" />保留经过 b 的路径</label>
    </Controls>
    <svg viewBox="0 0 520 210" className={`${pen.canvas} hidden! max-w-140 sm:block!`} role="img" aria-label={`x=${x}，ℓ=${g.loss}，求得 dℓ/dx=${g.gradX}，正确值 ${g.exact}`}>
      <defs><Arrow id={arrow} className="fill-muted" /></defs>
      <path d="M86 110H154" className={stage >= 3 ? pen.c : `${pen.axis} stroke-[1.5]`} markerEnd={`url(#${arrow})`} />
      <path d="M206 96L290 56" className={stage >= 2 && throughB ? pen.c : `${pen.axis} stroke-[1.5]`} markerEnd={`url(#${arrow})`} />
      <path d="M346 56L430 96" className={stage >= 1 && throughB ? pen.c : `${pen.axis} stroke-[1.5]`} markerEnd={`url(#${arrow})`} />
      <path d="M206 110H430" className={`${stage >= 1 && direct ? pen.c : pen.axis} stroke-[1.5] ${direct ? '' : '[stroke-dasharray:3_5] opacity-40'}`} markerEnd={`url(#${arrow})`} />
      <text x="248" y="68" className={throughB ? pen.textB : pen.muted}>×3</text>
      <text x="390" y="68" className={throughB ? pen.textB : pen.muted}>×1</text>
      <text x="318" y="128" textAnchor="middle" className={direct ? pen.textB : pen.muted}>×1{direct ? '' : '（已删去）'}</text>
      {!throughB && <text x="318" y="20" textAnchor="middle" className={pen.muted}>经 b 的贡献已删去</text>}
      {node(60, 110, 'x', x, `dℓ/dx = ${g.gradX}`)}
      {node(180, 110, 'a', g.a, `dℓ/da = ${g.gradA}`)}
      {node(318, 50, 'b', g.b, 'dℓ/db = 1')}
      {node(456, 110, 'ℓ', g.loss, 'dℓ/dℓ = 1')}
    </svg>
    <svg viewBox="0 0 320 330" className={`${pen.canvas} max-w-96 sm:hidden!`} role="img" aria-label={`共享节点 a 收到 ${g.direct}+${g.throughB}=${g.gradA}，反向阶段 ${stage}`}>
      <path d="M160 63V133" className={stage >= 3 ? pen.c : pen.axis} />
      <path d="M141 170L81 228M83 247H217" className={stage >= 2 && throughB ? pen.c : pen.axis} />
      <path d="M180 169L239 228" className={stage >= 1 && direct ? pen.c : pen.axis} />
      <text x="65" y="200" className={pen.textB}>经 b ×3</text><text x="220" y="200" className={pen.textB}>直接 ×1</text>
      {[{ name: 'x', x: 160, y: 43, value: x }, { name: 'a', x: 160, y: 153, value: g.a }, { name: 'b', x: 60, y: 247, value: g.b }, { name: 'ℓ', x: 260, y: 247, value: g.loss }].map(n => <g key={n.name}><circle cx={n.x} cy={n.y} r="23" className="fill-sunken stroke-rule-strong" /><text x={n.x} y={n.y + 5} textAnchor="middle">{n.name}={n.value}</text></g>)}
      <text x="160" y="102" textAnchor="middle">返回 x：{g.gradA}×2x={g.gradX}</text>
      <text x="160" y="303" textAnchor="middle">a 汇合：{g.direct}+{g.throughB}={g.gradA}</text>
    </svg>
    <Readout>{['0 · 前向值已就绪', '1 · 从 ℓ 返回两条直接依赖', '2 · 经 b 的贡献与直接贡献在 a 相加', '3 · a 的总梯度继续乘 2x 返回 x'][stage]}<br />dℓ/da = 直接 {g.direct} + 经 b {g.throughB} = {g.gradA} · dℓ/dx = {g.gradA} × 2x = {g.gradX} · 正确值 8x = {g.exact}{g.gradX === g.exact ? ' ✓' : ' ✗ 漏掉了一条路径'}</Readout>
  </LabFrame>
}
