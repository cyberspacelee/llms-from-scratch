import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function WarpBranchLab() {
  const [active, setActive] = useState(32)
  const [threshold, setThreshold] = useState(16)
  const [mode, setMode] = useState('threshold')
  const a = (lane: number) => mode === 'threshold' ? lane < threshold : lane % 2 === 0
  const countA = Array.from({ length: active }, (_, lane) => a(lane)).filter(Boolean).length
  const countB = active - countA
  return <LabFrame title="一个 warp 的两条分支">
    <Controls>
      <Range label="有效 lane 数" value={active} min={1} max={32} onChange={setActive} />
      <fieldset disabled={mode !== 'threshold'} className="disabled:opacity-50"><Range label="A 分支阈值" value={threshold} min={0} max={32} onChange={setThreshold} /></fieldset>
      <label className="block text-sm">分支条件<select value={mode} onChange={e => setMode(e.target.value)} className="mt-2 block h-9 w-full rounded border border-rule bg-paper px-2"><option value="threshold">lane 小于阈值</option><option value="even">偶数 lane</option></select></label>
    </Controls>
    <svg viewBox="0 0 320 248" className={`${pen.canvas} max-w-96`} role="img" aria-label={`A 分支 ${countA} 个 lane，B 分支 ${countB} 个 lane`}>
      {[0, 1].map(path => <g key={path}>
        <text x="10" y={20 + path * 116}>{path === 0 ? '路径 A' : '路径 B'} · 活跃 {path === 0 ? countA : countB}</text>
        {Array.from({ length: 32 }, (_, lane) => {
          const enabled = lane < active && (path === 0 ? a(lane) : !a(lane))
          return <g key={lane}>
            <rect x={10 + lane % 8 * 38} y={32 + Math.floor(lane / 8) * 19 + path * 116} width="34" height="16" rx="2" className={enabled ? path === 0 ? 'fill-accent-soft stroke-accent' : 'fill-accent2-soft stroke-accent2' : 'fill-sunken stroke-rule'} />
            <text x={27 + lane % 8 * 38} y={44 + Math.floor(lane / 8) * 19 + path * 116} textAnchor="middle" className={`text-[10px] ${enabled ? '' : pen.muted}`}>{lane}</text>
          </g>
        })}
      </g>)}
    </svg>
    <Readout>有效 {active}/32 · A {countA} · B {countB} · 非空路径 {(countA > 0 ? 1 : 0) + (countB > 0 ? 1 : 0)} · 图示活跃集合，不代表机器指令数或耗时</Readout>
  </LabFrame>
}
