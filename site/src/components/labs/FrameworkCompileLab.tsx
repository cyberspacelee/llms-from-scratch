import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { compileTrace } from '../../lib/framework-trace-model'
import { Toggle, Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function FrameworkCompileLab() {
  const [step, setStep] = useState(1), [broken, setBroken] = useState(false)
  const rows = compileTrace(step, broken), last = rows.at(-1)
  return <LabFrame title="输入契约怎样决定复用" hint="固定 shape 的教学 guard；实际 guards 由 PyTorch 决定">
    <Controls><Range label="已提交调用" value={step} min={0} max={4} onChange={setStep} /><Toggle label="切换正文 python_region 中断案例" checked={broken} onChange={value => setBroken(value)} /></Controls>
    <SvgCanvas viewBox="0 0 320 320" className={`${pen.canvas} max-w-96`} role="img" aria-label={`已执行 ${step} 次，保存 ${last?.graphs ?? 0} 种 shape 契约`}>
      {Array.from({ length: 4 }, (_, i) => <g key={i}>
        <rect x="10" y={12 + i * 76} width="300" height="64" rx="4" className={i < step ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} />
        <text x="22" y={35 + i * 76}>调用 {i + 1} · shape ({broken ? i === 2 ? 3 : 2 : i === 2 ? 4 : 3},)</text>
        <text x="22" y={58 + i * 76} className={rows[i]?.hit ? pen.textA : pen.muted}>{rows[i]?.route ?? '等待调用'}</text>
      </g>)}
    </SvgCanvas>
    {broken && <SvgCanvas viewBox="0 0 320 140" className={`${pen.canvas} max-w-96`} role="img" aria-label="编译乘二，禁用区域加一，编译乘三">
      {['图 A · ×2', 'Python/eager · +1', '图 B · ×3'].map((label, i) => <g key={i}><rect x="20" y={5 + i * 45} width="280" height="34" rx="3" className={i === 1 ? 'fill-accent2-soft stroke-accent2' : 'fill-info-soft stroke-info'} /><text x="160" y={28 + i * 45} textAnchor="middle">{label}</text></g>)}
    </SvgCanvas>}
    <Readout>{last ? <>输入=[{last.x.join(',')}] · y={broken ? '(2x+1)×3' : 'x²+3x'}=[{last.output.join(',')}] · 梯度=[{last.gradient.join(',')}]<br />缓存 {last.graphs} 个 shape 契约 · 第四次回到首次长度，可命中首次保存的契约。<br />{broken ? '切换后的函数是正文 segmented：每次调用跨图 A、禁用区域、图 B；fullgraph=True 拒绝这个中断。shape 版本数与单次调用的图分段数是两件事。' : '当前张量表达式可以作为一个区域；结果用 eager 多项式独立核对。'}</> : '尚无调用，也没有缓存契约。'}</Readout>
  </LabFrame>
}
