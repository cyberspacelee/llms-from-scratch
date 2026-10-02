import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { chatMask } from '../../lib/training-trace-model'
import { Toggle, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'


export default function ChatMaskLab() {
  const [onlySecond, setOnlySecond] = useState(false), [truncated, setTruncated] = useState(false), [row, setRow] = useState(0)
  const result = chatMask(onlySecond, truncated), selected = Math.min(row, result.valid.length - 1)
  const influencing = result.valid.flatMap((v, i) => v && selected <= i ? [i] : [])
  return <LabFrame title="回复目标与 prompt 梯度是两条账" hint="SYS / Q1 / Q2 的两轮教学模板">
    <Controls><Range label="选择输入行" min={0} max={result.valid.length - 1} value={selected} onChange={setRow} /><Toggle label="只监督第二轮" checked={onlySecond} onChange={value => setOnlySecond(value)} /><Toggle label="截断至第一个 assistant 头" checked={truncated} onChange={value => setTruncated(value)} /></Controls>
    <SvgCanvas viewBox={`0 0 360 ${48 + result.valid.length * 38}`} className={`${pen.canvas} max-w-110!`} role="img" aria-label="每行 input、下一目标、直接监督以及选中表示的后续梯度路径">
      <text x="8" y="22">行 · input → target</text><text x="290" y="22">loss</text>
      {result.valid.map((valid, i) => <g key={i}><rect x="4" y={32 + i * 38} width="352" height="33" className={i === selected ? 'fill-accent-soft stroke-accent stroke-2' : influencing.includes(i) ? 'fill-paper stroke-info' : 'fill-paper stroke-rule'} /><text x="12" y={54 + i * 38}>{i} · {result.tokens[i]} → {result.tokens[i + 1]}</text><text x="308" y={54 + i * 38} className={valid ? pen.textA : pen.muted}>{Number(valid)}</text></g>)}
    </SvgCanvas>
    <Readout>有效分母 {result.count || '0：平均未定义'} · 第 {selected} 行直接 logits 损失 {result.valid[selected] ? '有' : '无'} · 可受有效预测行 [{influencing.join(',')}] 影响。<br />独立梯度探针 hⱼ=(j+1)/10、uᵢ=mean(h≤ᵢ)、目标 1 的平方损失：∂L/∂h{selected}={fmt(result.gradient[selected], 6)}。该数用于验证因果依赖，不是 Decoder 的实测梯度。</Readout>
  </LabFrame>
}
