import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { grpoGroup } from '../../lib/advanced-interactive-model'

export default function GrpoLab() {
  const [equal, setEqual] = useState(false), [ratio, setRatio] = useState(1.4), [epsilon, setEpsilon] = useState(0.2), [selected, setSelected] = useState(0), [step, setStep] = useState(0)
  const r = grpoGroup(equal, ratio, epsilon, selected)
  return <LabFrame title="奖励、优势、概率比与裁剪">
    <Controls><Range label="步骤" value={step} min={0} max={3} onChange={setStep} format={v => ['奖励', '组内优势', '概率比', '受限更新'][v]} /><Range label="回复编号" value={selected} min={0} max={3} onChange={setSelected} /><Range label="ρ" value={ratio} min={0.4} max={1.6} step={0.05} onChange={setRatio} /><Range label="ε clip" value={epsilon} min={0.05} max={0.4} step={0.05} onChange={setEpsilon} /><Toggle label="四条全部无奖励" checked={equal} onChange={value => setEqual(value)} /></Controls>
    <SvgCanvas viewBox="0 0 280 230" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`选择回复 ${selected}，优势 ${r.advantage}，最小化损失的导数 ${r.lossDerivativeLogRatio}`}>
      <line x1="140" y1="15" x2="140" y2="205" className={pen.axis} />{r.rewards.map((reward, i) => <g key={i}><text x="8" y={35 + i * 48}>回复 {i}</text><rect x={step && r.advantages[i] < 0 ? 60 : 140} y={15 + i * 48} width={Math.abs(step ? r.advantages[i] : reward) * 80} height="25" className={i === selected ? 'fill-accent' : 'fill-info/40'} /><text x="235" y={34 + i * 48}>{step ? r.advantages[i] : reward}</text></g>)}<text x="8" y="226">0 轴两侧保留优势的正负方向</text>
    </SvgCanvas><Readout>均值={fmt(r.mean)}；总体标准差={fmt(r.std)}；A=[{r.advantages.join(', ')}]<br />ρA={fmt(r.raw)}；clip(ρ)A={fmt(r.restricted)}；min={fmt(r.objective)}<br />{step === 3 ? `最小化负目标：对 logρ 的偏导=${fmt(r.lossDerivativeLogRatio)}；${r.lossDerivativeLogRatio === 0 ? '此样本无推动（零优势或裁剪饱和）' : r.lossDerivativeLogRatio < 0 ? '梯度下降提高该回复概率比' : '梯度下降降低该回复概率比'}` : '先确定优势符号，再对两个乘积取 min'}</Readout>
  </LabFrame>
}
