import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { mlaPaths } from '../../lib/advanced-interactive-model'

export default function MlaPathsLab() {
  const [head, setHead] = useState(0), [absorbed, setAbsorbed] = useState(false), [step, setStep] = useState(0)
  const r = mlaPaths(head)
  return <LabFrame title="同一 MLA 参数，两条相等的路径">
    <Controls><Range label="头" value={head} min={0} max={1} onChange={setHead} /><Range label="步骤" value={step} min={0} max={3} onChange={setStep} format={v => ['缓存', '内容分数', 'softmax', '值输出'][v]} /><Toggle label="吸收投影" checked={absorbed} onChange={value => setAbsorbed(value)} /></Controls>
    <SvgCanvas viewBox="0 0 280 240" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`头 ${head}，${absorbed ? '吸收' : '显式'}路径，同一注意力输出`}>
      {r.latent.map((c, j) => <g key={j}><rect x="8" y={8 + j * 62} width="264" height="54" className="fill-sunken stroke-rule" /><text x="16" y={28 + j * 62}>位置 {j}：缓存 c=[{c.join(', ')}]</text><text x="16" y={48 + j * 62} className={pen.mono}>{step === 0 ? '另存共享位置键，2 维' : step === 1 ? `${absorbed ? 'q̂·c' : 'q·Uₖc'}=${r.explicitScores[j]}` : `a${j}=${fmt(r.weights[j], 6)}`}</text></g>)}<text x="8" y="220">{absorbed ? '先 Σ a·c，再 Uᵥ 投影' : '先恢复 Uᵥc，再 Σ a·v'}</text>
    </SvgCanvas><Readout>q=[{r.query.join(', ')}]；Uₖᵀq=[{r.projected.join(', ')}]<br />内容=[{r.explicitScores.join(', ')}]；位置=[{r.positional.join(', ')}]；softmax 前仍除以 2<br />{step === 3 ? `显式=[${r.explicit.map(v => fmt(v, 6)).join(', ')}]；吸收=[${r.absorbed.map(v => fmt(v, 6)).join(', ')}]；最大误差=${fmt(Math.max(...r.explicit.map((v, i) => Math.abs(v - r.absorbed[i]))), 9)}` : '当前头独立计算权重与潜在混合，不跨头共享 z'}<br />每位置持久缓存：c 2 + 位置键 2=4 元素；显式全头 K/V 与位置键=10 元素</Readout>
  </LabFrame>
}
