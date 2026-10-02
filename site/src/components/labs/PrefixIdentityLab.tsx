import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Readout, pen } from './Lab'


export default function PrefixIdentityLab() {
  const [different, setDifferent] = useState(false), [namespace, setNamespace] = useState(false)
  const same = !different && !namespace
  return <LabFrame title="相同局部块，是否有相同前缀身份？">
    <Controls><Toggle label="B 的第一块不同" checked={different} onChange={value => setDifferent(value)} /><Toggle label="B 使用不同模型/位置规则" checked={namespace} onChange={value => setNamespace(value)} /></Controls>
    <SvgCanvas viewBox="0 0 280 175" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`第二块 token 相同；完整前缀身份${same ? '相同' : '不同'}`}>
      {['A', 'B'].map((name, i) => <g key={name}><text x="4" y={30 + i * 80}>{name}</text><rect x="28" y={8 + i * 80} width="108" height="50" className="fill-sunken stroke-rule" /><text x="82" y={30 + i * 80} textAnchor="middle">{i && different ? '9,2,3,4' : '1,2,3,4'}</text><text x="82" y={49 + i * 80} textAnchor="middle">prefix h₀</text><path d={`M140 ${33 + i * 80}h18`} className={pen.axis} /><rect x="162" y={8 + i * 80} width="110" height="50" className={same ? 'fill-accent/15 stroke-accent' : 'fill-accent2/15 stroke-accent2'} /><text x="217" y={30 + i * 80} textAnchor="middle">5,6,7,8</text><text x="217" y={49 + i * 80} textAnchor="middle">hash(h₀,u₁)</text></g>)}
    </SvgCanvas><Readout>相同局部 ID ≠ 相同计算上下文。{same ? '命名空间与完整前缀一致，可作为候选命中' : '前缀或命名空间不同，不可共享本块 KV'}。图使用完整前缀相等性判断；不模拟哈希值或假设哈希无碰撞。</Readout>
  </LabFrame>
}
