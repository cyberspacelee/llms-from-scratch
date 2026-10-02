import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { moduleParameters } from '../../lib/framework-trace-model'
import { Toggle, Select, Controls, LabFrame, Readout, pen } from './Lab'

export default function FrameworkModuleLab() {
  const [selected, setSelected] = useState(0), [training, setTraining] = useState(false), [id, setId] = useState(0)
  const parameter = moduleParameters[selected]
  return <LabFrame title="模块名字、参数对象与使用位置">
    <Controls><Select label="登记参数" value={selected} onChange={e => setSelected(Number(e.target.value))} >{moduleParameters.map((p, i) => <option key={p.name} value={i}>{p.name}</option>)}</Select><Select label="embedding 行 ID" value={id} onChange={e => setId(Number(e.target.value))} >{[0, 1, 2, 3].map(value => <option key={value}>{value}</option>)}</Select><Toggle label="train 模式" checked={training} onChange={value => setTraining(value)} /></Controls>
    <SvgCanvas viewBox="0 0 320 318" className={`${pen.canvas} max-w-96`} role="img" aria-label={`选中 ${parameter.name}，身份 ${parameter.identity}，全部参数 20 个`}>
      <text x="160" y="22" textAnchor="middle">TinyWords · {training ? 'training=True' : 'training=False'}</text>
      {['embedding', 'norm', 'dropout', 'head'].map((name, i) => <g key={name}>
        <path d={`M34 ${i ? 63 + (i - 1) * 58 : 31}V${63 + i * 58}H57`} className={pen.axis} />
        <rect x="58" y={40 + i * 58} width="252" height="44" rx="4" className={parameter.owner === name ? 'fill-accent2-soft stroke-accent2' : 'fill-sunken stroke-rule'} />
        <text x="70" y={66 + i * 58}>{name} · {name === 'dropout' ? training ? '随机 mask ×2' : '恒等' : `${moduleParameters.filter(p => p.owner === name).reduce((s, p) => s + p.count, 0)} 个标量`}</text>
      </g>)}
      <text x="10" y="298">四个参数对象：8+2+2+8=20</text>
    </SvgCanvas>
    <Readout>{parameter.name} · shape {parameter.shape} · 对象身份 {parameter.identity}<br />eval 只改变层的模式，参数登记和 autograd 开关互不替代。<br />E[{id},:]：{id === 0 ? '输入位置 (0,0),(1,1),(1,2) 读取同一行；最后一处无直接损失' : id === 1 ? '(0,1),(1,0) 读取同一行' : id === 2 ? '(0,2) 读取此行但目标被忽略' : '本批未读取'}。<br />E[0] 与 E[2] 的值都是 (2,0)，仍是不同参数行。</Readout>
  </LabFrame>
}
