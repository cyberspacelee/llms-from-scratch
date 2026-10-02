import { useState } from 'react'
import { vectorBranch } from '../../lib/framework-tensor-model'
import { Button, Controls, LabFrame, Range, Readout } from './Lab'
import { TraceTable } from './DataViews'

/** The same vector, shared bias and branched loss as the F4 Python experiment. */
export default function AutogradBranchLab() {
  const [w0, setW0] = useState(1), [w1, setW1] = useState(2), [bias, setBias] = useState(1)
  const [seed, setSeed] = useState(1)
  const [grad, setGrad] = useState<{ w: number[]; b: number } | null>(null)
  const { u, square, product, sum, loss, upstream, branches, gu, gw, gb } = vectorBranch(w0, w1, bias, seed)
  const display = (values: number[]) => `(${values.map(value => Number(value.toFixed(4))).join(', ')})`
  return <LabFrame title="向量分支汇合，再归约共享偏置" hint="x=(2,−1) 固定；u=w⊙x+b；L=½(u₀²+u₀u₁−2)²">
    <Controls><Range label="参数 w₀" value={w0} min={-2} max={3} step={0.25} onChange={setW0} /><Range label="参数 w₁" value={w1} min={-2} max={3} step={0.25} onChange={setW1} /><Range label="共享参数 b" value={bias} min={-2} max={3} step={0.25} onChange={setBias} /><Range label="标量损失反向种子 g" step={0.5} value={seed} min={-2} max={2} onChange={setSeed} /></Controls>
    <TraceTable label="正文前向与两条分支的反向值" rows={[
      ['前向中间向量 u', display(u)], ['平方分支 + 乘积分支 → s', `${square} + ${product} = ${sum}`], ['损失 L → 上游 gₛ', `${loss} → ${upstream}`], ['u₀：平方贡献 + 乘积贡献', `${branches[0]} + ${branches[1]} = ${gu[0]}`], ['u₁：乘积贡献', String(gu[1])], ['乘输入 / 广播归约 → 参数梯度', `g_w=${display(gw)}；g_b=${gb}`],
    ]} />
    <div className="mt-3 flex flex-wrap gap-2"><Button primary onClick={() => setGrad(current => ({ w: gw.map((value, i) => value + (current?.w[i] ?? 0)), b: gb + (current?.b ?? 0) }))}>重新前向并 backward</Button><Button onClick={() => setGrad(null)}>清空叶子梯度</Button><Button onClick={() => { setW0(1); setW1(2); setBias(1); setSeed(1); setGrad(null) }}>恢复正文参数</Button></div>
    <Readout>当前 w.grad={grad ? display(grad.w) : 'None'}；b.grad={grad ? grad.b : 'None'}。本次贡献 g_w={display(gw)}，g_b={gb}。默认参数第一次反向得到 (40,−12)、32；第二次得到 (80,−24)、64。更改参数不会清空已有梯度；本实验每次重新构建前向图。</Readout>
  </LabFrame>
}
