import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import Formula from './Formula'
import { contraction } from './NumpyContractModel'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function NumpyContractionLab() {
  const [row, setRow] = useState(0)
  const [output, setOutput] = useState(0)
  const [k, setK] = useState(0)
  const result = contraction(row, output)
  return <LabFrame title="矩阵乘的保留轴与收缩轴">
    <Controls>
      <Range label="样本行 b" value={row} min={0} max={1} onChange={setRow} />
      <Range label="输出通道 m" value={output} min={0} max={1} onChange={setOutput} />
      <Range label="查看收缩位置 k" value={k} min={0} max={2} onChange={setK} />
    </Controls>
    <SvgCanvas viewBox="0 0 320 310" className={`${pen.canvas} max-w-96`} role="img" aria-label={`输出(${row},${output})，三项乘积${result.products.join('、')}，和${result.value}`}>
      <text x="12" y="20">X 的第 {row} 行 · 特征 k</text>
      <text x="12" y="99">W 的第 {output} 行 · 同一个 k</text>
      <text x="12" y="178">相乘后，沿 k 求和</text>
      {[result.x[row], result.w[output], result.products].map((values, group) => values.map((v, i) => <g key={`${group}-${i}`}>
        <rect x={12 + i * 99} y={35 + group * 79} width="91" height="31" rx="3" className={i === k ? 'fill-accent2-soft stroke-accent2' : group === 0 ? 'fill-accent-soft stroke-accent' : 'fill-info-soft stroke-info'} />
        <text x={57 + i * 99} y={56 + group * 79} textAnchor="middle">{v}</text>
      </g>))}
      <text x="12" y="259">Z[{row},{output}] = {result.value}</text>
      <text x="12" y="291" className={pen.muted}>保留 b、m；消去特征 k</text>
    </SvgCanvas>
    <Readout><Formula>{`Z_{${row},${output}}=\\sum_{k=0}^{2}X_{${row},k}W_{${output},k}=${result.value}`}</Formula><br />当前项 {result.x[row][k]} × {result.w[output][k]} = {result.products[k]} · X @ W.T · einsum('bd,md-&gt;bm', X, W)</Readout>
  </LabFrame>
}
