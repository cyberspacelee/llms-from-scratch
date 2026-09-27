import { useId, useState } from 'react'
import { transposeElement } from '../../lib/transpose-model'
import Formula from './Formula'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function TransposeLab() {
  const id = useId()
  const [row, setRow] = useState(2)
  const [col, setCol] = useState(5)
  const [pitch, setPitch] = useState(33)
  const [edge, setEdge] = useState(false)
  const element = transposeElement(row, col, pitch, edge)
  const gridX = 20, size = 144, cell = size / 32
  const validSize = edge ? 3 : 32
  return <LabFrame title="转置：一个元素怎样交换负责人" hint="35×67 输入 · block 32×8 · tile 32×32">
    <Controls>
      <Range label="tile 内输入行 r" value={row} min={0} max={31} onChange={setRow} />
      <Range label="tile 内输入列 c" value={col} min={0} max={31} onChange={setCol} />
      <label className="block text-sm" htmlFor={`${id}-pitch`}>共享行跨度
        <select id={`${id}-pitch`} value={pitch} onChange={event => setPitch(Number(event.target.value))} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2 text-ink">
          <option value={32}>32 列：无填充</option><option value={33}>33 列：填充一列</option>
        </select>
      </label>
      <label className="block text-sm" htmlFor={`${id}-tile`}>输入 tile
        <select id={`${id}-tile`} value={edge ? 'edge' : 'full'} onChange={event => setEdge(event.target.value === 'edge')} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2 text-ink">
          <option value="full">左上：32×32 全有效</option><option value="edge">右下：3×3 有效</option>
        </select>
      </label>
    </Controls>
    <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="转置数据搬运图，可横向滚动">
      <svg viewBox="0 0 360 666" className={`${pen.canvas} min-w-80`} role="img" aria-label={`元素 A(${element.inputRow},${element.inputCol}) 经 shared(${row},${col}) 写入转置输出；${element.valid ? '有效' : '边界外，不写输出'}`}>
        {[0, 1, 2].map(stage => {
          const y = 40 + stage * 215
          const output = stage === 2
          const selectedRow = output ? col : row, selectedCol = output ? row : col
          const title = ['① global input：沿行读', '② shared tile：换列读', '③ global output：沿行写'][stage]
          return <g key={stage}>
            <text x="20" y={y - 16} className="font-semibold">{title}</text>
            <rect x={gridX} y={y} width={size} height={size} className="fill-sunken stroke-rule-strong" />
            <rect x={gridX} y={y} width={validSize * cell} height={validSize * cell} className="fill-accent-soft" />
            {stage === 1
              ? <rect x={gridX + col * cell} y={y} width={cell} height={element.activeReaders * cell} className="fill-info-soft" />
              : <rect x={gridX} y={y + selectedRow * cell} width={selectedRow < validSize ? validSize * cell : 0} height={cell} className="fill-info-soft" />}
            {Array.from({ length: 33 }, (_, index) => <path key={index} d={`M${gridX + index * cell} ${y}v${size}M${gridX} ${y + index * cell}h${size}`} className="stroke-rule fill-none stroke-[0.5]" />)}
            <rect x={gridX + selectedCol * cell} y={y + selectedRow * cell} width={cell} height={cell} className={element.valid ? 'fill-accent2 stroke-accent2' : 'fill-muted stroke-ink'} />
            {stage === 1 && pitch === 33 && <rect x={gridX + size + 3} y={y} width={cell} height={size} className="fill-accent2-soft stroke-accent2" />}
            <text x="186" y={y + 16} className={pen.mono}>{stage === 0 ? `A[${element.inputRow}, ${element.inputCol}]` : stage === 1 ? `tile[${row}, ${col}]` : `B[${element.inputCol}, ${element.inputRow}]`}</text>
            <text x="186" y={y + 42}>{stage === 1 ? `bank ${element.bank}` : element.valid ? '选中元素有效' : '选中元素越界'}</text>
            <text x="186" y={y + 68} className={pen.mono}>{stage === 0 ? `tx=${element.producer.x}, ty=${element.producer.y}` : stage === 1 ? `行跨度 ${pitch}` : `tx=${element.consumer.x}, ty=${element.consumer.y}`}</text>
            <text x="186" y={y + 94}>{stage === 0 ? `装载循环 j=${element.producer.j}` : stage === 1 ? element.activeReaders ? `列读：${element.conflict} 路模型` : '无输出 lane 读取此列' : `写回循环 j=${element.consumer.j}`}</text>
            <text x="186" y={y + 120}>{stage === 0 ? element.valid ? '连续读取一行' : '装载零，不读越界' : stage === 1 ? pitch === 33 ? '右侧细条是填充列' : '同列落入同一 bank' : element.valid ? '写回负责人连续写行' : '不写越界输出'}</text>
            <text x="20" y={y + 161} className={pen.muted}>{stage === 1 ? `蓝色列：${element.activeReaders} 个有效写回 lane 的读位置` : '蓝色行：同一 warp 的有效读写方向'}</text>
          </g>
        })}
        <path d="M92 210v16m-5-5 5 5 5-5M92 425v16m-5-5 5 5 5-5" className={pen.axis} />
        <text x="115" y="222" className={pen.textB}>装载完成 → 全 block 屏障</text>
        <text x="115" y="437" className={pen.textA}>屏障后按转置映射消费</text>
      </svg>
    </div>
    <Readout>
      <Formula>{String.raw`A_{${element.inputRow},${element.inputCol}}\to\mathrm{tile}_{${row},${col}}\to B_{${element.inputCol},${element.inputRow}}`}</Formula>
      {' · '}{element.valid ? `输入元素偏移 ${element.inputWord}，输出元素偏移 ${element.outputWord}` : '无效位置装载零，输出不写'}
      {' · '}shared 元素偏移 {element.sharedWord}，bank {element.bank}
      {' · '}同一写回 warp 的 shared 列读：{element.activeReaders} 个活跃 lane，{element.activeReaders ? `${element.conflict} 路冲突模型` : '不发出列读取'}
    </Readout>
  </LabFrame>
}
