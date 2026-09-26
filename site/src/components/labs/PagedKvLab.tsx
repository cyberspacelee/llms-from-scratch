import { useId, useState } from 'react'
import { Arrow, Controls, LabFrame, pen, Range } from './Lab'

const physicalIds = [6, 2, 9, 4]
const blockSize = 4

/** Logical token positions → block_table → flat physical slots. */
export default function PagedKvLab() {
  const [length, setLength] = useState(10)
  const [prefix, setPrefix] = useState(1)
  const id = useId().replace(/:/g, '')

  const numBlocks = Math.ceil((length + 1) / blockSize)
  const shared = Math.min(prefix, Math.floor(length / blockSize))
  const table = physicalIds.slice(0, numBlocks)
  const nextBlock = Math.floor(length / blockSize)
  const slot = table[nextBlock] * blockSize + (length % blockSize)
  const [step, cell] = [180, 36]

  return (
    <LabFrame title="逻辑 token → 物理槽位" hint="块大小取 4；绿色首块可以被别的请求共享">
      <Controls>
        <Range label="当前序列长度" value={length} min={4} max={15} onChange={setLength} />
        <Range label="共享的完整前缀块" value={prefix} min={0} max={2} onChange={setPrefix} format={() => String(shared)} />
      </Controls>
      <svg viewBox="0 0 740 320" className={pen.canvas} role="img"
        aria-label={`block_table [${table.join(', ')}]，下一个 token 写入槽位 ${slot}`}>
        <defs><Arrow id={`${id}-arrow`} className="fill-info" /></defs>
        <text x="20" y="24" className="font-semibold">逻辑序列</text>
        {table.map((physical, block) => {
          const x = 20 + block * step
          return (
            <g key={block}>
              <text x={x} y={52} className={`${pen.mono} ${pen.muted}`}>logical block {block}</text>
              {Array.from({ length: blockSize }, (_, offset) => {
                const position = block * blockSize + offset
                const isNext = position === length
                const filled = position < length
                const fill = isNext ? 'fill-accent2' : filled ? (block < shared ? 'fill-accent' : 'fill-info') : 'fill-sunken'
                return (
                  <g key={offset}>
                    <rect x={x + offset * cell} y={62} width={cell - 4} height={36} rx="3" className={`${fill} stroke-rule-strong`} />
                    <text x={x + offset * cell + (cell - 4) / 2} y={85} textAnchor="middle"
                      className={`${pen.mono} ${filled || isNext ? 'fill-paper!' : pen.muted}`}>
                      {isNext ? 'NEXT' : filled ? position : '·'}
                    </text>
                  </g>
                )
              })}
              <path d={`M ${x + 70} 102 C ${x + 70} 140, ${x + 70} 150, ${x + 70} 182`} className="fill-none stroke-info stroke-[1.5]" markerEnd={`url(#${id}-arrow)`} />
              <text x={x} y={206} className={`${pen.mono} ${pen.muted}`}>physical block {physical}{block < shared ? ' · ref=2' : ''}</text>
              {Array.from({ length: blockSize }, (_, offset) => (
                <g key={offset}>
                  <rect x={x + offset * cell} y={216} width={cell - 4} height={36} rx="3"
                    className={`${block < shared ? 'fill-accent-soft' : 'fill-sunken'} stroke-rule-strong`} />
                  <text x={x + offset * cell + (cell - 4) / 2} y={239} textAnchor="middle" className={pen.mono}>
                    {physical * blockSize + offset}
                  </text>
                </g>
              ))}
            </g>
          )
        })}
        <text x="20" y="290" className="font-semibold">物理 KV 池</text>
        <text x="20" y="312" className={pen.muted}>上排数字是 token 的逻辑位置，下排是扁平的物理槽位编号。</text>
      </svg>
      <dl className="mt-4 grid grid-cols-[repeat(auto-fit,minmax(10rem,1fr))] gap-px overflow-hidden rounded-lg bg-rule">
        {[
          ['block_table', `[${table.join(', ')}]`],
          ['下一个 token 的逻辑位置', String(length)],
          ['下一个 token 的物理槽位', `${table[nextBlock]} × ${blockSize} + ${length % blockSize} = ${slot}`],
        ].map(([label, value]) => (
          <div key={label} className="flex flex-col-reverse bg-sunken px-3 py-2">
            <dt className="text-xs text-muted">{label}</dt>
            <dd className="m-0 font-mono text-sm wrap-anywhere">{value}</dd>
          </div>
        ))}
      </dl>
    </LabFrame>
  )
}
