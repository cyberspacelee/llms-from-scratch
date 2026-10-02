import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Select, Controls, LabFrame, pen } from './Lab'

const levels = [
  { name: 'CPU', detail: 'Host 准备输入、提交 kernel；调用返回不表示设备完成。', route: 'systems/engine-execution', link: '引擎如何准备与提交工作' },
  { name: 'GPU', detail: 'Device 执行提交的 grid；global memory 保存跨 block 使用的数据。', route: 'gpu/memory-and-sync', link: '地址、内存与同步' },
  { name: 'SM', detail: 'SM 承载已驻留的 block，并从可运行的 warp 中发射指令。', route: 'gpu/execution-model', link: '驻留与执行模型' },
  { name: 'block', detail: 'block 是线程协作与资源分配单位；shared 与 block 屏障在此范围内。', route: 'gpu/kernel-design', link: '分块、复用与屏障' },
  { name: 'thread', detail: 'thread 是逻辑程序实例；编号决定输出归属，不对应专属物理核心。', route: 'gpu/execution-model', link: '线程编号与输出负责人' },
]

export default function GpuHierarchyLab() {
  const [selected, setSelected] = useState(0)
  const current = levels[selected]
  const base = import.meta.env.BASE_URL.replace(/\/?$/, '/')
  return <LabFrame title="主机、设备与逻辑执行层次" hint="包含关系示意；数量不代表硬件规格">
    <Controls><Select label="观察层级" value={selected} onChange={event => setSelected(Number(event.target.value))} >{levels.map((level, index) => <option key={level.name} value={index}>{level.name}</option>)}</Select></Controls>
    <SvgCanvas viewBox="0 0 320 386" className={`${pen.canvas} max-w-96`} role="group" aria-label={`执行层次，当前 ${current.name}`}>
      {levels.map((level, index) => <g key={level.name} role="button" tabIndex={0} aria-label={`观察 ${level.name}`} aria-pressed={selected === index} onClick={() => setSelected(index)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSelected(index) } }} className="cursor-pointer focus:outline-2 focus:outline-info">
        <rect x={12 + index * 12} y={12 + index * 72} width={296 - index * 12} height="54" rx="3" className={selected === index ? 'fill-accent-soft stroke-accent stroke-2' : 'fill-sunken stroke-rule'} />
        <text x={26 + index * 12} y={45 + index * 72}>{level.name}</text>
        {index < 4 && <text x="158" y={80 + index * 72} className={pen.muted}>{index === 0 ? '提交工作' : index === 1 ? '包含 SM' : index === 2 ? '承载 block' : '包含 thread'}</text>}
      </g>)}
    </SvgCanvas>
    <div aria-live="polite" className="mt-3 border-t border-rule pt-3 text-sm"><p className="mt-0 mb-2">{current.detail}</p><a href={`${base}${current.route}/`} className="font-semibold text-accent">{current.link}</a></div>
  </LabFrame>
}
