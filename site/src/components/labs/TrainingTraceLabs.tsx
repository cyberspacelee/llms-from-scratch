import { useState } from 'react'
import { adamTrajectory, chatMask, checkpointComparison, ddpLedger, duplicateGroups, evaluationLedger, evaluationWindows, loraTrajectory, packing, stateAllocation } from '../../lib/training-trace-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

const vector = (values: number[]) => `(${values.map(x => fmt(x, 4)).join(', ')})`
function Trace({ rows, active = -1, label }: { rows: [string, string][]; active?: number; label: string }) {
  return <svg viewBox={`0 0 360 ${rows.length * 64 + 8}`} className={`${pen.canvas} max-w-110!`} role="img" aria-label={label}>{rows.map(([name, value], i) => <g key={name}>
    <rect x="4" y={4 + i * 64} width="352" height="54" rx="3" className={active === i ? 'fill-accent-soft stroke-accent' : 'fill-paper stroke-rule'} /><text x="16" y={24 + i * 64}>{name}</text><text x="16" y={45 + i * 64} className={pen.mono}>{value}</text>{i < rows.length - 1 && <path d={`M180 ${58 + i * 64}v8`} className={pen.axis} />}
  </g>)}</svg>
}

export function PackingLab() {
  const [isolated, setIsolated] = useState(true), [row, setRow] = useState(3), [stage, setStage] = useState(0)
  const result = packing(isolated)
  return <LabFrame title="两篇文档怎样成为一批预测" hint="A=[1,2,5]，B=[3,4,5]，EOS=5">
    <Controls><Range label="构造阶段" min={0} max={2} value={stage} onChange={setStage} /><Range label="查询 / 损失行" min={0} max={4} value={row} onChange={setRow} /><label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={isolated} onChange={e => setIsolated(e.target.checked)} />隔离文档</label></Controls>
    <Trace active={stage} label="文档拼接、标签移位和三类元数据" rows={[
      ['文档拼接（ID 5 不自动隔离）', 'A:1,2,5 │ B:3,4,5'], ['input → target（本行）', `${result.stream[row]} → ${result.stream[row + 1]} · 行 ${row}`], ['位置编号 / 直接损失', `p=${result.positions[row]} · loss mask=${Number(result.valid[row])}`],
    ]} />
    <svg viewBox="0 0 360 228" className={`${pen.canvas} max-w-110!`} role="img" aria-label={`行 ${row} 允许读入的同文档历史`}>
      {[0, 1, 2, 3, 4].map(j => <text key={j} x={93 + j * 52} y="19" textAnchor="middle">{j}</text>)}
      {result.allowed.map((keys, i) => <g key={i}><text x="3" y={50 + i * 39}>{i}·{i < 3 ? 'A' : 'B'}</text>{keys.map((allowed, j) => <g key={j}><rect x={71 + j * 52} y={27 + i * 39} width="44" height="32" className={allowed ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} strokeWidth={i === row ? 2 : 1} /><text x={93 + j * 52} y={49 + i * 39} textAnchor="middle">{allowed ? '1' : '0'}</text></g>)}</g>)}
    </svg>
    <Readout>{result.count} 个有效目标 · 行 {row} 可读键 [{result.allowed[row].flatMap((v, j) => v ? [j] : []).join(',')}]。{isolated ? '行 2 的 EOS→B 首项没有直接监督；B 的位置从 0 重新开始。' : '连续目标计入跨文档预测，B 读取 A 的历史。'} 两种选择定义不同目标。</Readout>
  </LabFrame>
}

export function AdamHistoryLab() {
  const [step, setStep] = useState(1), [reset, setReset] = useState(false), [coordinate, setCoordinate] = useState(0)
  const states = adamTrajectory(6, reset ? 1 : -1), state = states[step], previous = states[Math.max(0, step - 1)], i = coordinate
  const reference = adamTrajectory(6)[step]
  return <LabFrame title="同一参数的 AdamW 历史" hint="w₀=(1,−2)，ρ₁=.9，ρ₂=.99，η=.1，λ=.2">
    <Controls><Range label="更新步" min={0} max={6} value={step} onChange={setStep} /><Range label="参数坐标" min={0} max={1} value={coordinate} onChange={setCoordinate} /><label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={reset} onChange={e => setReset(e.target.checked)} />第 1 步后丢失优化器状态</label></Controls>
    <Trace label="数据梯度、矩状态、偏差校正与衰减更新" rows={[
      ['当前数据梯度 g（不含衰减）', fmt(state.g[i], 6)], ['一阶 m / 二阶原始矩 v', `${fmt(state.m[i], 6)} / ${fmt(state.v[i], 6)}`], ['偏差校正 m̂ / v̂', `${fmt(state.mh[i], 6)} / ${fmt(state.vh[i], 6)}`], ['参数：旧值 → 更新后值', `${fmt(previous.w[i], 6)} → ${fmt(state.w[i], 6)}`],
    ]} active={step ? 3 : 0} />
    <Readout>优化器步计数 k={state.k} · w={vector(state.w)} · 与完整历史最大差 {fmt(Math.max(...state.w.map((v, j) => Math.abs(v - reference.w[j]))), 6)}。第 0 步矩状态未形成；后续梯度由新 w 与同一四个目标重新计算。</Readout>
  </LabFrame>
}

export function CheckpointReplayLab() {
  const [omit, setOmit] = useState<'none' | 'optimizer' | 'rng' | 'cursor'>('none'), [step, setStep] = useState(0)
  const result = checkpointComparison(omit), reference = result.reference[step], restored = result.resumed[step]
  const firstMismatch = result.reference.findIndex((row, i) => row.document !== result.resumed[i].document || row.w !== result.resumed[i].w)
  return <LabFrame title="漏存状态后，哪一步首先偏离" hint="独立缩小例：单参数、三篇目标、5 步后保存，再运行 7 步">
    <Controls><label className="text-sm">恢复状态<select value={omit} onChange={e => setOmit(e.target.value as typeof omit)} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2"><option value="none">全部恢复</option><option value="optimizer">漏掉 m、v、优化器步数</option><option value="rng">漏掉 RNG</option><option value="cursor">漏掉当前顺序与游标</option></select></label><Range label="恢复后的更新序号" min={0} max={6} value={step} onChange={setStep} /></Controls>
    <svg viewBox="0 0 360 230" className={`${pen.canvas} max-w-110!`} role="img" aria-label="连续和恢复路径的七步参数轨迹">
      <line x1="30" y1="180" x2="335" y2="180" className={pen.axis} />
      <polyline points={result.reference.map((r, i) => `${40 + i * 46},${180 - r.w * 180}`).join(' ')} className={pen.a} /><polyline points={result.resumed.map((r, i) => `${40 + i * 46},${180 - r.w * 180}`).join(' ')} className={pen.b} />
      {result.reference.map((r, i) => <g key={i}><circle cx={40 + i * 46} cy={180 - r.w * 180} r={i === step ? 6 : 3} className="fill-accent" /><text x={40 + i * 46} y="203" textAnchor="middle">+{i + 1}</text></g>)}
      <text x="30" y="222" className={pen.muted}>实线：连续训练；虚线：恢复训练</text>
    </svg>
    <Readout>连续：文档 {reference.document}，loss={fmt(reference.loss, 6)}，w={fmt(reference.w, 6)}<br />恢复：文档 {restored.document}，loss={fmt(restored.loss, 6)}，w={fmt(restored.w, 6)}<br />{firstMismatch < 0 ? '七步参数与输入事件完全相同。' : `第一处偏离：恢复后第 ${firstMismatch + 1} 步。`} 重新初始化 RNG 不保证下一轮 shuffle 相同。</Readout>
  </LabFrame>
}

export function EvaluationReplayLab() {
  const [stride, setStride] = useState(2), [step, setStep] = useState(0), [model, setModel] = useState<'A' | 'B'>('A'), [included, setIncluded] = useState([true, true, true, true])
  const windows = evaluationWindows(stride), selected = Math.min(step, windows.length - 1), current = windows[selected]
  const visited = windows.slice(0, selected + 1).flatMap(w => w.targets), result = evaluationLedger(model, included)
  return <LabFrame title="重读上下文，只计新增目标" hint="D2 长度 7，输入上限 K=4；下方 NLL 是正文固定观测">
    <Controls><Range label="新增目标步幅 s" min={1} max={4} value={stride} onChange={setStride} /><Range label="窗口序号" min={0} max={windows.length - 1} value={selected} onChange={setStep} /><label className="text-sm">checkpoint<select value={model} onChange={e => setModel(e.target.value as 'A' | 'B')} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2"><option>A</option><option>B</option></select></label></Controls>
    <svg viewBox="0 0 360 244" className={`${pen.canvas} max-w-110!`} role="img" aria-label="输入 token、计分目标和累计覆盖的错位一格关系">
      {['输入 token', '新增目标', '累计计分'].map((name, i) => <text key={name} x="5" y={35 + i * 71}>{name}</text>)}
      {[0, 1, 2, 3, 4, 5, 6].map(j => <g key={j}>{[current.inputs.includes(j), current.targets.includes(j), visited.includes(j)].map((v, i) => <g key={i}><rect x={7 + j * 49} y={45 + i * 71} width="43" height="34" className={v ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} /><text x={28 + j * 49} y={68 + i * 71} textAnchor="middle">{j}</text></g>)}</g>)}
    </svg>
    <div className="mt-3 flex flex-wrap gap-4">{included.map((value, i) => <label key={i} className="text-sm"><input type="checkbox" checked={value} onChange={e => setIncluded(values => values.map((v, j) => j === i ? e.target.checked : v))} /> D{i + 1}</label>)}</div>
    <Readout>本窗输入 [{current.inputs.join(',')}] → 新目标 [{current.targets.join(',')}]；累计 {visited.length}/6 项。每个目标在完整回放中恰好一次。<br />{result.count ? `固定 s=2 观测：ΣNLL=${fmt(result.nll)}，N=${result.count}，token 均值=${fmt(result.mean!)}，PPL=${fmt(Math.exp(result.mean!), 4)}；文档等权均值=${fmt(result.documentMean!)}` : '无有效目标：NLL 与 PPL 未定义。'}<br />改变步幅仅重算覆盖；不同上下文的真实 NLL 必须重新运行模型，不能沿用固定观测。</Readout>
  </LabFrame>
}

export function ChatMaskLab() {
  const [onlySecond, setOnlySecond] = useState(false), [truncated, setTruncated] = useState(false), [row, setRow] = useState(0)
  const result = chatMask(onlySecond, truncated), selected = Math.min(row, result.valid.length - 1)
  const influencing = result.valid.flatMap((v, i) => v && selected <= i ? [i] : [])
  return <LabFrame title="回复目标与 prompt 梯度是两条账" hint="SYS / Q1 / Q2 的两轮教学模板">
    <Controls><Range label="选择输入行" min={0} max={result.valid.length - 1} value={selected} onChange={setRow} /><label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={onlySecond} onChange={e => setOnlySecond(e.target.checked)} />只监督第二轮</label><label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={truncated} onChange={e => setTruncated(e.target.checked)} />截断至第一个 assistant 头</label></Controls>
    <svg viewBox={`0 0 360 ${48 + result.valid.length * 38}`} className={`${pen.canvas} max-w-110!`} role="img" aria-label="每行 input、下一目标、直接监督以及选中表示的后续梯度路径">
      <text x="8" y="22">行 · input → target</text><text x="290" y="22">loss</text>
      {result.valid.map((valid, i) => <g key={i}><rect x="4" y={32 + i * 38} width="352" height="33" className={i === selected ? 'fill-accent-soft stroke-accent stroke-2' : influencing.includes(i) ? 'fill-paper stroke-info' : 'fill-paper stroke-rule'} /><text x="12" y={54 + i * 38}>{i} · {result.tokens[i]} → {result.tokens[i + 1]}</text><text x="308" y={54 + i * 38} className={valid ? pen.textA : pen.muted}>{Number(valid)}</text></g>)}
    </svg>
    <Readout>有效分母 {result.count || '0：平均未定义'} · 第 {selected} 行直接 logits 损失 {result.valid[selected] ? '有' : '无'} · 可受有效预测行 [{influencing.join(',')}] 影响。<br />独立梯度探针 hⱼ=(j+1)/10、uᵢ=mean(h≤ᵢ)、目标 1 的平方损失：∂L/∂h{selected}={fmt(result.gradient[selected], 6)}。该数用于验证因果依赖，不是 Decoder 的实测梯度。</Readout>
  </LabFrame>
}

export function LoraTraceLab() {
  const [step, setStep] = useState(0), [bothZero, setBothZero] = useState(false), [merged, setMerged] = useState(false), [target, setTarget] = useState(2)
  const result = loraTrajectory(2, bothZero)[step]
  const error = Math.max(...result.logits.map((v, i) => Math.abs(v - result.merged[i])))
  return <LabFrame title="零增量、首步梯度与合并" hint="正文 x=(1,2,3,4)，r=2，s=1，SGD η=.01">
    <Controls><Range label="已完成更新数" min={0} max={2} value={step} onChange={setStep} /><Range label="观察输出类别" min={0} max={5} value={target} onChange={setTarget} /><label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={bothZero} onChange={e => setBothZero(e.target.checked)} />A、B 都初始化为零</label><label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={merged} onChange={e => setMerged(e.target.checked)} />使用合并矩阵</label></Controls>
    <Trace label="冻结基座路径和低秩增量的数值" rows={merged ? [
      ['固定基座 + 固定 adapter → W′', 'W′ = W + BA'], ['合并输出 W′x（所选类）', fmt(result.merged[target], 6)], ['两路径输出差', fmt(error, 9)],
    ] : [
      ['冻结路径 Wx（所选类）', fmt(result.base[target], 6)], ['压缩表示 Ax', vector(result.compressed)], ['adapter 增量 B(Ax)（所选类）', fmt(result.delta[target], 6)], ['输出 Wx+B(Ax)', fmt(result.logits[target], 6)],
    ]} />
    <Readout>目标固定为下标 2 · CE={fmt(result.loss, 6)} · 当前 ∥∇A∥={fmt(result.gradA, 6)}，∥∇B∥={fmt(result.gradB, 6)}。{bothZero ? '两因子为零，所有步骤的数据梯度都为零。' : '初始 B=0，A 的数据梯度为零，B 先更新；第二次反向才让 A 获得梯度。'} 合并后禁止再加同一 adapter。</Readout>
  </LabFrame>
}

export function DuplicateGraphLab() {
  const [threshold, setThreshold] = useState(0.4), [chain, setChain] = useState(false), [selected, setSelected] = useState(0)
  const [assignments, setAssignments] = useState<Record<number, boolean>>({})
  const result = duplicateGroups(threshold, chain), points = [[70, 55], [180, 115], [290, 55], [90, 210], [270, 210]]
  const component = result.group[selected]
  const training = (group: number) => assignments[group] ?? group % 2 === 0
  return <LabFrame title="边的阈值与整组划分" hint="正文 A–E 的完整三词 shingle 复核">
    <Controls><Range label="Jaccard 阈值" min={0.1} max={1} step={0.1} value={threshold} onChange={value => { setThreshold(value); setAssignments({}) }} /><Range label="选择文档" min={0} max={4} value={selected} onChange={setSelected} /><label className="text-sm">选中整组的用途<select value={training(component) ? 'training' : 'validation'} onChange={e => setAssignments(current => ({ ...current, [component]: e.target.value === 'training' }))} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2"><option value="training">训练</option><option value="validation">验证</option></select></label><label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={chain} onChange={e => { setChain(e.target.checked); setAssignments({}) }} />独立链反例 A–B–C</label></Controls>
    <svg viewBox="0 0 360 260" className={`${pen.canvas} max-w-110!`} role="img" aria-label="Jaccard 达阈值的边及选中文档的连通分量">
      {result.edges.map(([i, j]) => <g key={`${i}-${j}`}><line x1={points[i][0]} y1={points[i][1]} x2={points[j][0]} y2={points[j][1]} className={result.group[i] === component ? pen.a : pen.axis} /><text x={(points[i][0] + points[j][0]) / 2} y={(points[i][1] + points[j][1]) / 2 - 9} textAnchor="middle" className={pen.mono}>{fmt(result.similarities[i][j], 1)}</text></g>)}
      {points.map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r="23" className={result.group[i] === component ? 'fill-accent-soft stroke-accent' : 'fill-paper stroke-rule'} strokeWidth={i === selected ? 3 : 1} /><text x={x} y={y + 5} textAnchor="middle">{'ABCDE'[i]}</text><text x={x} y={y + 40} textAnchor="middle" className={pen.muted}>{training(result.group[i]) ? '训练' : '验证'}</text></g>)}
    </svg>
    <Readout>选中组 [{result.group.flatMap((g, i) => g === component ? ['ABCDE'[i]] : []).join(',')}] · {new Set(result.group).size} 个不可拆分的组。{chain ? '链场景使用明确另设的相似度：AB=BC=.6、AC=0；同组不是两两相似。' : 'A=C 的 Jaccard=1，AB=BC=.4；阈值 .4 恢复正文组 {A,B,C}、{D}、{E}。'}<br />训练 [{result.group.flatMap((g, i) => training(g) ? ['ABCDE'[i]] : []).join(',')}]；验证 [{result.group.flatMap((g, i) => !training(g) ? ['ABCDE'[i]] : []).join(',')}]。移动选中组会移动组内全部记录。改阈值重置划分；默认按组最小 ID 的奇偶分配，初始结果与正文一致，交互变式不沿用正文 seed。</Readout>
  </LabFrame>
}

export function DdpReplayLab() {
  const [step, setStep] = useState(0), [wrong, setWrong] = useState(false)
  const result = ddpLedger(wrong, Math.min(step, 2))
  const names = ['统计 14 个有效目标', '微批 1：暂不同步', '微批 2：累积完毕', 'DDP 平均所有 rank', 'SGD：两侧同一次更新']
  return <LabFrame title="四个微批，只有一次全局更新" hint="rank 0：3+2 个 y=1；rank 1：5+4 个 y=3">
    <Controls><Range label="执行阶段" min={0} max={4} value={step} onChange={setStep} /><label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={wrong} onChange={e => setWrong(e.target.checked)} />错误：各 rank 先取局部均值</label></Controls>
    <Trace active={Math.min(step, 3)} label={names[step]} rows={[
      ['有效数计数归约', 'rank0:5 + rank1:9 = N:14'], ['局部梯度和（当前已处理微批）', `${fmt(result.localSums[0])} / ${fmt(result.localSums[1])}`], [wrong ? '错误局部分母 5 / 9' : '本地 sum × R/N', `${fmt(result.local[0], 6)} / ${fmt(result.local[1], 6)}`], ['同步平均 → SGD 参数', step >= 3 ? `${fmt(result.gradient, 6)} → ${step === 4 ? fmt(result.theta, 6) : '尚未 step()'}` : '等待所有微批，不提前更新'],
    ]} />
    <Readout>阶段：{names[step]}。单进程参考 g={fmt(result.reference, 6)}，θ₁={fmt(-0.1 * result.reference, 6)}。{step >= 3 ? `当前偏差 ${fmt(result.gradient - result.reference, 6)}` : '未完成累积的梯度不能与完整更新比较。'} 参数同步不能证明分母正确。</Readout>
  </LabFrame>
}

export function StateAllocationLab() {
  const [ranks, setRanks] = useState(4), [layout, setLayout] = useState(0)
  const bytes = stateAllocation(ranks), names = ['DDP', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3 / FSDP']
  return <LabFrame title="相同参数坐标，由谁保存哪一份状态" hint="十亿参数；参数 2B、梯度 2B、主参数+m+v 共 12B">
    <Controls><Range label="rank 数" min={1} max={8} value={ranks} onChange={setRanks} /><Range label="状态布局" min={0} max={3} value={layout} onChange={setLayout} /></Controls>
    <svg viewBox="0 0 360 242" className={`${pen.canvas} max-w-110!`} role="img" aria-label="各分片阶段每 rank 的常驻模型状态字节数">{bytes.map((value, i) => <g key={i}><text x="8" y={24 + i * 56}>{names[i]} · {fmt(value, 2)} GB/rank</text><rect x="8" y={34 + i * 56} width={value / 16 * 336} height="22" className={layout === i ? 'fill-accent stroke-accent' : 'fill-accent-soft stroke-rule'} /></g>)}</svg>
    <Readout>{names[layout]}：{fmt(bytes[layout], 3)} GB/rank；所有 rank 常驻账合计 {fmt(bytes[layout] * ranks, 3)} GB。这是指定精度的模型状态，不含激活、临时参数 gather、通信缓冲和 workspace。选中布局仍对每个坐标执行相同的全局梯度更新。</Readout>
  </LabFrame>
}
