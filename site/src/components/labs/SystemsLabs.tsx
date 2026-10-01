import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, fmt, pen, useWidth } from './Lab'
import { onlineAttention, packedRows, parallelMatrix, quantizedWeights, resourceLedger, roofline, serviceMetrics, speculativeTrace } from '../../lib/systems-interactive-model'

export function LedgerLab() {
  const [batch, setBatch] = useState(1), [length, setLength] = useState(5), [large, setLarge] = useState(false)
  const r = resourceLedger(batch, length, large), scale = Math.max(r.weights, r.kv)
  return <LabFrame title="容量、读取与运算分别记账">
    <Controls><Range label="请求数 B" value={batch} min={1} max={16} onChange={setBatch} /><Range label="已处理长度 T" value={length} min={1} max={large ? 16384 : 16} onChange={setLength} />
      <label className="text-sm"><input type="checkbox" checked={large} onChange={e => { setLarge(e.target.checked); setLength(e.target.checked ? 8192 : 5) }} /> Llama 3 8B 形状</label></Controls>
    <svg viewBox="0 0 280 150" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`权重 ${r.weights} 字节，KV ${r.kv} 字节`}>
      {[['权重容量', r.weights], ['KV 容量', r.kv]].map(([name, value], i) => <g key={name}><text x="8" y={24 + i * 66}>{name}</text><rect x="8" y={34 + i * 66} width={260 * Number(value) / scale} height="20" className={i ? 'fill-accent2' : 'fill-accent'} /></g>)}
    </svg>
    <Readout>容量：权重 {r.weights.toLocaleString()} B；KV {r.kv.toLocaleString()} B<br />当步逻辑读取：权重 + KV = {r.read.toLocaleString()} B；新增 KV 写入 {r.write.toLocaleString()} B<br />decode：线性 {r.linearFlops.toLocaleString()} + 注意力 {r.attentionFlops.toLocaleString()} FLOP<br />同长度 causal prefill 主项：{r.prefillFlops.toLocaleString()} FLOP</Readout>
  </LabFrame>
}

export function RooflineLab() {
  const [batch, setBatch] = useState(1), [length, setLength] = useState(8192)
  const r = roofline(batch, length), x = (i: number) => 30 + Math.log10(Math.max(0.1, i) / 0.1) / 4 * 230
  const y = (t: number) => 205 - t / 1000 * 165
  return <LabFrame title="同一 decode 负载在屋顶线上的位置" hint="规格理论上限；不含所有算子与额外流量">
    <Controls><Range label="B" value={batch} min={1} max={512} onChange={setBatch} /><Range label="T" value={length} min={1} max={16384} onChange={setLength} /></Controls>
    <svg viewBox="0 0 280 245" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`强度 ${fmt(r.intensity)} FLOP/byte，理论上限 ${fmt(r.ceiling)} TFLOP/s`}>
      <path d="M30 25V205H264" className={pen.axis} /><path d={`M${x(0.1)} ${y(0.335)}L${x(r.ridge)} ${y(989.5)}H260`} className={pen.a} />
      <circle cx={x(r.intensity)} cy={y(r.ceiling)} r="5" className="fill-accent2" /><line x1={x(r.ridge)} y1="35" x2={x(r.ridge)} y2="205" className={pen.guide} />
      {[0.1, 1, 10, 100, 1000].map(i => <text key={i} x={x(i)} y="225" textAnchor="middle" className={pen.mono}>{i}</text>)}<text x="35" y="20">TFLOP/s</text><text x="140" y="243" textAnchor="middle">FLOP/byte · 对数横轴</text>
    </svg>
    <Readout>I={fmt(r.intensity)}；脊点={fmt(r.ridge)}；上限={fmt(r.ceiling)} TFLOP/s<br />计算下界 {fmt(r.tCompute * 1000)} ms；搬运下界 {fmt(r.tMemory * 1000)} ms；取较大值 {fmt(Math.max(r.tCompute, r.tMemory) * 1000)} ms</Readout>
  </LabFrame>
}

export function OnlineSoftmaxLab() {
  const [position, setPosition] = useState(5), [block, setBlock] = useState(3), [step, setStep] = useState(0)
  const r = onlineAttention(position, block, step), s = r.state
  return <LabFrame title="在线 softmax：共同基准与重缩放">
    <Controls><Range label="查询绝对位置" value={position} min={3} max={5} onChange={setPosition} /><Range label="键块大小" value={block} min={1} max={6} onChange={v => { setBlock(v); setStep(0) }} /><Range label="已读块数" value={step} min={0} max={Math.ceil(6 / block)} onChange={setStep} /></Controls>
    <svg viewBox="0 0 280 170" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`已读 ${step} 块，m=${s.m}, l=${s.l}, u=${s.u}`}>
      {Array.from({ length: 6 }, (_, j) => <g key={j}><rect x={8 + j * 45} y="25" width="39" height="42" className={j > position ? 'fill-sunken stroke-rule [stroke-dasharray:3_3]' : j < step * block ? 'fill-accent/20 stroke-accent' : 'fill-paper stroke-rule'} /><text x={27 + j * 45} y="43" textAnchor="middle">k{j}</text><text x={27 + j * 45} y="59" textAnchor="middle">v={j + 1}</text></g>)}
      <text x="8" y="100">旧基准 {r.previous.m === -Infinity ? '−∞' : r.previous.m} → 新基准 {s.m === -Infinity ? '−∞' : s.m}</text><text x="8" y="125">旧 l、u × {fmt(s.factor, 6)}</text><text x="8" y="150">新块贡献 → 共同分母 → 输出</text>
    </svg>
    <Readout>m（正文 a）={s.m === -Infinity ? '−∞' : s.m}；l={fmt(s.l, 6)}；u={fmt(s.u, 6)}<br />o=u/l：{r.output === null ? '未定义（尚无可见键）' : fmt(r.output, 6)}；整行参照 {fmt(r.dense, 6)}<br />当前块可见键：{s.end > s.start ? `${s.start}–${s.end - 1}` : '空块，不进入分母'}</Readout>
  </LabFrame>
}

export function PackedExecutionLab() {
  const [decode, setDecode] = useState(false), [selected, setSelected] = useState(0), [stage, setStage] = useState(0)
  const rows = packedRows(decode), row = rows[Math.min(selected, rows.length - 1)]
  const [ref, width] = useWidth<HTMLDivElement>(), columns = width < 500 ? 3 : 5
  return <LabFrame title="token、逻辑位置、物理槽与 Graph 缓冲区">
    <div ref={ref}><Controls><label className="text-sm"><input type="checkbox" checked={decode} onChange={e => { setDecode(e.target.checked); setSelected(0) }} /> 下一轮 decode</label><Range label="packed 下标" value={Math.min(selected, rows.length - 1)} min={0} max={rows.length - 1} onChange={setSelected} /><Range label="执行阶段" value={stage} min={0} max={3} onChange={setStage} format={v => ['准备', 'eager', 'capture', 'replay'][v]} /></Controls>
      <svg viewBox={`0 0 ${columns * 88} ${Math.ceil(rows.length / columns) * 78 + 20}`} className={pen.canvas} role="img" aria-label={`选择请求 ${row.request} 的 ID ${row.id}，逻辑位置 ${row.position}，写槽 ${row.slot}`}>
        {rows.map((r, i) => <g key={i} transform={`translate(${(i % columns) * 88 + 4},${Math.floor(i / columns) * 78 + 8})`}><rect width="80" height="66" className={i === row.index ? 'fill-accent/20 stroke-accent stroke-2' : 'fill-sunken stroke-rule'} /><text x="6" y="18">{r.request} · ID {r.id}</text><text x="6" y="37">p={r.position}</text><text x="6" y="55">slot={r.slot}</text></g>)}
      </svg></div>
    <Readout>packed[{row.index}] → {row.request}；RoPE 位置 {row.position}；可读本请求位置 0–{row.position}<br />block table=[{row.blocks.join(', ')}]；{row.blocks[Math.floor(row.position / 4)]}×4+{row.position % 4}={row.slot}<br />{stage < 2 ? '本轮按有效长度准备并执行' : !decode ? '该固定快照的 prefill 使用 eager；本阶段不能捕获/重放 prefill' : `固定地址 buf_ids / buf_positions / buf_slots；Graph 4 行，有效 3 行；padding 行 slot=-1；${stage === 2 ? '捕获固定运算依赖' : '复制新值后 replay，仅前三行采样'}`}</Readout>
  </LabFrame>
}

export function QuantizationLab() {
  const [bits, setBits] = useState(3), [group, setGroup] = useState(8), [outlier, setOutlier] = useState(20), [asymmetric, setAsymmetric] = useState(false)
  const r = quantizedWeights(bits, group, outlier, asymmetric)
  return <LabFrame title="同一矩阵：编码、恢复、输出和存储">
    <Controls><Range label="位宽" value={bits} min={2} max={8} onChange={setBits} /><label className="text-sm">共享 scale 的元素数<select className="block h-9 w-full border border-rule bg-paper" value={group} onChange={e => setGroup(Number(e.target.value))}><option value={8}>全矩阵 8</option><option value={4}>每行 4</option><option value={2}>每组 2</option></select></label><Range label="第二行异常值幅度" value={outlier} min={1} max={20} onChange={setOutlier} /><label className="text-sm"><input type="checkbox" checked={asymmetric} onChange={e => setAsymmetric(e.target.checked)} /> 非对称无符号码</label></Controls>
    <svg viewBox="0 0 280 252" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`原值与恢复值，输出误差 ${r.output.map((v, i) => fmt(v - r.reference[i])).join(', ')}`}>
      {r.original.map((v, i) => <g key={i}><text x="6" y={24 + i * 26}>w{i}</text><line x1="42" y1={20 + i * 26} x2="264" y2={20 + i * 26} className={pen.grid} /><circle cx={153 + v / outlier * 105} cy={20 + i * 26} r="4" className="fill-accent" /><path d={`M${153 + r.reconstructed[i] / outlier * 105} ${15 + i * 26}v10`} className={pen.b} /></g>)}<text x="6" y="233">● 原值；虚线 恢复值；同一横轴</text>
    </svg>
    <Readout>码=[{r.codes.join(', ')}]；scale=[{r.scales.map(v => fmt(v, 4)).join(', ')}]；zero=[{r.zeros.join(', ')}]<br />Wx=[{r.reference.map(v => fmt(v)).join(', ')}]；恢复输出=[{r.output.map(v => fmt(v)).join(', ')}]<br />误差=[{r.output.map((v, i) => fmt(v - r.reference[i])).join(', ')}]；紧密打包码 + fp16 scale{asymmetric ? ' + uint8 zero' : ''}={r.bytes} B；原 bf16=16 B</Readout>
  </LabFrame>
}

export function CollectiveLab() {
  const [mode, setMode] = useState<'column' | 'row'>('column'), [step, setStep] = useState(0)
  const r = parallelMatrix(mode)
  return <LabFrame title="两张卡算同一矩阵">
    <Controls><label className="text-sm">切分轴<select value={mode} onChange={e => setMode(e.target.value as 'column' | 'row')} className="block h-9 w-full border border-rule bg-paper"><option value="column">输出列：拼接</option><option value="row">输入行：部分和</option></select></label><Range label="步骤" value={step} min={0} max={2} onChange={setStep} format={v => ['分发', '局部计算', '合并'][v]} /></Controls>
    <svg viewBox="0 0 280 200" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`${mode} 切分，两 rank 输出 ${r.output.join(', ')}`}>
      {[0, 1].map(rank => <g key={rank}><rect x="8" y={8 + rank * 72} width="264" height="60" className="fill-sunken stroke-rule" /><text x="18" y={30 + rank * 72}>rank {rank}：{mode === 'column' ? `输出坐标 ${rank * 2},${rank * 2 + 1}` : `输入坐标 ${rank * 2},${rank * 2 + 1}`}</text><text x="18" y={53 + rank * 72} className={pen.mono}>{step ? r.local[rank].join(', ') : '取得权重切片与所需输入'}</text></g>)}<text x="8" y="182" className={pen.textA}>{step === 2 ? `${mode === 'column' ? '拼接' : '相加'} → ${r.output.join(', ')}` : '局部结果保持 rank 身份'}</text>
    </svg><Readout>完整 xW=[{r.reference.join(', ')}]；合并误差={fmt(Math.max(...r.output.map((v, i) => Math.abs(v - r.reference[i]))))}<br />两 rank 都要完整输出时：{mode === 'column' ? 'all-gather' : 'all-reduce'}；每 rank 发/收各 {r.sentBytes} B（半精度，教学两卡）</Readout>
  </LabFrame>
}

export function ServiceTimelineLab() {
  const [request, setRequest] = useState(0), [event, setEvent] = useState(0), [ttft, setTtft] = useState(0.7), [tpot, setTpot] = useState(0.2), [total, setTotal] = useState(1)
  const all = serviceMetrics(ttft, tpot, total), r = all.requests[request]
  const selected = Math.min(event, r.events.length + 1)
  const time = selected === 0 ? r.arrival : selected === r.events.length + 1 ? r.end : r.events[selected - 1][0]
  return <LabFrame title="从客户端事件读出延迟和合格产出" hint="正文固定轨迹；不是实机测量">
    <Controls><Range label="请求" value={request} min={0} max={3} onChange={v => { setRequest(v); setEvent(0) }} format={v => 'ABCD'[v]} /><Range label="事件" value={selected} min={0} max={r.events.length + 1} onChange={setEvent} format={v => v === 0 ? '发送' : v > r.events.length ? (r.complete ? '完成' : '取消') : `输出 ${v}`} /><Range label="TTFT 上限 / s" value={ttft} min={0.1} max={1.5} step={0.1} onChange={setTtft} /><Range label="TPOT 上限 / s" value={tpot} min={0.05} max={0.5} step={0.05} onChange={setTpot} /><Range label="总延迟上限 / s" value={total} min={0.5} max={2} step={0.1} onChange={setTotal} /></Controls>
    <svg viewBox="0 0 280 195" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`选择请求 ${r.name} 时刻 ${time} s，是否满足 SLO ${r.qualified}`}>
      <line x1={32 + time * 150} x2={32 + time * 150} y1="4" y2="175" className={pen.guide} />
      {all.requests.map((row, i) => <g key={row.name}><text x="4" y={28 + i * 42}>{row.name}</text><line x1={32 + row.arrival * 150} x2={32 + row.end * 150} y1={23 + i * 42} y2={23 + i * 42} className={row.qualified ? pen.a : pen.axis} />{row.events.map((e, ei) => <g key={ei}><circle cx={32 + e[0] * 150} cy={23 + i * 42} r={request === i && selected === ei + 1 ? 6 : 4} className="fill-accent2" /><text x={32 + e[0] * 150} y={40 + i * 42} textAnchor="middle" className={pen.mono}>+{e[1]}</text></g>)}{!row.complete && <text x={32 + row.end * 150} y={27 + i * 42}>×</text>}</g>)}<text x="32" y="192">0</text><text x="257" y="192" textAnchor="end">1.5 s</text>
    </svg><Readout>{r.name}：所选事件 t={fmt(time, 2)} s；TTFT={r.ttft === null ? '未定义' : fmt(r.ttft)}；ITL=[{r.itl.map(v => fmt(v)).join(', ')}]；TPOT={r.tpot === null ? '未定义' : fmt(r.tpot)}<br />总延迟={fmt(r.total)} s；{r.complete ? (r.qualified ? 'SLO 合格' : 'SLO 不合格') : '取消，仍进入 4 条到达分母'}<br />合格 {all.requests.filter(v => v.qualified).length}/4 到达；goodput={fmt(all.goodput)} 请求/s；{fmt(all.tokenGoodput)} token/s（窗口 1.5 s）</Readout>
  </LabFrame>
}

export function SpeculativeReplayLab() {
  const [step, setStep] = useState(0), [scenario, setScenario] = useState(1)
  const draws = [[0.9, 0.5], [0.5, 0.9], [0.5, 0.5]][scenario], r = speculativeTrace(draws[0], draws[1], step)
  return <LabFrame title="两步验证与 KV 提交边界">
    <Controls><label className="text-sm">抽数轨迹<select value={scenario} onChange={e => { setScenario(Number(e.target.value)); setStep(0) }} className="block h-9 w-full border border-rule bg-paper"><option value={0}>第一项拒绝</option><option value={1}>第二项拒绝</option><option value={2}>全部接受</option></select></label><Range label="验证步骤" value={step} min={0} max={3} onChange={setStep} format={v => ['初始历史', '验证第 1 项', '验证第 2 项', '提交/补偿'][v]} /></Controls>
    <svg viewBox="0 0 280 145" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`提交 ID ${r.submitted.join(', ')}；有效 KV ${r.cached.join(', ')}`}>
      {[['ID', r.submitted], ['KV', r.cached]].map(([label, values], row) => <g key={String(label)}><text x="4" y={35 + row * 65}>{String(label)}</text>{(values as number[]).map((v, i) => <g key={i}><rect x={38 + i * 44} y={12 + row * 65} width="36" height="35" className={row ? 'fill-accent/20 stroke-accent' : 'fill-accent2/15 stroke-accent2'} /><text x={56 + i * 44} y={35 + row * 65} textAnchor="middle">{v}</text></g>)}</g>)}
    </svg><Readout>草稿=[0,2]；接受阈值=[5/6,2/3]；固定抽数=[{draws.join(', ')}]<br />已接受 {r.accepted} 项；{r.rejected ? '首次拒绝后停止，残差只含 ID 1' : '按真实已接受历史继续'}<br />有效 KV=[{r.cached.join(', ')}]；下轮待输入={r.pending === null ? '尚未提交新末项' : r.pending}；截断的是验证路径，已提交输出不回滚</Readout>
  </LabFrame>
}

export function PrefixIdentityLab() {
  const [different, setDifferent] = useState(false), [namespace, setNamespace] = useState(false)
  const same = !different && !namespace
  return <LabFrame title="相同局部块，是否有相同前缀身份？">
    <Controls><label className="text-sm"><input type="checkbox" checked={different} onChange={e => setDifferent(e.target.checked)} /> B 的第一块不同</label><label className="text-sm"><input type="checkbox" checked={namespace} onChange={e => setNamespace(e.target.checked)} /> B 使用不同模型/位置规则</label></Controls>
    <svg viewBox="0 0 280 175" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`第二块 token 相同；完整前缀身份${same ? '相同' : '不同'}`}>
      {['A', 'B'].map((name, i) => <g key={name}><text x="4" y={30 + i * 80}>{name}</text><rect x="28" y={8 + i * 80} width="108" height="50" className="fill-sunken stroke-rule" /><text x="82" y={30 + i * 80} textAnchor="middle">{i && different ? '9,2,3,4' : '1,2,3,4'}</text><text x="82" y={49 + i * 80} textAnchor="middle">prefix h₀</text><path d={`M140 ${33 + i * 80}h18`} className={pen.axis} /><rect x="162" y={8 + i * 80} width="110" height="50" className={same ? 'fill-accent/15 stroke-accent' : 'fill-accent2/15 stroke-accent2'} /><text x="217" y={30 + i * 80} textAnchor="middle">5,6,7,8</text><text x="217" y={49 + i * 80} textAnchor="middle">hash(h₀,u₁)</text></g>)}
    </svg><Readout>相同局部 ID ≠ 相同计算上下文。{same ? '命名空间与完整前缀一致，可作为候选命中' : '前缀或命名空间不同，不可共享本块 KV'}。图使用完整前缀相等性判断；不模拟哈希值或假设哈希无碰撞。</Readout>
  </LabFrame>
}
