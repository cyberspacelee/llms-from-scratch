import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { dpoLedger, frequencyCache, grpoGroup, hybridBoundary, mlaPaths, moeDispatch, multimodalPatch, sparseSelection, stateRecurrence } from '../../lib/advanced-interactive-model'

export function MoeDispatchLab() {
  const [step, setStep] = useState(0), [token, setToken] = useState(0), [crowded, setCrowded] = useState(false), [capacity, setCapacity] = useState(2), [drop, setDrop] = useState(false)
  const r = moeDispatch(crowded, capacity, drop), routes = r.routes.filter(v => v.token === token)
  return <LabFrame title="三行输入分发后仍回到原行">
    <Controls><Range label="阶段" value={step} min={0} max={3} onChange={setStep} format={v => ['路由', 'dispatch', '专家计算', '加权 gather'][v]} /><Range label="原 token 行" value={token} min={0} max={2} onChange={setToken} /><Range label="专家容量" value={capacity} min={1} max={3} onChange={setCapacity} /><label className="text-sm"><input type="checkbox" checked={crowded} onChange={e => setCrowded(e.target.checked)} /> 全部选择专家 0、1</label><label className="text-sm"><input type="checkbox" checked={drop} onChange={e => setDrop(e.target.checked)} /> 丢弃超容量贡献</label></Controls>
    <svg viewBox="0 0 280 250" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`专家命中 ${r.counts.join(', ')}，原行 ${token} 的输出 ${r.output[token].join(', ')}`}>
      {[0, 1, 2, 3].map(expert => <g key={expert}><rect x="8" y={8 + expert * 58} width="264" height="48" className={routes.some(v => v.expert === expert) ? 'fill-accent/15 stroke-accent' : 'fill-sunken stroke-rule'} /><text x="16" y={27 + expert * 58}>专家 {expert} · {r.counts[expert]} 次调用</text><text x="16" y={46 + expert * 58} className={pen.mono}>{r.routes.filter(v => v.expert === expert).map(v => `行${v.token}${v.dropped ? '×' : v.overflow ? '待' : ''}`).join(' / ')}</text></g>)}
    </svg><Readout>token {token} 输入=[{r.inputs[token].join(', ')}]<br />{routes.map(v => `专家${v.expert}：${step < 2 ? '持有原行号' : `[${v.expertOutput.join(',')}]`} × ${fmt(v.weight)}${v.dropped ? '（已丢弃）' : v.overflow ? '（额外批次等待后执行）' : ''}`).join('；')}<br />{step === 3 ? `gather 回原行=[${r.output[token].map(v => fmt(v)).join(', ')}]；未丢弃参照=[${r.reference[token].map(v => fmt(v)).join(', ')}]` : '返回时保留原行号，两个贡献相加'}</Readout>
  </LabFrame>
}

export function MlaPathsLab() {
  const [head, setHead] = useState(0), [absorbed, setAbsorbed] = useState(false), [step, setStep] = useState(0)
  const r = mlaPaths(head)
  return <LabFrame title="同一 MLA 参数，两条相等的路径">
    <Controls><Range label="头" value={head} min={0} max={1} onChange={setHead} /><Range label="步骤" value={step} min={0} max={3} onChange={setStep} format={v => ['缓存', '内容分数', 'softmax', '值输出'][v]} /><label className="text-sm"><input type="checkbox" checked={absorbed} onChange={e => setAbsorbed(e.target.checked)} /> 吸收投影</label></Controls>
    <svg viewBox="0 0 280 240" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`头 ${head}，${absorbed ? '吸收' : '显式'}路径，同一注意力输出`}>
      {r.latent.map((c, j) => <g key={j}><rect x="8" y={8 + j * 62} width="264" height="54" className="fill-sunken stroke-rule" /><text x="16" y={28 + j * 62}>位置 {j}：缓存 c=[{c.join(', ')}]</text><text x="16" y={48 + j * 62} className={pen.mono}>{step === 0 ? '另存共享位置键，2 维' : step === 1 ? `${absorbed ? 'q̂·c' : 'q·Uₖc'}=${r.explicitScores[j]}` : `a${j}=${fmt(r.weights[j], 6)}`}</text></g>)}<text x="8" y="220">{absorbed ? '先 Σ a·c，再 Uᵥ 投影' : '先恢复 Uᵥc，再 Σ a·v'}</text>
    </svg><Readout>q=[{r.query.join(', ')}]；Uₖᵀq=[{r.projected.join(', ')}]<br />内容=[{r.explicitScores.join(', ')}]；位置=[{r.positional.join(', ')}]；softmax 前仍除以 2<br />{step === 3 ? `显式=[${r.explicit.map(v => fmt(v, 6)).join(', ')}]；吸收=[${r.absorbed.map(v => fmt(v, 6)).join(', ')}]；最大误差=${fmt(Math.max(...r.explicit.map((v, i) => Math.abs(v - r.absorbed[i]))), 9)}` : '当前头独立计算权重与潜在混合，不跨头共享 z'}<br />每位置持久缓存：c 2 + 位置键 2=4 元素；显式全头 K/V 与位置键=10 元素</Readout>
  </LabFrame>
}

export function DpoLedgerLab() {
  const [lambda, setLambda] = useState(0.2), [eos, setEos] = useState(true), [selected, setSelected] = useState(0)
  const r = dpoLedger(lambda, eos)
  return <LabFrame title="逐 token 回复账目与固定参考差">
    <Controls><Range label="λ DPO" value={lambda} min={0.05} max={2} step={0.05} onChange={setLambda} /><Range label="所选预测行" value={selected} min={0} max={3} onChange={setSelected} format={v => ['提示内 ?', '首回复', 'EOS', 'padding'][v]} /><label className="text-sm"><input type="checkbox" checked={eos} onChange={e => setEos(e.target.checked)} /> EOS 属于回复</label></Controls>
    <svg viewBox="0 0 280 190" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`DPO margin ${fmt(r.margin)}，损失 ${fmt(r.loss)}`}>
      {['策略 chosen', '策略 rejected', '参考 chosen', '参考 rejected'].map((name, i) => <g key={name}><text x="8" y={22 + i * 43}>{name}</text><line x1="8" x2={8 + -r.logs[i] * 120} y1={32 + i * 43} y2={32 + i * 43} className={i < 2 ? pen.a : pen.b} /><text x="205" y={32 + i * 43} className={pen.mono}>{fmt(r.logs[i])}</text></g>)}
    </svg><Readout>预测行 {selected}：{selected === 1 ? '目标 4 或 5，计入回复' : selected === 2 ? (eos ? '目标 EOS，计入回复' : 'EOS 排除，已改变目标 span') : '不进入回复 log-prob 和；仍可能通过后续位置影响梯度'}<br />d=(logπ+−logπ−)−(logref+−logref−)={fmt(r.difference, 6)}<br />z={fmt(r.margin, 6)}；loss={fmt(r.loss, 6)}；chosen log-prob 偏导={fmt(r.gradient, 6)}；rejected 偏导={fmt(-r.gradient, 6)}</Readout>
  </LabFrame>
}

export function FrequencyCacheLab() {
  const [query, setQuery] = useState(5), [key, setKey] = useState(4), [scale, setScale] = useState(2), [recompute, setRecompute] = useState(false)
  const r = frequencyCache(query, Math.min(key, query), scale)
  const end = (angle: number) => [140 + Math.cos(angle) * 85, 115 - Math.sin(angle) * 85]
  const mixed = end(r.oldAngle), correct = end(r.newAngle)
  return <LabFrame title="频率改了，旧缓存仍是旧规则">
    <Controls><Range label="查询位置 i" value={query} min={1} max={8} onChange={setQuery} /><Range label="历史键 j" value={Math.min(key, query)} min={0} max={query} onChange={setKey} /><Range label="位置插值因子" value={scale} min={1} max={4} step={0.25} onChange={setScale} /><label className="text-sm"><input type="checkbox" checked={recompute} onChange={e => setRecompute(e.target.checked)} /> 用新规则重算完整前缀</label></Controls>
    <svg viewBox="0 0 280 230" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`混用分数 ${r.mixedScore}，新规则参照 ${r.correctScore}`}><circle cx="140" cy="115" r="85" className={pen.axis} /><path d={`M140 115L${correct.join(' ')}`} className={pen.a} /><path d={`M140 115L${(recompute ? correct : mixed).join(' ')}`} className={pen.b} /><text x="8" y="225">实线：新规则；虚线：实际缓存路径</text></svg>
    <Readout>ω=π/4；旧键+新查询角差={fmt(r.oldAngle)}；完整新规则角差={fmt(r.newAngle)}<br />实际分数={fmt(recompute ? r.correctScore : r.mixedScore, 6)}；参照={fmt(r.correctScore, 6)}；误差={fmt(recompute ? 0 : r.mixedScore - r.correctScore, 6)}<br />本图用 q=k=(1,0) 隔离单块角度；多层模型须重算前缀隐藏状态，不能只重旋转旧键</Readout>
  </LabFrame>
}

export function GrpoLab() {
  const [equal, setEqual] = useState(false), [ratio, setRatio] = useState(1.4), [epsilon, setEpsilon] = useState(0.2), [selected, setSelected] = useState(0), [step, setStep] = useState(0)
  const r = grpoGroup(equal, ratio, epsilon, selected)
  return <LabFrame title="奖励、优势、概率比与裁剪">
    <Controls><Range label="步骤" value={step} min={0} max={3} onChange={setStep} format={v => ['奖励', '组内优势', '概率比', '受限更新'][v]} /><Range label="回复编号" value={selected} min={0} max={3} onChange={setSelected} /><Range label="ρ" value={ratio} min={0.4} max={1.6} step={0.05} onChange={setRatio} /><Range label="ε clip" value={epsilon} min={0.05} max={0.4} step={0.05} onChange={setEpsilon} /><label className="text-sm"><input type="checkbox" checked={equal} onChange={e => setEqual(e.target.checked)} /> 四条全部无奖励</label></Controls>
    <svg viewBox="0 0 280 230" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`选择回复 ${selected}，优势 ${r.advantage}，最小化损失的导数 ${r.lossDerivativeLogRatio}`}>
      <line x1="140" y1="15" x2="140" y2="205" className={pen.axis} />{r.rewards.map((reward, i) => <g key={i}><text x="8" y={35 + i * 48}>回复 {i}</text><rect x={step && r.advantages[i] < 0 ? 60 : 140} y={15 + i * 48} width={Math.abs(step ? r.advantages[i] : reward) * 80} height="25" className={i === selected ? 'fill-accent' : 'fill-info/40'} /><text x="235" y={34 + i * 48}>{step ? r.advantages[i] : reward}</text></g>)}<text x="8" y="226">0 轴两侧保留优势的正负方向</text>
    </svg><Readout>均值={fmt(r.mean)}；总体标准差={fmt(r.std)}；A=[{r.advantages.join(', ')}]<br />ρA={fmt(r.raw)}；clip(ρ)A={fmt(r.restricted)}；min={fmt(r.objective)}<br />{step === 3 ? `最小化负目标：对 logρ 的偏导=${fmt(r.lossDerivativeLogRatio)}；${r.lossDerivativeLogRatio === 0 ? '此样本无推动（零优势或裁剪饱和）' : r.lossDerivativeLogRatio < 0 ? '梯度下降提高该回复概率比' : '梯度下降降低该回复概率比'}` : '先确定优势符号，再对两个乘积取 min'}</Readout>
  </LabFrame>
}

export function StateScanLab() {
  const [step, setStep] = useState(0), [selective, setSelective] = useState(false), [retention, setRetention] = useState(0.5), [scan, setScan] = useState(false)
  const rows = stateRecurrence(selective, retention), r = rows[step]
  return <LabFrame title="同一四步递推与仿射组合">
    <Controls><Range label="t" value={step} min={0} max={3} onChange={setStep} /><Range label="保持系数 a" value={retention} min={0} max={1.2} step={0.1} onChange={setRetention} /><label className="text-sm"><input type="checkbox" checked={selective} onChange={e => setSelective(e.target.checked)} /> 负输入保持旧状态且不写入</label><label className="text-sm"><input type="checkbox" checked={scan} onChange={e => setScan(e.target.checked)} /> 仿射组合路径</label></Controls>
    <svg viewBox="0 0 280 230" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`t=${step} 状态 ${r.state}，扫描参照 ${r.cumulativeB}`}>
      {rows.map((row, i) => <g key={i}><rect x="8" y={8 + i * 53} width="264" height="45" className={i === step ? 'fill-accent/15 stroke-accent' : 'fill-sunken stroke-rule'} /><text x="16" y={26 + i * 53}>t{i} · u={row.input} · (a,b)=({fmt(row.a, 1)},{row.b})</text><text x="16" y={44 + i * 53} className={pen.mono}>{scan ? `累计 (A,B)=(${fmt(row.cumulativeA)},${fmt(row.cumulativeB)})` : `${fmt(row.decayed)} + ${row.b} = ${fmt(row.state)}`}</text></g>)}
    </svg><Readout>旧状态={fmt(r.previous)}；衰减={fmt(r.decayed)}；新写入={r.b}；状态={fmt(r.state)}<br />(a₂,b₂)∘(a₁,b₁)=(a₂a₁,a₂b₁+b₂)；累计映射 s=A·0+B={fmt(r.cumulativeB)}<br />串行与仿射组合误差={fmt(r.state - r.cumulativeB, 9)}；本图不测并行扫描速度</Readout>
  </LabFrame>
}

export function HybridBoundaryLab() {
  const [boundary, setBoundary] = useState(4), [full, setFull] = useState(true), [window, setWindow] = useState(true), [recursive, setRecursive] = useState(true)
  const r = hybridBoundary(boundary, full, window, recursive)
  const componentNames = ['FULL', 'SWA', '递归快照'], enabled = [full, window, recursive]
  return <LabFrame title="混合缓存的共同安全边界" hint="独立扩展示例：8 项前缀；SWA 窗口 3">
    <Controls><Range label="候选已处理边界 b" value={boundary} min={0} max={8} onChange={setBoundary} />{componentNames.map((name, i) => <label key={name} className="text-sm"><input type="checkbox" checked={enabled[i]} onChange={e => [setFull, setWindow, setRecursive][i](e.target.checked)} /> {name}</label>)}</Controls>
    <svg viewBox="0 0 280 235" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`候选边界 ${boundary}，各组件投票 ${r.votes.join(', ')}，可复用 ${r.valid}`}>
      {componentNames.map((name, row) => <g key={name}><text x="8" y={22 + row * 70}>{name} · {enabled[row] ? (r.votes[row] ? '通过' : '拒绝') : '未启用'}</text>{Array.from({ length: 9 }, (_, i) => { const available = row === 0 ? i < 8 : row === 1 ? r.windowAvailable.includes(i) : r.checkpoints.includes(i); return <g key={i}><rect x={8 + i * 29} y={30 + row * 70} width="24" height="25" className={available ? 'fill-accent/20 stroke-accent' : 'fill-sunken stroke-rule [stroke-dasharray:3_3]'} /><text x={20 + i * 29} y={48 + row * 70} textAnchor="middle">{i}</text></g> })}</g>)}<text x="8" y="230">FULL/SWA：token 下标；快照：处理边界</text>
    </svg><Readout>FULL 需要 [0,{boundary})；SWA 需要最近 {'{' + r.requiredWindow.join(', ') + '}'}；递归状态需要恰好 b={boundary} 的快照<br />共同结果：{r.valid ? '可从此边界继续' : '不能复用此边界，应查更早的独立候选或重算'}<br />本章两层模型实际保留：递归矩阵 4 个元素 + 每位置 KV 3 个元素；8 位置时共 28 元素。图中 SWA 是新增第三种组件，非原两层模型</Readout>
  </LabFrame>
}

export function SparseSelectionLab() {
  const [selected, setSelected] = useState([0, 3, 7])
  const r = sparseSelection(selected)
  return <LabFrame title="候选集合改变分母与输出">
    <Controls><label className="text-sm">固定 indexer 场景<select value={selected.join(',')} onChange={e => setSelected(e.target.value ? e.target.value.split(',').map(Number) : [])} className="block h-9 w-full border border-rule bg-paper"><option value="0,3,7">top-3：0、3、7</option><option value="0,4,7">漏读 3：0、4、7</option><option value="5,6,7">最近窗口</option><option value="0,1,2,3,4,5,6,7">全部读取</option><option value="">清空集合</option></select></label></Controls>
    <div className="mt-3 grid grid-cols-4 gap-2">{r.weights.map((_, j) => <label key={j} className="text-sm"><input type="checkbox" checked={selected.includes(j)} onChange={e => setSelected(e.target.checked ? [...selected, j].sort((a, b) => a - b) : selected.filter(v => v !== j))} /> 键 {j}</label>)}</div>
    <svg viewBox="0 0 280 252" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`保留 ${selected.join(', ')}，丢弃质量 ${r.discarded}，输出 ${r.output}`}>
      {r.probabilities.map((p, j) => <g key={j}><text x="8" y={25 + j * 26}>k{j}</text><rect x="38" y={9 + j * 26} width={p * 225} height="16" className="fill-accent" /><path d={`M38 ${29 + j * 26}h${r.weights[j] / 12 * 225}`} className={pen.b} /></g>)}<text x="8" y="234">实柱：集合内概率；虚线：完整概率</text>
    </svg><Readout>indexer 仅决定集合，主指数=[{r.weights.join(', ')}]；分母={r.denominator}<br />被丢弃质量 δ={fmt(r.discarded)}；稀疏输出={r.output === null ? '未定义：禁止空集合' : fmt(r.output, 6)}；完整输出={fmt(r.full, 6)}<br />输出差={r.error === null ? '未定义' : fmt(r.error, 6)}；固定值界 2Mδ={fmt(r.bound)}（M=70）</Readout>
  </LabFrame>
}

export function MultimodalAlignmentLab() {
  const [patch, setPatch] = useState(0), [position, setPosition] = useState(6), [swapped, setSwapped] = useState(false), [gradient, setGradient] = useState(false)
  const r = multimodalPatch(patch, swapped), names = ['USER', 'z0', 'z1', 'z2', 'z3', 'QUESTION', 'ASSISTANT', 'ANSWER', 'EOS']
  return <LabFrame title="像素块、视觉行与回答监督">
    <Controls><Range label="patch" value={patch} min={0} max={3} onChange={setPatch} /><Range label="预测行" value={position} min={0} max={8} onChange={setPosition} /><label className="text-sm"><input type="checkbox" checked={swapped} onChange={e => setSwapped(e.target.checked)} /> 交换图像上下半部</label><label className="text-sm"><input type="checkbox" checked={gradient} onChange={e => setGradient(e.target.checked)} /> 显示梯度路径</label></Controls>
    <svg viewBox="0 0 280 370" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`patch ${patch} 位于语言模型输入 ${r.position}，预测行 ${position}`}>
      {Array.from({ length: 16 }, (_, i) => { const selected = r.indices.includes(i), v = (swapped ? (i + 8) % 16 : i) / 15; return <g key={i}><rect x={8 + i % 4 * 33} y={8 + Math.floor(i / 4) * 33} width="30" height="30" fill={`rgb(${Math.round(v * 180 + 45)},${Math.round(v * 180 + 45)},${Math.round(v * 180 + 45)})`} className={selected ? 'stroke-accent stroke-3' : 'stroke-rule'} /></g> })}<text x="150" y="45">patch {patch}</text><text x="150" y="69">h ∈ R⁶</text><text x="150" y="93">projector</text><text x="150" y="117">z ∈ R⁸</text>
      {names.map((name, i) => <g key={i} transform={`translate(${8 + i % 3 * 90},${160 + Math.floor(i / 3) * 65})`}><rect width="83" height="50" className={i === r.position ? 'fill-accent/20 stroke-accent stroke-2' : i <= position ? 'fill-info/10 stroke-info' : 'fill-sunken stroke-rule [stroke-dasharray:3_3]'} /><text x="41" y="20" textAnchor="middle" className="text-[11px]">{name}</text><text x="41" y="39" textAnchor="middle">p={i}{i === position ? ' ←' : ''}</text></g>)}<text x="8" y="365">{gradient ? '回复 loss → Decoder → z → projector' : '蓝框：当前行因果可见；虚线：未来'}</text>
    </svg><Readout>patch 索引=[{r.indices.join(', ')}]；像素=[{r.pixels.map(v => fmt(v)).join(', ')}]；h 第一维均值={fmt(r.mean, 6)}<br />六维特征 → 八维视觉行 z{patch}，进入位置 {r.position}；本图只核对形状，未伪造其余特征值<br />当前预测目标：{position === 6 ? (swapped ? 'ANSWER（应为上半部）' : 'ANSWER（应为下半部）') : position === 7 ? 'EOS' : '忽略/无下一目标'}；有效 logits 行=[6,7]<br />{gradient ? '视觉行无直接文字标签；回复 loss 仍可回传连接器。冻结视觉编码器不要求 detach 连接器输出' : '换图必须重算视觉特征与前缀 KV；相同问题文字不代表缓存相同'}</Readout>
  </LabFrame>
}
