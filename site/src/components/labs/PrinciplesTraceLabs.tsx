import { useState } from 'react'
import { decoderResidual, modernGate, permutationAttention, sequenceLedger } from '../../lib/principles-trace-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

const vector = (values: number[]) => `(${values.map(x => fmt(x, 4)).join(', ')})`
function Trace({ rows, active = -1, label }: { rows: [string, string][]; active?: number; label: string }) {
  return <svg viewBox={`0 0 360 ${rows.length * 64 + 8}`} className={`${pen.canvas} max-w-110!`} role="img" aria-label={label}>
    {rows.map(([name, value], i) => <g key={name}>
      <rect x="4" y={4 + i * 64} width="352" height="54" rx="3" className={active === i ? 'fill-accent-soft stroke-accent' : 'fill-paper stroke-rule'} />
      <text x="16" y={24 + i * 64}>{name}</text><text x="16" y={45 + i * 64} className={pen.mono}>{value}</text>
      {i < rows.length - 1 && <path d={`M180 ${58 + i * 64}v8`} className={pen.axis} />}
    </g>)}
  </svg>
}

export function TokenLookupLab() {
  const ids = [231, 140, 171, 256, 257], pieces = ['E7', '8C', 'AB', '20 73', '61 74']
  const embedding = [[0.1, 0.3], [-0.2, 0.4], [0.5, 0.1], [0.2, -0.1], [-0.3, 0.2]]
  const [position, setPosition] = useState(3), [special, setSpecial] = useState(false)
  return <LabFrame title="从字节片段到 embedding 行" hint="猫 sat · 两次合并后的固定词表">
    <Controls><Range label="正文 token 位置" min={0} max={4} value={position} onChange={setPosition} /><label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={special} onChange={e => setSpecial(e.target.checked)} />附加 BOS / EOS</label></Controls>
    <Trace label={`第 ${position} 个 token 选 embedding 第 ${ids[position]} 行`} rows={[
      ['UTF-8 片段（十六进制）', pieces[position]], ['词表 ID → 参数行', `${ids[position]} → E[${ids[position]}, :]`], ['查表输出', vector(embedding[position])], ['序列形状', special ? '(7, 2) · BOS=258, EOS=259' : '(5, 2) · 5 个普通 token'],
    ]} active={2} />
    <Readout>ID 是行地址，不乘入向量。两份输入的 ID 256 共用参数行；上游 (1,2) 与 (3,−1) 汇合为 (4,1)。其余四行数值仅为本实验补充的查表例。</Readout>
  </LabFrame>
}

export function SequencePplLab() {
  const [row, setRow] = useState(0), [valid, setValid] = useState([true, true, true]), [pads, setPads] = useState(2)
  const p = [0.6, 0.7, 0.6], histories = ['BOS', 'BOS,A', 'BOS,A,B'], targets = ['A', 'B', 'EOS']
  const result = sequenceLedger(valid)
  return <LabFrame title="三个下一 token 目标的概率账" hint="自然对数 · 未加权 NLL">
    <Controls><Range label="预测行 i" min={0} max={2} value={row} onChange={setRow} /><Range label="PAD 占位数" min={0} max={5} value={pads} onChange={setPads} /></Controls>
    <div className="mt-3 flex flex-wrap gap-4">{targets.map((name, i) => <label key={name} className="text-sm"><input type="checkbox" checked={valid[i]} onChange={e => setValid(current => current.map((v, j) => j === i ? e.target.checked : v))} /> 计分目标 {name}</label>)}</div>
    <Trace label={`历史 ${histories[row]} 预测 ${targets[row]}，汇总 ${result.count} 个有效目标`} rows={[
      ['已知历史 → 下一目标', `${histories[row]} → ${targets[row]}`], ['条件概率 → 本行 NLL', `${p[row]} → ${fmt(-Math.log(p[row]), 6)} nat`], ['损失和 / 有效目标数', `${fmt(result.sum, 6)} / ${result.count}`], ['平均 NLL → PPL', result.mean === null ? '未定义：零有效目标' : `${fmt(result.mean, 6)} → ${fmt(result.ppl!, 6)}`],
    ]} active={1} />
    <Readout>数组长度 {3 + pads}，PAD 不改变有效分母。{result.count === 3 ? `整段 P(A,B,EOS|BOS)=${fmt(result.joint, 3)}` : '取消任一真实目标后，当前乘积不再代表完整文档概率。'} · 有效概率的连乘 {fmt(result.joint, 6)}</Readout>
  </LabFrame>
}

export function ResidualTraceLab() {
  const [position, setPosition] = useState(0), [stage, setStage] = useState(0)
  const result = decoderResidual(position)
  return <LabFrame title="一个位置的两次残差与词表头" hint="P4 的二维手算参数；不是 P11 的现代块">
    <Controls><Range label="输入位置" min={0} max={2} value={position} onChange={setPosition} /><Range label="前向阶段" min={0} max={5} value={stage} onChange={setStage} /></Controls>
    <Trace active={stage} label="输入、注意力写入、残差、FFN 写入、残差与词表 logits 的数值路径" rows={[
      ['原始残差流 X', vector(result.input)], ['注意力写入 branch', vector(result.branch)], ['第一次相加 H=X+branch', vector(result.first)], ['FFN 写入 GELU(1),0', vector(result.ffn)], ['第二次相加 Y=H+FFN', vector(result.second)], ['末层 Norm → 四类 logits', vector(result.logits)],
    ]} />
    <Readout>目标 {['A', 'B', 'EOS'][position]} · 正确类别概率 {fmt(result.probabilities[position + 1], 6)} · NLL {fmt(result.nll, 6)}。注意力写入为零的 A 位置仍保留原状态 (0,2)。</Readout>
  </LabFrame>
}

export function PermutationLab() {
  const [swapped, setSwapped] = useState(true), [causal, setCausal] = useState(false), [position, setPosition] = useState(false), [move, setMove] = useState(false), [row, setRow] = useState(1)
  const result = permutationAttention(swapped, causal, position, move)
  const baseline = permutationAttention(false, causal, position, false)
  const expected = result.permutation.map(i => baseline.output[i])
  const error = Math.max(...result.output.map((x, i) => Math.abs(x - expected[i])))
  return <LabFrame title="交换内容，还是同时交换槽位结构" hint="X=(1,2,3)，位置表 E=(0,1,0)，单位 Q/K/V">
    <Controls>{[['交换 BOS 与 A', swapped, setSwapped], ['causal mask', causal, setCausal], ['加位置表', position, setPosition], ['mask / 位置表随身份移动', move, setMove]].map(([name, value, setter]) => <label key={String(name)} className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={value as boolean} onChange={e => (setter as (v: boolean) => void)(e.target.checked)} />{String(name)}</label>)}<Range label="查询槽位" min={0} max={2} value={row} onChange={setRow} /></Controls>
    <svg viewBox="0 0 360 185" className={`${pen.canvas} max-w-110!`} role="img" aria-label="按固定槽位或随身份移动的 mask 显示分数和允许集合">
      {result.permutation.map((id, j) => <text key={j} x={122 + j * 76} y="20" textAnchor="middle">{['BOS', 'A', 'B'][id]}</text>)}
      {result.scores.map((scores, i) => <g key={i}><text x="4" y={54 + i * 46}>槽 {i}</text>{scores.map((score, j) => <g key={j}><rect x={86 + j * 76} y={32 + i * 46} width="70" height="38" className={result.allowed[i][j] ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} strokeWidth={row === i ? 2 : 1} /><text x={121 + j * 76} y={56 + i * 46} textAnchor="middle" className={pen.mono}>{result.allowed[i][j] ? score : '−∞'}</text></g>)}</g>)}
    </svg>
    <Readout>当前输入 {vector(result.values)}<br />查询行概率 {vector(result.probabilities[row])} · 输出 {vector(result.output)}<br />单纯重排旧输出 {vector(expected)} · 最大差 {fmt(error, 6)}</Readout>
  </LabFrame>
}

export function GqaMapLab() {
  const [kv, setKv] = useState(1), [query, setQuery] = useState(0), [length, setLength] = useState(2)
  const key = Math.floor(query / (2 / kv))
  return <LabFrame title="两个 Q 头映射到哪些 KV 参数" hint="B=L=1，nq=2，dh=2；缓存不保存 Q">
    <Controls><Range label="选择 Q 头" min={0} max={1} value={query} onChange={setQuery} /><label className="text-sm">KV 头数<select value={kv} onChange={e => setKv(Number(e.target.value))} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2"><option value="1">1 · 共用 KV</option><option value="2">2 · 独立 KV</option></select></label><Range label="缓存长度 T" min={1} max={8} value={length} onChange={setLength} /></Controls>
    <svg viewBox="0 0 360 170" className={`${pen.canvas} max-w-110!`} role="img" aria-label={`Q${query} 读取 KV${key}`}>
      {[0, 1].map(i => <g key={i}><rect x="10" y={24 + i * 74} width="88" height="42" className={query === i ? 'fill-accent-soft stroke-accent' : 'fill-paper stroke-rule'} /><text x="24" y={50 + i * 74}>Q 头 {i}</text><path d={`M98 ${45 + i * 74}L245 ${kv === 1 ? 80 : 45 + i * 74}`} className={query === i ? pen.a : pen.axis} /></g>)}
      {Array.from({ length: kv }, (_, i) => <g key={i}><rect x="245" y={kv === 1 ? 59 : 24 + i * 74} width="105" height="42" className={key === i ? 'fill-accent-soft stroke-accent' : 'fill-paper stroke-rule'} /><text x="257" y={kv === 1 ? 85 : 50 + i * 74}>KV 组 {i}</text></g>)}
    </svg>
    <Readout>2×B×L×T×nkv×dh = 2×1×1×{length}×{kv}×2 = {4 * length * kv} 个缓存元素。Q/O 参数仍为 16；K/V 为 {8 * kv}。共享 KV 不强制两个 Q 使用相同 softmax 权重。</Readout>
  </LabFrame>
}

export function SwiGluLab() {
  const [multiplier, setMultiplier] = useState(1), [stage, setStage] = useState(0)
  const result = modernGate(multiplier)
  return <LabFrame title="同一个位置的 gate 与 up 怎样相乘" hint="沿用 A 位置 H=(2p,2)；up/down 仍为 I₂">
    <Controls><Range label="gate 权重倍数" min={-2} max={2} step={0.25} value={multiplier} onChange={setMultiplier} /><Range label="计算阶段" min={0} max={4} value={stage} onChange={setStage} /></Controls>
    <Trace active={stage} label="SwiGLU 的两路投影、激活与逐元素乘法" rows={[
      ['第二次 RMSNorm 的输入 H', vector(result.h)], ['两路输入 r（up=r）', vector(result.r)], ['SiLU(gate) = gate·sigmoid(gate)', vector(result.silu)], ['逐元素乘 up，再经 down', vector(result.branch)], ['相加写回残差流', vector(result.output)],
    ]} />
    <Readout>gate={vector(result.gate)}。倍数 1 恢复正文 Y≈(1.644970,2.752987)；倍数 0 令 FFN 写入为零；负 gate 可以产生负写入，不是仅用 sigmoid(gate)×up。</Readout>
  </LabFrame>
}
