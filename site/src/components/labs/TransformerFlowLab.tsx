import { useState } from 'react'
import { decoderLedger } from '../../lib/chapter-labs-model'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

const stages = [
  { name: '输入嵌入', shape: '(1, U, 12)', code: 'x = self.embedding(ids)', meaning: 'ID 是离散索引；查表后每个位置获得 12 维可训练向量。' },
  { name: '注意力前 RMSNorm', shape: '(1, U, 12)', code: 'n = self.norm_attention(x)', meaning: '只规范每行特征尺度；原始 x 保留在残差旁路。' },
  { name: 'Q / K / V 与 RoPE', shape: 'Q (1,4,U,4)；K/V (1,nkv,U,4)', code: 'q, k = apply_rope(q, positions), apply_rope(k, positions)', meaning: '先投影与拆头，按绝对位置旋转 Q/K；V 不旋转。' },
  { name: '读取每层 KV 缓存', shape: 'K/V (1,nkv,P+U,4)', code: 'k = torch.cat((cache[0], k), -2)', meaning: '历史 K 已经旋转；当前层只接入自己的历史。' },
  { name: '分数、因果 mask、softmax', shape: '(1,4,U,P+U)', code: 'scores.masked_fill(~mask, -torch.inf).softmax(-1)', meaning: '每行在允许键上归一化，查询 i 只读到键 P+i。' },
  { name: '加权 V、合头、输出投影', shape: '(1,U,16) → (1,U,12)', code: 'branch = self.output(y.transpose(1,2).reshape(B,U,-1))', meaning: '注意力用时间轴混合信息；O 投影把 16 维写回 12 维残差流。' },
  { name: '第一条残差', shape: '(1,U,12)', code: 'z = x + branch', meaning: '加法两侧形状完全相同；下一分支读取的是新状态 z。' },
  { name: 'FFN 前 RMSNorm 与 SwiGLU', shape: '12 → 20 → 12', code: 'f = self.ff(self.norm_ff(z))', meaning: '同一组权重分别处理每个位置；gate 与 up 在 20 维逐元素相乘。' },
  { name: '第二条残差', shape: '(1,U,12)', code: 'x = z + f', meaning: '得到本层输出；第二层重复整个块，各层参数与缓存独立。' },
  { name: '末层 RMSNorm 与词表头', shape: '(1,U,8)', code: 'logits = self.head(self.norm(x))', meaning: '每个位置为下一 token 的 8 个候选打分；绑定头复用输入嵌入权重。' },
]

export default function TransformerFlowLab() {
  const [stage, setStage] = useState(0)
  const [length, setLength] = useState(3)
  const [past, setPast] = useState(0)
  const [kv, setKv] = useState(2)
  const ledger = decoderLedger(kv, length, past)
  const current = stages[stage]
  return <LabFrame title="跟踪一个完整现代块" hint="B=1，d=12，nq=4，dh=4，dff=20，L=2，V=8">
    <Controls>
      <label className="text-sm">计算阶段<select className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2" value={stage} onChange={e => setStage(Number(e.target.value))}>{stages.map((s, i) => <option key={s.name} value={i}>{i + 1} · {s.name}</option>)}</select></label>
      <label className="text-sm">KV 头数<select className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2" value={kv} onChange={e => setKv(Number(e.target.value))}>{[1, 2, 4].map(n => <option key={n} value={n}>{n} · {n === 1 ? 'MQA' : n === 2 ? 'GQA' : 'MHA'}</option>)}</select></label>
      <Range label="新增长度 U" min={1} max={5} value={length} onChange={setLength} />
      <Range label="历史长度 P" min={0} max={5} value={past} onChange={setPast} />
    </Controls>
    <svg viewBox="0 0 380 540" className={`${pen.canvas} max-w-110!`} role="img" aria-label={`完整块的数据路径，当前阶段 ${current.name}`}>
      {stages.map((s, i) => <g key={s.name}>
        {i < stages.length - 1 && <path d={`M200 ${34 + i * 50}v16m-4 -5l4 5 4 -5`} className={pen.axis} />}
        <rect x="62" y={10 + i * 50} width="286" height="32" rx="3" className={i === stage ? 'fill-accent-soft stroke-accent stroke-2' : 'fill-paper stroke-rule'} />
        <text x="76" y={31 + i * 50} className={i === stage ? pen.textA : ''}>{i + 1}. {s.name}</text>
      </g>)}
      <path d="M62 26H18V326H62M62 326H34V426H62" className={pen.guide} />
      <text x="20" y="520" className={pen.muted}>左侧：x 和 z 的残差旁路；一个块重复两层</text>
    </svg>
    <p className="mt-3 text-sm">{current.meaning}</p>
    <Readout>{current.code}<br />形状：{current.shape.replaceAll('nkv', String(kv)).replaceAll('P+U', String(past + length)).replaceAll('U', String(length))}<br />总参数 {ledger.parameters} · 全部两层缓存 {ledger.cacheElements} 个元素 · 每层当前分数 {ledger.scoreElements} 个元素</Readout>
  </LabFrame>
}
