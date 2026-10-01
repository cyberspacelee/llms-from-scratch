export function resourceLedger(batch: number, length: number, large = false) {
  const linear = large ? 7504658432 : 352
  const blocks = large ? 6979321856 : 288
  const head = large ? 525336576 : 64
  const weights = large ? 16060522496 : 744
  const kvPerToken = large ? 131072 : 16
  const attentionPerPair = large ? 524288 : 32
  const kv = batch * length * kvPerToken
  return { weights, kv, write: batch * kvPerToken, read: 2 * linear + kv,
    linearFlops: 2 * batch * linear, attentionFlops: attentionPerPair * batch * length,
    prefillFlops: batch * (2 * length * blocks + 2 * head + attentionPerPair * length * (length + 1) / 2) }
}

export function roofline(batch: number, length: number) {
  const r = resourceLedger(batch, length, true)
  const flops = r.linearFlops + r.attentionFlops
  const intensity = flops / r.read
  return { ...r, intensity, ridge: 989.5 / 3.35, tCompute: flops / 989.5e12,
    tMemory: r.read / 3.35e12, ceiling: Math.min(989.5, 3.35 * intensity) }
}

export function onlineAttention(position: number, block: number, count: number) {
  let m = -Infinity, l = 0, u = 0
  const states = [{ m, l, u, factor: 0, start: 0, end: 0 }]
  for (let start = 0; start < 6; start += block) {
    const end = Math.min(6, start + block, position + 1)
    if (end <= start) { states.push({ m, l, u, factor: 1, start, end: start }); continue }
    const nextM = Math.max(m, end - 1)
    const factor = l === 0 ? 0 : Math.exp(m - nextM)
    l *= factor; u *= factor
    for (let j = start; j < end; j++) { const p = Math.exp(j - nextM); l += p; u += p * (j + 1) }
    m = nextM
    states.push({ m, l, u, factor, start, end })
  }
  const state = states[Math.min(count, states.length - 1)]
  let numerator = 0, denominator = 0
  for (let j = 0; j <= position; j++) { numerator += Math.exp(j - position) * (j + 1); denominator += Math.exp(j - position) }
  return { state, previous: states[Math.max(0, Math.min(count - 1, states.length - 1))], states,
    output: state.l > 0 ? state.u / state.l : null, dense: numerator / denominator }
}

export const packedRequests = [
  { name: 'A', cached: 0, ids: [11, 12, 13], blocks: [2] },
  { name: 'B', cached: 4, ids: [25, 26], blocks: [5, 7] },
  { name: 'C', cached: 0, ids: [31, 32, 33, 34], blocks: [9] },
]
export function packedRows(decode = false) {
  let offset = 0
  return packedRequests.flatMap((r, ri) => {
    const ids = decode ? [14, 27, 35].slice(ri, ri + 1) : r.ids
    const base = r.cached + (decode ? r.ids.length : 0)
    const blocks = decode && ri === 2 ? [9, 10] : r.blocks
    return ids.map((id, i) => { const position = base + i; return { request: r.name, id, position,
      index: offset++, slot: blocks[Math.floor(position / 4)] * 4 + position % 4,
      blocks, visible: position + 1 } })
  })
}

export function roundEven(value: number) {
  const low = Math.floor(value), fraction = value - low
  return fraction === 0.5 ? (low % 2 === 0 ? low : low + 1) : Math.round(value)
}
export function quantizedWeights(bits: number, group: number, outlier: number, asymmetric = false) {
  const original = [-1, -0.5, 0.5, 1, -outlier, outlier, -1, 1]
  const reconstructed: number[] = [], codes: number[] = [], scales: number[] = [], zeros: number[] = []
  for (let start = 0; start < 8; start += group) {
    const values = original.slice(start, start + group)
    const qmax = asymmetric ? 2 ** bits - 1 : 2 ** (bits - 1) - 1
    const min = Math.min(0, ...values), max = Math.max(0, ...values)
    const scale = (asymmetric ? max - min : Math.max(...values.map(Math.abs))) / qmax || 1
    const zero = asymmetric ? Math.max(0, Math.min(qmax, roundEven(-min / scale))) : 0
    scales.push(scale); zeros.push(zero)
    for (const value of values) {
      const code = Math.max(asymmetric ? 0 : -qmax, Math.min(qmax, roundEven(value / scale) + zero))
      codes.push(code); reconstructed.push(scale * (code - zero))
    }
  }
  const multiply = (w: number[]) => [0, 1].map(r => w.slice(r * 4, r * 4 + 4).reduce((s, v, i) => s + v * i, 0))
  const output = multiply(reconstructed), reference = multiply(original)
  return { original, reconstructed, codes, scales, zeros, output, reference,
    bytes: Math.ceil(8 * bits / 8) + scales.length * (asymmetric ? 3 : 2) }
}

export function parallelMatrix(mode: 'column' | 'row') {
  const x = [1, 2, 3, 4]
  const w = Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) => 4 * r + c))
  const reference = w[0].map((_, c) => x.reduce((s, v, r) => s + v * w[r][c], 0))
  const local = mode === 'column' ? [reference.slice(0, 2), reference.slice(2)] : [0, 1].map(rank => w[0].map((_, c) => x.reduce((s, v, r) => s + (Math.floor(r / 2) === rank ? v * w[r][c] : 0), 0)))
  const output = mode === 'column' ? local.flat() : local[0].map((v, i) => v + local[1][i])
  return { x, w, local, output, reference, sentBytes: mode === 'column' ? 4 : 8 }
}

export const serviceTrace = [
  { name: 'A', arrival: 0, dequeue: 0.1, events: [[1, 1], [1.2, 1], [1.5, 1]], end: 1.5, complete: true },
  { name: 'B', arrival: 0.1, dequeue: 0.4, events: [[0.6, 1]], end: 0.6, complete: true },
  { name: 'C', arrival: 0.2, dequeue: 0.5, events: [[0.8, 1], [1.1, 3]], end: 1.1, complete: true },
  { name: 'D', arrival: 0.3, dequeue: null, events: [] as number[][], end: 0.9, complete: false },
]
export function serviceMetrics(ttftLimit: number, tpotLimit: number, totalLimit: number) {
  const requests = serviceTrace.map(r => {
    const tokens = r.events.reduce((s, e) => s + e[1], 0)
    const ttft = tokens ? r.events[0][0] - r.arrival : null
    const tpot = tokens > 1 ? (r.events.at(-1)![0] - r.events[0][0]) / (tokens - 1) : null
    const itl = r.events.slice(1).map((e, i) => e[0] - r.events[i][0])
    const total = r.end - r.arrival
    return { ...r, tokens, ttft, tpot, itl, total,
      qualified: r.complete && ttft !== null && ttft <= ttftLimit + 1e-12 && total <= totalLimit + 1e-12 && (tpot === null || tpot <= tpotLimit + 1e-12) }
  })
  const qualified = requests.filter(r => r.qualified)
  return { requests, goodput: qualified.length / 1.5, tokenGoodput: qualified.reduce((s, r) => s + r.tokens, 0) / 1.5 }
}

export function speculativeTrace(first: number, second: number, step: number) {
  const draft = [0, 2], thresholds = [5 / 6, 2 / 3], random = [first, second]
  let accepted = 0
  while (accepted < step && accepted < 2 && random[accepted] < thresholds[accepted]) accepted++
  const rejected = accepted < Math.min(step, 2)
  const submitted = rejected ? [1, 2, ...draft.slice(0, accepted), 1] : step >= 3 ? [1, 2, 0, 2, 2] : [1, 2, ...draft.slice(0, accepted)]
  const cached = [1, 2, ...draft.slice(0, accepted)]
  return { draft, thresholds, random, accepted, rejected, submitted, cached,
    pending: rejected || step >= 3 ? submitted.at(-1) : null }
}

export function pageSnapshot(step: number) {
  let a = [0, 1], b: number[] = []
  const memory: (number | null)[][] = [[1, 2, 3, 4], [5, 6, null, null], [null, null, null, null]]
  if (step >= 1) b = [...a]
  if (step >= 2) { memory[2] = [...memory[1]]; a = [0, 2]; memory[2][2] = 7 }
  if (step >= 3) memory[1][2] = 70
  if (step >= 4) a = []
  if (step >= 5) b = []
  const refs = memory.map((_, block) => [...a, ...b].filter(value => value === block).length)
  const logical = (table: number[], length: number) => Array.from({ length }, (_, position) => memory[table[Math.floor(position / 4)]][position % 4])
  return { a, b, memory, refs, free: refs.flatMap((count, i) => count ? [] : [i]),
    aValues: a.length ? logical(a, step >= 2 ? 7 : 6) : [],
    bValues: b.length ? logical(b, step >= 3 ? 7 : 6) : [] }
}
