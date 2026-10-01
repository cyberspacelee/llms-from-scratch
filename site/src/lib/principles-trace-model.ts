export const normalize = (scores: number[]) => {
  const max = Math.max(...scores)
  const values = scores.map(x => Math.exp(x - max))
  const sum = values.reduce((a, b) => a + b, 0)
  return values.map(x => x / sum)
}

export function sequenceLedger(valid: boolean[], probabilities = [0.6, 0.7, 0.6]) {
  const count = valid.filter(Boolean).length
  const nll = probabilities.reduce((sum, p, i) => sum + (valid[i] ? -Math.log(p) : 0), 0)
  return { count, sum: nll, mean: count ? nll / count : null, ppl: count ? Math.exp(nll / count) : null, joint: Math.exp(-nll) }
}

export function decoderResidual(position: number) {
  const s = 1 / Math.sqrt(2), gelu = 0.8413447460685429
  const input = [[2, 0], [0, 2], [2, 4]][position]
  const branch = [[s, -s], [0, 0], [-s / 3, s / 3]][position]
  const first = input.map((x, i) => x + branch[i])
  const second = [first[0] + gelu, first[1]]
  const difference = second[0] - second[1]
  const r = difference / Math.sqrt(difference * difference + 4)
  const logits = [0, r, -r, 0], probabilities = normalize(logits)
  return { input, branch, first, ffn: [gelu, 0], second, logits, probabilities, nll: -Math.log(probabilities[position + 1]) }
}

export function permutationAttention(swapped: boolean, causal: boolean, position: boolean, moveStructure: boolean) {
  const permutation = swapped ? [1, 0, 2] : [0, 1, 2]
  const original = [1, 2, 3], offsets = [0, 1, 0]
  const values = permutation.map((identity, slot) => original[identity] + (position ? offsets[moveStructure ? identity : slot] : 0))
  const scores = values.map(q => values.map(k => q * k))
  const allowed = scores.map((row, i) => row.map((_, j) => !causal || (moveStructure ? permutation[j] <= permutation[i] : j <= i)))
  const probabilities = scores.map((row, i) => normalize(row.map((x, j) => allowed[i][j] ? x : -Infinity)))
  const output = probabilities.map(row => row.reduce((sum, p, j) => sum + p * values[j], 0))
  return { permutation, values, scores, allowed, probabilities, output }
}

export function modernGate(gateMultiplier = 1) {
  const p = normalize([1 / Math.sqrt(2), 0])[0]
  const h = [2 * p, 2]
  const rms = Math.sqrt(h.reduce((s, x) => s + x * x, 0) / 2 + 1)
  const r = h.map(x => x / rms)
  const gate = r.map(x => gateMultiplier * x)
  const silu = gate.map(x => x / (1 + Math.exp(-x)))
  const branch = silu.map((x, i) => x * r[i])
  return { p, h, r, gate, silu, branch, output: h.map((x, i) => x + branch[i]) }
}

export function cachedChapterAttention(past: number, length: number, row: number, wrong: boolean) {
  const position = past + row, q = [Math.cos(position), Math.sin(position)]
  const scores = Array.from({ length: past + length }, (_, j) => Math.cos(j - position) / Math.SQRT2)
  const allowed = scores.map((_, j) => j <= (wrong ? row : position))
  const probabilities = normalize(scores.map((s, j) => allowed[j] ? s : -Infinity))
  const output = [probabilities.reduce((sum, p, j) => sum + p * j, 0), 0]
  return { position, q, scores, allowed, probabilities, output }
}
