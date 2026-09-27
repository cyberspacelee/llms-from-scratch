/**
 * Two attention heads over four positions. The input rows are position one-hots, so each
 * head's projection rows are its q and k vectors directly (d_h = 2).
 */
export type MaskMode = 'causal' | 'bidirectional'
export const tokens = ['BOS', '猫', '吃', '鱼'] as const

const angle = (p: number) => (p * Math.PI) / 3
/** Head 0 turns each query one step ahead of its keys, so it prefers the previous position. */
export const previousHead = {
  q: tokens.map((_, i) => [Math.cos(angle(i)), Math.sin(angle(i))]),
  k: tokens.map((_, j) => [Math.cos(angle(j + 1)), Math.sin(angle(j + 1))]),
}
/** Head 1 gives BOS a long key along the shared query direction. */
export const firstHead = {
  q: tokens.map(() => [1, 0]),
  k: tokens.map((_, j) => (j === 0 ? [1, 0] : [0, 1])),
}

export function headWeights(head: { q: number[][]; k: number[][] }, mode: MaskMode, scale: number) {
  if (!(scale > 0 && scale <= 8) || (mode !== 'causal' && mode !== 'bidirectional')) throw new RangeError('scale in (0, 8] and a mask mode required')
  const T = head.q.length
  return head.q.map((q, i) => {
    const scores = head.k.map((k, j) => (mode === 'causal' && j > i ? -Infinity : (scale * (q[0] * k[0] + q[1] * k[1])) / Math.SQRT2))
    const top = Math.max(...scores)
    const exps = scores.map(s => (s === -Infinity ? 0 : Math.exp(s - top)))
    const total = exps.reduce((a, b) => a + b, 0)
    return Array.from({ length: T }, (_, j) => exps[j] / total)
  })
}
