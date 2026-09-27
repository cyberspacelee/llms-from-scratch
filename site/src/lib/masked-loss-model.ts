export type LossReduction = 'none' | 'sum' | 'mean'
export const lossLogits = [[2, 1, 0], [0, 1, 2], [1, 1, 1]] as const

export function maskedLoss(targets: readonly number[], ignored: readonly boolean[], reduction: LossReduction) {
  if (targets.length !== 3 || ignored.length !== 3 || targets.some(value => !Number.isInteger(value) || value < 0 || value > 2)
      || !['none', 'sum', 'mean'].includes(reduction)) throw new RangeError('Expected 3 targets in 0–2, masks, and a reduction')
  const rows = lossLogits.map((logits, index) => {
    const maximum = Math.max(...logits)
    const exps = logits.map(value => Math.exp(value - maximum))
    const denominator = exps.reduce((sum, value) => sum + value, 0)
    const probability = exps.map(value => value / denominator)
    const rawLoss = maximum + Math.log(denominator) - logits[targets[index]]
    return { logits, probability, rawLoss, loss: ignored[index] ? 0 : rawLoss }
  })
  const valid = ignored.filter(value => !value).length
  const total = rows.reduce((sum, row) => sum + row.loss, 0)
  // Teaching policy: all ignored returns zero; raw framework mean has zero denominator.
  const result = reduction === 'none' ? rows.map(row => row.loss) : reduction === 'sum' ? total : total / Math.max(1, valid)
  return { rows, valid, total, result, empty: valid === 0 }
}
