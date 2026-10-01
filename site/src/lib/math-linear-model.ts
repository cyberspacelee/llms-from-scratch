import { softmaxLoss } from './math-labs-model.ts'

export const linearX = [[1, 2], [3, 4], [-1, 0]]
export const linearW = [[1, -1], [2, 0]]
export const linearB = [0.5, -1]
export const linearZ = linearX.map(row => linearW.map((w, k) => w.reduce((s, v, j) => s + v * row[j], linearB[k])))

export function linearElement(row: number, output: number) {
  if (!Number.isInteger(row) || row < 0 || row > 2 || !Number.isInteger(output) || output < 0 || output > 1) throw new RangeError('Invalid output cell')
  const terms = linearX[row].map((value, j) => value * linearW[output][j])
  return { terms, bias: linearB[output], value: linearZ[row][output] }
}

export function gradientContributions(weight: number, step: number, mean: boolean) {
  if (!Number.isInteger(weight) || weight < 0 || weight > 3 || !Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Invalid weight or step')
  const k = Math.floor(weight / 2), j = weight % 2, divisor = mean ? 3 : 1
  const terms = linearX.map((row, i) => linearZ[i][k] * row[j] / divisor)
  const gradient = linearW.map((w, output) => w.map((_, input) => linearX.slice(0, step).reduce((s, row, i) => s + linearZ[i][output] * row[input] / divisor, 0)))
  return { k, j, terms, divisor, partial: terms.slice(0, step).reduce((s, value) => s + value, 0), gradient }
}

export function localDifference(h: number) {
  if (!(h > 0) || !Number.isFinite(h)) throw new RangeError('Positive finite h required')
  return { quotient: ((2 + h) ** 2 - 4) / (2 * h), derivative: 2, predicted: 4 + 2 * h, actual: 0.5 * ((2 + h) ** 2 + 4) }
}

type Parameters = { w1: number[][]; b1: number[]; w2: number[][]; b2: number[] }
export function xorTraining(eta: number, steps: number) {
  if (!Number.isFinite(eta) || eta <= 0 || !Number.isInteger(steps) || steps < 0 || steps > 100) throw new RangeError('Positive eta and 0..100 steps required')
  const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]], labels = [0, 1, 1, 0]
  const forward = (parameters: Parameters) => inputs.map((x, i) => {
    const pre = parameters.w1.map((row, k) => row.reduce((s, w, j) => s + w * x[j], parameters.b1[k]))
    const hidden = pre.map(value => Math.max(0, value))
    const logits = parameters.w2.map((row, k) => row.reduce((s, w, j) => s + w * hidden[j], parameters.b2[k]))
    return { x, pre, hidden, logits, ...softmaxLoss(logits, 1, labels[i]) }
  })
  let parameters: Parameters = { w1: [[1, -1], [-1, 1]], b1: [0, 0], w2: [[0, 0], [-1, -1]], b2: [0, 0] }
  const initialRows = forward(parameters), initialLoss = initialRows.reduce((s, row) => s + row.loss / 4, 0)
  const snapshots = []
  for (let t = 0; t <= steps; t++) {
    const rows = forward(parameters)
    const gradients: Parameters = { w1: [[0, 0], [0, 0]], b1: [0, 0], w2: [[0, 0], [0, 0]], b2: [0, 0] }
    for (const row of rows) {
      const g2 = row.gradient.map(value => value / 4)
      const g1 = row.pre.map((value, j) => value > 0 ? g2.reduce((s, g, k) => s + g * parameters.w2[k][j], 0) : 0)
      for (let k = 0; k < 2; k++) {
        gradients.b2[k] += g2[k]; gradients.b1[k] += g1[k]
        for (let j = 0; j < 2; j++) { gradients.w2[k][j] += g2[k] * row.hidden[j]; gradients.w1[k][j] += g1[k] * row.x[j] }
      }
    }
    const next: Parameters = { w1: parameters.w1.map((row, k) => row.map((v, j) => v - eta * gradients.w1[k][j])), b1: parameters.b1.map((v, k) => v - eta * gradients.b1[k]), w2: parameters.w2.map((row, k) => row.map((v, j) => v - eta * gradients.w2[k][j])), b2: parameters.b2.map((v, k) => v - eta * gradients.b2[k]) }
    snapshots.push({ parameters, gradients, next, rows, loss: rows.reduce((s, row) => s + row.loss / 4, 0) })
    parameters = next
  }
  return { ...snapshots[steps], initialRows, initialLoss }
}
