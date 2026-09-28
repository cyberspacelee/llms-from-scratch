import assert from 'node:assert/strict'
import { descend, sharedGraph, softmaxLoss } from '../src/lib/math-labs-model.ts'
import { rotate, spectralTransform, lowRank, conditioning, cachedAttention, decoderLedger, budgetLoss } from '../src/lib/chapter-labs-model.ts'

const close = (a, b, tol = 1e-12) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`)
// M2 example: logits [0, log 2, log 3] give probabilities 1/6, 2/6, 3/6.
const s = softmaxLoss([0, Math.log(2), Math.log(3)], 1, 2)
s.probs.forEach((p, i) => close(p, (i + 1) / 6))
close(s.loss, -Math.log(0.5)); close(s.gradient[2], -0.5); close(s.gradient[0], 1 / 6)
close(s.gradient.reduce((a, b) => a + b, 0), 0)
// Adding a constant to every logit leaves loss and gradient unchanged.
const shifted = softmaxLoss([1000, 1000 + Math.log(2), 1000 + Math.log(3)], 1, 2)
close(shifted.loss, s.loss, 1e-9)
// Temperature 2 on the same logits: gradient carries the 1/tau factor.
const t = softmaxLoss([0, Math.log(2), Math.log(3)], 2, 2)
const r = [1, Math.SQRT2, Math.sqrt(3)], z = r.reduce((a, b) => a + b, 0)
close(t.probs[2], Math.sqrt(3) / z); close(t.gradient[2], (Math.sqrt(3) / z - 1) / 2)
// M4 quadratic: distinguish convergence, oscillation, the threshold, and divergence.
assert.equal(descend(0.2, [2, 1], 12).verdict, 'converge')
assert.equal(descend(0.45, [2, 1], 12).verdict, 'oscillate')
assert.equal(descend(0.5, [2, 1], 12).verdict, 'boundary')
assert.equal(descend(0.6, [2, 1], 12).verdict, 'diverge')
const run = descend(0.2, [2, 1], 30)
close(run.path[30][0], 2 * 0.8 ** 30); close(run.losses[30], 0.5 * ((2 * 0.8 ** 30) ** 2 + 4 * (0.2 ** 30) ** 2))
assert.ok(run.losses[30] < 1e-5)
// M6 shared node at x = 2: forward 4, 12, 16; full gradient 16, dropping a path gives 4 or 12.
assert.deepEqual(sharedGraph(2, true, true), { a: 4, b: 12, loss: 16, direct: 1, throughB: 3, gradA: 4, gradX: 16, exact: 16 })
assert.equal(sharedGraph(2, true, false).gradX, 4)
assert.equal(sharedGraph(2, false, true).gradX, 12)
assert.throws(() => softmaxLoss([1, 2], 0, 0), RangeError)
console.log('math lab models: softmax CE and gradient, descent regimes, shared-node backprop passed')
const eigenvector = rotate([0, 1], 37)
const transformed = spectralTransform(eigenvector, 37, [1, 9])
transformed.output.forEach((x, i) => close(x, 9 * eigenvector[i]))
const inverse = rotate(rotate([2, -3], 37), -37)
close(inverse[0], 2); close(inverse[1], -3)
for (let rank = 0; rank <= 3; rank++) {
  const result = lowRank(rank)
  close(result.residual.flat().reduce((sum, x) => sum + x * x, 0), result.error ** 2)
}
close(lowRank(1).error, Math.sqrt(5)); close(lowRank(3).energy, 1)
close(conditioning(4, 1e-6).outputError, 0.01)
for (let past = 0; past <= 5; past++) for (let length = 1; length <= 4; length++) for (let row = 0; row < length; row++) {
  const chunk = cachedAttention(past, length, row)
  const full = cachedAttention(0, past + length, past + row)
  chunk.probabilities.forEach((p, j) => close(p, full.probabilities[j]))
  close(chunk.probabilities.reduce((a, b) => a + b, 0), 1)
  assert.equal(chunk.allowed.filter(Boolean).length, past + row + 1)
}
assert.notDeepEqual(cachedAttention(3, 2, 0).output, cachedAttention(3, 2, 0, true).output)
assert.deepEqual(decoderLedger(2, 5, 0), { parameters: 2748, cacheElements: 160, scoreElements: 100 })
console.log('chapter labs: eigen directions, SVD residual energy, conditioning, full/chunk attention and model ledger passed')
const budget = budgetLoss(10, 100, 0.6, 0.4)
const optimal = budgetLoss(budget.optimalSize, 100, 0.6, 0.4)
close(budget.dataSize, 10)
assert.ok(optimal.loss < budgetLoss(budget.optimalSize * 0.9, 100, 0.6, 0.4).loss)
assert.ok(optimal.loss < budgetLoss(budget.optimalSize * 1.1, 100, 0.6, 0.4).loss)
close(budgetLoss(10, 400, 0.6, 0.4).optimalSize / budget.optimalSize, 4 ** 0.4)
