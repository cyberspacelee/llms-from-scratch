import assert from 'node:assert/strict'
import { eventProbability, bayesPaths, sequencePath, batchDistribution, informationCost } from '../src/lib/math-probability-model.ts'
import { linearElement, gradientContributions, localDifference, xorTraining } from '../src/lib/math-linear-model.ts'
import { compileTrace, headCoordinate, moduleParameters } from '../src/lib/framework-trace-model.ts'
const close = (a, b, tolerance = 1e-11) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`)

close(eventProbability('positive', 'D').conditional, 1 / 3)
assert.equal(eventProbability('positive', 'firstTwo').independent, true)
assert.equal(eventProbability('positive', 'negative').independent, false)
assert.equal(eventProbability('positive', 'empty').conditional, null)
close(bayesPaths(0.75).posteriorD, 0.5); close(bayesPaths(0.6).positive, 0.6); close(bayesPaths(0.6).posteriorD, 1 / 3)
close(sequencePath(3).joint, 0.252); close(sequencePath(3).nll, -Math.log(0.252))
for (const size of [1, 4, 16, 64]) {
  const result = batchDistribution(size, 'independent')
  close(result.rows.reduce((s, r) => s + r.probability, 0), 1)
  close(result.mean, 2); close(result.variance, 3 / size)
}
close(batchDistribution(4, 'independent').closeProbability, 27 / 64)
close(batchDistribution(64, 'copied').variance, 3); close(batchDistribution(4, 'complete').variance, 0)
const info = informationCost(0.75, 'all', false)
close(info.entropy, Math.log(2)); close(info.crossEntropy, -0.5 * Math.log(0.75 * 0.25)); close(info.kl, info.crossEntropy - info.entropy)
close(info.conditionalEntropy, 0.4773856262211096)
close(informationCost(0.5, 'all', true).entropy, 1)
assert.equal(informationCost(0, 'all', false).kl, Infinity)
assert.equal(informationCost(1, 'E', false).crossEntropy, 0)
close(linearElement(1, 0).value, -0.5); close(linearElement(1, 1).value, 5)
assert.deepEqual(gradientContributions(1, 3, false).gradient, [[-1.5, -3], [19, 22]])
close(gradientContributions(1, 3, true).partial, -1)
close(gradientContributions(3, 3, true).partial, 22 / 3)
assert.equal(gradientContributions(0, 0, true).partial, 0)
close(localDifference(0.1).quotient, 2.05); close(localDifference(0.1).actual, 4.205)
const initial = xorTraining(0.3, 0), next = xorTraining(0.3, 1)
close(initial.loss, 1.003204434039084); close(initial.gradients.w2[1][0], -(1 - 1 / (1 + Math.E)) / 4)
close(next.loss, 0.9236409233667829)
assert.deepEqual(next.rows.map(r => r.probs[1] > r.probs[0] ? 1 : 0), [1, 0, 0, 1])
for (let offset = 0; offset < 8; offset++) {
  const coordinate = headCoordinate(offset)
  assert.equal(coordinate.head * 2 + coordinate.t * 4 + coordinate.channel, offset)
}
const trace = compileTrace(4, false)
assert.deepEqual(trace.map(row => row.hit), [false, true, false, true])
assert.deepEqual(trace[0].output, [-2, 0, 10]); assert.deepEqual(trace[0].gradient, [-1, 3, 7])
assert.equal(trace[3].graphs, 2)
assert.deepEqual(compileTrace(1, true)[0].output, [9, 15])
assert.equal(moduleParameters.reduce((s, p) => s + p.count, 0), 20)
assert.throws(() => bayesPaths(-1), RangeError); assert.throws(() => batchDistribution(0, 'independent'), RangeError)
assert.throws(() => headCoordinate(8), RangeError)
console.log('Math/framework interactions: events, Bayes, chain, exact batch distribution, information boundaries, linear contributions, XOR update, head coordinates and compile trace passed.')
