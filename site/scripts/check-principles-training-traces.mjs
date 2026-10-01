import assert from 'node:assert/strict'
import { cachedChapterAttention, decoderResidual, modernGate, permutationAttention, sequenceLedger } from '../src/lib/principles-trace-model.ts'
import { adamTrajectory, chatMask, checkpointComparison, ddpLedger, duplicateGroups, evaluationLedger, evaluationWindows, loraTrajectory, packing, stateAllocation } from '../src/lib/training-trace-model.ts'
import { sampleDistribution } from '../src/lib/sampling-model.ts'

const near = (actual, expected, epsilon = 1e-6) => assert.ok(Math.abs(actual - expected) < epsilon, `${actual} != ${expected}`)
const sequence = sequenceLedger([true, true, true])
const sampled = sampleDistribution([0.1, 0.4, 0.3, 0.2].map(Math.log), { temperature: 0.5, topK: 4, topP: 1, minP: 0, penalty: 1, history: [1] })
sampled.final.forEach((p, i) => near(p, [1 / 30, 16 / 30, 9 / 30, 4 / 30][i]))
near(sequence.sum, -Math.log(0.252)); near(sequence.ppl, 0.252 ** (-1 / 3))
assert.equal(sequenceLedger([false, false, false]).ppl, null)
near(decoderResidual(1).first[0], 0); near(decoderResidual(1).probabilities[2], 0.3878, 0.0001)
for (const causal of [false, true]) for (const position of [false, true]) {
  const baseline = permutationAttention(false, causal, position, false)
  const result = permutationAttention(true, causal, position, true)
  result.output.forEach((x, i) => near(x, baseline.output[result.permutation[i]], 1e-12))
}
assert.ok(Math.abs(permutationAttention(true, true, false, false).output[0] - permutationAttention(false, true, false, false).output[1]) > 0.1)
near(modernGate().output[0], 1.644970); near(modernGate().output[1], 2.752987)
assert.deepEqual(modernGate(0).branch, [0, 0])
near(cachedChapterAttention(3, 2, 0, false).output[0], 2.061223)
near(cachedChapterAttention(3, 2, 0, true).output[0], 0)
for (let past = 0; past < 6; past++) for (let length = 1; length < 5; length++) for (let row = 0; row < length; row++) {
  const chunk = cachedChapterAttention(past, length, row, false)
  const full = cachedChapterAttention(0, past + length, past + row, false)
  near(chunk.output[0], full.output[0], 1e-12)
}
assert.equal(packing(true).count, 4); assert.equal(packing(false).count, 5)
assert.deepEqual(packing(true).allowed[3], [false, false, false, true, false])
const adam = adamTrajectory(2)
near(adam[1].w[0], 0.88); near(adam[1].w[1], -1.86)
assert.ok(Math.abs(adam[2].w[0] - adamTrajectory(2, 1)[2].w[0]) > 1e-4)
const complete = checkpointComparison('none')
assert.deepEqual(complete.reference, complete.resumed)
for (const omit of ['optimizer', 'rng', 'cursor']) {
  const result = checkpointComparison(omit)
  assert.ok(result.reference.some((row, i) => row.w !== result.resumed[i].w || row.document !== result.resumed[i].document), `omitting ${omit} must reveal a mismatch`)
}
for (let stride = 1; stride <= 4; stride++) {
  const windows = evaluationWindows(stride)
  assert.deepEqual(windows.flatMap(row => row.targets), [1, 2, 3, 4, 5, 6])
  windows.forEach(row => assert.ok(row.inputs.length <= 4))
}
near(evaluationLedger('A', [true, true, true, true]).mean, 2.5)
near(evaluationLedger('B', [true, true, true, true]).mean, 2.2)
near(evaluationLedger('B', [true, true, true, false]).mean, 2.08)
assert.equal(evaluationLedger('A', [false, false, false, false]).mean, null)
assert.deepEqual(chatMask(false, false).valid.flatMap((v, i) => v ? [i] : []), [4, 5, 9, 10])
assert.equal(chatMask(false, true).count, 0)
assert.ok(chatMask(true, false).gradient[0] < 0)
const chat = chatMask(false, false), hidden = Array.from({ length: 11 }, (_, i) => (i + 1) / 10)
const causalLoss = values => chat.valid.reduce((loss, valid, i) => {
  const mean = values.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1)
  return loss + (valid ? 0.5 * (mean - 1) ** 2 / chat.count : 0)
}, 0)
const perturb = delta => hidden.map((v, i) => v + (i === 0 ? delta : 0))
near((causalLoss(perturb(1e-5)) - causalLoss(perturb(-1e-5))) / 2e-5, chat.gradient[0], 1e-8)
const lora = loraTrajectory(2)
near(lora[0].loss, 1.555142); near(lora[1].loss, 1.179401)
assert.equal(lora[0].gradA, 0); assert.ok(lora[0].gradB > 0); assert.ok(lora[1].gradA > 0)
lora.forEach(row => row.logits.forEach((v, i) => near(v, row.merged[i], 1e-12)))
loraTrajectory(2, true).forEach(row => { assert.equal(row.gradA, 0); assert.equal(row.gradB, 0) })
assert.deepEqual(duplicateGroups(0.4, false).group, [0, 0, 0, 3, 4])
assert.deepEqual(duplicateGroups(0.5, false).group, [0, 1, 0, 3, 4])
assert.deepEqual(duplicateGroups(0.6, true).group, [0, 0, 0, 3, 4])
near(ddpLedger(false, 2).gradient, -32 / 14); near(ddpLedger(true, 2).theta, 0.2)
assert.deepEqual(stateAllocation(4), [16, 7, 5.5, 4])
console.log('principles/training SVG models: numeric cases, counterexamples and boundaries passed')
