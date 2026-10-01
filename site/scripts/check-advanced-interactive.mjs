import assert from 'node:assert/strict'
import { dpoLedger, frequencyCache, grpoGroup, hybridBoundary, mlaPaths, moeDispatch, multimodalPatch, sparseSelection, stateRecurrence } from '../src/lib/advanced-interactive-model.ts'
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-8, `${a} != ${b}`)
const moe = moeDispatch(false, 2, false)
assert.deepEqual(moe.counts, [2, 1, 2, 1]); assert.deepEqual(moe.output, moe.reference)
close(moe.output[0][0], 1 + 1 / (1 + Math.exp(1)))
assert.equal(moeDispatch(true, 2, true).routes.filter(r => r.dropped).length, 2)
assert.deepEqual(moeDispatch(true, 2, false).output, moeDispatch(true, 3, false).output)
for (const head of [0, 1]) {
  const r = mlaPaths(head); assert.deepEqual(r.explicitScores, r.absorbedScores)
  r.explicit.forEach((v, i) => close(v, r.absorbed[i])); close(r.weights.reduce((s, v) => s + v, 0), 1)
}
close(dpoLedger(0.2, true).difference, Math.log(44 / 27)); close(dpoLedger(0.2, false).difference, Math.log(11 / 6))
assert.ok(dpoLedger(0.2, true).gradient < 0)
assert.notEqual(frequencyCache(5, 4, 2).mixedScore, frequencyCache(5, 4, 2).correctScore)
close(frequencyCache(5, 4, 1).mixedScore, frequencyCache(5, 4, 1).correctScore)
assert.deepEqual(grpoGroup(false, 1.4, 0.2, 0).advantages, [1, -1, -1, 1])
assert.equal(grpoGroup(false, 1.4, 0.2, 0).lossDerivativeLogRatio, 0)
close(grpoGroup(false, 0.6, 0.2, 1).objective, -0.8)
assert.equal(grpoGroup(true, 1.4, 0.2, 0).objective, 0)
close(stateRecurrence(false, 0.5).at(-1).state, 3.125); close(stateRecurrence(true, 0.5).at(-1).state, 4.25)
for (const selective of [false, true]) for (const a of [0, 0.5, 1, 1.2]) stateRecurrence(selective, a).forEach(r => close(r.state, r.cumulativeB))
assert.equal(hybridBoundary(0, true, true, true).valid, true)
assert.deepEqual(hybridBoundary(4, true, true, true).votes, [true, false, true])
assert.equal(hybridBoundary(5, true, true, false).valid, true)
assert.equal(hybridBoundary(5, true, true, true).valid, false)
const sparse = sparseSelection([0, 3, 7]); close(sparse.output, 340 / 7); close(sparse.discarded, 5 / 12)
close(sparseSelection([0, 4, 7]).output, 320 / 6); assert.equal(sparseSelection([]).output, null)
assert.throws(() => sparseSelection([0, 0]))
for (let p = 0; p < 4; p++) {
  const patch = multimodalPatch(p, false); close(patch.mean, [2.5, 4.5, 10.5, 12.5][p] / 15)
  assert.equal(patch.position, p + 1); close(multimodalPatch(p, true).mean, [10.5, 12.5, 2.5, 4.5][p] / 15)
}
console.log('advanced interactive: MoE, MLA, DPO, frequency, GRPO, recurrence, hybrid boundary, sparse and patch checks passed')
