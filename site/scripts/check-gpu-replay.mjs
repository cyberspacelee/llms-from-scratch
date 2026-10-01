import assert from 'node:assert/strict'
import { barrierTrace, epilogue, timingBoundary } from '../src/lib/gpu-replay-model.ts'

assert.ok(barrierTrace('protected').at(-1).reads.every(read => read.valid))
assert.equal(barrierTrace('publish').at(-1).reads[0].value, 0)
assert.equal(barrierTrace('reclaim').at(-1).reads[1].value, 9)
assert.equal(epilogue(0, 0, 1, true, 2).result, 113)
assert.equal(epilogue(0, 5, 1, false, 2).result, 0)
for (let row = 0; row < 5; row++) for (let col = 0; col < 6; col++) {
  assert.equal(epilogue(row, col, -2, true, 2).result, epilogue(row, col, -2, false, 2).result)
}
assert.equal(epilogue(0, 0, 1, true, 2).bytes, 120)
assert.equal(epilogue(0, 0, 1, false, 2).bytes, 600)
assert.equal(timingBoundary(0).duration, 0.4)
assert.ok(Math.abs(timingBoundary(1).duration - 2) < 1e-12)
assert.equal(timingBoundary(2).duration, 7)
assert.equal(timingBoundary(3).duration, 8.4)
assert.throws(() => epilogue(0, 6, 0, true, 2), RangeError)
assert.throws(() => timingBoundary(4), RangeError)
console.log('GPU replay: barrier counterexamples, all 30 fused outputs, traffic and timing scopes verified.')
