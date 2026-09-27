import assert from 'node:assert/strict'
import { kernelLifecycle } from '../src/lib/kernel-lifecycle.ts'

const first = kernelLifecycle(4, 0)
assert.equal(first.sharedBytes, 128)
assert.equal(first.accumulatorSlots, 16)
assert.equal(first.validLoads, 32)
assert.equal(first.sample, 0)
assert.equal(kernelLifecycle(4, 2).sample, 20)
assert.equal(kernelLifecycle(4, 2).canOverwrite, false)
assert.equal(kernelLifecycle(4, 3).canOverwrite, true)
assert.equal(kernelLifecycle(4, 4).sample, 20)
assert.equal(kernelLifecycle(4, 6).sample, 112)
assert.equal(kernelLifecycle(4, 6).outputWritten, false)
assert.equal(kernelLifecycle(4, 8).outputWritten, true)
assert.equal(kernelLifecycle(4, 4, true).validLoads, 9)
assert.equal(kernelLifecycle(4, 4, true).paddingSlots, 23)
assert.equal(kernelLifecycle(4, 8, true).validOutputs, 2)
for (let tile = 1; tile <= 4; tile++) {
  const final = kernelLifecycle(tile, Math.ceil(7 / tile) * 4)
  assert.equal(final.completedK, 7)
  assert.equal(final.sample, 112)
}
assert.throws(() => kernelLifecycle(0, 0), RangeError)
assert.throws(() => kernelLifecycle(4, 9), RangeError)
console.log('Kernel lifecycle: load/publish/consume/reclaim/write, padding, resources and accumulation passed.')
