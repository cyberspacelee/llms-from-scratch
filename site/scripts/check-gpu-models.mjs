import assert from 'node:assert/strict'
import { residency, streamSchedule, schedulerFrames } from '../src/lib/gpu-models.ts'

const limits = { warps: 64, registers: 65536, sharedKiB: 96, blocks: 32 }
assert.equal(residency(256, 32, 16, limits).occupancy, 0.75)
assert.equal(residency(256, 64, 16, limits).blocks, 4)
assert.equal(residency(256, 32, 32, limits).blocks, 3)
assert.equal(residency(256, 32, 0, limits).blocks, 8)
assert.equal(residency(256, 32, 128, limits).blocks, 0)
assert.equal(residency(33, 32, 0, limits).warpsPerBlock, 2)
assert.throws(() => residency(0, 32, 0, limits), RangeError)
assert.throws(() => residency(32, 1, 0, { ...limits, registers: NaN }), RangeError)
assert.equal(streamSchedule(5, 3, 2, false, false).total, 10)
assert.equal(streamSchedule(5, 3, 2, true, true).total, 7)
assert.equal(streamSchedule(5, 3, 2, true, false).safe, false)
assert.equal(streamSchedule(3, 5, 2, true, false).ordered, false)
assert.throws(() => streamSchedule(0, 1, 1, true, true), RangeError)
console.log('GPU teaching models: resource ceilings, zero residency, stream order and missing dependency verified.')
for (const frame of schedulerFrames) {
  const resident = frame.sms.flat()
  assert.ok(frame.sms.every(sm => sm.length <= 2))
  assert.equal(new Set([...resident, ...frame.done]).size, resident.length + frame.done.length)
  assert.ok([...resident, ...frame.done].every(block => block >= 0 && block < 6))
}
assert.equal(schedulerFrames.at(-1).done.length, 6)
assert.deepEqual(schedulerFrames[2].sms, schedulerFrames[1].sms)
