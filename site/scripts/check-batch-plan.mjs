import assert from 'node:assert/strict'
import { batchPlan } from '../src/lib/batch-plan.ts'
assert.equal(batchPlan(5, 2, false).weighted, 9)
assert.equal(batchPlan(5, 2, false).unweighted, 35 / 3)
assert.equal(batchPlan(5, 2, true).retained, 4)
assert.equal(batchPlan(5, 2, true).weighted, 5)
assert.equal(batchPlan(1, 8, true).weighted, null)
for (let n = 1; n <= 10; n++) for (let b = 1; b <= 8; b++) for (const drop of [false, true]) {
  const plan = batchPlan(n, b, drop)
  const indices = plan.batches.flatMap(batch => batch.indices)
  assert.deepEqual(indices, Array.from({ length: plan.retained }, (_, i) => i))
  assert.equal(plan.retained + plan.dropped, n)
  assert.ok(plan.batches.every(batch => drop ? batch.indices.length === b : batch.indices.length <= b))
}
assert.throws(() => batchPlan(0, 2, false), RangeError)
console.log('Batch plan: tail samples, empty loader and sample-weighted means verified.')
