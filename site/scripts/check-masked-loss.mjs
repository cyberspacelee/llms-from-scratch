import assert from 'node:assert/strict'
import { maskedLoss } from '../src/lib/masked-loss-model.ts'
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`)
const targets = [0, 1, 0, 1, 0, 0], ignored = [false, false, true, false, false, true]
const base = maskedLoss(targets, ignored, 'mean')
const s = 1 / Math.sqrt(2)
close(base.result, Math.log(Math.exp(s) + Math.exp(-s) + 2) - s); close(base.total, 4 * base.result)
assert.equal(base.valid, 4); assert.equal(base.rows[2].loss, 0)
for (const row of base.rows) close(row.probability.reduce((a, b) => a + b, 0), 1)
const changed = [...targets]; changed[1] = 0
close(maskedLoss(changed, ignored, 'mean').result - base.result, 1 / Math.sqrt(2) / 2)
close(maskedLoss(targets, ignored.map(() => false), 'mean').result, base.result)
assert.deepEqual(maskedLoss(targets, ignored.map(() => true), 'none').result, Array(6).fill(0))
assert.equal(maskedLoss(targets, ignored.map(() => true), 'mean').result, 0)
assert.equal(maskedLoss(targets, ignored.map(() => true), 'sum').result, 0)
assert.throws(() => maskedLoss([4, 1, 0, 1, 0, 0], ignored, 'mean'), RangeError)
assert.throws(() => maskedLoss(targets.slice(0, 3), ignored, 'mean'), RangeError)
console.log('Masked CE model: six F5 positions, four classes, exact logits, targets, effective denominator, reductions and all-ignored policy passed.')
