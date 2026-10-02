import assert from 'node:assert/strict'
import { tensorLayout, vectorBranch } from '../src/lib/framework-tensor-model.ts'
assert.deepEqual(tensorLayout(false, 1, 2).traversal, [0, 1, 2, 3, 4, 5])
assert.deepEqual(tensorLayout(true, 2, 1).traversal, [0, 3, 1, 4, 2, 5])
for (const transposed of [false, true]) {
  const { rows, cols } = tensorLayout(transposed, 0, 0)
  const offsets = new Set()
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) offsets.add(tensorLayout(transposed, r, c).offset)
  assert.equal(offsets.size, 6)
}
const branch = vectorBranch(1, 2, 1, 1)
assert.deepEqual(branch.u, [3, -1])
assert.equal(branch.loss, 8)
assert.deepEqual(branch.gu, [20, 12])
assert.deepEqual(branch.gw, [40, -12])
assert.equal(branch.gb, 32)
assert.deepEqual(vectorBranch(1, 2, 1, 0.5).gw, [20, -6])
assert.throws(() => vectorBranch(1, 2, Infinity, 1), RangeError)
assert.throws(() => tensorLayout(true, 3, 0), RangeError)
console.log('framework tensor models: strides, storage coverage, branch VJP and accumulation passed')
