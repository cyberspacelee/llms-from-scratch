import assert from 'node:assert/strict'
import { tensorLayout, branchGradient } from '../src/lib/framework-tensor-model.ts'
assert.deepEqual(tensorLayout(false, 1, 2).traversal, [0, 1, 2, 3, 4, 5])
assert.deepEqual(tensorLayout(true, 2, 1).traversal, [0, 3, 1, 4, 2, 5])
for (const transposed of [false, true]) {
  const { rows, cols } = tensorLayout(transposed, 0, 0)
  const offsets = new Set()
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) offsets.add(tensorLayout(transposed, r, c).offset)
  assert.equal(offsets.size, 6)
}
assert.deepEqual(branchGradient(2, 1, 7), { square: 4, linear: 6, output: 10, squareGradient: 4, linearGradient: 3, gradient: 7, accumulated: 14 })
assert.equal(branchGradient(-2, 2, 0).gradient, -2)
assert.throws(() => tensorLayout(true, 3, 0), RangeError)
console.log('framework tensor models: strides, storage coverage, branch VJP and accumulation passed')
