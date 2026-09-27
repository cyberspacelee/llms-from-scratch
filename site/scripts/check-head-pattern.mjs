import assert from 'node:assert/strict'
import { firstHead, headWeights, previousHead } from '../src/lib/head-pattern-model.ts'

const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} vs ${b}`)
const argmax = row => row.indexOf(Math.max(...row))
for (const mode of ['causal', 'bidirectional']) for (const head of [previousHead, firstHead]) {
  for (const row of headWeights(head, mode, 2)) close(row.reduce((a, b) => a + b, 0), 1)
}
const prev = headWeights(previousHead, 'causal', 4)
const first = headWeights(firstHead, 'causal', 4)
for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) { assert.equal(prev[i][j], 0); assert.equal(first[i][j], 0) }
close(prev[0][0], 1)
for (let i = 1; i < 4; i++) { assert.equal(argmax(prev[i]), i - 1); assert.equal(argmax(first[i]), 0) }
// Row 1 of the first-token head: scores 4/sqrt2 and 0 over two visible keys.
close(first[1][0], 1 / (1 + Math.exp(-4 / Math.SQRT2)))
// Without the causal mask, the previous-token head at position 0 finds no earlier key and spreads out.
const open = headWeights(previousHead, 'bidirectional', 4)
assert.ok(open[0][3] > 0 && open[1][3] > 0)
assert.throws(() => headWeights(firstHead, 'causal', 0), RangeError)
console.log('head pattern model: row sums, causal zeros, previous-token and first-token heads passed')
