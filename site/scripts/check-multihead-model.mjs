import assert from 'node:assert/strict'
import { headCost, locate, misplacedRows, rowSources, stageLayout } from '../src/lib/multihead-model.ts'

const config = { length: 4, heads: 2, headWidth: 2 }
const dot = (coord, stride) => coord.reduce((sum, value, axis) => sum + value * stride[axis], 0)
// Every stage coordinate must address the same storage offset through that stage's strides.
for (const stage of ['linear', 'split', 'heads', 'merged', 'wrong']) {
  const { shape, stride } = stageLayout(stage, config)
  const seen = new Set()
  for (let t = 0; t < 4; t++) for (let c = 0; c < 4; c++) {
    const place = locate(stage, config, t, c)
    assert.equal(dot(place.coord, stride), place.offset, `${stage} ${t},${c}`)
    place.coord.forEach((value, axis) => assert.ok(value >= 0 && value < shape[axis]))
    seen.add(place.coord.join(','))
  }
  assert.equal(seen.size, 16, `${stage} is a bijection`)
}
assert.deepEqual(stageLayout('heads', config).stride, [2, 4, 1])
assert.equal(stageLayout('heads', config).contiguous, false)
assert.deepEqual(locate('heads', config, 1, 2).coord, [1, 1, 0])
// The textbook shape walk from P3: (2,4,4) -> (2,4,2,2) -> (2,2,4,2).
assert.deepEqual(stageLayout('split', config).shape, [4, 2, 2])
assert.deepEqual(stageLayout('heads', config).shape, [2, 4, 2])
for (let row = 0; row < 4; row++) for (let head = 0; head < 2; head++)
  assert.deepEqual(rowSources('heads', config, head, row), [{ token: row, head }])
assert.equal(misplacedRows('heads', config), 0)
// Reshaping without the transpose: head 0, row 1 holds token 0's head-1 features.
assert.deepEqual(rowSources('wrong', config, 0, 1), [{ token: 0, head: 1 }])
assert.deepEqual(locate('wrong', config, 0, 2).coord, [0, 1, 0])
assert.equal(misplacedRows('wrong', config), 6)
assert.equal(misplacedRows('wrong', { length: 3, heads: 1, headWidth: 2 }), 0)
assert.deepEqual(headCost(config), { width: 4, scoreElements: 32, projectionWeights: 64 })
assert.equal(headCost({ length: 4, heads: 4, headWidth: 1 }).projectionWeights, 64)
assert.throws(() => locate('linear', config, 4, 0), RangeError)
assert.throws(() => rowSources('linear', config, 0, 0), RangeError)
console.log('multi-head axis model: strides, bijections, per-head row sources and wrong reshape passed')
