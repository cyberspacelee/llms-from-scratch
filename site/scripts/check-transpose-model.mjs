// node --experimental-strip-types site/scripts/check-transpose-model.mjs
import assert from 'node:assert/strict'
import { transposeElement } from '../src/lib/transpose-model.ts'

for (const pitch of [32, 33]) for (const edge of [false, true]) {
  let valid = 0
  const output = new Set()
  for (let row = 0; row < 32; row++) for (let col = 0; col < 32; col++) {
    const element = transposeElement(row, col, pitch, edge)
    assert.equal(element.producer.y + element.producer.j, row)
    assert.equal(element.producer.x, col)
    assert.equal(element.consumer.x, row)
    assert.equal(element.consumer.y + element.consumer.j, col)
    assert.equal(element.bank, element.sharedWord % 32)
    const readers = edge ? col < 3 ? 3 : 0 : 32
    assert.equal(element.activeReaders, readers)
    assert.equal(new Set(element.columnReadBanks).size, readers === 0 ? 0 : pitch === 32 ? 1 : readers)
    assert.equal(element.conflict, readers === 0 ? 0 : pitch === 32 ? readers : 1)
    if (element.valid) { valid++; output.add(element.outputWord) }
  }
  assert.equal(valid, edge ? 9 : 1024)
  assert.equal(output.size, valid)
}
assert.equal(transposeElement(2, 5, 32, false).bank, 5)
assert.equal(transposeElement(2, 5, 33, false).bank, 7)
assert.equal(transposeElement(2, 2, 33, true).valid, true)
assert.equal(transposeElement(3, 2, 33, true).valid, false)
assert.throws(() => transposeElement(32, 0, 33, false), RangeError)
console.log('transpose model: producer/consumer coverage, 32/33 banks, edge masks passed')
