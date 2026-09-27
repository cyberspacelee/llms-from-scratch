import assert from 'node:assert/strict'
import { matrixIndex, threadCoordinates, threadNumber } from '../src/components/labs/GpuIndexModel.ts'

const writes = []
for (let by = 0; by < 3; by++) for (let bx = 0; bx < 2; bx++) {
  for (let ty = 0; ty < 2; ty++) for (let tx = 0; tx < 4; tx++) {
    const index = matrixIndex(bx, by, tx, ty)
    if (index.valid) writes.push(index.offset)
  }
}
assert.deepEqual(writes.sort((a, b) => a - b), Array.from({ length: 35 }, (_, i) => i))
assert.deepEqual(matrixIndex(1, 0, 3, 0), { row: 0, col: 7, offset: 7, valid: false, offsetOnly: true })
assert.deepEqual(matrixIndex(1, 2, 2, 0), { row: 4, col: 6, offset: 34, valid: true, offsetOnly: true })
assert.equal(threadNumber(0, 0, 1, 8, 4), 32)
for (const [width, height, depth] of [[8, 4, 2], [8, 8, 1], [16, 4, 1], [32, 2, 1]]) {
  const seen = new Set()
  for (let z = 0; z < depth; z++) for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const t = threadNumber(x, y, z, width, height)
    const coordinates = threadCoordinates(t, width, height)
    assert.deepEqual(coordinates, { x, y, z, warp: t < 32 ? 0 : 1, lane: t % 32 })
    seen.add(t)
  }
  assert.equal(seen.size, 64)
}
assert.deepEqual(threadCoordinates(32, 8, 8), { x: 0, y: 4, z: 0, warp: 1, lane: 0 })
assert.deepEqual(threadCoordinates(32, 16, 4), { x: 0, y: 2, z: 0, warp: 1, lane: 0 })
assert.deepEqual(threadCoordinates(32, 32, 2), { x: 0, y: 1, z: 0, warp: 1, lane: 0 })
console.log('GPU indexing: 35 unique outputs, offset-only counterexample, 3D round trip and warp layouts passed.')
