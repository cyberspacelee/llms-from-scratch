import assert from 'node:assert/strict'
import { storageSelection, axisCalculation, contraction } from '../src/components/labs/NumpyContractModel.ts'

for (const mode of ['view', 'copy', 'advanced', 'assign']) for (let cell = 0; cell < 4; cell++) {
  for (const value of [-5, 0, 15]) {
    const result = storageSelection(mode, cell, value)
    assert.equal(result.selected[cell], value)
    assert.equal(result.source[[1, 2, 4, 5][cell]], ['view', 'assign'].includes(mode) ? value : [1, 2, 4, 5][cell])
    assert.equal(result.shared, mode === 'view')
  }
}
assert.deepEqual(axisCalculation('feature', 1, true).values, [11, 22, 33, 14, 25, 36])
assert.deepEqual(axisCalculation('sample', 1, true).values, [101, 102, 103, 204, 205, 206])
assert.equal(axisCalculation('invalid', 1, true).valid, false)
assert.deepEqual(axisCalculation('mean', 0, false).values, [2.5, 3.5, 4.5])
assert.equal(axisCalculation('mean', 1, true).shape, '(2,1)')
assert.equal(axisCalculation('mean', 1, false).shape, '(2,)')
for (let row = 0; row < 2; row++) for (let output = 0; output < 2; output++) {
  assert.equal(contraction(row, output).value, [[-2, 4], [-2, 13]][row][output])
}
console.log('NumPy labs: selection writeback, broadcast, axis shapes and contraction verified.')
