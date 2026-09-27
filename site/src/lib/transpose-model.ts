/** The exact dim3(32, 8) mapping used by transpose_reduce.cu. */
export function transposeElement(row: number, col: number, pitch: number, edge: boolean) {
  if (![row, col].every(value => Number.isInteger(value) && value >= 0 && value < 32) ||
      ![32, 33].includes(pitch)) throw new RangeError('Expected tile coordinates and pitch 32 or 33')
  const blockX = edge ? 2 : 0, blockY = edge ? 1 : 0
  const inputRow = blockY * 32 + row, inputCol = blockX * 32 + col
  const activeReaders = edge ? col < 3 ? 3 : 0 : 32
  return {
    inputRow, inputCol, valid: inputRow < 35 && inputCol < 67,
    producer: { x: col, y: row % 8, j: Math.floor(row / 8) * 8 },
    consumer: { x: row, y: col % 8, j: Math.floor(col / 8) * 8 },
    sharedWord: row * pitch + col,
    bank: (row * pitch + col) % 32,
    inputWord: inputRow * 67 + inputCol,
    outputWord: inputCol * 35 + inputRow,
    columnReadBanks: Array.from({ length: activeReaders }, (_, lane) => (lane * pitch + col) % 32),
    activeReaders,
    conflict: activeReaders === 0 ? 0 : pitch === 32 ? activeReaders : 1,
  }
}
