export function matrixIndex(bx: number, by: number, tx: number, ty: number) {
  const row = by * 2 + ty
  const col = bx * 4 + tx
  const offset = row * 7 + col
  return { row, col, offset, valid: row < 5 && col < 7, offsetOnly: offset < 35 }
}

export function threadCoordinates(thread: number, width: number, height: number) {
  return { x: thread % width, y: Math.floor(thread / width) % height, z: Math.floor(thread / (width * height)), warp: Math.floor(thread / 32), lane: thread % 32 }
}

export function threadNumber(x: number, y: number, z: number, width: number, height: number) {
  return x + width * (y + height * z)
}
