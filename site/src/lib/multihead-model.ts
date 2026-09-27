/** Axis bookkeeping of multi-head attention for one batch element, as row-major storage offsets. */
export type HeadStage = 'linear' | 'split' | 'heads' | 'merged' | 'wrong'

export type HeadConfig = { length: number; heads: number; headWidth: number }

export function checkConfig({ length, heads, headWidth }: HeadConfig) {
  for (const value of [length, heads, headWidth])
    if (!Number.isInteger(value) || value < 1 || value > 8) throw new RangeError('T, n_q and d_h must be integers in 1–8')
}

const stridesOf = (shape: number[]) => shape.map((_, axis) => shape.slice(axis + 1).reduce((a, b) => a * b, 1))

/** Shape and element strides (per batch element) of each stage. */
export function stageLayout(stage: HeadStage, config: HeadConfig) {
  checkConfig(config)
  const { length: T, heads: H, headWidth: D } = config
  const d = H * D
  switch (stage) {
    case 'linear': return { shape: [T, d], stride: [d, 1], contiguous: true }
    case 'split': return { shape: [T, H, D], stride: [d, D, 1], contiguous: true }
    // transpose(1, 2) swaps strides without moving data
    case 'heads': return { shape: [H, T, D], stride: [D, d, 1], contiguous: false }
    case 'merged': return { shape: [T, d], stride: [d, 1], contiguous: true }
    case 'wrong': return { shape: [H, T, D], stride: stridesOf([H, T, D]), contiguous: true }
  }
}

/**
 * Where the linear-output element (token t, feature c) appears in a stage.
 * Returns the stage coordinate, and the (token, feature) it is read as.
 */
export function locate(stage: HeadStage, config: HeadConfig, t: number, c: number) {
  checkConfig(config)
  const { length: T, heads: H, headWidth: D } = config
  const d = H * D
  if (!Number.isInteger(t) || !Number.isInteger(c) || t < 0 || t >= T || c < 0 || c >= d) throw new RangeError('Element outside (T, d)')
  const head = Math.floor(c / D), within = c % D
  const offset = t * d + c
  switch (stage) {
    case 'linear':
    case 'merged': return { coord: [t, c], offset, head, token: t }
    case 'split': return { coord: [t, head, within], offset, head, token: t }
    case 'heads': return { coord: [head, t, within], offset, head, token: t }
    case 'wrong': {
      // reshape(H, T, D) of the (T, d) buffer reinterprets the same flat offsets
      const h = Math.floor(offset / (T * D)), row = Math.floor((offset % (T * D)) / D)
      return { coord: [h, row, offset % D], offset, head: h, token: row }
    }
  }
}

/**
 * Source (token, head) pairs found in one row of a per-head view. The correct transpose
 * gives exactly the pair (row, head); the direct reshape fills rows with other tokens or heads.
 */
export function rowSources(stage: HeadStage, config: HeadConfig, head: number, row: number) {
  if (stage !== 'heads' && stage !== 'wrong') throw new RangeError('Only per-head stages have head rows')
  const { length: T, heads: H, headWidth: D } = config
  const sources = new Map<string, { token: number; head: number }>()
  for (let t = 0; t < T; t++) for (let c = 0; c < H * D; c++) {
    const place = locate(stage, config, t, c)
    if (place.coord[0] === head && place.coord[1] === row) sources.set(`${t}:${Math.floor(c / D)}`, { token: t, head: Math.floor(c / D) })
  }
  return [...sources.values()]
}

/** Rows of a per-head view whose contents are not (that row's token, that panel's head). */
export function misplacedRows(stage: HeadStage, config: HeadConfig) {
  let count = 0
  for (let h = 0; h < config.heads; h++) for (let row = 0; row < config.length; row++) {
    const sources = rowSources(stage, config, h, row)
    if (sources.length !== 1 || sources[0].token !== row || sources[0].head !== h) count++
  }
  return count
}

/** Attention cost of the split: score elements per batch element, and projection weights. */
export function headCost({ length: T, heads: H, headWidth: D }: HeadConfig) {
  checkConfig({ length: T, heads: H, headWidth: D })
  const d = H * D
  return { width: d, scoreElements: H * T * T, projectionWeights: 4 * d * d }
}
