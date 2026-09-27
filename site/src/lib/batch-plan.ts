export function batchPlan(samples: number, batchSize: number, dropLast: boolean) {
  if (![samples, batchSize].every(value => Number.isInteger(value) && value > 0)) throw new RangeError('Expected positive integer counts')
  const batches = []
  for (let start = 0; start < samples; start += batchSize) {
    const indices = Array.from({ length: Math.min(batchSize, samples - start) }, (_, i) => start + i)
    if (dropLast && indices.length < batchSize) break
    const sum = indices.reduce((total, i) => total + (2 * i - 3) ** 2, 0)
    batches.push({ indices, sum, mean: sum / indices.length })
  }
  const retained = batches.reduce((count, b) => count + b.indices.length, 0)
  return { batches, retained, dropped: samples - retained,
    weighted: retained ? batches.reduce((sum, b) => sum + b.sum, 0) / retained : null,
    unweighted: batches.length ? batches.reduce((sum, b) => sum + b.mean, 0) / batches.length : null }
}
