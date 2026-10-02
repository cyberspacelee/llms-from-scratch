import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { sampleDistribution } from '../src/lib/sampling-model.ts'
import { cachedChapterAttention } from '../src/lib/principles-trace-model.ts'
import { onlineAttention } from '../src/lib/systems-interactive-model.ts'
import { vectorBranch } from '../src/lib/framework-tensor-model.ts'
import { maskedLoss } from '../src/lib/masked-loss-model.ts'

const root = fileURLToPath(new URL('../../', import.meta.url))
const regenerated = spawnSync('uv', ['run', 'python', 'scripts/export_site_traces.py', '--check'], { cwd: root, encoding: 'utf8' })
assert.equal(regenerated.status, 0, regenerated.stderr || regenerated.stdout)
const cases = JSON.parse(readFileSync(new URL('../src/data/reference-traces.json', import.meta.url), 'utf8'))
assert.equal(cases.schemaVersion, 1)
assert.equal(cases.dtype, 'float64')
const close = (a, b) => {
  if (Array.isArray(a)) { assert.equal(a.length, b.length); a.forEach((value, index) => close(value, b[index])); return }
  assert.ok(Number.isFinite(a) && Math.abs(a - b) <= cases.atol, `${a} != Python ${b}`)
}
for (const entry of cases.sampling) close(sampleDistribution(entry.logits, entry.options).final, entry.probabilities)
for (const entry of cases.cachedAttention) close(cachedChapterAttention(entry.past, entry.length, entry.row, false).output, entry.output)
for (const entry of cases.onlineAttention) close(onlineAttention(entry.position, entry.block, Math.ceil(6 / entry.block)).output, entry.output)
for (const entry of cases.maskedLoss) {
  const actual = maskedLoss(entry.targets, entry.ignored, 'mean')
  assert.equal(actual.valid, entry.valid)
  close(actual.total, entry.total)
  if (entry.mean !== null) close(actual.result, entry.mean)
  else { assert.equal(actual.empty, true); assert.equal(actual.result, 0) }
}
for (const entry of cases.vectorBranch) {
  const actual = vectorBranch(entry.w0, entry.w1, entry.bias, entry.seed)
  for (const key of ['u', 'loss', 'gu', 'gw', 'gb']) close(actual[key], entry[key])
}
console.log(`${cases.sampling.length + cases.cachedAttention.length + cases.onlineAttention.length + cases.maskedLoss.length + cases.vectorBranch.length} Python/TypeScript reference cases passed (sampling, RoPE/cache, online softmax, masked loss, vector autograd)`)
