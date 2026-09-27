import assert from 'node:assert/strict'
import { sampleDistribution } from '../src/lib/sampling-model.ts'

const close = (actual, expected) => actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-12, `${actual} vs ${expected}`))
const logits = [0.4, 0.3, 0.2, 0.1].map(Math.log)
const base = { temperature: 1, topK: 4, topP: 1, minP: 0, penalty: 1, history: [] }
// The same cases that code/principles/generation.py asserts with PyTorch.
close(sampleDistribution(logits, { ...base, topP: 0.6 }).final, [4 / 7, 3 / 7, 0, 0])
close(sampleDistribution(logits, { ...base, topK: 2 }).final, [4 / 7, 3 / 7, 0, 0])
close(sampleDistribution(logits, { ...base, minP: 0.6 }).final, [4 / 7, 3 / 7, 0, 0])
close(sampleDistribution(logits, { ...base, temperature: 0.5, minP: 0.6 }).final, [1, 0, 0, 0])
close(sampleDistribution(logits, { ...base, temperature: 0.5 }).final, [0.16, 0.09, 0.04, 0.01].map(v => v / 0.3))
close(sampleDistribution(logits, { ...base, penalty: 2, history: [0, 0] }).final, [0.16, 0.3, 0.2, 0.1].map(v => v / 0.76))
// top-p = 0.1 still keeps the maximum; top-p = 0.85 keeps three tokens (0.9 sits on a rounding edge).
assert.equal(sampleDistribution(logits, { ...base, topP: 0.1 }).support, 1)
assert.equal(sampleDistribution(logits, { ...base, topP: 0.85 }).support, 3)
// A positive logit is divided instead of multiplied.
const positive = sampleDistribution([2, 1], { ...base, topK: 2, penalty: 2, history: [0] })
close(positive.penalized, [1, 1]); close(positive.final, [0.5, 0.5])
const shifted = sampleDistribution(logits.map(v => v + 3), { ...base })
close(shifted.final, sampleDistribution(logits, base).final)
assert.ok(Math.abs(sampleDistribution(logits, base).entropy - 1.2798542258336676) < 1e-12)
assert.throws(() => sampleDistribution(logits, { ...base, temperature: 0 }), RangeError)
assert.throws(() => sampleDistribution(logits, { ...base, history: [4] }), RangeError)
console.log('sampling model: penalty, temperature, top-k, top-p crossing, min-p and entropy passed')
