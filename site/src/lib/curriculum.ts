/** Routes describe a reader's goal; folder order only organizes the library. */
export const learningPaths = [
  {
    id: 'first-model', title: '从文本组装一个模型',
    description: '追踪 token、预测目标、注意力、位置与结构，最后运行完整 Decoder。',
    steps: ['principles/tokenization', 'principles/language-modeling', 'principles/attention', 'principles/decoder', 'principles/attention-order', 'principles/sinusoidal', 'principles/rope', 'principles/normalization', 'principles/head-sharing', 'principles/gated-ffn', 'principles/generation', 'principles/kv-cache', 'principles/complete-transformer'],
  },
  {
    id: 'text-training', title: '训练并恢复一个文本模型',
    description: '从目标与数据窗口出发，审计原料后完成更新、评估与恢复。',
    steps: ['training/data', 'training/data-engineering', 'training/optimization', 'training/pretraining', 'training/evaluation', 'training/checkpoint'],
  },
  {
    id: 'adaptation', title: '适配已有模型',
    description: '确定回复监督，选择参数更新方式，再比较教师、偏好和规则奖励。',
    steps: ['post-training/instruction-tuning', 'post-training/lora', 'post-training/distillation', 'post-training/dpo', 'post-training/reasoning-rl'],
  },
  {
    id: 'serving', title: '解释一轮生成与服务',
    description: '确认生成与缓存先修后，沿资源、请求和执行理解服务，并用同负载评价。',
    steps: ['systems/ledger', 'systems/accelerator', 'systems/flash-attention', 'systems/engine-runtime', 'systems/paged-cache', 'systems/engine-execution', 'systems/serving-evaluation'],
  },
  {
    id: 'kernel', title: '写并测量一个 kernel',
    description: '以输出负责人、访存、分块和真实计时验证执行改动。',
    steps: ['gpu/execution-model', 'gpu/memory-and-sync', 'gpu/kernel-design', 'gpu/measurement'],
  },
] as const

export function routeFor(path: string) {
  return learningPaths.find(route => (route.steps as readonly string[]).includes(path))
}
