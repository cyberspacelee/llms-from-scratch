export type InfraChapter = {
  slug: string
  nav: string
  title: string
  question: string
}

export const aiInfraChapters: InfraChapter[] = [
  {
    slug: 'ai-infra',
    nav: '导读',
    title: '从一次请求到一簇机器',
    question: '这条路线为什么按记账、架构、硬件、引擎、集群来拆',
  },
  {
    slug: 'ai-infra-01',
    nav: '01 三本账',
    title: '一次生成的三本账',
    question: '一次生成里，时间、容量和数据搬移各记在哪里',
  },
  {
    slug: 'ai-infra-02',
    nav: '02 架构',
    title: '架构改写哪一笔账',
    question: '注意力变体、分数矩阵和专家结构怎样改写容量',
  },
  {
    slug: 'ai-infra-03',
    nav: '03 负载',
    title: '同一模型的两套账单',
    question: 'prefill、decode 和训练为什么不是同一笔资源账',
  },
  {
    slug: 'ai-infra-04',
    nav: '04 加速器',
    title: '算力、带宽与片上存储',
    question: '一张加速器先给出哪些算力和带宽上限',
  },
  {
    slug: 'ai-infra-05',
    nav: '05 Kernel',
    title: '让读进来的数据被复用',
    question: '分块、融合和 Tensor Core 怎样少访问显存',
  },
  {
    slug: 'ai-infra-06',
    nav: '06 测量',
    title: '先写下可证伪的假设',
    question: '时间线、单 kernel 和编译产物各自回答什么',
  },
  {
    slug: 'ai-infra-07',
    nav: '07 推理引擎',
    title: '把空转填上',
    question: '调度、分页 KV、量化和投机解码各改变哪一笔账',
  },
  {
    slug: 'ai-infra-08',
    nav: '08 集群',
    title: '从一卡到一簇',
    question: '并行、网络、训练和端边云怎样为容量支付通信',
  },
]
