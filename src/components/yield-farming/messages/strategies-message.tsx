import type { YieldStrategy } from '@/lib/mock-data'

type StrategiesMessageProps = {
  strategies: Array<YieldStrategy>
  title?: string
  description?: string
  max?: number
}

const formatTVL = (num: number): string => {
  if (num >= 1_000_000_000) {
    return `$${(num / 1_000_000_000).toFixed(1)}B`
  }
  if (num >= 1_000_000) {
    return `$${(num / 1_000_000).toFixed(1)}M`
  }
  return `$${num.toFixed(0)}`
}

export function StrategiesMessage({
  strategies,
  title = 'Popular strategies',
  description = 'Demo picks based on simulated market data.',
  max = 3,
}: StrategiesMessageProps) {
  const popularStrategies = strategies.slice(0, max)

  const lines = [
    title,
    description,
    '',
    ...popularStrategies.map((strategy, index) => {
      const tokens = strategy.tokens.map((token) => token.symbol).join(' / ')
      return `${index + 1}. ${strategy.name} — ${strategy.apy}% APY, ${strategy.riskLevel} risk, TVL ${formatTVL(strategy.tvl)} (${tokens})`
    }),
  ]

  return (
    <div className="space-y-3">
      <div className="whitespace-pre-wrap">{lines.join('\n')}</div>
    </div>
  )
}
