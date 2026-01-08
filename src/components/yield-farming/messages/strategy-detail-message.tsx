import { findStrategyById } from '@/lib/mock-data'

type StrategyDetailMessageProps = {
  strategyId: string
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

export function StrategyDetailMessage({ strategyId }: StrategyDetailMessageProps) {
  const strategy = findStrategyById(strategyId)

  if (!strategy) {
    return (
      <div className="whitespace-pre-wrap text-muted-foreground">
        {'Strategy not found.\nThis demo strategy is no longer available.'}
      </div>
    )
  }

  const tokens = strategy.tokens.map((token) => token.symbol).join(' / ')
  const lines = [
    strategy.name,
    `${strategy.protocol} · ${strategy.classification}`,
    strategy.description,
    '',
    `APY: ${strategy.apy}%`,
    `Risk: ${strategy.riskLevel}`,
    `TVL: ${formatTVL(strategy.tvl)}`,
    `Tokens: ${tokens}`,
    '',
    'How it works:',
    ...strategy.steps.map(
      (step, index) => `${index + 1}) ${step.description}`,
    ),
  ]

  return (
    <div className="space-y-3">
      <div className="whitespace-pre-wrap">{lines.join('\n')}</div>
    </div>
  )
}
