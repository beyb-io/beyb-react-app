import type { TokenBalance } from '@/hooks/use-token-balances'
import { formatUSDCompact } from '@/lib/format'

type BalanceMessageProps = {
  balances: Array<TokenBalance>
  totalUSD: number
  isLoading: boolean
  error?: Error | null
  address?: string | null
  isConnected: boolean
}

const shortenAddress = (address: string) => {
  return `${address.slice(0, 6)}...${address.slice(-4)}`
}

export function BalanceMessage({
  balances,
  totalUSD,
  isLoading,
  error,
  address,
  isConnected,
}: BalanceMessageProps) {
  if (!isConnected) {
    return (
      <div className="space-y-2">
        <div className="whitespace-pre-wrap">
          {'Wallet balance\nConnect a wallet to load live balances.'}
        </div>
      </div>
    )
  }

  const headerLines = [
    'Wallet balance',
    `Address: ${address ? shortenAddress(address) : 'Connected wallet'}`,
    `Total: ${formatUSDCompact(totalUSD)}`,
  ]

  if (error) {
    headerLines.push('Error: Failed to load balances.')
  }

  const tokenLines = isLoading
    ? ['Loading token balances...']
    : balances.length === 0
      ? ['No balances available for this wallet.']
      : balances.map(
          (token) =>
            `- ${token.symbol}: ${token.balance} (${token.valueUSD.toFixed(2)} USD)`,
        )

  return (
    <div className="space-y-3">
      <div className="whitespace-pre-wrap">{headerLines.join('\n')}</div>
      <div className="whitespace-pre-wrap">{tokenLines.join('\n')}</div>
    </div>
  )
}
