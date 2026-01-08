import { createContext, useCallback, useContext, useMemo } from 'react'
import { keccak256, stringToHex } from 'viem'
import { useAccount, useSendTransaction } from 'wagmi'
import type { ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { findStrategyById } from '@/lib/mock-data'
import { ChatConnectButton } from '@/components/yield-farming/chat-connect-button'

type TransactionConfirmProviderProps = {
  strategyId: string
  disabled?: boolean
  amount?: string
  tokenSymbol?: string
  children: ReactNode
}

const formatAddress = (value: string) => {
  if (value.length <= 16) return value
  return `${value.slice(0, 8)}...${value.slice(-6)}`
}

type TransactionConfirmContextValue = {
  strategyName: string
  simulatedAmount: string
  simulatedToken: string
  fakeTo: `0x${string}`
  fakeData: `0x${string}`
  isConnected: boolean
  isPending: boolean
  error: Error | null
  hash?: string
  disabled: boolean
  handleConfirm: () => void
}

const TransactionConfirmContext =
  createContext<TransactionConfirmContextValue | null>(null)

const useTransactionConfirmContext = () => {
  return useContext(TransactionConfirmContext)
}

export function TransactionConfirmProvider({
  strategyId,
  disabled = false,
  amount,
  tokenSymbol,
  children,
}: TransactionConfirmProviderProps) {
  const { isConnected } = useAccount()
  const { sendTransaction, isPending, error } = useSendTransaction()
  const strategy = findStrategyById(strategyId)
  const simulatedToken = tokenSymbol ?? strategy?.tokens[0]?.symbol ?? 'TOKEN'
  const simulatedAmount = amount ?? '100'
  const strategyName = strategy?.name ?? 'Demo strategy'

  const fakeTo = useMemo<`0x${string}`>(() => {
    const hash = keccak256(stringToHex(`beyb:${strategyId}`))
    return `0x${hash.slice(-40)}`
  }, [strategyId])

  const fakeData = useMemo(
    () =>
      stringToHex(
        `DEMO_DEPOSIT:${strategyId}:${simulatedAmount}:${simulatedToken}`,
      ),
    [strategyId, simulatedAmount, simulatedToken],
  )

  const handleConfirm = useCallback(() => {
    if (!isConnected || disabled) return
    sendTransaction({
      to: fakeTo,
      value: 0n,
      data: fakeData,
    })
  }, [disabled, fakeData, fakeTo, isConnected, sendTransaction])

  const value: TransactionConfirmContextValue = {
    strategyName,
    simulatedAmount,
    simulatedToken,
    fakeTo,
    fakeData,
    isConnected,
    isPending,
    error: error ?? null,
    disabled,
    handleConfirm,
  }

  return (
    <TransactionConfirmContext.Provider value={value}>
      {children}
    </TransactionConfirmContext.Provider>
  )
}

export function TransactionConfirmBody() {
  const context = useTransactionConfirmContext()
  if (!context) return null

  const details = [
    'Confirm transaction',
    'This opens your wallet for a simulated deposit.',
    '',
    `Strategy: ${context.strategyName}`,
    `Amount: ${context.simulatedAmount} ${context.simulatedToken}`,
    'Value: 0 BNB',
  ]

  return (
    <div className="space-y-3">
      <div className="whitespace-pre-wrap">{details.join('\n')}</div>
      <div className="space-y-1 text-xs font-mono text-muted-foreground">
        <div>To: {context.fakeTo}</div>
        <div>Data: {context.fakeData}</div>
      </div>
      {!context.isConnected && (
        <div className="text-xs text-destructive">
          Connect your wallet to continue.
        </div>
      )}
      {context.isPending && (
        <div className="text-xs text-muted-foreground">
          Waiting for wallet confirmation...
        </div>
      )}
      {context.error && (
        <div className="text-xs text-destructive">
          {context.error.message}
        </div>
      )}
      {context.hash && (
        <div className="text-xs text-muted-foreground">
          Tx hash: {formatAddress(context.hash)}
        </div>
      )}
    </div>
  )
}

export function TransactionConfirmActions() {
  const context = useTransactionConfirmContext()
  if (!context) return null

  if (!context.isConnected) {
    return <ChatConnectButton />
  }

  return (
    <Button
      disabled={
        context.disabled ||
        context.isPending ||
        Boolean(context.hash)
      }
      onClick={context.handleConfirm}
    >
      {context.isPending ? 'Waiting...' : 'Confirm in wallet'}
    </Button>
  )
}
