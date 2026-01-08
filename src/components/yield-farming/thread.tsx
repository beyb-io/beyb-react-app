import { useCallback, useEffect, useRef, useState } from 'react'
import { IconLoader2, IconSend } from '@tabler/icons-react'
import {
  PromptInput,
  PromptInputAction,
  PromptInputActions,
  PromptInputTextarea,
} from '@/components/prompt-kit/prompt-input'
import { Message, MessageContent } from '@/components/prompt-kit/message'
import { PromptSuggestion } from '@/components/prompt-kit/prompt-suggestion'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'
import { simulateAiRequest } from '@/lib/fake-ai'
import { MOCK_STRATEGIES } from '@/lib/mock-data'
import { useTokenBalances } from '@/hooks/use-token-balances'
import { BalanceMessage } from '@/components/yield-farming/messages/balance-message'
import { formatUSDCompact } from '@/lib/format'
import { StrategiesMessage } from '@/components/yield-farming/messages/strategies-message'
import { StrategyDetailMessage } from '@/components/yield-farming/messages/strategy-detail-message'
import {
  TransactionConfirmActions,
  TransactionConfirmBody,
  TransactionConfirmProvider,
} from '@/components/yield-farming/messages/transaction-confirm-message'
import { TextShimmer } from '@/components/prompt-kit/text-shimmer'
import { ChatConnectButton } from '@/components/yield-farming/chat-connect-button'
import { ConnectButton } from '@rainbow-me/rainbowkit'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  type:
    | 'text'
    | 'balance'
    | 'strategies'
    | 'strategy-detail'
    | 'strategy-confirm'
  content?: string
  strategyId?: string
  title?: string
}

const THINKING_DELAY_MS = 500

const delay = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms)
  })

const createMessageId = () => {
  if (typeof globalThis.crypto.randomUUID === 'function') {
    return globalThis.crypto.randomUUID()
  }
  return `msg_${Date.now()}_${Math.random().toString(16).slice(2)}`
}

export function YieldFarmingThread() {
  const [messages, setMessages] = useState<Array<ChatMessage>>(() => [
    {
      id: createMessageId(),
      role: 'assistant',
      type: 'text',
      content:
        'Welcome to BEYB. Explore curated yield strategies, track your wallet balance, and get guided ideas for putting idle assets to work.',
    },
    {
      id: createMessageId(),
      role: 'assistant',
      type: 'strategies',
      title: 'What now we have',
    },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [thinkingLabel, setThinkingLabel] = useState<string | null>(null)
  const [pendingBalanceRequestId, setPendingBalanceRequestId] = useState<
    string | null
  >(null)
  const [hasBalanceSnapshot, setHasBalanceSnapshot] = useState(false)
  const [balanceSnapshotAddress, setBalanceSnapshotAddress] = useState<
    string | null
  >(null)
  const endRef = useRef<HTMLDivElement | null>(null)

  const startThinking = useCallback((label: string) => {
    setThinkingLabel(label)
    setIsLoading(true)
  }, [])

  const stopThinking = useCallback(() => {
    setIsLoading(false)
    setThinkingLabel(null)
  }, [])

  const { connectedAddress, mockBalances, positions } = useAppStore()
  const {
    balances,
    totalUSD,
    isLoading: balancesLoading,
    error: balancesError,
  } = useTokenBalances()
  const popularStrategies = MOCK_STRATEGIES.slice(0, 3)
  const isWalletConnected = Boolean(connectedAddress)

  const balanceSuggestionLabel = connectedAddress
    ? balancesLoading && !hasBalanceSnapshot
      ? 'Get balance - ...'
      : `Get balance - ${formatUSDCompact(totalUSD)}`
    : 'Get balance'

  const submitMessage = async (override?: string) => {
    const trimmed = (override ?? input).trim()
    if (!trimmed || isLoading) return

    const userMessage = {
      id: createMessageId(),
      role: 'user' as const,
      type: 'text' as const,
      content: trimmed,
    }

    setInput('')
    const normalized = trimmed.toLowerCase()
    const isBalanceQuery =
      normalized.includes('balance') || normalized.includes('wallet')
    const isPopularStrategiesQuery =
      normalized.includes('popular') && normalized.includes('strateg')

    if (isBalanceQuery) {
      setMessages((prev) => [...prev, userMessage])
      const shouldWaitForBalances =
        Boolean(connectedAddress) && !hasBalanceSnapshot && balancesLoading
      if (shouldWaitForBalances) {
        setPendingBalanceRequestId(createMessageId())
        startThinking('Loading balances...')
        return
      }
      setMessages((prev) => [
        ...prev,
        { id: createMessageId(), role: 'assistant', type: 'balance' },
      ])
      return
    }

    if (isPopularStrategiesQuery) {
      setMessages((prev) => [...prev, userMessage])
      startThinking('Loading strategies...')
      await delay(THINKING_DELAY_MS)
      stopThinking()
      setMessages((prev) => [
        ...prev,
        { id: createMessageId(), role: 'assistant', type: 'strategies' },
      ])
      return
    }

    setMessages((prev) => [...prev, userMessage])
    startThinking('Thinking...')

    try {
      const response = await simulateAiRequest({
        prompt: trimmed,
        walletAddress: connectedAddress,
        balances: mockBalances,
        positions,
      })

      setMessages((prev) => [
        ...prev,
        {
          id: createMessageId(),
          role: 'assistant' as const,
          type: 'text' as const,
          content: response.message,
        },
      ])
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: createMessageId(),
          role: 'assistant' as const,
          type: 'text' as const,
          content:
            error instanceof Error
              ? `Demo backend error: ${error.message}`
              : 'Demo backend error: Unable to fetch a response.',
        },
      ])
    } finally {
      stopThinking()
    }
  }

  const showStrategyDetails = async (strategyId: string) => {
    if (isLoading) return
    startThinking('Loading strategy...')
    await delay(THINKING_DELAY_MS)
    stopThinking()
    setMessages((prev) => [
      ...prev,
      {
        id: createMessageId(),
        role: 'assistant',
        type: 'strategy-detail',
        strategyId,
      },
    ])
  }

  const showStrategyConfirm = async (strategyId: string) => {
    if (isLoading) return
    startThinking('Preparing transaction...')
    await delay(THINKING_DELAY_MS)
    stopThinking()
    setMessages((prev) => [
      ...prev,
      {
        id: createMessageId(),
        role: 'assistant',
        type: 'strategy-confirm',
        strategyId,
      },
    ])
  }

  const handleSubmit = (event?: any) => {
    event?.preventDefault()
    void submitMessage()
  }

  const handleSuggestion = (value: string) => {
    void submitMessage(value)
  }

  const isSubmitDisabled = isLoading || input.trim().length === 0

  useEffect(() => {
    if (connectedAddress !== balanceSnapshotAddress) {
      setBalanceSnapshotAddress(connectedAddress)
      setHasBalanceSnapshot(false)
    }
  }, [connectedAddress, balanceSnapshotAddress])

  useEffect(() => {
    if (!connectedAddress || balancesLoading || hasBalanceSnapshot) return
    setHasBalanceSnapshot(true)
  }, [connectedAddress, balancesLoading, hasBalanceSnapshot])

  useEffect(() => {
    if (!pendingBalanceRequestId || balancesLoading) return
    stopThinking()
    setPendingBalanceRequestId(null)
    setMessages((prev) => [
      ...prev,
      { id: createMessageId(), role: 'assistant', type: 'balance' },
    ])
  }, [balancesLoading, pendingBalanceRequestId, stopThinking])

  useEffect(() => {
    if (!endRef.current) return
    requestAnimationFrame(() => {
      setTimeout(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
      }, 50)
    })
  }, [messages, isLoading, balancesLoading, hasBalanceSnapshot])

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto bg-background px-4 pb-6 pt-8">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
          {messages.map((message) =>
            message.role === 'user' ? (
              <Message key={message.id} className="justify-end">
                <MessageContent className="max-w-[80%] bg-primary/15 text-foreground">
                  {message.content ?? ''}
                </MessageContent>
              </Message>
            ) : message.type === 'balance' ? (
              <Message key={message.id}>
                {/* <MessageAvatar src="/logo192.png" alt="BEYB" fallback="BY" /> */}
                <div className="flex max-w-[80%] flex-col gap-2">
                  <MessageContent>
                    <BalanceMessage
                      balances={balances}
                      totalUSD={totalUSD}
                      isLoading={balancesLoading && !hasBalanceSnapshot}
                      error={balancesError}
                      address={connectedAddress}
                      isConnected={isWalletConnected}
                    />
                  </MessageContent>
                  {!isWalletConnected && (
                    <div className="flex flex-wrap gap-2">
                      <ChatConnectButton />
                    </div>
                  )}
                </div>
              </Message>
            ) : message.type === 'strategies' ? (
              <Message key={message.id}>
                {/* <MessageAvatar src="/logo192.png" alt="BEYB" fallback="BY" /> */}
                <div className="flex max-w-[80%] flex-col gap-2">
                  <MessageContent>
                    <StrategiesMessage
                      strategies={popularStrategies}
                      title={message.title}
                    />
                  </MessageContent>
                  <div className="flex flex-wrap gap-2">
                    {popularStrategies.map((strategy) => (
                      <Button
                        key={strategy.id}
                        disabled={isLoading}
                        onClick={() => showStrategyDetails(strategy.id)}
                      >
                        Details: {strategy.name}
                      </Button>
                    ))}
                  </div>
                </div>
              </Message>
            ) : message.type === 'strategy-detail' ? (
              <Message key={message.id}>
                {/* <MessageAvatar src="/logo192.png" alt="BEYB" fallback="BY" /> */}
                <div className="flex max-w-[80%] flex-col gap-2">
                  <MessageContent>
                    {message.strategyId ? (
                      <StrategyDetailMessage strategyId={message.strategyId} />
                    ) : (
                      <div className="text-muted-foreground">
                        Strategy details are unavailable.
                      </div>
                    )}
                  </MessageContent>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      disabled={isLoading || !message.strategyId}
                      onClick={() =>
                        message.strategyId
                          ? showStrategyConfirm(message.strategyId)
                          : null
                      }
                    >
                      Get yield
                    </Button>
                  </div>
                </div>
              </Message>
            ) : message.type === 'strategy-confirm' ? (
              <Message key={message.id}>
                {/* <MessageAvatar src="/logo192.png" alt="BEYB" fallback="BY" /> */}
                {message.strategyId ? (
                  <TransactionConfirmProvider
                    strategyId={message.strategyId}
                    disabled={isLoading}
                  >
                    <div className="flex max-w-[80%] flex-col gap-2">
                      <MessageContent>
                        <TransactionConfirmBody />
                      </MessageContent>
                      <div className="flex flex-wrap gap-2">
                        <TransactionConfirmActions />
                      </div>
                    </div>
                  </TransactionConfirmProvider>
                ) : (
                  <MessageContent className="max-w-[80%]">
                    <div className="text-muted-foreground">
                      Transaction details are unavailable.
                    </div>
                  </MessageContent>
                )}
              </Message>
            ) : (
              <Message key={message.id}>
                {/* <MessageAvatar src="/logo192.png" alt="BEYB" fallback="BY" /> */}
                <MessageContent className="max-w-[80%]" markdown>
                  {message.content ?? ''}
                </MessageContent>
              </Message>
            ),
          )}

          {isLoading && (
            <Message>
              {/* <MessageAvatar src="/logo192.png" alt="BEYB" fallback="BY" /> */}
              <MessageContent className="max-w-[80%]">
                <TextShimmer className="text-sm">
                  {thinkingLabel ?? 'Thinking...'}
                </TextShimmer>
              </MessageContent>
            </Message>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-border bg-black/5 p-4">
        <div className="mx-auto w-full max-w-3xl space-y-4">
          <div className="flex flex-nowrap gap-2 overflow-x-auto pb-1 md:flex-wrap md:overflow-visible [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
            <PromptSuggestion
              variant="outline"
              className="shrink-0 bg-white"
              disabled={isLoading}
              onClick={() => handleSuggestion('Show wallet balance')}
            >
              {balanceSuggestionLabel}
            </PromptSuggestion>
            <PromptSuggestion
              variant="outline"
              className="shrink-0 bg-white"
              disabled={isLoading}
              onClick={() => handleSuggestion('Show popular strategies')}
            >
              Show popular strategies
            </PromptSuggestion>

            <ConnectButton.Custom>
              {({
                account,
                chain,
                openAccountModal,
                openChainModal,
                openConnectModal,
                authenticationStatus,
                mounted,
              }) => {
                const ready = mounted && authenticationStatus !== 'loading'
                const connected =
                  ready &&
                  account &&
                  chain &&
                  (!authenticationStatus ||
                    authenticationStatus === 'authenticated')

                if (!ready) return null
                if (!connected) {
                  return (
                    <PromptSuggestion
                      variant="default"
                      className="shrink-0 md:ml-auto"
                      disabled={isLoading}
                      onClick={openConnectModal}
                    >
                      Connect wallet
                    </PromptSuggestion>
                  )
                }
                if (chain.unsupported) {
                  return (
                    <PromptSuggestion
                      variant="outline"
                      className="shrink-0 md:ml-auto"
                      disabled={isLoading}
                      onClick={openChainModal}
                    >
                      Wrong network
                    </PromptSuggestion>
                  )
                }
                return (
                  <PromptSuggestion
                    variant="outline"
                    className="shrink-0 md:ml-auto"
                    disabled={isLoading}
                    onClick={openAccountModal}
                  >
                    Wallet: {account.displayName}
                  </PromptSuggestion>
                )
              }}
            </ConnectButton.Custom>
          </div>
          <PromptInput
            value={input}
            onValueChange={setInput}
            onSubmit={handleSubmit}
            isLoading={isLoading}
            disabled={isLoading}
          >
            <PromptInputTextarea placeholder="Ask about yield farming strategies..." />
            <PromptInputActions className="justify-end">
              <PromptInputAction tooltip="Send message">
                <Button
                  size="icon"
                  disabled={isSubmitDisabled}
                  onClick={(event) => handleSubmit(event)}
                >
                  {isLoading ? (
                    <IconLoader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <IconSend className="h-4 w-4" />
                  )}
                </Button>
              </PromptInputAction>
            </PromptInputActions>
          </PromptInput>
        </div>
      </div>
    </div>
  )
}
