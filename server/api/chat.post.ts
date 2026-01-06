import { anthropic } from '@ai-sdk/anthropic';
import {
  convertToCoreMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  streamText,
  tool,
} from 'ai';
import { z } from 'zod';
import {
  executeGetWalletBalances,
  executeClassifyTokens,
  executeFindYieldStrategies,
  executeGetStrategyDetails,
  executePreviewDeposit,
  executeDeposit,
  executeGetPositions,
  executeWithdrawFromPosition,
} from '~/lib/ai-tools';

const SYSTEM_PROMPT = `You are a helpful and knowledgeable DeFi yield farming assistant. Your role is to help users understand yield farming, discover opportunities, and manage their positions.

Key guidelines:
- Be friendly and educational, especially for users new to DeFi
- Explain concepts clearly without being condescending
- When a wallet is connected, use the available tools to provide personalized recommendations
- Always mention risks when discussing strategies
- Use the preview_deposit tool before executing any deposits
- Format numbers clearly (e.g., "5.2% APY", "$1,234.56")
- When showing strategies, highlight the key differences (APY, risk level, protocol)

Remember:
- This is a demo/prototype with simulated transactions
- All transactions are on testnet (no real money)
- Tool calls help provide rich, interactive experiences

When wallet is NOT connected:
- Provide general educational content about yield farming
- Explain different strategies and protocols
- Encourage users to connect their wallet for personalized recommendations

When wallet IS connected:
- Use tools to read balances and suggest strategies based on their actual holdings
- Provide specific, actionable recommendations
- Guide users through deposit and withdrawal flows step by step`;

export default defineEventHandler(async (event) => {
  try {
    const body = await readBody(event);
    const { messages, walletAddress, mockBalances, positions } = body;

    if (!messages || !Array.isArray(messages)) {
      throw createError({
        statusCode: 400,
        message: 'Invalid request: messages array required'
      });
    }

    const apiKey = process.env.ANTHROPIC_API_KEY;
    const hasApiKey = apiKey && apiKey !== 'sk-ant-your-key-here';
    if (!hasApiKey) {
      const balances = Array.isArray(mockBalances) ? mockBalances : [];
      const positionsList = Array.isArray(positions) ? positions : [];
      const totalPositionsValue = positionsList.reduce(
        (sum, position) => sum + Number(position?.currentValue || 0),
        0
      );
      const balancesResult = executeGetWalletBalances(balances);
      const tokenSymbols = balances.map((balance) => balance?.token?.symbol).filter(Boolean);
      const strategyResult = executeFindYieldStrategies({
        tokens: tokenSymbols.length > 0 ? tokenSymbols : undefined,
      });
      const suggestedStrategies = [...strategyResult.strategies]
        .sort((a, b) => b.apy - a.apy)
        .slice(0, 3);

      const formatUsd = (value: number) =>
        `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      const formatAddress = (address: string) =>
        address.length > 10 ? `${address.slice(0, 6)}...${address.slice(-4)}` : address;

      let demoText = 'Demo mode: ANTHROPIC_API_KEY is not set, so responses are scripted.';

      if (walletAddress) {
        demoText += `\n\nWallet: ${formatAddress(walletAddress)}`;
        demoText += `\nTotal balance: ${formatUsd(balancesResult.totalValueUSD)}`;
        if (balancesResult.balances.length > 0) {
          const balanceLines = balancesResult.balances.map((balance) => {
            const symbol = balance.token?.symbol || 'TOKEN';
            return `- ${symbol}: ${balance.balance} (~${formatUsd(balance.valueUSD)})`;
          });
          demoText += `\n\nBalances:\n${balanceLines.join('\n')}`;
        } else {
          demoText += '\n\nBalances: none yet. Connect a wallet to generate mock balances.';
        }

        if (positionsList.length > 0) {
          demoText += `\n\nActive positions: ${positionsList.length} (total ${formatUsd(totalPositionsValue)})`;
        } else {
          demoText += '\n\nActive positions: none yet.';
        }

        if (suggestedStrategies.length > 0) {
          const strategyLines = suggestedStrategies.map(
            (strategy) =>
              `- ${strategy.name} (${strategy.protocol}) — ${strategy.apy.toFixed(1)}% APY, ${strategy.riskLevel} risk`
          );
          demoText += `\n\nSuggested strategies:\n${strategyLines.join('\n')}`;
        }
      } else {
        demoText +=
          '\n\nYield farming means putting tokens into protocols to earn rewards. Common strategies include lending, staking, and providing liquidity.';
        demoText +=
          '\nConnect a wallet to see personalized mock balances and strategy suggestions.';
      }

      demoText += '\n\nThis is a demo with simulated transactions. Always consider risks and impermanent loss.';

      const stream = createUIMessageStream({
        execute({ writer }) {
          writer.write({ type: 'start' });
          writer.write({ type: 'start-step' });
          writer.write({ type: 'text-start', id: 'text-1' });
          writer.write({ type: 'text-delta', id: 'text-1', delta: demoText });
          writer.write({ type: 'text-end', id: 'text-1' });
          writer.write({ type: 'finish-step' });
          writer.write({ type: 'finish', finishReason: 'stop' });
        },
      });

      return createUIMessageStreamResponse({ stream });
    }

    // Add wallet context to system prompt
    let systemPrompt = SYSTEM_PROMPT;
    if (walletAddress) {
      systemPrompt += `\n\nWallet Status: CONNECTED
Connected Address: ${walletAddress}
Available Balances: ${JSON.stringify(mockBalances || [])}
Active Positions: ${positions?.length || 0} position(s)

Use the available tools to help this user manage their yield farming positions.`;
    } else {
      systemPrompt += `\n\nWallet Status: NOT CONNECTED
Provide general information and encourage the user to connect their wallet for personalized recommendations.`;
    }

    // Define tools using Vercel AI SDK format
    const result = streamText({
      model: anthropic('claude-3-5-sonnet-20241022'),
      messages: convertToCoreMessages(messages),
      system: systemPrompt,
      maxTokens: 4096,
      tools: {
        get_wallet_balances: tool({
          description: 'Get token balances for the connected wallet',
          parameters: z.object({}),
          execute: async () => executeGetWalletBalances(mockBalances || []),
        }),
        classify_tokens: tool({
          description: 'Get educational info about token classifications',
          parameters: z.object({
            tokens: z.array(z.string()).optional(),
          }),
          execute: async ({ tokens }) => executeClassifyTokens(tokens),
        }),
        find_yield_strategies: tool({
          description: 'Find yield farming strategies',
          parameters: z.object({
            tokens: z.array(z.string()).optional(),
            minAPY: z.number().optional(),
            riskLevel: z.enum(['Low', 'Medium', 'High']).optional(),
          }),
          execute: async (params) => executeFindYieldStrategies(params),
        }),
        get_strategy_details: tool({
          description: 'Get detailed info about a specific strategy',
          parameters: z.object({
            strategyId: z.string(),
          }),
          execute: async ({ strategyId }) => executeGetStrategyDetails(strategyId),
        }),
        preview_deposit: tool({
          description: 'Preview a deposit transaction',
          parameters: z.object({
            strategyId: z.string(),
            amount: z.string(),
            tokenSymbol: z.string(),
          }),
          execute: async ({ strategyId, amount, tokenSymbol }) =>
            executePreviewDeposit(strategyId, amount, tokenSymbol),
        }),
        execute_deposit: tool({
          description: 'Execute a deposit (simulated)',
          parameters: z.object({
            strategyId: z.string(),
            amount: z.string(),
            tokenSymbol: z.string(),
          }),
          execute: async () => {
            // This will be called from client-side with store action
            return { success: false, error: 'Use client-side deposit action' };
          },
        }),
        get_positions: tool({
          description: 'Get all active positions',
          parameters: z.object({}),
          execute: async () => executeGetPositions(positions || []),
        }),
        withdraw_from_position: tool({
          description: 'Withdraw from a position (simulated)',
          parameters: z.object({
            positionId: z.string(),
          }),
          execute: async () => {
            // This will be called from client-side with store action
            return { success: false, error: 'Use client-side withdraw action' };
          },
        }),
      },
    });

    return result.toUIMessageStreamResponse();
  } catch (error) {
    console.error('API Error:', error);
    throw createError({
      statusCode: 500,
      message: error instanceof Error ? error.message : 'Unknown error occurred'
    });
  }
});
