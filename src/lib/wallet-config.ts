import { getDefaultConfig } from '@rainbow-me/rainbowkit';
import {
  injectedWallet,
  metaMaskWallet,
  okxWallet,
  rabbyWallet,
  rainbowWallet,
  trustWallet,
  walletConnectWallet,
} from '@rainbow-me/rainbowkit/wallets';
import { bsc } from 'wagmi/chains';

// Configure custom RPC for BSC with better rate limits
const bscWithCustomRpc = {
  ...bsc,
  rpcUrls: {
    default: {
      http: [
        // Ankr public RPC (better rate limits than thirdweb)
        'https://rpc.ankr.com/bsc',
      ],
    },
    public: {
      http: [
        'https://rpc.ankr.com/bsc',
        'https://bsc-dataseed1.binance.org',
        'https://bsc-dataseed2.binance.org',
      ],
    },
  },
} as const;

// Configure wagmi and RainbowKit for BSC mainnet
export const wagmiConfig = getDefaultConfig({
  appName: 'Yield Farming Assistant',
  projectId: import.meta.env.VITE_WALLETCONNECT_PROJECT_ID || 'YOUR_PROJECT_ID',
  wallets: [
    {
      groupName: 'Recommended',
      wallets: [
        metaMaskWallet,
        rainbowWallet,
        trustWallet,
        okxWallet,
        rabbyWallet,
        walletConnectWallet,
        injectedWallet,
      ],
    },
  ],
  chains: [bscWithCustomRpc],
  ssr: false,
});

// Custom theme for RainbowKit matching base-maia style
export const rainbowKitTheme = {
  accentColor: 'oklch(0.488 0.243 264.376)', // Primary purple from theme
  accentColorForeground: 'white',
  borderRadius: 'medium',
  fontStack: 'system',
} as const;
