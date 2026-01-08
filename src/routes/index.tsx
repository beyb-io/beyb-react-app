'use client';

import { createFileRoute } from '@tanstack/react-router';
import { useAccount as useWagmiAccount } from 'wagmi';
import { useEffect } from 'react';
import { YieldFarmingHeader } from '@/components/yield-farming/header';
import { YieldFarmingThread } from '@/components/yield-farming/thread';
import { useAppStore } from '@/lib/store';

export const Route = createFileRoute('/')({ component: YieldFarmingApp });

function YieldFarmingApp() {
  const { address, isConnected } = useWagmiAccount();
  const { setConnectedAddress } = useAppStore();

  // Update store when wallet connection changes
  useEffect(() => {
    if (isConnected && address) {
      setConnectedAddress(address);
    } else {
      setConnectedAddress(null);
    }
  }, [address, isConnected, setConnectedAddress]);

  return (
    <div className="flex h-dvh flex-col bg-background">
      {/* <YieldFarmingHeader /> */}

      <div className="flex-1 overflow-hidden">
        <YieldFarmingThread />
      </div>
    </div>
  );
}
