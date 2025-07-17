import { useState, useCallback, useEffect, useRef } from 'react'
import ThemeProvider from '@/components/ThemeProvider'
import TabVisibilityProvider from '@/contexts/TabVisibilityProvider'
import ApiKeyAlert from '@/components/ApiKeyAlert'
import StatusIndicator from '@/components/status/StatusIndicator'
import { healthCheckInterval } from '@/lib/constants'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import SiteHeader from '@/features/SiteHeader'
import { InvalidApiKeyError, RequireApiKeError } from '@/api/lightrag'

import GraphViewer from '@/features/GraphViewer'
import DocumentManager from '@/features/DocumentManager'
import RetrievalTesting from '@/features/RetrievalTesting'
import ApiSite from '@/features/ApiSite'

import { Tabs, TabsContent } from '@/components/ui/Tabs'

function App() {
  const message = useBackendState.use.message()
  const enableHealthCheck = useSettingsStore.use.enableHealthCheck()
  const currentTab = useSettingsStore.use.currentTab()
  const [apiKeyAlertOpen, setApiKeyAlertOpen] = useState(false)
  const healthCheckInitializedRef = useRef(false); // Prevent duplicate health checks in Vite dev mode

  const handleApiKeyAlertOpenChange = useCallback((open: boolean) => {
    setApiKeyAlertOpen(open)
    if (!open) {
      useBackendState.getState().clear()
    }
  }, [])

  // Track component mount status with useRef
  const isMountedRef = useRef(true);

  // Set up mount/unmount status tracking
  useEffect(() => {
    isMountedRef.current = true;

    // Handle page reload/unload
    const handleBeforeUnload = () => {
      isMountedRef.current = false;
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      isMountedRef.current = false;
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  // Health check - can be disabled
  useEffect(() => {
    // Only execute if health check is enabled and ApiKeyAlert is closed
    if (!enableHealthCheck || apiKeyAlertOpen) return;

    // Health check function
    const performHealthCheck = async () => {
      try {
        // Only perform health check if component is still mounted
        if (isMountedRef.current) {
          await useBackendState.getState().check();
        }
      } catch (error) {
        console.error('Health check error:', error);
      }
    };

    // On first mount or when enableHealthCheck becomes true and apiKeyAlertOpen is false,
    // perform an immediate health check
    if (!healthCheckInitializedRef.current) {
      healthCheckInitializedRef.current = true;
      // Immediate health check on first load
      performHealthCheck();
    }

    // Set interval for periodic execution
    const interval = setInterval(performHealthCheck, healthCheckInterval * 1000);
    return () => clearInterval(interval);
  }, [enableHealthCheck, apiKeyAlertOpen]);

  const handleTabChange = useCallback(
    (tab: string) => useSettingsStore.getState().setCurrentTab(tab as any),
    []
  )

  useEffect(() => {
    if (message) {
      if (message.includes(InvalidApiKeyError) || message.includes(RequireApiKeError)) {
        setApiKeyAlertOpen(true)
      }
    }
  }, [message])

  return (
    <ThemeProvider>
      <TabVisibilityProvider>
        <main className="flex h-screen w-screen overflow-hidden">
          <Tabs
            defaultValue={currentTab}
            className="!m-0 flex grow flex-col !p-0 overflow-hidden"
            onValueChange={handleTabChange}
          >
            <SiteHeader />
            <div className="relative grow">
              <TabsContent value="documents" className="absolute top-0 right-0 bottom-0 left-0 overflow-auto">
                <DocumentManager />
              </TabsContent>
              <TabsContent value="knowledge-graph" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                <GraphViewer />
              </TabsContent>
              <TabsContent value="retrieval" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                <RetrievalTesting />
              </TabsContent>
              <TabsContent value="api" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                <ApiSite />
              </TabsContent>
            </div>
          </Tabs>
          {enableHealthCheck && <StatusIndicator />}
          <ApiKeyAlert open={apiKeyAlertOpen} onOpenChange={handleApiKeyAlertOpenChange} />
        </main>
      </TabVisibilityProvider>
    </ThemeProvider>
  )
}

export default App