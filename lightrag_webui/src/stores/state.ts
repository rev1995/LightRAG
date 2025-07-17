import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { checkHealth, LightragStatus } from '@/api/lightrag'
import { useSettingsStore } from './settings'

interface BackendState {
  health: boolean
  message: string | null
  messageTitle: string | null
  status: LightragStatus | null
  lastCheckTime: number
  pipelineBusy: boolean
  coreVersion: string | null
  apiVersion: string | null
  webuiTitle: string | null
  webuiDescription: string | null

  check: () => Promise<boolean>
  clear: () => void
  setErrorMessage: (message: string, messageTitle: string) => void
  setPipelineBusy: (busy: boolean) => void
  setVersion: (coreVersion: string | null, apiVersion: string | null) => void
  setCustomTitle: (webuiTitle: string | null, webuiDescription: string | null) => void
}

const useBackendStateStoreBase = create<BackendState>()((set) => ({
  health: true,
  message: null,
  messageTitle: null,
  lastCheckTime: Date.now(),
  status: null,
  pipelineBusy: false,
  coreVersion: null,
  apiVersion: null,
  webuiTitle: null,
  webuiDescription: null,

  check: async () => {
    const health = await checkHealth()
    if (health.status === 'healthy') {
      // Update version information if health check returns it
      if (health.core_version || health.api_version) {
        set({
          coreVersion: health.core_version || null,
          apiVersion: health.api_version || null
        });
      }

      // Update custom title information if health check returns it
      if ('webui_title' in health || 'webui_description' in health) {
        set({
          webuiTitle: 'webui_title' in health ? (health.webui_title ?? null) : null,
          webuiDescription: 'webui_description' in health ? (health.webui_description ?? null) : null
        });
      }

      // Extract and store backend max graph nodes limit
      if (health.configuration?.max_graph_nodes) {
        const maxNodes = parseInt(health.configuration.max_graph_nodes, 10)
        if (!isNaN(maxNodes) && maxNodes > 0) {
          const currentBackendMaxNodes = useSettingsStore.getState().backendMaxGraphNodes

          // Only update if the backend limit has actually changed
          if (currentBackendMaxNodes !== maxNodes) {
            useSettingsStore.getState().setBackendMaxGraphNodes(maxNodes)

            // Auto-adjust current graphMaxNodes if it exceeds the new backend limit
            const currentMaxNodes = useSettingsStore.getState().graphMaxNodes
            if (currentMaxNodes > maxNodes) {
              useSettingsStore.getState().setGraphMaxNodes(maxNodes, true)
            }
          }
        }
      }

      set({
        health: true,
        message: null,
        messageTitle: null,
        lastCheckTime: Date.now(),
        status: health,
        pipelineBusy: health.pipeline_busy
      })
      return true
    }
    set({
      health: false,
      message: health.message,
      messageTitle: 'Backend Health Check Error!',
      lastCheckTime: Date.now(),
      status: null
    })
    return false
  },

  clear: () => {
    set({ health: true, message: null, messageTitle: null })
  },

  setErrorMessage: (message: string, messageTitle: string) => {
    set({ health: false, message, messageTitle })
  },

  setPipelineBusy: (busy: boolean) => {
    set({ pipelineBusy: busy })
  },

  setVersion: (coreVersion: string | null, apiVersion: string | null) => {
    set({ coreVersion, apiVersion })
  },

  setCustomTitle: (webuiTitle: string | null, webuiDescription: string | null) => {
    set({ webuiTitle, webuiDescription })
  }
}))

const useBackendState = createSelectors(useBackendStateStoreBase)

export { useBackendState }