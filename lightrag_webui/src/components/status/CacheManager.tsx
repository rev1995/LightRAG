import * as React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Checkbox from '@/components/ui/Checkbox'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { clearCache } from '@/api/lightrag'
import { useTranslation } from 'react-i18next'
import { Trash2Icon, RefreshCwIcon } from 'lucide-react'

interface CacheMode {
  id: string
  label: string
  description: string
  checked: boolean
}

export default function CacheManager() {
  const { t } = useTranslation()
  const [isLoading, setIsLoading] = React.useState(false)
  const [lastClearTime, setLastClearTime] = React.useState<string | null>(null)
  const [clearResult, setClearResult] = React.useState<{ status: string; message: string } | null>(null)

  const [cacheModes, setCacheModes] = React.useState<CacheMode[]>([
    {
      id: 'query',
      label: 'Query Cache',
      description: 'Clear cached LLM responses for queries',
      checked: true
    },
    {
      id: 'entity_extract',
      label: 'Entity Extraction Cache',
      description: 'Clear cached entity extraction results',
      checked: true
    },
    {
      id: 'relation_extract',
      label: 'Relation Extraction Cache',
      description: 'Clear cached relation extraction results',
      checked: true
    },
    {
      id: 'summary',
      label: 'Summary Cache',
      description: 'Clear cached document and entity summaries',
      checked: true
    }
  ])

  const handleModeToggle = (modeId: string, checked: boolean) => {
    setCacheModes((prev: CacheMode[]) => 
      prev.map((mode: CacheMode) => 
        mode.id === modeId ? { ...mode, checked } : mode
      )
    )
  }

  const handleSelectAll = (checked: boolean) => {
    setCacheModes((prev: CacheMode[]) => prev.map((mode: CacheMode) => ({ ...mode, checked })))
  }

  const handleClearCache = async () => {
    const selectedModes = cacheModes
      .filter(mode => mode.checked)
      .map(mode => mode.id)

    if (selectedModes.length === 0) {
      setClearResult({
        status: 'warning',
        message: 'Please select at least one cache mode to clear'
      })
      return
    }

    setIsLoading(true)
    setClearResult(null)

    try {
      const result = await clearCache(selectedModes)
      setClearResult(result)
      setLastClearTime(new Date().toLocaleString())
    } catch (error) {
      setClearResult({
        status: 'error',
        message: error instanceof Error ? error.message : 'Failed to clear cache'
      })
    } finally {
      setIsLoading(false)
    }
  }

  const selectedCount = cacheModes.filter(mode => mode.checked).length
  const allSelected = cacheModes.every(mode => mode.checked)

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Trash2Icon className="h-5 w-5" />
          Cache Management
        </CardTitle>
        <CardDescription>
          Clear LLM response cache with different modes to free up memory and ensure fresh responses
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Select All Toggle */}
        <div className="flex items-center space-x-2">
          <Checkbox
            id="select-all"
            checked={allSelected}
            onCheckedChange={handleSelectAll}
          />
          <label htmlFor="select-all" className="text-sm font-medium">
            Select All Cache Modes
          </label>
        </div>

        {/* Cache Mode Options */}
        <div className="space-y-3">
          {cacheModes.map((mode) => (
            <div key={mode.id} className="flex items-start space-x-3">
              <Checkbox
                id={mode.id}
                checked={mode.checked}
                onCheckedChange={(checked) => handleModeToggle(mode.id, checked)}
                className="mt-1"
              />
              <div className="flex-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label 
                        htmlFor={mode.id} 
                        className="text-sm font-medium cursor-help"
                      >
                        {mode.label}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="right">
                      <p>{mode.description}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <p className="text-xs text-muted-foreground mt-1">
                  {mode.description}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Clear Button */}
        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            {selectedCount} of {cacheModes.length} modes selected
          </div>
          <Button
            onClick={handleClearCache}
            disabled={isLoading || selectedCount === 0}
            variant="destructive"
            size="sm"
            className="flex items-center gap-2"
          >
            {isLoading ? (
              <RefreshCwIcon className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2Icon className="h-4 w-4" />
            )}
            {isLoading ? 'Clearing...' : 'Clear Selected Cache'}
          </Button>
        </div>

        {/* Result Display */}
        {clearResult && (
          <div className={`p-3 rounded-md text-sm ${
            clearResult.status === 'success' 
              ? 'bg-green-50 text-green-800 border border-green-200'
              : clearResult.status === 'warning'
              ? 'bg-yellow-50 text-yellow-800 border border-yellow-200'
              : 'bg-red-50 text-red-800 border border-red-200'
          }`}>
            <div className="font-medium">
              {clearResult.status === 'success' ? '✅ Success' : 
               clearResult.status === 'warning' ? '⚠️ Warning' : '❌ Error'}
            </div>
            <div className="mt-1">{clearResult.message}</div>
          </div>
        )}

        {/* Last Clear Time */}
        {lastClearTime && (
          <div className="text-xs text-muted-foreground">
            Last cleared: {lastClearTime}
          </div>
        )}

        {/* Cache Information */}
        <div className="text-xs text-muted-foreground space-y-1">
          <p><strong>Query Cache:</strong> Stores LLM responses for repeated queries</p>
          <p><strong>Entity Extraction Cache:</strong> Stores extracted entities from documents</p>
          <p><strong>Relation Extraction Cache:</strong> Stores extracted relationships between entities</p>
          <p><strong>Summary Cache:</strong> Stores document and entity summaries</p>
        </div>
      </CardContent>
    </Card>
  )
} 