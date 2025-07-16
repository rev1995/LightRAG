import { useCallback } from 'react'
import { QueryMode, QueryRequest } from '@/api/lightrag'
// Removed unused import for Text component
import Checkbox from '@/components/ui/Checkbox'
import Input from '@/components/ui/Input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { useSettingsStore } from '@/stores/settings'
import { useTranslation } from 'react-i18next'
import Textarea from '@/components/ui/Textarea'

export default function QuerySettings() {
  const { t } = useTranslation()
  const querySettings = useSettingsStore((state) => state.querySettings)

  const handleChange = useCallback((key: keyof QueryRequest, value: any) => {
    useSettingsStore.getState().updateQuerySettings({ [key]: value })
  }, [])

  return (
    <Card className="flex shrink-0 flex-col min-w-[220px]">
      <CardHeader className="px-4 pt-4 pb-2">
        <CardTitle>{t('retrievePanel.querySettings.parametersTitle')}</CardTitle>
        <CardDescription className="sr-only">{t('retrievePanel.querySettings.parametersDescription')}</CardDescription>
      </CardHeader>
      <CardContent className="m-0 flex grow flex-col p-0 text-xs">
        <div className="relative size-full">
          <div className="absolute inset-0 flex flex-col gap-2 overflow-auto px-2">
            {/* Query Mode */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="query_mode_select" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.queryMode')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.queryModeTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Select
                value={querySettings.mode}
                onValueChange={(v) => handleChange('mode', v as QueryMode)}
              >
                <SelectTrigger
                  id="query_mode_select"
                  className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="naive">{t('retrievePanel.querySettings.queryModeOptions.naive')}</SelectItem>
                    <SelectItem value="local">{t('retrievePanel.querySettings.queryModeOptions.local')}</SelectItem>
                    <SelectItem value="global">{t('retrievePanel.querySettings.queryModeOptions.global')}</SelectItem>
                    <SelectItem value="hybrid">{t('retrievePanel.querySettings.queryModeOptions.hybrid')}</SelectItem>
                    <SelectItem value="mix">{t('retrievePanel.querySettings.queryModeOptions.mix')}</SelectItem>
                    <SelectItem value="bypass">{t('retrievePanel.querySettings.queryModeOptions.bypass')}</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </>

            {/* User Prompt */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="user_prompt" className="ml-1 cursor-help">
                      User Prompt
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>Custom prompt to guide LLM on how to process retrieved results. Does not participate in RAG retrieval phase.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Textarea
                id="user_prompt"
                value={querySettings.user_prompt || ''}
                onChange={(e) => handleChange('user_prompt', e.target.value)}
                placeholder="e.g., For diagrams, use mermaid format with English/Pinyin node names and Chinese display labels"
                className="min-h-[60px] resize-none"
                rows={3}
              />
            </>

            {/* Enable Rerank */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="enable_rerank" className="ml-1 cursor-help">
                      Enable Rerank
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>Enable reranking for retrieved text chunks to improve relevance</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="enable_rerank"
                  checked={querySettings.enable_rerank !== false}
                  onCheckedChange={(checked) => handleChange('enable_rerank', checked)}
                />
                <label htmlFor="enable_rerank" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                  Enable
                </label>
              </div>
            </>

            {/* Response Format */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="response_format_select" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.responseFormat')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.responseFormatTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Select
                value={querySettings.response_type}
                onValueChange={(v) => handleChange('response_type', v)}
              >
                <SelectTrigger
                  id="response_format_select"
                  className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="Multiple Paragraphs">{t('retrievePanel.querySettings.responseFormatOptions.multipleParagraphs')}</SelectItem>
                    <SelectItem value="Single Paragraph">{t('retrievePanel.querySettings.responseFormatOptions.singleParagraph')}</SelectItem>
                    <SelectItem value="Bullet Points">{t('retrievePanel.querySettings.responseFormatOptions.bulletPoints')}</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </>

            {/* Top K */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="top_k" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.topK')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.topKTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <Input
                  id="top_k"
                  type="number"
                  value={querySettings.top_k ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange('top_k', value === '' ? '' : parseInt(value) || 0)
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(parseInt(value))) {
                      handleChange('top_k', 1)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.topKPlaceholder')}
                />
              </div>
            </>

            {/* Chunk Top K */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="chunk_top_k" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.chunkTopK')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.chunkTopKTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <Input
                  id="chunk_top_k"
                  type="number"
                  value={querySettings.chunk_top_k ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange('chunk_top_k', value === '' ? '' : parseInt(value) || 0)
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(parseInt(value))) {
                      handleChange('chunk_top_k', 1)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.chunkTopKPlaceholder')}
                />
              </div>
            </>

            {/* Max Entity Tokens */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="max_entity_tokens" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.maxEntityTokens')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.maxEntityTokensTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <Input
                  id="max_entity_tokens"
                  type="number"
                  value={querySettings.max_entity_tokens ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange('max_entity_tokens', value === '' ? '' : parseInt(value) || 0)
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(parseInt(value))) {
                      handleChange('max_entity_tokens', 1000)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.maxEntityTokensPlaceholder')}
                />
              </div>
            </>

            {/* Max Relation Tokens */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="max_relation_tokens" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.maxRelationTokens')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.maxRelationTokensTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <Input
                  id="max_relation_tokens"
                  type="number"
                  value={querySettings.max_relation_tokens ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange('max_relation_tokens', value === '' ? '' : parseInt(value) || 0)
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(parseInt(value))) {
                      handleChange('max_relation_tokens', 1000)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.maxRelationTokensPlaceholder')}
                />
              </div>
            </>

            {/* Max Total Tokens */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="max_total_tokens" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.maxTotalTokens')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.maxTotalTokensTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <Input
                  id="max_total_tokens"
                  type="number"
                  value={querySettings.max_total_tokens ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange('max_total_tokens', value === '' ? '' : parseInt(value) || 0)
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(parseInt(value))) {
                      handleChange('max_total_tokens', 32000)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.maxTotalTokensPlaceholder')}
                />
              </div>
            </>

            {/* History Turns */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="history_turns" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.historyTurns')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.historyTurnsTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <Input
                  id="history_turns"
                  type="number"
                  value={querySettings.history_turns ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange('history_turns', value === '' ? '' : parseInt(value) || 0)
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(parseInt(value))) {
                      handleChange('history_turns', 3)
                    }
                  }}
                  min={0}
                  placeholder={t('retrievePanel.querySettings.historyTurnsPlaceholder')}
                />
              </div>
            </>

            {/* Only Need Context */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="only_need_context" className="ml-1 cursor-help">
                      Only Need Context
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>Return only retrieved context without generating a response</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="only_need_context"
                  checked={querySettings.only_need_context || false}
                  onCheckedChange={(checked) => handleChange('only_need_context', checked)}
                />
                <label htmlFor="only_need_context" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                  Enable
                </label>
              </div>
            </>

            {/* Only Need Prompt */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="only_need_prompt" className="ml-1 cursor-help">
                      Only Need Prompt
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>Return only the generated prompt without producing a response</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="only_need_prompt"
                  checked={querySettings.only_need_prompt || false}
                  onCheckedChange={(checked) => handleChange('only_need_prompt', checked)}
                />
                <label htmlFor="only_need_prompt" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                  Enable
                </label>
              </div>
            </>

            {/* Stream */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="stream" className="ml-1 cursor-help">
                      Stream Response
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>Enable streaming output for real-time responses</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="stream"
                  checked={querySettings.stream || false}
                  onCheckedChange={(checked) => handleChange('stream', checked)}
                />
                <label htmlFor="stream" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                  Enable
                </label>
              </div>
            </>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
