import { defineConfig } from 'vite'
import path from 'path'
import { webuiPrefix } from './src/lib/constants'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  base: webuiPrefix,
  build: {
    outDir: path.resolve(__dirname, '../lightrag/api/webui'),
    emptyOutDir: true,
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'graph-vendor': ['sigma', 'graphology', '@react-sigma/core'],
          'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-popover', '@radix-ui/react-select', '@radix-ui/react-tabs'],
          'utils-vendor': ['axios', 'i18next', 'zustand', 'clsx', 'tailwind-merge'],
          'feature-graph': ['./src/features/GraphViewer'],
          'feature-documents': ['./src/features/DocumentManager'],
          'feature-retrieval': ['./src/features/RetrievalTesting'],
          'mermaid-vendor': ['mermaid'],
          'markdown-vendor': [
            'react-markdown',
            'rehype-react',
            'remark-gfm',
            'remark-math',
            'react-syntax-highlighter'
          ]
        },
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    }
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})