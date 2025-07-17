import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import AppRouter from './AppRouter'
import { initializeI18n } from './i18n';

initializeI18n().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <AppRouter />
    </StrictMode>
  )
});