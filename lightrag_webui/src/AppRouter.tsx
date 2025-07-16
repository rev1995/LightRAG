import { HashRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'sonner'
import App from './App'
import ThemeProvider from '@/components/ThemeProvider'

const AppRouter = () => {
  return (
    <ThemeProvider>
      <Router>
        <Routes>
          <Route path="/*" element={<App />} />
        </Routes>
        <Toaster
          position="bottom-center"
          theme="system"
          closeButton
          richColors
        />
      </Router>
    </ThemeProvider>
  )
}

export default AppRouter
