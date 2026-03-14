import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: (failureCount, error: any) => {
        // Don't retry on 401 errors (authentication issues)
        if (error?.response?.status === 401) {
          return false
        }
        return failureCount < 1
      },
      refetchOnWindowFocus: false, // Disable refetch on tab switch to prevent logout
      refetchOnMount: true, // Still refetch when component mounts
      refetchOnReconnect: false, // Don't refetch on reconnect to avoid logout issues
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
