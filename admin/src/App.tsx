import { Routes, Route, Navigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { useAuthStore } from '@/store/auth'
import Layout from '@/components/Layout'
import Login from '@/pages/Login'
import Dashboard from '@/pages/Dashboard'
import KnowledgeBase from '@/pages/KnowledgeBase'
import Prompts from '@/pages/Prompts'
import Calls from '@/pages/Calls'
import Analytics from '@/pages/Analytics'
import Sessions from '@/pages/Sessions'
import SIPSettings from '@/pages/SIPSettings'
import Settings from '@/pages/Settings'
import Logs from '@/pages/Logs'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, token, user } = useAuthStore()
  const [isChecking, setIsChecking] = useState(true)
  
  useEffect(() => {
    // Wait for Zustand to rehydrate from localStorage
    let attempts = 0
    const maxAttempts = 10 // Max 500ms wait
    
    const checkAuth = () => {
      attempts++
      const storeState = useAuthStore.getState()
      const storedToken = localStorage.getItem('admin_token')
      
      // If we have a token in localStorage but store says not authenticated,
      // wait a bit more for rehydration (but not forever)
      if (storedToken && !storeState.isAuthenticated && attempts < maxAttempts) {
        setTimeout(checkAuth, 50)
        return
      }
      
      setIsChecking(false)
    }
    
    // Initial check after a short delay
    const timer = setTimeout(checkAuth, 50)
    return () => clearTimeout(timer)
  }, [])
  
  // Check if we have a valid token - use localStorage as source of truth
  // Only check once after rehydration completes
  if (isChecking) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-slate-500">Loading...</div>
      </div>
    )
  }
  
  // After checking, use store state (which should be synced with localStorage)
  const storedToken = localStorage.getItem('admin_token')
  const hasToken = token || storedToken
  const hasUser = user !== null
  
  // Consider authenticated if we have token and user
  const isAuth = hasToken && hasUser && (isAuthenticated || (storedToken && user))
  
  if (!isAuth) {
    return <Navigate to="/login" replace />
  }
  
  return <Layout>{children}</Layout>
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/knowledge-base"
        element={
          <ProtectedRoute>
            <KnowledgeBase />
          </ProtectedRoute>
        }
      />
      <Route
        path="/prompts"
        element={
          <ProtectedRoute>
            <Prompts />
          </ProtectedRoute>
        }
      />
      <Route
        path="/calls"
        element={
          <ProtectedRoute>
            <Calls />
          </ProtectedRoute>
        }
      />
      <Route
        path="/analytics"
        element={
          <ProtectedRoute>
            <Analytics />
          </ProtectedRoute>
        }
      />
      <Route
        path="/sessions"
        element={
          <ProtectedRoute>
            <Sessions />
          </ProtectedRoute>
        }
      />
      <Route
        path="/sip-settings"
        element={
          <ProtectedRoute>
            <SIPSettings />
          </ProtectedRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ProtectedRoute>
            <Settings />
          </ProtectedRoute>
        }
      />
      <Route
        path="/logs"
        element={
          <ProtectedRoute>
            <Logs />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
