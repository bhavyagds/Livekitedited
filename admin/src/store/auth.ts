import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface User {
  id: string
  email: string
  name: string
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  login: (token: string, user: User) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (token: string, user: User) => {
        // Store in both localStorage and Zustand persist
        localStorage.setItem('admin_token', token)
        set({ user, token, isAuthenticated: true })
      },
      logout: () => {
        localStorage.removeItem('admin_token')
        set({ user: null, token: null, isAuthenticated: false })
      },
    }),
    {
      name: 'admin-auth',
      partialize: (state) => ({ 
        user: state.user, 
        token: state.token, 
        isAuthenticated: !!state.token && !!state.user 
      }),
      onRehydrateStorage: () => (state, error) => {
        if (error) {
          console.error('Failed to rehydrate auth store:', error)
          return
        }
        // Sync localStorage token with store on rehydration
        if (state) {
          const storedToken = localStorage.getItem('admin_token')
          // Always sync token from localStorage on rehydration
          if (storedToken) {
            state.token = storedToken
            // If we have token and user, we're authenticated
            // If user is missing but token exists, try to restore from persisted state
            if (!state.user && storedToken) {
              // Try to get user from persisted state
              try {
                const persisted = localStorage.getItem('admin-auth')
                if (persisted) {
                  const parsed = JSON.parse(persisted)
                  if (parsed.state?.user) {
                    state.user = parsed.state.user
                  }
                }
              } catch (e) {
                // Ignore parse errors
              }
            }
            state.isAuthenticated = !!state.user && !!storedToken
          } else {
            // No token in localStorage, clear everything
            state.token = null
            state.user = null
            state.isAuthenticated = false
          }
        }
      },
    }
  )
)
