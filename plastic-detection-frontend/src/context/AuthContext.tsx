import { createContext, useContext, useEffect, useMemo, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import { Api, type AuthResponse } from '../services/api';

interface AuthState {
  user: AuthResponse['user'] | null;
  token: string | null;
  refreshToken: string | null;
  loading: boolean;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const STORAGE_KEY = 'rewater_auth_state';

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [state, setState] = useState<AuthState>({
    user: null,
    token: null,
    refreshToken: null,
    loading: true,
  });

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      try {
        const parsed = JSON.parse(raw) as AuthState;
        setState({ ...parsed, loading: false });
      } catch (error) {
        console.error('Failed to parse auth state', error);
        localStorage.removeItem(STORAGE_KEY);
        setState(prev => ({ ...prev, loading: false }));
      }
    } else {
      setState(prev => ({ ...prev, loading: false }));
    }
  }, []);

  const persistState = useCallback((nextState: AuthState) => {
    setState(nextState);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(nextState));
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const response = await Api.login(email, password);
    persistState({
      user: response.user,
      token: response.token,
      refreshToken: response.refreshToken || null,
      loading: false,
    });
  }, [persistState]);

  const logout = useCallback(() => {
    persistState({
      user: null,
      token: null,
      refreshToken: null,
      loading: false,
    });
  }, [persistState]);

  const value = useMemo<AuthContextValue>(
    () => ({
      ...state,
      login,
      logout,
    }),
    [state.user, state.token, state.refreshToken, state.loading, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
