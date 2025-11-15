import { useState } from 'react';
import type { FormEvent } from 'react';
import { Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Lock, Mail, ArrowRight } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Login.css';

const Login = () => {
  const { login, user, loading } = useAuth();
  const [email, setEmail] = useState('admin@rewater.com');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const state = location.state as { from?: { pathname?: string } } | undefined;
  const from = state?.from?.pathname || '/dashboard';

  if (!loading && user) {
    return <Navigate to={from} replace />;
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError('');
    setSubmitting(true);

    try {
      await login(email, password);
      navigate(from, { replace: true });
    } catch (err) {
      console.error(err);
      setError('Invalid credentials or server unavailable.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-wrapper">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">🌊</div>
          <h1>Welcome to ReWater</h1>
          <p className="subtitle">Sign in to monitor and manage plastic detection intelligence.</p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="form-field">
            <span>Email</span>
            <div className="input-wrapper">
              <Mail size={18} />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
              />
            </div>
          </label>

          <label className="form-field">
            <span>Password</span>
            <div className="input-wrapper">
              <Lock size={18} />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
              />
            </div>
          </label>

          {error && <div className="form-error">{error}</div>}

          <button className="btn btn-primary btn-cta" type="submit" disabled={submitting}>
            {submitting ? 'Signing in...' : 'Sign In'}
            <ArrowRight size={18} />
          </button>
        </form>

        <div className="auth-footer">
          <p className="subtitle">Need access? Contact the ReWater platform administrator.</p>
        </div>
      </div>
    </div>
  );
};

export default Login;
