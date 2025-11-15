import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import DashboardHome from './components/DashboardHome';
import MapInterface from './components/MapInterface';
import DetectionDetails from './components/DetectionDetails';
import Analytics from './components/Analytics';
import UploadInterface from './components/UploadInterface';
import AdminPanel from './components/AdminPanel';
import PublicPortal from './components/PublicPortal';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider } from './context/AuthContext';
import Login from './components/Login';
import NotFound from './components/NotFound';
import './App.css';

function App() {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route
              path="/dashboard"
              element={(
                <ProtectedRoute>
                  <DashboardHome />
                </ProtectedRoute>
              )}
            />
            <Route
              path="/map"
              element={(
                <ProtectedRoute>
                  <MapInterface />
                </ProtectedRoute>
              )}
            />
            <Route
              path="/detection/:id"
              element={(
                <ProtectedRoute>
                  <DetectionDetails />
                </ProtectedRoute>
              )}
            />
            <Route
              path="/analytics"
              element={(
                <ProtectedRoute>
                  <Analytics />
                </ProtectedRoute>
              )}
            />
            <Route
              path="/upload"
              element={(
                <ProtectedRoute>
                  <UploadInterface />
                </ProtectedRoute>
              )}
            />
            <Route
              path="/admin"
              element={(
                <ProtectedRoute roles={['admin']}>
                  <AdminPanel />
                </ProtectedRoute>
              )}
            />
            <Route path="/public" element={<PublicPortal />} />
          </Route>
          <Route path="/login" element={<Login />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </AuthProvider>
    </Router>
  );
}

export default App;
