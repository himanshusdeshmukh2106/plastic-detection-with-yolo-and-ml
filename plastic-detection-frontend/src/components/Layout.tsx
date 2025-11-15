import { useState } from 'react';
import { Link, useLocation, Outlet, useNavigate } from 'react-router-dom';
import { 
  LayoutDashboard, Map, BarChart3, Upload, Settings, 
  Menu, X, Bell, User, Search, LogOut, Globe
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Layout.css';

const Layout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const navItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/map', icon: Map, label: 'Map View' },
    { path: '/analytics', icon: BarChart3, label: 'Analytics' },
    { path: '/upload', icon: Upload, label: 'Upload' },
    { path: '/admin', icon: Settings, label: 'Admin', roles: ['admin'] },
    { path: '/public', icon: Globe, label: 'Public Portal', public: true },
  ];

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const filteredNavItems = navItems.filter((item) => {
    if (item.public) {
      return true;
    }
    if (item.roles && item.roles.length > 0) {
      return !!user && item.roles.includes(user.role);
    }
    return !!user;
  });

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon">🌊</div>
            {sidebarOpen && <span className="logo-text">ReWater</span>}
          </div>
        </div>

        <nav className="sidebar-nav">
          {filteredNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-item ${isActive ? 'active' : ''}`}
                title={item.label}
              >
                <Icon size={20} />
                {sidebarOpen && <span>{item.label}</span>}
              </Link>
            );
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <div className="main-wrapper">
        {/* Top Bar */}
        <header className="topbar">
          <button 
            className="menu-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>

          {user && (
            <div className="topbar-search">
              <Search size={18} />
              <input 
                type="text" 
                placeholder="Search detections, locations..." 
              />
            </div>
          )}

          <div className="topbar-actions">
            {user ? (
              <>
                <button className="topbar-btn">
                  <Bell size={20} />
                  <span className="notification-badge">3</span>
                </button>
                <div className="user-chip">
                  <div className="user-avatar">
                    {user.name
                      .split(' ')
                      .map((part) => part[0])
                      .join('')
                      .slice(0, 2)
                      .toUpperCase()}
                  </div>
                  <div className="user-details">
                    <span className="user-name">{user.name}</span>
                    <span className="user-role">{user.role}</span>
                  </div>
                  <button className="logout-btn" onClick={handleLogout}>
                    <LogOut size={18} />
                  </button>
                </div>
              </>
            ) : (
              <Link to="/login" className="btn btn-primary">
                <User size={18} />
                Sign In
              </Link>
            )}
          </div>
        </header>

        {/* Page Content */}
        <main className="main-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;
