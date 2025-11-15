import { useState } from 'react';
import { 
  Users, Settings, Database, Activity,
  Server, Download, Trash2, RefreshCw
} from 'lucide-react';
import './AdminPanel.css';

const AdminPanel = () => {
  const [activeTab, setActiveTab] = useState('system');

  return (
    <div className="admin-panel">
      <div className="admin-header">
        <div>
          <h1>Admin Panel</h1>
          <p className="subtitle">System management and configuration</p>
        </div>
      </div>

      <div className="admin-content">
        {/* Tabs */}
        <div className="admin-tabs">
          <button
            className={`tab-btn ${activeTab === 'system' ? 'active' : ''}`}
            onClick={() => setActiveTab('system')}
          >
            <Server size={18} />
            System
          </button>
          <button
            className={`tab-btn ${activeTab === 'users' ? 'active' : ''}`}
            onClick={() => setActiveTab('users')}
          >
            <Users size={18} />
            Users
          </button>
          <button
            className={`tab-btn ${activeTab === 'database' ? 'active' : ''}`}
            onClick={() => setActiveTab('database')}
          >
            <Database size={18} />
            Database
          </button>
          <button
            className={`tab-btn ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            <Settings size={18} />
            Settings
          </button>
        </div>

        {/* System Tab */}
        {activeTab === 'system' && (
          <div className="tab-content">
            <div className="admin-grid">
              <div className="card admin-card">
                <h3><Activity size={20} /> System Health</h3>
                <div className="health-items">
                  <div className="health-item">
                    <span>API Server</span>
                    <span className="badge badge-success">Running</span>
                  </div>
                  <div className="health-item">
                    <span>ML Models</span>
                    <span className="badge badge-success">Loaded (5)</span>
                  </div>
                  <div className="health-item">
                    <span>Database</span>
                    <span className="badge badge-success">Connected</span>
                  </div>
                  <div className="health-item">
                    <span>Storage</span>
                    <span className="badge badge-warning">78% Used</span>
                  </div>
                </div>
              </div>

              <div className="card admin-card">
                <h3><Server size={20} /> Server Stats</h3>
                <div className="stats-list">
                  <div className="stat-item">
                    <label>CPU Usage</label>
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ width: '45%' }}></div>
                    </div>
                    <span>45%</span>
                  </div>
                  <div className="stat-item">
                    <label>Memory</label>
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ width: '62%' }}></div>
                    </div>
                    <span>62%</span>
                  </div>
                  <div className="stat-item">
                    <label>Disk Space</label>
                    <div className="progress-bar">
                      <div className="progress-fill warning" style={{ width: '78%' }}></div>
                    </div>
                    <span>78%</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="card admin-card">
              <h3>System Actions</h3>
              <div className="action-grid">
                <button className="admin-action-btn">
                  <RefreshCw size={20} />
                  <span>Restart Services</span>
                </button>
                <button className="admin-action-btn">
                  <Download size={20} />
                  <span>Export Logs</span>
                </button>
                <button className="admin-action-btn">
                  <Database size={20} />
                  <span>Backup Database</span>
                </button>
                <button className="admin-action-btn danger">
                  <Trash2 size={20} />
                  <span>Clear Cache</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Users Tab */}
        {activeTab === 'users' && (
          <div className="tab-content">
            <div className="card admin-card">
              <div className="card-header">
                <h3><Users size={20} /> User Management</h3>
                <button className="btn btn-primary">Add User</button>
              </div>
              <div className="users-table">
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Email</th>
                      <th>Role</th>
                      <th>Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Admin User</td>
                      <td>admin@rewater.com</td>
                      <td><span className="badge badge-danger">Admin</span></td>
                      <td><span className="badge badge-success">Active</span></td>
                      <td>
                        <button className="btn-icon">Edit</button>
                      </td>
                    </tr>
                    <tr>
                      <td>Field Operator</td>
                      <td>operator@rewater.com</td>
                      <td><span className="badge badge-info">Operator</span></td>
                      <td><span className="badge badge-success">Active</span></td>
                      <td>
                        <button className="btn-icon">Edit</button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Database Tab */}
        {activeTab === 'database' && (
          <div className="tab-content">
            <div className="admin-grid">
              <div className="card admin-card">
                <h3><Database size={20} /> Database Info</h3>
                <div className="info-list">
                  <div className="info-row">
                    <span>Total Records</span>
                    <strong>15,234</strong>
                  </div>
                  <div className="info-row">
                    <span>Database Size</span>
                    <strong>2.4 GB</strong>
                  </div>
                  <div className="info-row">
                    <span>Last Backup</span>
                    <strong>2 hours ago</strong>
                  </div>
                  <div className="info-row">
                    <span>Connection Pool</span>
                    <strong>8/20 active</strong>
                  </div>
                </div>
              </div>

              <div className="card admin-card">
                <h3>Database Actions</h3>
                <div className="action-list">
                  <button className="list-action-btn">
                    <Download size={18} />
                    <span>Export All Data</span>
                  </button>
                  <button className="list-action-btn">
                    <RefreshCw size={18} />
                    <span>Optimize Tables</span>
                  </button>
                  <button className="list-action-btn">
                    <Database size={18} />
                    <span>Create Backup</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <div className="tab-content">
            <div className="card admin-card">
              <h3><Settings size={20} /> System Settings</h3>
              <div className="settings-form">
                <div className="form-group">
                  <label>Detection Confidence Threshold</label>
                  <input type="range" min="0" max="100" defaultValue="60" />
                  <span>60%</span>
                </div>
                <div className="form-group">
                  <label>Auto-cleanup Threshold (items/m²)</label>
                  <input type="number" defaultValue="50" />
                </div>
                <div className="form-group">
                  <label>Alert Email</label>
                  <input type="email" defaultValue="alerts@rewater.com" />
                </div>
                <div className="form-group">
                  <label className="checkbox-label">
                    <input type="checkbox" defaultChecked />
                    <span>Enable Email Notifications</span>
                  </label>
                </div>
                <div className="form-group">
                  <label className="checkbox-label">
                    <input type="checkbox" defaultChecked />
                    <span>Auto-backup Daily</span>
                  </label>
                </div>
                <button className="btn btn-primary">Save Settings</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AdminPanel;
