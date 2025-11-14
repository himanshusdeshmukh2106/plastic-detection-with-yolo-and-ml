import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import MobileCapture from './components/MobileCapture'
import Dashboard from './components/Dashboard'
import './App.css'

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-container">
            <h1 className="logo">🌍 Plastic Detection</h1>
            <div className="nav-links">
              <Link to="/" className="nav-link">📱 Mobile</Link>
              <Link to="/dashboard" className="nav-link">📊 Dashboard</Link>
            </div>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<MobileCapture />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
