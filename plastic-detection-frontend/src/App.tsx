import { type ReactElement } from 'react'
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import MobileCapture from './components/MobileCapture.tsx'
import Dashboard from './components/Dashboard.tsx'
import './App.css'

function App(): ReactElement {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-container">
            <h1 className="logo">Plastic Watch</h1>
            <div className="nav-links">
              <NavLink
                to="/"
                end
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                Mobile Capture
              </NavLink>
              <NavLink
                to="/dashboard"
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                Impact Dashboard
              </NavLink>
            </div>
          </div>
        </nav>

        <main className="app-main">
          <Routes>
            <Route path="/" element={<MobileCapture />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
