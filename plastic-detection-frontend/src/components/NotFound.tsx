import { Link } from 'react-router-dom';
import { Compass } from 'lucide-react';
import './NotFound.css';

const NotFound = () => (
  <div className="not-found-page">
    <div className="not-found-card card">
      <div className="icon-wrapper">
        <Compass size={48} />
      </div>
      <h1>Page Not Found</h1>
      <p className="subtitle">The page you are looking for might have been moved, removed, or is temporarily unavailable.</p>
      <Link to="/" className="btn btn-primary">Back to Dashboard</Link>
    </div>
  </div>
);

export default NotFound;
