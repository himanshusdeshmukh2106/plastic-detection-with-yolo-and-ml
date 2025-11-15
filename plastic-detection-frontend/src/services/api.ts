const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:5000';

interface RequestOptions extends RequestInit {
  authToken?: string | null;
  expectsJson?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { authToken, expectsJson = true, headers, ...rest } = options;

  const finalHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(headers as Record<string, string> | undefined),
  };

  if (rest.body instanceof FormData) {
    delete finalHeaders['Content-Type'];
  }

  if (authToken) {
    finalHeaders['Authorization'] = `Bearer ${authToken}`;
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: finalHeaders,
    ...rest,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed with status ${response.status}`);
  }

  if (!expectsJson) {
    return response.blob() as unknown as T;
  }

  return response.json() as Promise<T>;
}

export interface AuthResponse {
  token: string;
  refreshToken?: string;
  user: {
    id: string;
    name: string;
    email: string;
    role: string;
  };
}

export interface DetectionRecord {
  id: number;
  waste_type: string;
  confidence: number;
  latitude: number;
  longitude: number;
  timestamp: string;
  image_path?: string;
  density?: number;
  status?: string;
  notes?: string;
  source?: string;
  plastic_breakdown?: Array<{ type: string; percentage: number }>;
  estimated_mass_kg?: number;
  nearest_landmark?: string;
}

export interface StatsResponse {
  total_detections?: number;
  today_detections?: number;
  week_detections?: number;
  previous_week?: number;
  active_hotspots?: number;
  cleanup_ratio?: number;
  coverage_area?: number;
  last_detection?: string;
  system_uptime?: number;
  avg_density?: number;
  teams_deployed?: number;
}

export interface CollectionChannelStatus {
  channel: string;
  status: 'operational' | 'degraded' | 'offline';
  lastIngest?: string;
  successRate?: number;
  queueDepth?: number;
  automated?: boolean;
  notes?: string;
}

export interface AutoIngestJob {
  id: string;
  source: string;
  frequencyMinutes: number;
  nextRun: string;
  status: 'scheduled' | 'running' | 'failed';
}

export interface CollectionStatusResponse {
  channels: CollectionChannelStatus[];
  autoIngestJobs: AutoIngestJob[];
}

export interface ScheduleAutoIngestPayload {
  source: string;
  frequencyMinutes: number;
  startImmediately?: boolean;
}

export interface CleanupTeam {
  id: string;
  name: string;
  location: { lat: number; lng: number };
  region: string;
  status: 'available' | 'busy' | 'offline';
  capacity: number;
  current_load: number;
  members: number;
  equipment: string[];
  assigned_detections: number[];
  total_cleaned: number;
  efficiency_rating: number;
}

export interface Hotspot {
  id: string;
  latitude: number;
  longitude: number;
  detection_count: number;
  radius: number;
  severity: 'medium' | 'high' | 'critical';
  detections: number[];
  nearest_landmark: string;
}

export interface AssignmentResult {
  team: CleanupTeam;
  detection: DetectionRecord;
  distance: number;
}

export const Api = {
  login(email: string, password: string) {
    return request<AuthResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  },

  refresh(refreshToken: string) {
    return request<AuthResponse>('/auth/refresh', {
      method: 'POST',
      body: JSON.stringify({ refreshToken }),
    });
  },

  fetchStats(range = 'all', authToken?: string) {
    return request<StatsResponse>(`/stats?range=${range}`, {
      method: 'GET',
      authToken,
    });
  },

  fetchHistory(limit = 1000, authToken?: string) {
    return request<{ history: DetectionRecord[] }>(`/history?limit=${limit}`, {
      method: 'GET',
      authToken,
    });
  },

  fetchDetection(id: string | number, authToken?: string) {
    return request<DetectionRecord>(`/detections/${id}`, {
      method: 'GET',
      authToken,
    });
  },

  saveDetection(payload: Partial<DetectionRecord>, authToken?: string) {
    return request<DetectionRecord>('/save-detection', {
      method: 'POST',
      body: JSON.stringify(payload),
      authToken,
    });
  },

  updateDetection(id: number, payload: Partial<DetectionRecord>, authToken?: string) {
    return request<DetectionRecord>(`/detections/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
      authToken,
    });
  },

  exportDetections(params: URLSearchParams, authToken?: string) {
    return request<Blob>(`/export?${params.toString()}`, {
      method: 'GET',
      authToken,
      expectsJson: false,
    });
  },

  fetchHotspots(authToken?: string) {
    return request('/hotspots', {
      method: 'GET',
      authToken,
    });
  },

  fetchSystemStatus(authToken?: string) {
    return request('/system-status', {
      method: 'GET',
      authToken,
    });
  },

  fetchAnalytics(range = 'month', authToken?: string) {
    return request(`/analytics?range=${range}`, {
      method: 'GET',
      authToken,
    });
  },

  uploadDetection(formData: FormData, authToken?: string) {
    return request<{ prediction?: string; confidence?: number; [key: string]: unknown }>('/predict', {
      method: 'POST',
      body: formData,
      authToken,
      headers: {
        Accept: 'application/json',
      },
    });
  },

  triggerAlert(payload: { detectionId: number; teamId?: string; message?: string }, authToken?: string) {
    return request('/alerts', {
      method: 'POST',
      body: JSON.stringify(payload),
      authToken,
    });
  },

  fetchCollectionStatus(authToken?: string) {
    return request<CollectionStatusResponse>('/collection/status', {
      method: 'GET',
      authToken,
    });
  },

  scheduleAutoIngest(payload: ScheduleAutoIngestPayload, authToken?: string) {
    return request('/collection/ingest-jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
      authToken,
    });
  },

  // Teams
  fetchTeams(authToken?: string) {
    return request<{ teams: CleanupTeam[] }>('/teams', {
      method: 'GET',
      authToken,
    });
  },

  fetchTeam(teamId: string, authToken?: string) {
    return request<CleanupTeam>(`/teams/${teamId}`, {
      method: 'GET',
      authToken,
    });
  },

  updateTeam(teamId: string, payload: Partial<CleanupTeam>, authToken?: string) {
    return request<CleanupTeam>(`/teams/${teamId}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
      authToken,
    });
  },

  completeTeamAssignment(teamId: string, authToken?: string) {
    return request<{ success: boolean; team: CleanupTeam; completed_count: number }>(`/teams/${teamId}/complete`, {
      method: 'POST',
      authToken,
    });
  },

  // Assignment
  assignDetection(detectionId: number, authToken?: string) {
    return request<{ success: boolean; assignment: AssignmentResult }>('/assign', {
      method: 'POST',
      body: JSON.stringify({ detection_id: detectionId }),
      authToken,
    });
  },

  bulkAssignDetections(detectionIds?: number[], authToken?: string) {
    return request<{ success: boolean; assigned_count: number; failed_count: number; assignments: AssignmentResult[] }>('/assign/bulk', {
      method: 'POST',
      body: JSON.stringify({ detection_ids: detectionIds }),
      authToken,
    });
  },

  // Hotspots
  calculateHotspots(radius = 5, authToken?: string) {
    return request<{ hotspots: Hotspot[]; total_hotspots: number; radius_km: number }>(`/hotspots/calculate?radius=${radius}`, {
      method: 'GET',
      authToken,
    });
  },
};
