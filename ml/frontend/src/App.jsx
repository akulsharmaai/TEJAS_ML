import React, { useState, useEffect, useCallback } from 'react';
import {
  Train,
  AlertTriangle,
  Activity,
  RotateCcw,
  Sparkles,
  ShieldAlert,
  Server,
  Wrench,
  Zap,
  Timer,
  RefreshCw,
  Database,
  FileText,
  BadgeCheck,
  Camera,
  Calendar,
  Lock,
  Search,
  Building,
  Radio,
  Sliders
} from 'lucide-react';
import {
  DEPARTMENT_DATA,
  SEVERITY_LEVELS
} from './constants';
import { EmergencyBanner } from './components/EmergencyBanner';
import { CautionOrders } from './pages/CautionOrders';
import { useNotification } from './context/NotificationContext';

const API_BASE_URL = `http://${window.location.hostname}:8000`;


// Real Master Pilot Registered Asset Presets
const OFFICER_PRESETS = [
  {
    id: 'p1',
    label: 'Bridge Structural Distress',
    asset_id: 'AST-001877',
    defect_type: 'structural crack indication',
    defect_severity: 'HIGH',
    observation: 'Active structural crack propagating along bearing abutment under bridge girder.',
    dept: 'Engineering',
    tag: 'ECoR (Bridge)'
  },
  {
    id: 'p2',
    label: 'Signal Red Aspect Failure',
    asset_id: 'AST-001238',
    defect_type: 'LED aspect partial failure',
    defect_severity: 'CRITICAL',
    observation: 'Red aspect LED array cluster 35% dark during night inspection.',
    dept: 'S&T',
    tag: 'SR (Signal)'
  },
  {
    id: 'p3',
    label: 'OHE Catenary Alignment',
    asset_id: 'AST-002740',
    defect_type: 'geometry deviation',
    defect_severity: 'MEDIUM',
    observation: 'Cantilever assembly geometry deviation under dynamic crosswind.',
    dept: 'Traction',
    tag: 'NR (Traction)'
  },
  {
    id: 'p4',
    label: 'Routine Sleeper Inspection',
    asset_id: 'AST-003140',
    defect_type: 'missing/loose fastening',
    defect_severity: 'LOW',
    observation: 'Minor ERC clip looseness on track curve section.',
    dept: 'Engineering',
    tag: 'WR (Track)'
  }
];

const getNowLocalDateTime = () => {
  const now = new Date();
  now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
  return now.toISOString().slice(0, 16);
};

export default function App() {
  const { connectionStatus, latestTelemetry } = useNotification();
  const [activeTab, setActiveTab] = useState('incident_report'); // 'incident_report' | 'caution_orders'

  // Primary Officer Incident Form State

  const [officerForm, setOfficerForm] = useState({

    asset_id: 'AST-001877',
    defect_type: 'structural crack indication',
    defect_severity: 'HIGH',
    officer_observation: 'Active structural crack propagating along bearing abutment under bridge girder.',
    inspection_datetime: getNowLocalDateTime(),
    days_since_defect: 3,
    inspection_image_available: 1,
    task_id: 'INC-2026-08-001'
  });

  // Verified Asset Telemetry from Registry (READ-ONLY)
  const [retrievedAsset, setRetrievedAsset] = useState(null);
  const [assetFetchError, setAssetFetchError] = useState(null);

  // Asset Search / Autocomplete
  const [assetSearchQuery, setAssetSearchQuery] = useState('');
  const [assetSearchResults, setAssetSearchResults] = useState([]);
  const [showSearchResults, setShowSearchResults] = useState(false);

  // ML Prediction Output
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendHealth, setBackendHealth] = useState({ online: false, checking: true, details: null });

  // Accordion Toggles
  const [showDeveloperPanel, setShowDeveloperPanel] = useState(false);

  // Periodic Health Check (Runs once on mount + every 30s)
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendHealth({ online: true, checking: false, details: data });
      } else {
        setBackendHealth({ online: false, checking: false, details: null });
      }
    } catch {
      setBackendHealth({ online: false, checking: false, details: null });
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Fetch asset details whenever Asset ID changes
  const fetchAssetContext = useCallback(async (assetId) => {
    if (!assetId || assetId.trim().length < 3) {
      setRetrievedAsset(null);
      setAssetFetchError(null);
      return;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/assets/${encodeURIComponent(assetId.trim())}`);
      if (res.ok) {
        const data = await res.json();
        setRetrievedAsset(data);
        setAssetFetchError(null);
      } else {
        setRetrievedAsset(null);
        setAssetFetchError(`Asset ID '${assetId}' not found in registry.`);
      }
    } catch (err) {
      setRetrievedAsset(null);
      setAssetFetchError(`Asset service unavailable: ${err.message}`);
    }
  }, []);

  useEffect(() => {
    fetchAssetContext(officerForm.asset_id);
  }, [officerForm.asset_id, fetchAssetContext]);



  const handleAssetSearch = async (query) => {
    setSearchQuery(query);
    if (!query || query.length < 2) {
      setSearchResults([]);
      return;
    }
    setIsSearching(true);
    try {
      const res = await fetch(`${API_BASE_URL}/assets?query=${encodeURIComponent(query)}&limit=8`);
      if (res.ok) {
        const data = await res.json();
        setSearchResults(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsSearching(false);
    }
  };

  const selectSearchedAsset = (asset) => {
    setOfficerForm(prev => ({
      ...prev,
      asset_id: asset.asset_id,
      task_id: `INC-2026-08-${asset.asset_id.slice(-3)}`
    }));
    setRetrievedAsset(asset);
    setSearchQuery('');
    setSearchResults([]);
  };

  const handleOfficerChange = (e) => {
    const { name, value, type, checked } = e.target;
    setOfficerForm(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value
    }));
  };
  const handleInputChange = handleOfficerChange;

  const handlePresetSelect = (preset) => {
    setOfficerForm({
      asset_id: preset.asset_id,
      defect_type: preset.defect_type,
      defect_severity: preset.defect_severity,
      officer_observation: preset.observation,
      inspection_datetime: getNowLocalDateTime(),
      days_since_defect: 3,
      num_open_defects: 1,
      inspection_image_available: 1,
      task_id: `INC-2026-08-${preset.asset_id.slice(-3)}`
    });
    setError(null);
  };

  const handleReset = () => {
    setOfficerForm({
      asset_id: 'AST-001877',
      defect_type: 'structural crack indication',
      defect_severity: 'HIGH',
      officer_observation: 'Active structural crack propagating along bearing abutment under bridge girder.',
      inspection_datetime: getNowLocalDateTime(),
      days_since_defect: 3,
      num_open_defects: 1,
      inspection_image_available: 1,
      task_id: 'INC-2026-08-1877'
    });
    setPrediction(null);
    setError(null);
  };

  const handleSubmitIncident = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const payload = {
        task_id: officerForm.task_id,
        asset_id: officerForm.asset_id,
        defect_type: officerForm.defect_type,
        defect_severity: officerForm.defect_severity,
        officer_observation: officerForm.officer_observation,
        days_since_defect: parseFloat(officerForm.days_since_defect) || 0,
        inspection_image_available: Boolean(officerForm.inspection_image_available)
      };

      const res = await fetch(`${API_BASE_URL}/predict/officer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Inference call failed');
      }

      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      setError({ message: err.message || 'Failed to submit incident report' });
    } finally {
      setLoading(false);
    }
  };
  const handleSubmit = handleSubmitIncident;


  const activeDept = retrievedAsset?.department || 'Engineering';
  const availableDefects = DEPARTMENT_DATA[activeDept]?.defects || DEPARTMENT_DATA['Engineering'].defects;

  return (
    <div className="app-container">
      <EmergencyBanner />
      {/* Top Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="ir-logo-badge">
            <Train size={28} />
          </div>
          <div className="header-title-group">
            <h1>
              Indian Railways — AI Automatic Block Planning
              <span className="sih-tag">TEJAS SIH26027</span>
            </h1>
            <p>Field Officer Maintenance Incident Reporting & Sub-Second WebSocket Telemetry</p>
          </div>
        </div>

        <div className="header-right">
          <div className="status-pill" title="Live WebSocket Telemetry Engine">
            <span className={`status-dot ${connectionStatus === 'CONNECTED' ? 'online' : 'offline'}`}></span>
            <span>
              {connectionStatus === 'CONNECTED'
                ? 'WebSocket Stream Active (<100ms)'
                : `Telemetry WS (${connectionStatus})`}
            </span>
          </div>
          <div className="status-pill" title="FastAPI ML Backend Status">
            <span className={`status-dot ${backendHealth.online ? 'online' : 'offline'}`}></span>
            <span>
              {backendHealth.online
                ? `TEJAS Engine Ready`
                : 'FastAPI Offline'}
            </span>
            <button
              type="button"
              className="refresh-health-btn"
              onClick={checkHealth}
              title="Refresh Connection Status"
            >
              <RefreshCw size={13} />
            </button>
          </div>
        </div>
      </header>

      {/* Module Navigation Tabs */}
      <div style={{
        display: 'flex',
        gap: '12px',
        padding: '0 24px',
        marginBottom: '20px',
        borderBottom: '1px solid var(--border-subtle)'
      }}>
        <button
          onClick={() => setActiveTab('incident_report')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '12px 20px',
            background: activeTab === 'incident_report' ? 'var(--bg-card-alt)' : 'transparent',
            border: 'none',
            borderBottom: activeTab === 'incident_report' ? '3px solid var(--ir-cyan)' : '3px solid transparent',
            color: activeTab === 'incident_report' ? 'var(--text-primary)' : 'var(--text-secondary)',
            fontWeight: 700,
            fontSize: '0.95rem',
            cursor: 'pointer'
          }}
        >
          <FileText size={18} color={activeTab === 'incident_report' ? 'var(--ir-cyan)' : 'var(--text-muted)'} />
          Maintenance Incident & Block Planner
        </button>

        <button
          onClick={() => setActiveTab('caution_orders')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '12px 20px',
            background: activeTab === 'caution_orders' ? 'var(--bg-card-alt)' : 'transparent',
            border: 'none',
            borderBottom: activeTab === 'caution_orders' ? '3px solid var(--ir-cyan)' : '3px solid transparent',
            color: activeTab === 'caution_orders' ? 'var(--text-primary)' : 'var(--text-secondary)',
            fontWeight: 700,
            fontSize: '0.95rem',
            cursor: 'pointer'
          }}
        >
          <ShieldAlert size={18} color={activeTab === 'caution_orders' ? 'var(--ir-cyan)' : 'var(--text-muted)'} />
          Caution Orders (TSR) & Form T/409
        </button>
      </div>

      {activeTab === 'caution_orders' ? (
        <CautionOrders />
      ) : (
        <>
          {/* Quick Incident Preset Scenarios */}

      <div className="presets-section">
        <div className="presets-label-bar">
          <BadgeCheck size={16} color="#38bdf8" />
          <span>Quick Incident Scenarios (Real Pilot Railway Assets):</span>
        </div>
        <div className="preset-buttons">
          {OFFICER_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`preset-btn ${officerForm.asset_id === p.asset_id ? 'active' : ''}`}
              onClick={() => handlePresetSelect(p)}
              title={`Load registered ${p.dept} asset ${p.asset_id}`}
            >
              <span className="preset-btn-title">{p.label}</span>
              <span className="preset-btn-tag">[{p.tag}]</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main 2-Column Dashboard */}
      <div className="main-grid">
        {/* Left Column: Officer Incident Form */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <FileText size={18} color="#38bdf8" />
              <span>Section A & B: Incident Reporting & Asset Telemetry</span>
            </div>
            <span className="workflow-badge">
              OFFICER MINIMAL INPUT WORKFLOW
            </span>
          </div>

          {error && (
            <div className="error-banner">
              <AlertTriangle size={20} color="#ef4444" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <div className="error-title">{error.title}</div>
                <div className="error-desc">{error.message}</div>
                {error.details && error.details.length > 0 && (
                  <ul className="error-details-list">
                    {error.details.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}

          <form onSubmit={handleSubmitIncident}>
            {/* Section 1: Asset Selection & Identification */}
            <div className="form-section">
              <div className="form-section-title">
                <Search size={14} color="#38bdf8" />
                <span>1. Identify Railway Asset</span>
              </div>

              <div className="form-row">
                <div className="form-group" style={{ position: 'relative' }}>
                  <label className="form-label">
                    Asset ID <span className="unit">(e.g. AST-001877, AST-001238)</span>
                  </label>
                  <input
                    type="text"
                    name="asset_id"
                    placeholder="Enter Asset ID..."
                    value={officerForm.asset_id}
                    onChange={handleOfficerChange}
                    className="form-input"
                    required
                  />
                  {assetFetchError && (
                    <div className="field-error-text">
                      ⚠️ {assetFetchError}
                    </div>
                  )}
                </div>

                <div className="form-group">
                  <label className="form-label">
                    Registry Lookup / Station Search
                  </label>
                  <div style={{ position: 'relative' }}>
                    <input
                      type="text"
                      placeholder="Type station name, code, or asset..."
                      value={assetSearchQuery}
                      onChange={(e) => searchAssets(e.target.value)}
                      className="form-input"
                    />
                    {showSearchResults && assetSearchResults.length > 0 && (
                      <div className="search-dropdown">
                        {assetSearchResults.map((ast) => (
                          <div
                            key={ast.asset_id}
                            className="search-result-item"
                            onClick={() => selectAssetFromSearch(ast)}
                          >
                            <div className="search-item-top">
                              <strong>{ast.asset_id}</strong> — {ast.asset_type}
                            </div>
                            <div className="search-item-sub">
                              Station: {ast.station_code} ({ast.station_name}) | Dept: {ast.department} | Zone: {ast.zone}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Section 2: Auto-Retrieved Verified Asset Telemetry (READ-ONLY) */}
            <div className="retrieved-asset-banner">
              <div className="retrieved-banner-header">
                <div className="retrieved-banner-title">
                  <Database size={15} color="#38bdf8" />
                  <span>2. TEJAS Auto-Resolved Infrastructure Telemetry</span>
                </div>
                <div className="readonly-badge">
                  <Lock size={11} />
                  <span>READ-ONLY • VERIFIED DATASET</span>
                </div>
              </div>

              {retrievedAsset ? (
                <div className="retrieved-grid">
                  <div className="retrieved-pill">
                    <span className="pill-k"><Building size={11} style={{ display: 'inline', marginRight: 3 }} /> Department:</span>
                    <span className="pill-v highlight">{retrievedAsset.department}</span>
                  </div>
                  <div className="retrieved-pill">
                    <span className="pill-k"><Wrench size={11} style={{ display: 'inline', marginRight: 3 }} /> Asset Category:</span>
                    <span className="pill-v">{retrievedAsset.asset_type}</span>
                  </div>
                  <div className="retrieved-pill">
                    <span className="pill-k"><Train size={11} style={{ display: 'inline', marginRight: 3 }} /> Location / Station:</span>
                    <span className="pill-v">{retrievedAsset.station_code} ({retrievedAsset.station_name})</span>
                  </div>
                  <div className="retrieved-pill">
                    <span className="pill-k"><Radio size={11} style={{ display: 'inline', marginRight: 3 }} /> Zone / State:</span>
                    <span className="pill-v">{retrievedAsset.zone} ({retrievedAsset.state})</span>
                  </div>
                  <div className="retrieved-pill">
                    <span className="pill-k"><Calendar size={11} style={{ display: 'inline', marginRight: 3 }} /> Asset Age:</span>
                    <span className="pill-v">{retrievedAsset.asset_age_years} years</span>
                  </div>
                  <div className="retrieved-pill">
                    <span className="pill-k"><ShieldAlert size={11} style={{ display: 'inline', marginRight: 3 }} /> Criticality Tier:</span>
                    <span className="pill-v">{retrievedAsset.asset_criticality} ({retrievedAsset.asset_criticality_score}/100)</span>
                  </div>
                  <div className="retrieved-pill">
                    <span className="pill-k"><Activity size={11} style={{ display: 'inline', marginRight: 3 }} /> Daily Traffic:</span>
                    <span className="pill-v">{retrievedAsset.scheduled_services_count_proxy ?? retrievedAsset.scheduled_daily_trains} trains/day</span>
                  </div>
                  <div className="retrieved-pill">
                    <span className="pill-k"><Timer size={11} style={{ display: 'inline', marginRight: 3 }} /> Overdue Status:</span>
                    <span className="pill-v">{retrievedAsset.maintenance_overdue_days} days overdue</span>
                  </div>
                  <div className="retrieved-pill">
                    <span className="pill-k"><AlertTriangle size={11} style={{ display: 'inline', marginRight: 3 }} /> Lifetime Failures:</span>
                    <span className="pill-v">{retrievedAsset.failures_last_365d ?? 0} (365d) | {retrievedAsset.same_defect_recurrences_365d ?? 0} recurrences</span>
                  </div>
                </div>
              ) : (
                <div className="retrieved-empty-state">
                  <Database size={24} color="#64748b" style={{ marginBottom: 6 }} />
                  <div>Enter a registered Asset ID above to automatically resolve verified infrastructure facts.</div>
                </div>
              )}
            </div>

            {/* Section 3: Observed Defect Details (Officer Input Only) */}
            <div className="form-section">
              <div className="form-section-title">
                <Wrench size={14} color="#38bdf8" />
                <span>3. Observed Defect & Field Findings</span>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">
                    Observed Defect Type <span className="unit">(Filtered for {activeDept})</span>
                  </label>
                  <select
                    name="defect_type"
                    value={officerForm.defect_type}
                    onChange={handleOfficerChange}
                    className="form-select"
                  >
                    {availableDefects.map(d => <option key={d} value={d}>{d}</option>)}
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Observed Defect Severity</label>
                  <select
                    name="defect_severity"
                    value={officerForm.defect_severity}
                    onChange={handleOfficerChange}
                    className="form-select"
                  >
                    {SEVERITY_LEVELS.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                  <label className="form-label">
                    Officer Field Observations / Notes
                  </label>
                  <textarea
                    name="officer_observation"
                    rows={2}
                    placeholder="Enter physical observations, exact defect location, crack propagation, or component details..."
                    value={officerForm.officer_observation}
                    onChange={handleOfficerChange}
                    className="form-textarea"
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Inspection Date & Time</label>
                  <input
                    type="datetime-local"
                    name="inspection_datetime"
                    value={officerForm.inspection_datetime}
                    onChange={handleOfficerChange}
                    className="form-input"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">
                    Days Since Defect Detected
                  </label>
                  <input
                    type="number"
                    name="days_since_defect"
                    min="0"
                    value={officerForm.days_since_defect}
                    onChange={handleOfficerChange}
                    className="form-input"
                    required
                  />
                </div>
              </div>
            </div>


            {/* Submit Action Bar */}
            <div className="form-actions">
              <button
                type="submit"
                className="btn-primary"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <div className="spinner"></div>
                    <span>Resolving Telemetry & Assessing...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    <span>RUN TEJAS ASSESSMENT</span>
                  </>
                )}
              </button>

              <button
                type="button"
                className="btn-secondary"
                onClick={handleReset}
              >
                <RotateCcw size={16} />
                <span>Reset</span>
              </button>
            </div>
          </form>
        </div>

        {/* Right Column: TEJAS Decision & SHAP Explainability */}
        <div className="card results-card">
          <div className="card-header">
            <div className="card-title">
              <Activity size={18} color="#f59e0b" />
              <span>Section E & F: TEJAS AI Decision Output</span>
            </div>
            {prediction && (
              <span className="evaluated-badge">
                ● 4 Champion ML Models Evaluated
              </span>
            )}
          </div>

          {!prediction && !loading && (
            <div className="result-placeholder">
              <Server className="placeholder-icon" />
              <h3 style={{ color: '#cbd5e1', marginBottom: 6, fontFamily: 'var(--font-heading)' }}>
                Awaiting Incident Submission
              </h3>
              <p style={{ fontSize: '0.82rem', maxWidth: 340 }}>
                Enter an Asset ID and observed defect severity on the left, then click <strong>RUN TEJAS ASSESSMENT</strong>. TEJAS will auto-resolve 30+ infrastructure parameters, execute all 4 champion models, and return the decision support deck.
              </p>
            </div>
          )}

          {loading && (
            <div className="result-placeholder">
              <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3, marginBottom: 16 }}></div>
              <h3 style={{ color: '#cbd5e1', marginBottom: 6, fontFamily: 'var(--font-heading)' }}>
                Evaluating Incident...
              </h3>
              <p style={{ fontSize: '0.82rem' }}>
                Auto-resolving telemetry, applying 165-dim pipeline, and generating TreeSHAP feature attributions...
              </p>
            </div>
          )}

          {prediction && !loading && (
            <div className="prediction-deck">
              <div className="result-hero critical">
                <div className="hero-top">
                  <div>
                    <div className="score-title">Predicted Urgency Score</div>
                    <div className="score-big">
                      {prediction.urgency_score?.toFixed(2) ?? 'N/A'}
                      <span className="target-subval" style={{fontSize: '1.2rem'}}> / 100</span>
                    </div>
                  </div>
                  <div className="hero-badges">
                    <span className="priority-pill critical" style={{background: '#334155'}}>
                      TASK ID: {prediction.task_id}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Developer Diagnostics (Separated & Collapsed at the bottom) */}
      <div className="dev-toggle-bar">
        <button
          type="button"
          className="dev-toggle-btn"
          onClick={() => setShowDeveloperPanel(!showDeveloperPanel)}
        >
          <Sliders size={13} />
          <span>{showDeveloperPanel ? 'Hide System Diagnostics' : 'Show Advanced System & Model Diagnostics (Developer / Demo Mode)'}</span>
        </button>
      </div>

      {showDeveloperPanel && (
        <div className="dev-panel card">
          <div className="card-header">
            <div className="card-title">
              <Zap size={16} color="#f59e0b" />
              <span>TEJAS Architectural Metadata & Pipeline Contracts</span>
            </div>
          </div>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', lineHeight: 1.6 }}>
            <p><strong>ColumnTransformer Pipeline:</strong> Trained HistGBM directly estimating maintenance Urgency Score from 7 raw inputs.</p>
            <p><strong>Champion ML Models Loaded:</strong> HistGradientBoosting (Urgency Score 0-100).</p>
            <p><strong>Asset Registry:</strong> Verified physical assets across Indian Railway Zones.</p>
          </div>
        </div>
      )}
        </>
      )}
    </div>
  );
}

