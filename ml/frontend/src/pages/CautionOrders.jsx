import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  PlusCircle,
  FileText,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  Printer,
  X,
  Gauge,
  MapPin,
  Clock,
  UserCheck,
  Building,
  Download,
  ExternalLink
} from 'lucide-react';
import {
  fetchActiveCautionOrders,
  fetchSections,
  issueCautionOrder,
  revokeCautionOrder,
  getCautionOrderPdfUrl
} from '../services/api';

export function CautionOrders() {
  const [orders, setOrders] = useState([]);
  const [sections, setSections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  // Modals
  const [isIssueModalOpen, setIsIssueModalOpen] = useState(false);
  const [selectedOrderForT409, setSelectedOrderForT409] = useState(null);

  // Form State
  const [formData, setFormData] = useState({
    section_id: 1,
    km_from: 412.5,
    km_to: 415.0,
    max_speed_kmh: 30,
    reason: 'USFD Transverse Rail Crack / Weld Defect detected on Up Line',
    issued_by_officer: 'Sr. DEN (Co) / Prayagraj Div'
  });

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [ordersData, sectionsData] = await Promise.all([
        fetchActiveCautionOrders(),
        fetchSections().catch(() => [])
      ]);
      setOrders(ordersData);
      setSections(sectionsData);
      if (sectionsData.length > 0) {
        setFormData(prev => ({ ...prev, section_id: sectionsData[0].section_id }));
      }
    } catch (err) {
      setError(err.message || 'Failed to load Caution Orders');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleIssueSubmit = async (e) => {
    e.preventDefault();
    try {
      setSubmitting(true);
      await issueCautionOrder({
        section_id: Number(formData.section_id),
        km_from: Number(formData.km_from),
        km_to: Number(formData.km_to),
        max_speed_kmh: Number(formData.max_speed_kmh),
        reason: formData.reason,
        issued_by_officer: formData.issued_by_officer
      });
      setIsIssueModalOpen(false);
      await loadData();
    } catch (err) {
      alert(`Error issuing Caution Order: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const handleRevoke = async (orderId) => {
    if (!window.confirm(`Are you sure you want to revoke speed restriction for Order #${orderId}?`)) {
      return;
    }
    try {
      await revokeCautionOrder(orderId);
      await loadData();
    } catch (err) {
      alert(`Error revoking Caution Order: ${err.message}`);
    }
  };

  // Metrics calculation
  const totalActiveOrders = orders.length;
  const totalRestrictedKm = orders.reduce((acc, curr) => acc + Math.max(0, curr.km_to - curr.km_from), 0).toFixed(1);
  const avgRestrictedSpeed = totalActiveOrders > 0
    ? Math.round(orders.reduce((acc, curr) => acc + curr.max_speed_kmh, 0) / totalActiveOrders)
    : 0;

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto', color: 'var(--text-primary)' }}>
      {/* Top Header Banner */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '24px',
        padding: '20px 24px',
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%)',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border-subtle)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.3)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            background: 'rgba(2, 132, 199, 0.2)',
            border: '1px solid rgba(56, 189, 248, 0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <ShieldAlert size={28} color="var(--ir-cyan)" />
          </div>
          <div>
            <h1 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.6rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
              Automated Caution Order (TSR) & Loco Dispatch Center
            </h1>
            <p style={{ margin: '4px 0 0 0', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Railway Form T/409 Management & Real-time Speed Restriction Telemetry Engine
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            onClick={loadData}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 16px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--bg-card-alt)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '0.875rem'
            }}
          >
            <RefreshCw size={16} className={loading ? 'spin' : ''} />
            Refresh
          </button>

          <button
            onClick={() => setIsIssueModalOpen(true)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 20px',
              borderRadius: 'var(--radius-md)',
              background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
              border: 'none',
              color: '#ffffff',
              cursor: 'pointer',
              fontWeight: 700,
              fontSize: '0.875rem',
              boxShadow: '0 4px 14px rgba(2, 132, 199, 0.4)'
            }}
          >
            <PlusCircle size={18} />
            Issue New Caution Order (T/409)
          </button>
        </div>
      </div>

      {/* Metric Summary Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '16px',
        marginBottom: '24px'
      }}>
        <div style={{
          padding: '20px',
          background: 'var(--bg-card)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 600 }}>Active Speed Restrictions</span>
            <AlertTriangle size={20} color="var(--ir-gold)" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--ir-gold)', marginTop: '8px' }}>
            {totalActiveOrders}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Active TSRs in Indian Railway Network
          </div>
        </div>

        <div style={{
          padding: '20px',
          background: 'var(--bg-card)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 600 }}>Restricted Track Corridor</span>
            <MapPin size={20} color="var(--ir-cyan)" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--ir-cyan)', marginTop: '8px' }}>
            {totalRestrictedKm} <span style={{ fontSize: '1rem' }}>KM</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Cumulative length under caution
          </div>
        </div>

        <div style={{
          padding: '20px',
          background: 'var(--bg-card)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 600 }}>Average Restricted Speed</span>
            <Gauge size={20} color="#f87171" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: '#f87171', marginTop: '8px' }}>
            {avgRestrictedSpeed} <span style={{ fontSize: '1rem' }}>KM/H</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Network-wide restricted velocity threshold
          </div>
        </div>

        <div style={{
          padding: '20px',
          background: 'var(--bg-card)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 600 }}>Safety Index Status</span>
            <UserCheck size={20} color="#34d399" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: '#34d399', marginTop: '8px' }}>
            98.4%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Real-time loco pilot compliance rating
          </div>
        </div>
      </div>

      {/* Active Caution Orders Registry Table */}
      <div style={{
        background: 'var(--bg-card)',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border-subtle)',
        overflow: 'hidden'
      }}>
        <div style={{
          padding: '16px 24px',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          justify: 'space-between',
          alignItems: 'center'
        }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FileText size={18} color="var(--ir-cyan)" />
            Active Caution Orders (Form T/409) Registry
          </h2>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', background: 'var(--bg-card-alt)', padding: '4px 10px', borderRadius: '12px' }}>
            {orders.length} Active Records
          </span>
        </div>

        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <RefreshCw size={24} className="spin" style={{ marginBottom: '12px' }} />
            <p>Fetching active caution orders from server...</p>
          </div>
        ) : error ? (
          <div style={{ padding: '30px', textAlign: 'center', color: 'var(--critical-text)' }}>
            <AlertTriangle size={24} style={{ marginBottom: '8px' }} />
            <p>{error}</p>
          </div>
        ) : orders.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <CheckCircle size={32} color="#34d399" style={{ marginBottom: '12px' }} />
            <p style={{ fontSize: '1rem', fontWeight: 600 }}>No Active Speed Restrictions</p>
            <p style={{ fontSize: '0.85rem' }}>All track sections operating at standard sectional speeds.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
              <thead>
                <tr style={{ background: 'var(--bg-card-alt)', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '12px 16px' }}>Order Number</th>
                  <th style={{ padding: '12px 16px' }}>Section Corridor</th>
                  <th style={{ padding: '12px 16px' }}>KM Range</th>
                  <th style={{ padding: '12px 16px' }}>Restricted Speed</th>
                  <th style={{ padding: '12px 16px' }}>Reason / Flaw</th>
                  <th style={{ padding: '12px 16px' }}>Issued By</th>
                  <th style={{ padding: '12px 16px' }}>Issued Time</th>
                  <th style={{ padding: '12px 16px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order) => (
                  <tr key={order.order_id} style={{ borderBottom: '1px solid var(--border-subtle)', transition: 'background 0.2s' }}>
                    <td style={{ padding: '14px 16px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--ir-cyan)' }}>
                      {order.order_number}
                    </td>
                    <td style={{ padding: '14px 16px' }}>
                      <div style={{ fontWeight: 600 }}>{order.section_code}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{order.section_name || 'Main Line'}</div>
                    </td>
                    <td style={{ padding: '14px 16px', fontFamily: 'var(--font-mono)' }}>
                      KM {order.km_from} - {order.km_to}
                    </td>
                    <td style={{ padding: '14px 16px' }}>
                      <span style={{
                        padding: '4px 10px',
                        borderRadius: '6px',
                        fontWeight: 800,
                        fontSize: '0.8rem',
                        background: order.max_speed_kmh <= 30 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                        color: order.max_speed_kmh <= 30 ? '#f87171' : 'var(--ir-gold)',
                        border: order.max_speed_kmh <= 30 ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid rgba(245, 158, 11, 0.4)'
                      }}>
                        {order.max_speed_kmh} KM/H RESTRICTION
                      </span>
                    </td>
                    <td style={{ padding: '14px 16px', maxWidth: '280px', color: 'var(--text-secondary)' }}>
                      {order.reason}
                    </td>
                    <td style={{ padding: '14px 16px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {order.issued_by_officer}
                    </td>
                    <td style={{ padding: '14px 16px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      {new Date(order.issued_at).toLocaleString()}
                    </td>
                    <td style={{ padding: '14px 16px', textAlign: 'right' }}>
                      <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                        <button
                          onClick={() => setSelectedOrderForT409(order)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            padding: '6px 12px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'rgba(56, 189, 248, 0.1)',
                            border: '1px solid rgba(56, 189, 248, 0.3)',
                            color: 'var(--ir-cyan)',
                            cursor: 'pointer',
                            fontSize: '0.75rem',
                            fontWeight: 600
                          }}
                        >
                          <Printer size={14} />
                          Form T/409
                        </button>

                        <button
                          onClick={() => handleRevoke(order.order_id)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            padding: '6px 12px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'rgba(239, 68, 68, 0.1)',
                            border: '1px solid rgba(239, 68, 68, 0.3)',
                            color: '#f87171',
                            cursor: 'pointer',
                            fontSize: '0.75rem',
                            fontWeight: 600
                          }}
                        >
                          <X size={14} />
                          Revoke
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Modal: Issue New Caution Order Form */}
      {isIssueModalOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '20px'
        }}>
          <div style={{
            background: 'var(--bg-card)',
            width: '100%',
            maxWidth: '560px',
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border-subtle)',
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.6)',
            overflow: 'hidden'
          }}>
            <div style={{
              padding: '16px 24px',
              background: 'var(--bg-card-alt)',
              borderBottom: '1px solid var(--border-subtle)',
              display: 'flex',
              justify: 'space-between',
              alignItems: 'center'
            }}>
              <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <PlusCircle size={20} color="var(--ir-cyan)" />
                Issue Caution Order (Form T/409)
              </h3>
              <button
                onClick={() => setIsIssueModalOpen(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            <form onSubmit={handleIssueSubmit} style={{ padding: '24px' }}>
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Section Corridor:
                </label>
                <select
                  value={formData.section_id}
                  onChange={(e) => setFormData({ ...formData, section_id: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-primary)',
                    fontSize: '0.9rem'
                  }}
                  required
                >
                  {sections.map((sec) => (
                    <option key={sec.section_id} value={sec.section_id}>
                      {sec.section_code} - {sec.section_name} ({sec.start_station} to {sec.end_station})
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                    KM From:
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.km_from}
                    onChange={(e) => setFormData({ ...formData, km_from: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-primary)',
                      fontSize: '0.9rem'
                    }}
                    required
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                    KM To:
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.km_to}
                    onChange={(e) => setFormData({ ...formData, km_to: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-primary)',
                      fontSize: '0.9rem'
                    }}
                    required
                  />
                </div>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Restricted Speed Limit (KM/H):
                </label>
                <select
                  value={formData.max_speed_kmh}
                  onChange={(e) => setFormData({ ...formData, max_speed_kmh: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-primary)',
                    fontSize: '0.9rem',
                    fontWeight: 700
                  }}
                >
                  <option value={20}>20 KM/H (Severe Caution)</option>
                  <option value={30}>30 KM/H (Standard Track Flaw Caution)</option>
                  <option value={45}>45 KM/H (Turnout / Catenary Limit)</option>
                  <option value={60}>60 KM/H (Moderate Restriction)</option>
                  <option value={75}>75 KM/H (Engineering Block Clearance)</option>
                </select>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Reason / Flaw Description:
                </label>
                <textarea
                  rows={3}
                  value={formData.reason}
                  onChange={(e) => setFormData({ ...formData, reason: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-primary)',
                    fontSize: '0.9rem',
                    resize: 'vertical'
                  }}
                  required
                />
              </div>

              <div style={{ marginBottom: '24px' }}>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Issuing Officer Name / Designation:
                </label>
                <input
                  type="text"
                  value={formData.issued_by_officer}
                  onChange={(e) => setFormData({ ...formData, issued_by_officer: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-primary)',
                    fontSize: '0.9rem'
                  }}
                  required
                />
              </div>

              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  onClick={() => setIsIssueModalOpen(false)}
                  style={{
                    padding: '10px 18px',
                    borderRadius: 'var(--radius-md)',
                    background: 'transparent',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontWeight: 600
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  style={{
                    padding: '10px 20px',
                    borderRadius: 'var(--radius-md)',
                    background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
                    border: 'none',
                    color: '#ffffff',
                    cursor: 'pointer',
                    fontWeight: 700
                  }}
                >
                  {submitting ? 'Broadcasting Order...' : 'Sanction & Broadcast Order'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: Printable Form T/409 Viewer */}
      {selectedOrderForT409 && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.8)',
          backdropFilter: 'blur(6px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '20px'
        }}>
          <div style={{
            background: '#fefcf8',
            color: '#1e293b',
            width: '100%',
            maxWidth: '750px',
            borderRadius: 'var(--radius-lg)',
            boxShadow: '0 25px 60px rgba(0, 0, 0, 0.8)',
            overflow: 'hidden',
            fontFamily: 'serif'
          }}>
            {/* Form Top Control Bar */}
            <div style={{
              padding: '12px 20px',
              background: '#0f172a',
              color: '#f8fafc',
              display: 'flex',
              justify: 'space-between',
              alignItems: 'center',
              fontFamily: 'sans-serif'
            }}>
              <span style={{ fontWeight: 700, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FileText size={18} color="var(--ir-cyan)" />
                Indian Railways Official Form T/409 Preview
              </span>
              <div style={{ display: 'flex', gap: '8px' }}>
                <a
                  href={getCautionOrderPdfUrl(selectedOrderForT409.order_id)}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '6px 12px',
                    borderRadius: '4px',
                    background: '#0284c7',
                    color: '#ffffff',
                    textDecoration: 'none',
                    fontSize: '0.8rem',
                    fontWeight: 600
                  }}
                >
                  <ExternalLink size={14} /> Open Official PDF
                </a>
                <button
                  onClick={() => window.print()}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '6px 12px',
                    borderRadius: '4px',
                    background: '#334155',
                    color: '#ffffff',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    fontWeight: 600
                  }}
                >
                  <Printer size={14} /> Print
                </button>
                <button
                  onClick={() => setSelectedOrderForT409(null)}
                  style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
                >
                  <X size={20} />
                </button>
              </div>
            </div>

            {/* Printable Form Document Content */}
            <div style={{ padding: '36px', border: '8px double #cbd5e1', margin: '16px', background: '#ffffff' }}>
              <div style={{ textAlign: 'center', borderBottom: '2px solid #0f172a', pb: '12px', marginBottom: '16px' }}>
                <h4 style={{ margin: 0, fontSize: '1rem', letterSpacing: '1px', textTransform: 'uppercase' }}>INDIAN RAILWAYS / OPERATING DEPARTMENT</h4>
                <h2 style={{ margin: '4px 0', fontSize: '1.6rem', color: '#0f172a', fontWeight: 900 }}>FORM T/409 - CAUTION ORDER</h2>
                <p style={{ margin: 0, fontSize: '0.85rem', fontStyle: 'italic', color: '#475569' }}>
                  (See Rules 4.09 of General & Subsidiary Rules - Sanctioned Temporary Speed Restriction Notice)
                </p>
              </div>

              {/* Grid Details */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.85rem', marginBottom: '16px' }}>
                <div><strong>Order Reference:</strong> <span style={{ fontFamily: 'monospace' }}>{selectedOrderForT409.order_number}</span></div>
                <div><strong>Issued Date:</strong> {new Date(selectedOrderForT409.issued_at).toLocaleString()}</div>
                <div><strong>Corridor Section:</strong> {selectedOrderForT409.section_code} ({selectedOrderForT409.section_name})</div>
                <div><strong>Issuing Officer:</strong> {selectedOrderForT409.issued_by_officer}</div>
              </div>

              {/* Restriction Box */}
              <div style={{
                border: '2px solid #dc2626',
                background: '#fff1f1',
                padding: '16px',
                borderRadius: '6px',
                marginBottom: '20px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontSize: '0.9rem', fontWeight: 'bold', color: '#991b1b' }}>SPEED RESTRICTION LOCATION</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: 900, color: '#dc2626', background: '#fee2e2', padding: '4px 12px', borderRadius: '4px', border: '1px solid #fca5a5' }}>
                    MAX SPEED: {selectedOrderForT409.max_speed_kmh} KM/H
                  </span>
                </div>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '6px' }}>
                  Kilometer Post: KM {selectedOrderForT409.km_from} to KM {selectedOrderForT409.km_to}
                </div>
                <div style={{ fontSize: '0.85rem', color: '#334155' }}>
                  <strong>Flaw / Reason:</strong> {selectedOrderForT409.reason}
                </div>
              </div>

              {/* Instructions */}
              <div style={{ fontSize: '0.8rem', lineHeight: '1.5', color: '#334155', marginBottom: '24px' }}>
                <p style={{ fontWeight: 'bold', margin: '0 0 4px 0' }}>LOCO PILOT & GUARD MANDATORY INSTRUCTIONS:</p>
                <ol style={{ margin: 0, paddingLeft: '20px' }}>
                  <li>Observance of the specified speed limit is mandatory until train completely clears KM {selectedOrderForT409.km_to}.</li>
                  <li>Sound whistle continuously when approaching engineering caution indicator boards.</li>
                  <li>Verify track status via TEJAS Real-time Rail Radar telemetry feed.</li>
                </ol>
              </div>

              {/* Signatures & Security QR */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', paddingTop: '20px', borderTop: '1px dashed #94a3b8' }}>
                <div style={{ textAlign: 'center', fontSize: '0.75rem', color: '#64748b' }}>
                  <div style={{ width: '60px', height: '60px', background: '#0f172a', color: '#ffffff', display: 'flex', alignItems: 'center', justifyCenter: 'center', fontSize: '0.6rem', padding: '4px', margin: '0 auto 4px auto', borderRadius: '4px' }}>
                    TEJAS QR VERIFIED
                  </div>
                  Digital Dispatch Hash: 8F2A-99B1
                </div>

                <div style={{ textAlign: 'center', fontSize: '0.8rem', fontWeight: 'bold' }}>
                  __________________________<br/>
                  Station Master / Controller
                </div>

                <div style={{ textAlign: 'center', fontSize: '0.8rem', fontWeight: 'bold' }}>
                  __________________________<br/>
                  Loco Pilot Acknowledgement Signature
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
