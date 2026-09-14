import React, { useEffect } from 'react';
import { useNotification } from '../context/NotificationContext';
import { ShieldAlert, AlertTriangle, Radio, X } from 'lucide-react';

export const EmergencyBanner = () => {
  const { currentBannerAlert, sirenActive, dismissBanner, connectionStatus } = useNotification();

  useEffect(() => {
    if (currentBannerAlert) {
      // 25 seconds timer: Allows full speech announcement (~12-15s) plus 5+ seconds afterwards
      const timer = setTimeout(() => {
        dismissBanner();
      }, 25000);
      return () => clearTimeout(timer);
    }
  }, [currentBannerAlert, dismissBanner]);


  if (!currentBannerAlert) return null;


  return (
    <div style={{
      position: 'fixed',
      top: '16px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 99999,
      width: '90%',
      maxWidth: '900px',
      backgroundColor: '#7f1d1d',
      border: '2px solid #ef4444',
      borderRadius: '12px',
      padding: '16px 20px',
      boxShadow: '0 12px 36px rgba(239, 68, 68, 0.45)',
      color: '#ffffff',
      animation: 'pulseAlert 1s infinite alternate',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <style>{`
        @keyframes pulseAlert {
          0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.4); border-color: #ef4444; }
          100% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.9); border-color: #fca5a5; }
        }
      `}</style>
      
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            backgroundColor: '#dc2626',
            padding: '10px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            animation: 'spinRadio 2s linear infinite'
          }}>
            <ShieldAlert size={28} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{
                backgroundColor: '#ef4444',
                color: '#fff',
                fontSize: '0.75rem',
                fontWeight: 'bold',
                padding: '2px 8px',
                borderRadius: '4px',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                CRITICAL BROADCAST ({currentBannerAlert.urgencyScore}% URGENCY)
              </span>
              <span style={{ fontSize: '0.8rem', color: '#fca5a5' }}>
                Sub-Second WebSocket Feed (&lt;100ms)
              </span>
            </div>
            <h3 style={{ margin: '4px 0 2px 0', fontSize: '1.2rem', fontWeight: '700', color: '#ffffff' }}>
              🚨 {currentBannerAlert.title} [{currentBannerAlert.id}]
            </h3>
            <p style={{ margin: 0, fontSize: '0.9rem', color: '#fecaca' }}>
              {currentBannerAlert.message}
            </p>
          </div>
        </div>

        <button
          onClick={dismissBanner}
          style={{
            backgroundColor: 'rgba(255, 255, 255, 0.15)',
            border: 'none',
            color: '#fff',
            padding: '6px 12px',
            borderRadius: '6px',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '4px'
          }}
        >
          <X size={16} /> Dismiss
        </button>
      </div>

      <div style={{
        marginTop: '12px',
        paddingTop: '10px',
        borderTop: '1px solid rgba(255, 255, 255, 0.2)',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '8px',
        fontSize: '0.8rem',
        color: '#fee2e2'
      }}>
        <div><strong>Location:</strong> {currentBannerAlert.routeLocation || 'Varanasi Mainline'}</div>
        <div><strong>Subsystem:</strong> {currentBannerAlert.subsystem || 'P.Way'}</div>
        <div><strong>Time:</strong> {currentBannerAlert.reportedExactTime || 'Just now'}</div>
        <div><strong>Action:</strong> {currentBannerAlert.recommendedAction || 'Emergency Restriction'}</div>
      </div>
    </div>
  );
};
