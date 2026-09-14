import React, { createContext, useContext, useEffect, useState, useRef, useCallback } from 'react';

const NotificationContext = createContext({
  connectionStatus: 'DISCONNECTED',
  latestTelemetry: null,
  emergencyAlerts: [],
  currentBannerAlert: null,
  sirenActive: false,
  triggerNotification: () => {},
  dismissBanner: () => {},
  clearAlerts: () => {}
});

export const useNotification = () => useContext(NotificationContext);

export const NotificationProvider = ({ children }) => {
  const [connectionStatus, setConnectionStatus] = useState('DISCONNECTED');
  const [latestTelemetry, setLatestTelemetry] = useState(null);
  const [emergencyAlerts, setEmergencyAlerts] = useState([]);
  const [currentBannerAlert, setCurrentBannerAlert] = useState(null);
  const [sirenActive, setSirenActive] = useState(false);

  const wsRef = useRef(null);
  const audioRef = useRef(null);

  // Voice Text-to-Speech Emergency Announcement
  const speakEmergencyVoice = useCallback((defectData) => {
    try {
      if (!('speechSynthesis' in window)) return;
      
      // Cancel any ongoing speech
      window.speechSynthesis.cancel();

      const title = defectData?.title || 'Defect Detected';
      const location = defectData?.routeLocation || 'Mainline Track';
      
      const spokenText = `Attention Control Office! Emergency Alert. ${title} detected on ${location}. Immediate safety restriction required.`;
      
      const utterance = new SpeechSynthesisUtterance(spokenText);
      utterance.rate = 1.0;
      utterance.pitch = 1.1;
      utterance.volume = 1.0;

      // Select an English voice if available
      const voices = window.speechSynthesis.getVoices();
      const engVoice = voices.find(v => v.lang.includes('en'));
      if (engVoice) utterance.voice = engVoice;

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Text-to-speech announcement failed:', e);
    }
  }, []);

  const triggerNotification = useCallback((defectData) => {
    console.log('🚨 Sub-Second Emergency Alert Triggered via WebSocket (<100ms):', defectData);
    
    // Add to historical emergency alerts stack
    setEmergencyAlerts(prev => [defectData, ...prev.slice(0, 49)]);
    
    // Active high-urgency banner and voice alert activation
    if (defectData.urgencyScore >= 90.0 || defectData.type === 'EMERGENCY') {
      setCurrentBannerAlert(defectData);
      setSirenActive(true);
      speakEmergencyVoice(defectData);
    }
  }, [speakEmergencyVoice]);


  const dismissBanner = useCallback(() => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setCurrentBannerAlert(null);
    setSirenActive(false);
  }, []);


  const clearAlerts = useCallback(() => {
    setEmergencyAlerts([]);
  }, []);

  useEffect(() => {
    const WS_URL = 'ws://localhost:8000/ws/telemetry';
    let socket = null;
    let reconnectTimeout = null;

    const connectWebSocket = () => {
      setConnectionStatus('CONNECTING');
      try {
        socket = new WebSocket(WS_URL);
        wsRef.current = socket;

        socket.onopen = () => {
          console.log('⚡ Connected to TEJAS Sub-Second Telemetry WebSocket Stream');
          setConnectionStatus('CONNECTED');
        };

        socket.onmessage = (event) => {
          const startTime = performance.now();
          try {
            const data = JSON.parse(event.data);
            const latency = performance.now() - startTime;
            console.debug(`[WebSocket Payload Received] Latency: ${latency.toFixed(2)}ms`, data);

            if (data.event_type === 'DEFECT_REPORTED' || data.type === 'EMERGENCY' || data.urgencyScore >= 90) {
              triggerNotification(data);
            } else if (data.event_type === 'TELEMETRY_METRIC') {
              setLatestTelemetry(data);
            }
          } catch (err) {
            console.error('Failed to parse WebSocket JSON payload:', err);
          }
        };

        socket.onerror = (err) => {
          console.warn('WebSocket error encountered:', err);
          setConnectionStatus('ERROR');
        };

        socket.onclose = () => {
          console.log('WebSocket connection closed. Reconnecting in 3s...');
          setConnectionStatus('DISCONNECTED');
          reconnectTimeout = setTimeout(connectWebSocket, 3000);
        };
      } catch (err) {
        console.error('WebSocket initialization failed:', err);
        reconnectTimeout = setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (wsRef.current) wsRef.current.close();
    };
  }, [triggerNotification]);

  return (
    <NotificationContext.Provider
      value={{
        connectionStatus,
        latestTelemetry,
        emergencyAlerts,
        currentBannerAlert,
        sirenActive,
        triggerNotification,
        dismissBanner,
        clearAlerts
      }}
    >
      {children}
    </NotificationContext.Provider>
  );
};
