// app/dashboard/page.js
//
// Galamsey Sentinel — Command Center Dashboard
// Self-contained page. All three components (Map, Alerts, Charts) live here.
// When your project grows, cut each section into its own file under
// frontend/components/ and replace the inline definitions with imports.
//
// Dependencies (add to frontend/package.json if not already present):
//   npm install leaflet react-leaflet recharts
//
// Environment variable (frontend/.env.local):
//   NEXT_PUBLIC_API_URL=http://localhost:5000

"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

// ─── Constants ────────────────────────────────────────────────────────────────

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

// Ghana study region centre for initial map view.
const GHANA_CENTRE = [6.5, -1.5];
const INITIAL_ZOOM = 8;

// SPM severity colour scale — matches backend spm_severity strings.
const SPM_COLOURS = {
  clean             : "#22c55e",
  slightly_turbid   : "#84cc16",
  moderately_turbid : "#eab308",
  highly_turbid     : "#f97316",
  extremely_turbid  : "#ef4444",
};

const ALERT_COLOURS = {
  CRITICAL : "#ef4444",
  HIGH     : "#f97316",
  MEDIUM   : "#eab308",
  NONE     : "#22c55e",
};

// Mock turbidity history for the chart (replaced with real API data in Phase 7).
const MOCK_TURBIDITY_HISTORY = [
  { date: "Jan", pra: 28,  ankobra: 19,  birim: 35  },
  { date: "Feb", pra: 42,  ankobra: 24,  birim: 61  },
  { date: "Mar", pra: 68,  ankobra: 31,  birim: 88  },
  { date: "Apr", pra: 115, ankobra: 44,  birim: 142 },
  { date: "May", pra: 182, ankobra: 58,  birim: 203 },
  { date: "Jun", pra: 241, ankobra: 73,  birim: 178 },
  { date: "Jul", pra: 196, ankobra: 62,  birim: 155 },
  { date: "Aug", pra: 158, ankobra: 48,  birim: 121 },
];

// =============================================================================
// LEAFLET MAP COMPONENT
// Dynamically imported with ssr: false because Leaflet requires window/document
// and crashes during Next.js server-side rendering.
// =============================================================================

function MapComponentInner({ onMapClick, predictions, isLoading }) {
  const { MapContainer, TileLayer, Marker, Popup, useMapEvents } = require("react-leaflet");
  const L = require("leaflet");

  // Fix Leaflet's default marker icon path broken by webpack.
  useEffect(() => {
    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl : "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
      iconUrl       : "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
      shadowUrl     : "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
    });
  }, []);

  // Custom marker icons by prediction class.
  const makeIcon = (colour, size = 28) =>
    L.divIcon({
      className : "",
      html      : `
        <div style="
          width:${size}px; height:${size}px;
          background:${colour};
          border:3px solid #fff;
          border-radius:50% 50% 50% 0;
          transform:rotate(-45deg);
          box-shadow:0 2px 8px rgba(0,0,0,0.5);
        "></div>`,
      iconSize   : [size, size],
      iconAnchor : [size / 2, size],
    });

  const iconByClass = {
    galamsey : makeIcon("#ef4444"),
    forest   : makeIcon("#22c55e"),
    water    : makeIcon("#3b82f6"),
    pending  : makeIcon("#94a3b8"),
  };

  // Inner click handler using the Leaflet hook (must be inside MapContainer).
  function ClickHandler() {
    useMapEvents({
      click(e) {
        if (!isLoading) onMapClick(e.latlng.lat, e.latlng.lng);
      },
    });
    return null;
  }

  return (
    <MapContainer
      center={GHANA_CENTRE}
      zoom={INITIAL_ZOOM}
      style={{ width: "100%", height: "100%", borderRadius: "0 0 8px 8px" }}
    >
      {/* ESRI World Imagery satellite basemap — free, no API key required. */}
      <TileLayer
        attribution='Tiles &copy; Esri &mdash; Source: Esri, Maxar, GeoEye'
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
      />

      {/* Leaflet Geoscale label overlay so place names are visible. */}
      <TileLayer
        attribution=""
        url="https://stamen-tiles.a.ssl.fastly.net/toner-labels/{z}/{x}/{y}.png"
        opacity={0.4}
      />

      <ClickHandler />

      {predictions.map((p) => (
        <Marker
          key={p.request_id}
          position={[p.coordinates.lat, p.coordinates.lon]}
          icon={iconByClass[p.prediction] || iconByClass.pending}
        >
          <Popup maxWidth={260}>
            <div style={{ fontFamily: "monospace", fontSize: "12px", lineHeight: 1.6 }}>
              <strong style={{ fontSize: "13px" }}>
                {p.prediction.toUpperCase()}
              </strong>
              <br />
              Confidence : {(p.confidence * 100).toFixed(1)}%<br />
              SPM        : {p.spm_mean.toFixed(2)} mg/L<br />
              Severity   : {p.spm_severity.replace(/_/g, " ")}<br />
              Alert      : <span style={{ color: ALERT_COLOURS[p.alert.level] }}>
                {p.alert.level}
              </span><br />
              Lat / Lon  : {p.coordinates.lat.toFixed(5)}, {p.coordinates.lon.toFixed(5)}<br />
              <span style={{ color: "#888" }}>{p.request_id}</span>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}

// Dynamic import — disables SSR for the entire Leaflet tree.
const MapComponent = dynamic(() => Promise.resolve(MapComponentInner), {
  ssr    : false,
  loading: () => (
    <div style={styles.mapPlaceholder}>
      <span style={{ color: "#64748b", fontSize: "14px" }}>
        Loading satellite map…
      </span>
    </div>
  ),
});

// =============================================================================
// ALERTS PANEL COMPONENT
// =============================================================================

function AlertsPanel({ predictions }) {
  const triggered = predictions
    .filter((p) => p.alert.triggered)
    .slice()
    .reverse(); // newest first

  return (
    <div style={styles.panel}>
      <div style={styles.panelHeader}>
        <span style={styles.panelIcon}>🚨</span>
        <span style={styles.panelTitle}>Active Alerts</span>
        {triggered.length > 0 && (
          <span style={{
            ...styles.badge,
            backgroundColor: ALERT_COLOURS.HIGH,
          }}>
            {triggered.length}
          </span>
        )}
      </div>

      <div style={styles.panelBody}>
        {triggered.length === 0 ? (
          <div style={styles.emptyState}>
            <div style={{ fontSize: "28px", marginBottom: "8px" }}>✅</div>
            <p style={{ color: "#64748b", fontSize: "13px", margin: 0 }}>
              No active alerts. Click anywhere on the map to run a prediction.
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {triggered.map((p) => (
              <AlertCard key={p.request_id} prediction={p} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function AlertCard({ prediction: p }) {
  const colour = ALERT_COLOURS[p.alert.level];
  return (
    <div style={{
      ...styles.alertCard,
      borderLeft: `4px solid ${colour}`,
    }}>
      <div style={styles.alertCardHeader}>
        <span style={{ ...styles.alertLevel, color: colour }}>
          {p.alert.level}
        </span>
        <span style={styles.alertClass}>
          {p.prediction.toUpperCase()}
        </span>
      </div>
      <p style={styles.alertMessage}>{p.alert.message}</p>
      <div style={styles.alertMeta}>
        <span>📍 {p.coordinates.lat.toFixed(4)}, {p.coordinates.lon.toFixed(4)}</span>
        <span>💧 SPM {p.spm_mean.toFixed(1)} mg/L</span>
      </div>
    </div>
  );
}

// =============================================================================
// TURBIDITY CHART COMPONENT
// =============================================================================

function TurbidityChart({ predictions }) {
  // Merge live prediction SPM values into the chart's time series.
  // Each live prediction appends a "Live" entry showing the latest SPM reading.
  const liveSPM = predictions.length > 0
    ? predictions[predictions.length - 1].spm_mean
    : null;

  const chartData = liveSPM !== null
    ? [...MOCK_TURBIDITY_HISTORY, { date: "Live", pra: liveSPM, ankobra: null, birim: null }]
    : MOCK_TURBIDITY_HISTORY;

  return (
    <div style={styles.panel}>
      <div style={styles.panelHeader}>
        <span style={styles.panelIcon}>📊</span>
        <span style={styles.panelTitle}>River Turbidity (SPM mg/L)</span>
      </div>

      <div style={{ ...styles.panelBody, padding: "0.5rem 0 0 0" }}>
        <div style={styles.chartLegend}>
          {[
            { key: "pra",     colour: "#f97316", label: "Pra"     },
            { key: "ankobra", colour: "#3b82f6", label: "Ankobra" },
            { key: "birim",   colour: "#a855f7", label: "Birim"   },
          ].map(({ key, colour, label }) => (
            <span key={key} style={styles.legendItem}>
              <span style={{ ...styles.legendDot, backgroundColor: colour }} />
              {label}
            </span>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 4, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              axisLine={{ stroke: "#333" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1a2e",
                border          : "1px solid #333",
                borderRadius    : "6px",
                color           : "#e2e8f0",
                fontSize        : "12px",
              }}
              formatter={(v, name) =>
                v !== null ? [`${v.toFixed(1)} mg/L`, name] : ["N/A", name]
              }
            />
            {/* 100 mg/L alert threshold line */}
            <ReferenceLine
              y={100}
              stroke="#ef4444"
              strokeDasharray="6 3"
              label={{ value: "Alert", fill: "#ef4444", fontSize: 10 }}
            />
            <Line dataKey="pra"     stroke="#f97316" strokeWidth={2} dot={false} connectNulls />
            <Line dataKey="ankobra" stroke="#3b82f6" strokeWidth={2} dot={false} connectNulls />
            <Line dataKey="birim"   stroke="#a855f7" strokeWidth={2} dot={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>

        <p style={styles.chartNote}>
          Historical data (2025 composite). Live point appended on each prediction.
          100 mg/L red line = alert threshold.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// STATS BAR COMPONENT
// =============================================================================

function StatsBar({ predictions, isLoading }) {
  const total     = predictions.length;
  const galamsey  = predictions.filter((p) => p.prediction === "galamsey").length;
  const alerts    = predictions.filter((p) => p.alert.triggered).length;
  const avgSPM    = total > 0
    ? (predictions.reduce((s, p) => s + p.spm_mean, 0) / total).toFixed(1)
    : "--";

  const stats = [
    { label: "Queries",       value: total,       colour: "#94a3b8" },
    { label: "Galamsey",      value: galamsey,    colour: "#ef4444" },
    { label: "Active Alerts", value: alerts,      colour: "#f97316" },
    { label: "Avg SPM",       value: `${avgSPM} mg/L`, colour: "#3b82f6" },
  ];

  return (
    <div style={styles.statsBar}>
      {stats.map(({ label, value, colour }) => (
        <div key={label} style={styles.statItem}>
          <span style={{ ...styles.statValue, color: colour }}>{value}</span>
          <span style={styles.statLabel}>{label}</span>
        </div>
      ))}
      {isLoading && (
        <div style={styles.loadingPill}>
          <span style={styles.spinner} />
          Querying model…
        </div>
      )}
    </div>
  );
}

// =============================================================================
// PREDICTION RESULT TOAST
// =============================================================================

function ResultToast({ result, onDismiss }) {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 6000);
    return () => clearTimeout(timer);
  }, [result, onDismiss]);

  if (!result) return null;

  const colour = ALERT_COLOURS[result.alert.level];

  return (
    <div style={{ ...styles.toast, borderLeft: `4px solid ${colour}` }}>
      <div style={styles.toastHeader}>
        <strong style={{ color: colour }}>{result.alert.level}</strong>
        <span style={{ color: "#94a3b8", fontSize: "12px" }}>
          {result.prediction.toUpperCase()} — {(result.confidence * 100).toFixed(0)}% conf.
        </span>
        <button onClick={onDismiss} style={styles.toastClose}>✕</button>
      </div>
      <p style={styles.toastBody}>{result.alert.message}</p>
      <span style={styles.toastMeta}>
        SPM {result.spm_mean.toFixed(2)} mg/L · {result.spm_severity.replace(/_/g, " ")}
      </span>
    </div>
  );
}

// =============================================================================
// MAIN DASHBOARD PAGE
// =============================================================================

export default function DashboardPage() {
  const [predictions, setPredictions] = useState([]);
  const [isLoading,   setIsLoading  ] = useState(false);
  const [error,       setError      ] = useState(null);
  const [latestResult,setLatestResult] = useState(null);

  // ── API call: POST /api/predict ─────────────────────────────────────────────
  const handleMapClick = useCallback(async (lat, lon) => {
    if (isLoading) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method  : "POST",
        headers : { "Content-Type": "application/json" },
        body    : JSON.stringify({ lat, lon }),
      });

      const json = await response.json();

      if (!response.ok || json.status === "error") {
        const errMsg = json.error?.message
          || json.error?.details?.[0]
          || `Server returned ${response.status}`;
        setError(errMsg);
        return;
      }

      // Flatten data + envelope fields so components can use a single object.
      const result = {
        request_id: json.request_id,
        timestamp : json.timestamp,
        ...json.data,
      };

      setPredictions((prev) => [...prev, result]);
      setLatestResult(result);

    } catch (err) {
      setError(
        err.message.includes("fetch")
          ? `Cannot reach API at ${API_URL}. Is the backend running?`
          : err.message
      );
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div style={styles.root}>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.headerLogo}>🛰</span>
          <div>
            <h1 style={styles.headerTitle}>Galamsey Sentinel</h1>
            <p style={styles.headerSubtitle}>
              AI-Driven Satellite Surveillance · Ghana · {new Date().getFullYear()}
            </p>
          </div>
        </div>
        <div style={styles.headerRight}>
          <span style={styles.statusDot} />
          <span style={{ color: "#22c55e", fontSize: "13px" }}>
            API {API_URL.replace("http://", "")}
          </span>
        </div>
      </header>

      {/* ── Stats bar ──────────────────────────────────────────────────────── */}
      <StatsBar predictions={predictions} isLoading={isLoading} />

      {/* ── Error banner ───────────────────────────────────────────────────── */}
      {error && (
        <div style={styles.errorBanner}>
          ⚠ {error}
          <button onClick={() => setError(null)} style={styles.errorDismiss}>✕</button>
        </div>
      )}

      {/* ── Main grid ──────────────────────────────────────────────────────── */}
      <div style={styles.grid}>

        {/* Left column — map */}
        <div style={styles.mapColumn}>
          <div style={styles.mapHeader}>
            <span style={{ color: "#94a3b8", fontSize: "13px" }}>
              🖱 Click anywhere on the map to run a prediction
            </span>
            {isLoading && <span style={styles.mapLoadingBadge}>Analysing…</span>}
          </div>
          <div style={styles.mapContainer}>
            <MapComponent
              onMapClick={handleMapClick}
              predictions={predictions}
              isLoading={isLoading}
            />
          </div>
        </div>

        {/* Right column — panels */}
        <div style={styles.rightColumn}>
          <AlertsPanel  predictions={predictions} />
          <TurbidityChart predictions={predictions} />
        </div>

      </div>

      {/* ── Result toast ───────────────────────────────────────────────────── */}
      <ResultToast
        result={latestResult}
        onDismiss={() => setLatestResult(null)}
      />

    </div>
  );
}

// =============================================================================
// STYLES
// All layout and cosmetic styles in one place so you can scan and edit them
// without hunting through JSX. Moved out of JSX for readability.
// =============================================================================

const styles = {
  // Page shell
  root: {
    backgroundColor : "#0a0a0f",
    minHeight       : "100vh",
    color           : "#e2e8f0",
    fontFamily      : "'Courier New', 'Lucida Console', monospace",
    display         : "flex",
    flexDirection   : "column",
    position        : "relative",
  },

  // Header
  header: {
    display         : "flex",
    alignItems      : "center",
    justifyContent  : "space-between",
    padding         : "1rem 1.5rem",
    borderBottom    : "1px solid #1e293b",
    backgroundColor : "#0d0d17",
  },
  headerLeft: {
    display    : "flex",
    alignItems : "center",
    gap        : "1rem",
  },
  headerLogo: {
    fontSize : "2rem",
  },
  headerTitle: {
    fontSize   : "1.4rem",
    fontWeight : "700",
    margin     : 0,
    color      : "#f8fafc",
    letterSpacing: "0.05em",
  },
  headerSubtitle: {
    fontSize : "11px",
    color    : "#475569",
    margin   : "2px 0 0",
    letterSpacing: "0.08em",
    textTransform: "uppercase",
  },
  headerRight: {
    display    : "flex",
    alignItems : "center",
    gap        : "8px",
  },
  statusDot: {
    width           : "8px",
    height          : "8px",
    borderRadius    : "50%",
    backgroundColor : "#22c55e",
    boxShadow       : "0 0 6px #22c55e",
    display         : "inline-block",
  },

  // Stats bar
  statsBar: {
    display         : "flex",
    alignItems      : "center",
    gap             : "0",
    backgroundColor : "#0d0d17",
    borderBottom    : "1px solid #1e293b",
    padding         : "0 1.5rem",
    flexWrap        : "wrap",
  },
  statItem: {
    display        : "flex",
    flexDirection  : "column",
    alignItems     : "center",
    padding        : "0.6rem 1.5rem",
    borderRight    : "1px solid #1e293b",
    minWidth       : "90px",
  },
  statValue: {
    fontSize   : "1.3rem",
    fontWeight : "700",
    lineHeight : 1.1,
  },
  statLabel: {
    fontSize  : "10px",
    color     : "#475569",
    marginTop : "2px",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  },
  loadingPill: {
    display        : "flex",
    alignItems     : "center",
    gap            : "8px",
    marginLeft     : "auto",
    backgroundColor: "#1e293b",
    padding        : "6px 14px",
    borderRadius   : "20px",
    fontSize       : "12px",
    color          : "#94a3b8",
  },
  spinner: {
    display        : "inline-block",
    width          : "12px",
    height         : "12px",
    border         : "2px solid #334155",
    borderTopColor : "#3b82f6",
    borderRadius   : "50%",
    animation      : "spin 0.8s linear infinite",
  },

  // Error banner
  errorBanner: {
    display        : "flex",
    alignItems     : "center",
    justifyContent : "space-between",
    backgroundColor: "#450a0a",
    color          : "#fca5a5",
    padding        : "0.6rem 1.5rem",
    fontSize       : "13px",
    borderBottom   : "1px solid #7f1d1d",
  },
  errorDismiss: {
    background  : "none",
    border      : "none",
    color       : "#fca5a5",
    cursor      : "pointer",
    fontSize    : "14px",
    padding     : "0 4px",
  },

  // Main grid
  grid: {
    display             : "grid",
    gridTemplateColumns : "1fr 340px",
    gap                 : "0",
    flex                : "1",
    overflow            : "hidden",
  },

  // Map column
  mapColumn: {
    display       : "flex",
    flexDirection : "column",
    borderRight   : "1px solid #1e293b",
  },
  mapHeader: {
    display        : "flex",
    alignItems     : "center",
    justifyContent : "space-between",
    padding        : "0.5rem 1rem",
    backgroundColor: "#0d0d17",
    borderBottom   : "1px solid #1e293b",
    minHeight      : "36px",
  },
  mapLoadingBadge: {
    backgroundColor: "#1e3a5f",
    color          : "#93c5fd",
    padding        : "2px 10px",
    borderRadius   : "12px",
    fontSize       : "11px",
  },
  mapContainer: {
    flex     : "1",
    minHeight: "600px",
    position : "relative",
  },
  mapPlaceholder: {
    width           : "100%",
    height          : "100%",
    minHeight       : "600px",
    display         : "flex",
    alignItems      : "center",
    justifyContent  : "center",
    backgroundColor : "#111827",
  },

  // Right column
  rightColumn: {
    display       : "flex",
    flexDirection : "column",
    overflow      : "auto",
    backgroundColor: "#0d0d17",
  },

  // Shared panel
  panel: {
    borderBottom  : "1px solid #1e293b",
    display       : "flex",
    flexDirection : "column",
  },
  panelHeader: {
    display        : "flex",
    alignItems     : "center",
    gap            : "8px",
    padding        : "0.85rem 1.1rem",
    borderBottom   : "1px solid #1e293b",
    backgroundColor: "#111827",
  },
  panelIcon: {
    fontSize: "16px",
  },
  panelTitle: {
    fontSize   : "13px",
    fontWeight : "600",
    color      : "#cbd5e1",
    flex       : 1,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  },
  panelBody: {
    padding  : "1rem",
    overflowY: "auto",
    maxHeight: "280px",
  },
  badge: {
    display      : "inline-flex",
    alignItems   : "center",
    justifyContent: "center",
    width        : "20px",
    height       : "20px",
    borderRadius : "50%",
    fontSize     : "11px",
    fontWeight   : "700",
    color        : "#fff",
  },

  // Empty state
  emptyState: {
    textAlign  : "center",
    padding    : "1.5rem 1rem",
    color      : "#475569",
  },

  // Alert card
  alertCard: {
    backgroundColor: "#111827",
    borderRadius   : "6px",
    padding        : "10px 12px",
  },
  alertCardHeader: {
    display        : "flex",
    alignItems     : "center",
    justifyContent : "space-between",
    marginBottom   : "4px",
  },
  alertLevel: {
    fontSize   : "11px",
    fontWeight : "700",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
  },
  alertClass: {
    fontSize  : "11px",
    color     : "#475569",
    fontFamily: "monospace",
  },
  alertMessage: {
    fontSize  : "12px",
    color     : "#94a3b8",
    margin    : "4px 0",
    lineHeight: 1.5,
  },
  alertMeta: {
    display  : "flex",
    gap      : "12px",
    fontSize : "11px",
    color    : "#475569",
    marginTop: "6px",
  },

  // Chart
  chartLegend: {
    display       : "flex",
    gap           : "12px",
    padding       : "0 1rem 8px",
    flexWrap      : "wrap",
  },
  legendItem: {
    display   : "flex",
    alignItems: "center",
    gap       : "5px",
    fontSize  : "11px",
    color     : "#64748b",
  },
  legendDot: {
    width       : "8px",
    height      : "8px",
    borderRadius: "50%",
    display     : "inline-block",
  },
  chartNote: {
    fontSize  : "10px",
    color     : "#334155",
    padding   : "4px 1rem 0.5rem",
    margin    : 0,
    lineHeight: 1.4,
  },

  // Toast
  toast: {
    position       : "fixed",
    bottom         : "1.5rem",
    right          : "1.5rem",
    backgroundColor: "#0d1117",
    border         : "1px solid #1e293b",
    borderRadius   : "8px",
    padding        : "12px 16px",
    maxWidth       : "360px",
    boxShadow      : "0 8px 32px rgba(0,0,0,0.6)",
    zIndex         : 9999,
    animation      : "slideUp 0.25s ease",
  },
  toastHeader: {
    display        : "flex",
    alignItems     : "center",
    justifyContent : "space-between",
    gap            : "8px",
    marginBottom   : "6px",
  },
  toastBody: {
    fontSize  : "12px",
    color     : "#94a3b8",
    margin    : "0 0 6px",
    lineHeight: 1.5,
  },
  toastMeta: {
    fontSize: "11px",
    color   : "#475569",
  },
  toastClose: {
    background  : "none",
    border      : "none",
    color       : "#475569",
    cursor      : "pointer",
    fontSize    : "13px",
    padding     : "0",
    marginLeft  : "auto",
  },
};