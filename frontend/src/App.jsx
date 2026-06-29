import React, { useState, useEffect, useRef } from 'react';
import {
  Activity, Shield, Layers, Cpu, LayoutDashboard,
  Volume2, VolumeX, AlertTriangle, Play, RefreshCw,
  TrendingUp, TrendingDown, ChevronRight, Settings, Info, Zap
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, BarChart, Bar, Cell,
  PieChart as RePieChart, Pie, ScatterChart, Scatter,
  LineChart, Line, ReferenceLine
} from 'recharts';
import Chart from 'react-apexcharts';

// ─── Ticker Metadata (Matching backend & full company names) ──────────────
const TICKERS = {
  "AAPL": "Apple Inc.",
  "MSFT": "Microsoft Corp.",
  "NVDA": "NVIDIA Corp.",
  "GOOGL": "Alphabet Inc.",
  "META": "Meta Platforms",
  "JPM": "JPMorgan Chase",
  "V": "Visa Inc.",
  "AMZN": "Amazon.com Inc.",
  "TSLA": "Tesla Inc.",
  "JNJ": "Johnson & Johnson"
};

const CHART_COLORS = ['#2563EB', '#00C896', '#EF4444', '#F59E0B', '#3B82F6', '#10B981', '#F87171', '#06B6D4'];

// ─── Interactive Watchlist Card Component ──────────────────────────────────
function WatchlistRow({ symbol, name, price, pctChange, active, onClick }) {
  const isUp = pctChange >= 0;
  return (
    <div
      onClick={onClick}
      className={`group flex items-center justify-between p-4 rounded-xl cursor-pointer transition-all duration-300 select-none mb-3 ${
        active
          ? 'bg-accent-light border-2 border-accent/30 shadow-card'
          : 'bg-[var(--surface)] hover:bg-[var(--surface-sub)] hover:shadow-soft border-2 border-transparent'
      }`}
    >
      <div className="flex items-center gap-4 min-w-0">
        <div 
          className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 transition-all duration-300 ${
            active ? 'text-white shadow-glow' : 'bg-[var(--surface-sub)] text-[var(--text-primary)]'
          }`}
          style={active ? { background: 'linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%)' } : {}}
        >
          {symbol.substring(0,2)}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-bold text-[var(--text-primary)]">{symbol}</div>
          <div className="text-xs text-[var(--text-secondary)] truncate max-w-[90px]">{name}</div>
        </div>
      </div>
      <div className="text-right flex flex-col items-end gap-1.5">
        <span className="text-sm text-[var(--text-primary)] font-bold">${price ? price.toFixed(2) : '--'}</span>
        <span
          className={`text-xs font-bold px-2.5 py-1 rounded-lg transition-all duration-300 ${
            isUp ? 'bg-bull-light text-bull' : 'bg-bear-light text-bear'
          }`}
        >
          {isUp ? '↑ +' : '↓ '}{pctChange ? Math.abs(pctChange).toFixed(2) : '0.00'}%
        </span>
      </div>
    </div>
  );
}

// ─── Header Indicators / Market Indices Bar ─────────────────────────────────
function MarketIndicesBar({ tickers }) {
  return (
    <div className="h-14 border-b border-[var(--border)] bg-[var(--surface)]/90 backdrop-blur-sm flex items-center px-8 overflow-hidden gap-8 select-none shrink-0 z-10 shadow-soft">
      <div className="flex items-center gap-3 shrink-0">
        <span className="w-2.5 h-2.5 rounded-full bg-accent animate-pulse shadow-accent-glow" />
        <span className="text-xs font-bold tracking-wide" style={{ color: '#1E40AF' }}>DEMO MODE</span>
      </div>
      <div className="h-6 w-px bg-[var(--border)] shrink-0" />
      <div className="flex items-center gap-8 overflow-hidden flex-1 ticker-wrap">
        <div className="flex items-center gap-8 ticker-inner">
          {tickers.concat(tickers).map((t, idx) => {
            const isUp = t.pctChange >= 0;
            return (
              <div key={idx} className="flex items-center gap-3 shrink-0 text-xs">
                <span className="text-[var(--text-secondary)] font-bold">{t.ticker}</span>
                <span className="text-[var(--text-primary)] font-semibold">${t.price?.toFixed(2)}</span>
                <span className={`font-bold text-xs px-2.5 py-1 rounded-lg ${isUp ? 'bg-bull-light text-bull' : 'bg-bear-light text-bear'}`}>
                  {isUp ? '↑ +' : '↓ '}{Math.abs(t.pctChange || 0).toFixed(2)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── Custom Recharts Tooltip ────────────────────────────────────────────────
function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[var(--surface)] border-2 border-[var(--border)] rounded-xl p-4 shadow-card-hover text-xs min-w-[150px]">
      <div className="text-[var(--text-muted)] text-xs border-b border-[var(--border)] pb-2 mb-2 font-semibold">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex justify-between gap-6 py-1">
          <span style={{ color: p.color || 'var(--text-secondary)' }} className="font-semibold">{p.name}</span>
          <span className="text-[var(--text-primary)] font-bold">{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Shimmer Skeleton ───────────────────────────────────────────────────────
function Sk({ h = 'h-24', w = 'w-full', className = '' }) {
  return <div className={`skeleton ${h} ${w} ${className}`} />;
}

// ─── Panel Layout Wrapper ───────────────────────────────────────────────────
function Panel({ title, subtitle, action, children, className = '', noPad = false }) {
  return (
    <div className={`bg-[var(--surface)] rounded-2xl overflow-hidden flex flex-col shadow-card hover:shadow-card-hover transition-all duration-300 border border-[var(--border)] ${className}`}>
      {title && (
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)] shrink-0">
          <div>
            <div className="text-sm font-bold text-[var(--text-primary)]">{title}</div>
            {subtitle && <div className="text-xs text-[var(--text-secondary)] mt-1">{subtitle}</div>}
          </div>
          {action}
        </div>
      )}
      <div className={noPad ? 'flex-1 overflow-auto' : 'p-6 flex-1 overflow-auto'}>
        {children}
      </div>
    </div>
  );
}

// ─── Metric Box Component ──────────────────────────────────────────────────
function Metric({ label, value, sub, up, bull, bear, accent }) {
  const valColor = bull ? 'text-bull' : bear ? 'text-bear' : accent ? 'text-accent' : 'text-[var(--text-primary)]';
  const subColor = up === true ? 'text-bull' : up === false ? 'text-bear' : 'text-[var(--text-secondary)]';
  const bgColor = bull ? 'bg-bull-dim' : bear ? 'bg-bear-dim' : 'bg-accent-dim';
  
  return (
    <div className={`${bgColor} border border-[var(--border)] rounded-2xl px-6 py-5 flex flex-col gap-2.5 min-w-0 shadow-soft hover:shadow-card transition-all duration-300 hover:-translate-y-1`}>
      <div className="text-xs font-bold uppercase tracking-wide text-[var(--text-muted)]">{label}</div>
      <div className={`text-3xl font-bold tracking-tight ${valColor}`}>{value}</div>
      {sub && <div className={`text-xs leading-none truncate ${subColor} font-medium`}>{sub}</div>}
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [ticker, setTicker] = useState('AAPL'); // Shared global ticker for TA, Risk, ML
  const [theme, setTheme] = useState('light'); // Theme state
  
  // Zoom state persistence for ApexCharts
  const [chartZoom, setChartZoom] = useState({ min: undefined, max: undefined });
  
  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);
  
  // Real-time synchronization state
  const [liveTickers, setLiveTickers] = useState([]);
  const [lastSync, setLastSync] = useState('--:--:--');
  const [intervalNonce, setIntervalNonce] = useState(0);
  const [isConnected, setIsConnected] = useState(true);

  // Audio system state
  const [isMuted, setIsMuted] = useState(true);
  const audioCtxRef = useRef(null);
  const bgmIntervalRef = useRef(null);
  const beatRef = useRef(0);

  // Tabs Data State
  const [overviewPeriod, setOverviewPeriod] = useState('1Y');
  const [overviewData, setOverviewData] = useState(null);
  const [overviewLoading, setOverviewLoading] = useState(true);

  const [taPeriod, setTaPeriod] = useState('1Y');
  const [taIndicators, setTaIndicators] = useState(['SMA20', 'BB']);
  const [taData, setTaData] = useState(null);
  const [taLoading, setTaLoading] = useState(true);

  const [riskConf, setRiskConf] = useState(95);
  const [riskPortVal, setRiskPortVal] = useState(1000000);
  const [riskData, setRiskData] = useState(null);
  const [riskLoading, setRiskLoading] = useState(true);

  const [portTickers, setPortTickers] = useState(['AAPL', 'MSFT', 'NVDA', 'JPM', 'AMZN', 'JNJ']);
  const [portRf, setPortRf] = useState(5);
  const [portData,    setPortData]    = useState(null);
  const [portLoading, setPortLoading] = useState(true);
  const [portTrigger, setPortTrigger] = useState(0);
  const [portMode, setPortMode] = useState('maxSharpe');

  const [mlData, setMlData] = useState(null);
  const [mlTraining, setMlTraining] = useState(false);

  // ─── Sound Synthesizer Controls ───────────────────────────────────────────
  const initAudio = () => {
    if (audioCtxRef.current) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioCtxRef.current = new AudioContext();
  };

  const playBgm = () => {
    initAudio();
    if (isMuted || !audioCtxRef.current) return;
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
    if (bgmIntervalRef.current) clearInterval(bgmIntervalRef.current);

    const stepTime = 60 / 120 / 2; // 120 bpm eighth notes
    bgmIntervalRef.current = setInterval(() => {
      if (isMuted || !audioCtxRef.current) return;
      const bassSeq = [55, 55, 110, 110, 65.4, 65.4, 130.8, 130.8, 49, 49, 98, 98, 73.4, 73.4, 146.8, 146.8];
      const freq = bassSeq[beatRef.current % bassSeq.length];

      try {
        const osc = audioCtxRef.current.createOscillator();
        const filter = audioCtxRef.current.createBiquadFilter();
        const gain = audioCtxRef.current.createGain();

        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(freq, audioCtxRef.current.currentTime);
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(380, audioCtxRef.current.currentTime);
        filter.Q.setValueAtTime(3, audioCtxRef.current.currentTime);
        gain.gain.setValueAtTime(0.015, audioCtxRef.current.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtxRef.current.currentTime + stepTime - 0.015);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(audioCtxRef.current.destination);

        osc.start();
        osc.stop(audioCtxRef.current.currentTime + stepTime);

        // Subtle melody notes
        if (beatRef.current % 8 === 0) {
          const melody = [440, 493.88, 523.25, 587.33, 659.25, 587.33];
          const mFreq = melody[(beatRef.current / 8 | 0) % melody.length];
          const mOsc = audioCtxRef.current.createOscillator();
          const mGain = audioCtxRef.current.createGain();

          mOsc.type = 'sine';
          mOsc.frequency.setValueAtTime(mFreq, audioCtxRef.current.currentTime);
          mGain.gain.setValueAtTime(0.006, audioCtxRef.current.currentTime);
          mGain.gain.linearRampToValueAtTime(0.001, audioCtxRef.current.currentTime + stepTime * 3);

          mOsc.connect(mGain);
          mGain.connect(audioCtxRef.current.destination);
          mOsc.start();
          mOsc.stop(audioCtxRef.current.currentTime + stepTime * 3);
        }
      } catch (e) {}
      beatRef.current++;
    }, stepTime * 1000);
  };

  useEffect(() => {
    if (isMuted) {
      if (bgmIntervalRef.current) clearInterval(bgmIntervalRef.current);
      bgmIntervalRef.current = null;
    } else {
      playBgm();
    }
    return () => { if (bgmIntervalRef.current) clearInterval(bgmIntervalRef.current); };
  }, [isMuted]);

  // ─── 3-Second Live Ticker Sync (Only ticker ribbon and watchlist rows refresh) ──
  useEffect(() => {
    const clock = setInterval(() => setIntervalNonce(prev => prev + 1), 3000);
    return () => clearInterval(clock);
  }, []);

  useEffect(() => {
    fetch(`/api/live-tickers?n_intervals=${intervalNonce}`)
      .then(res => res.json())
      .then(data => {
        setLiveTickers(data.tickers || []);
        setLastSync(data.timestamp || '--:--:--');
        setIsConnected(true);
      })
      .catch(() => setIsConnected(false));
  }, [intervalNonce]);

  // ─── Tab 1: Overview Data (Independent 60s slow fetch + period triggers) ───
  useEffect(() => {
    let active = true;
    setOverviewLoading(true);
    const getOverview = () => {
      console.log('[Overview] Fetching data for period:', overviewPeriod);
      fetch(`/api/overview?period=${overviewPeriod}&n_intervals=0`)
        .then(res => {
          console.log('[Overview] Response status:', res.status);
          return res.json();
        })
        .then(data => {
          console.log('[Overview] Data received:', data);
          if (active) {
            setOverviewData(data);
            setOverviewLoading(false);
          }
        })
        .catch((error) => { 
          console.error('[Overview] Fetch error:', error);
          if (active) setOverviewLoading(false); 
        });
    };
    getOverview();
    const reload = setInterval(getOverview, 60000);
    return () => { active = false; clearInterval(reload); };
  }, [overviewPeriod]);

  // ─── Tab 2: Technical Analysis Data ───────────────────────────────────────
  useEffect(() => {
    let active = true;
    setTaLoading(true);
    fetch(`/api/technical?ticker=${ticker}&period=${taPeriod}&n_intervals=0`)
      .then(res => res.json())
      .then(data => {
        if (active) {
          setTaData(data);
          setTaLoading(false);
        }
      })
      .catch(() => { if (active) setTaLoading(false); });
    return () => { active = false; };
  }, [ticker, taPeriod]);

  // ─── Tab 3: Risk Analytics Data ───────────────────────────────────────────
  useEffect(() => {
    let active = true;
    setRiskLoading(true);
    fetch(`/api/risk?ticker=${ticker}&confidence=${riskConf}&portfolio_value=${riskPortVal}&n_intervals=0`)
      .then(res => res.json())
      .then(data => {
        if (active) {
          setRiskData(data);
          setRiskLoading(false);
        }
      })
      .catch(() => { if (active) setRiskLoading(false); });
    return () => { active = false; };
  }, [ticker, riskConf, riskPortVal]);

  // ─── Tab 4: Portfolio Optimizer Data ──────────────────────────────────────
  useEffect(() => {
    // Only fetch if portTrigger > 0 (user clicked OPTIMIZE at least once)
    if (portTrigger === 0) return;
    
    let active = true;
    setPortLoading(true);
    fetch(`/api/portfolio?tickers=${portTickers.join(',')}&rf=${portRf}`)
      .then(res => res.json())
      .then(data => {
        if (active) {
          setPortData(data);
          setPortLoading(false);
        }
      })
      .catch((err) => { 
        if (active) {
          console.error('Portfolio optimization failed:', err);
          setPortLoading(false);
        }
      });
    return () => { active = false; };
  }, [portTrigger]);

  // ─── Tab 5: ML Predictions Pipeline ──────────────────────────────────────
  const runMlModel = () => {
    setMlTraining(true);
    console.log('[ML] Starting model training for ticker:', ticker);
    
    // Add timeout to prevent infinite loading
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000); // 90 second timeout
    
    fetch(`/api/ml?ticker=${ticker}`, { signal: controller.signal })
      .then(res => {
        clearTimeout(timeoutId);
        console.log('[ML] Response status:', res.status);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log('[ML] Model training complete:', data);
        setMlData(data);
        setMlTraining(false);
      })
      .catch((error) => {
        clearTimeout(timeoutId);
        console.error('[ML] Training failed:', error);
        setMlTraining(false);
        alert(`ML model training failed: ${error.message}\n\nThis might be due to insufficient data or a timeout. Try selecting a different ticker or refreshing the page.`);
      });
  };

  useEffect(() => {
    if (activeTab === 'ml') {
      runMlModel();
    }
  }, [activeTab, ticker]);

  // ─── Formatted Data for ApexCharts to prevent index mapping exceptions ────
  const getApexSeries = () => {
    if (!taData?.dates) return [];
    
    const candles = taData.dates.map((d, i) => ({
      x: new Date(d).getTime(),
      y: [taData.ohlc.open[i], taData.ohlc.high[i], taData.ohlc.low[i], taData.ohlc.close[i]]
    }));

    const result = [{
      name: 'OHLC',
      type: 'candlestick',
      data: candles
    }];

    const addLine = (label, values) => {
      result.push({
        name: label,
        type: 'line',
        data: taData.dates.map((d, i) => ({ x: new Date(d).getTime(), y: values[i] }))
      });
    };

    if (taIndicators.includes('SMA20')) addLine('SMA 20', taData.indicators.sma20);
    if (taIndicators.includes('SMA50')) addLine('SMA 50', taData.indicators.sma50);
    if (taIndicators.includes('EMA9'))  addLine('EMA 9',  taData.indicators.ema9);
    if (taIndicators.includes('BB')) {
      addLine('BB Upper', taData.indicators.bb_up);
      addLine('BB Lower', taData.indicators.bb_low);
    }

    return result;
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[var(--bg)] text-[var(--text-primary)] overflow-hidden font-sans select-none antialiased transition-colors duration-300">
      
      {/* ─── HEADER BAR (Navbar with Branding & Top Navigation Pills) ───────── */}
      <header className="h-20 border-b border-[var(--border)] bg-[var(--surface)]/95 backdrop-blur-md flex items-center justify-between px-8 shrink-0 z-30 shadow-soft">
        
        {/* Brand/Logo */}
        <div className="flex items-center gap-4 w-64 shrink-0">
          <div 
            className="w-12 h-12 rounded-2xl flex items-center justify-center shadow-glow"
            style={{ background: 'linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%)' }}
          >
            <Activity className="w-6 h-6 text-white" strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-[var(--text-primary)]">FinanceIQ</h1>
            <p className="text-xs uppercase tracking-wider text-[var(--text-secondary)] leading-none mt-0.5">Quant Terminal</p>
          </div>
        </div>

        {/* ─── HORIZONTAL TOP NAVIGATION PILLS ─── */}
        <nav className="flex items-center gap-2 bg-[var(--surface-sub)] p-2 rounded-xl shadow-soft">
          {[
            { id: 'overview',  label: 'Overview',  Icon: LayoutDashboard },
            { id: 'ta',        label: 'Technical', Icon: Activity },
            { id: 'risk',      label: 'Risk',      Icon: Shield },
            { id: 'portfolio', label: 'Portfolio', Icon: Layers },
            { id: 'ml',        label: 'Predict',   Icon: Cpu }
          ].map(({ id, label, Icon }) => {
            const active = activeTab === id;
            return (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center gap-2.5 px-6 py-3 rounded-lg text-sm font-bold transition-all duration-300 ${
                  active
                    ? 'text-white shadow-glow'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface)]'
                }`}
                style={active ? { background: 'linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%)' } : {}}
              >
                <Icon className="w-4 h-4" strokeWidth={active ? 2.5 : 2} />
                <span>{label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Header System Controls */}
        <div className="flex items-center gap-4 shrink-0">
          <div className="flex items-center gap-3 bg-[var(--surface-sub)] px-4 py-2.5 rounded-xl">
            <span className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-accent animate-pulse shadow-accent-glow' : 'bg-bear'}`} />
            <span className="text-[var(--text-secondary)] font-semibold text-xs">{isConnected ? 'CONNECTED' : 'OFFLINE'}</span>
          </div>
          
          {/* Download Report Button */}
          <button
            onClick={() => {
              const tickers = Object.keys(TICKERS).slice(0, 4).join(',');
              // Show alert that PDF generation takes time
              alert('⏱️ PDF Report Generation\n\nGenerating your comprehensive report...\n\nThis may take 2-3 minutes as it:\n• Downloads fresh market data\n• Trains ML models\n• Runs portfolio optimization\n• Creates professional charts\n\nPlease wait - the download will start automatically!');
              window.open(`http://localhost:8000/api/download-report?tickers=${tickers}`, '_blank');
            }}
            className="flex items-center gap-2.5 px-5 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 text-white shadow-glow hover:shadow-xl"
            style={{ background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)' }}
            title="Download comprehensive PDF report (takes 2-3 min)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span>Download Report</span>
          </button>
          
          {/* Theme Toggle */}
          <button
            onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            className="w-11 h-11 rounded-xl flex items-center justify-center bg-[var(--surface-sub)] text-[var(--text-secondary)] hover:bg-accent-dim hover:text-accent transition-all duration-300"
            title="Toggle theme"
          >
            {theme === 'light' ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            )}
          </button>
          
          <button
            onClick={() => setIsMuted(!isMuted)}
            className={`w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300 ${
              isMuted 
                ? 'bg-[var(--surface-sub)] text-[var(--text-secondary)] hover:bg-accent-dim hover:text-accent' 
                : 'bg-accent-light text-accent'
            }`}
            title={isMuted ? 'Unmute audio' : 'Mute audio'}
          >
            {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
          </button>
        </div>
      </header>

      {/* Live Index/Market Tickers Row */}
      <MarketIndicesBar tickers={liveTickers} />

      {/* ─── MAIN WORKSPACE AND SIDE WATCHLIST PANEL ──────────────────────── */}
      <div className="flex-1 flex overflow-hidden min-w-0">
        
        {/* CENTER MAIN WORKSPACE */}
        <main className="flex-1 overflow-y-auto bg-[var(--bg)] border-r border-[var(--border)] min-w-0">
          <div className="p-8 space-y-8 max-w-[1500px] mx-auto">
            
            {/* Active Asset Info Segment */}
            <div className="flex items-center justify-between pb-5">
              <div className="flex items-center gap-5">
                <div className="flex items-center gap-4 bg-[var(--surface)] px-5 py-4 rounded-2xl shadow-card border border-[var(--border)]">
                  <div className="text-2xl font-bold text-[var(--text-primary)] tracking-tight">{ticker}</div>
                  <div className="h-6 w-px bg-[var(--border)]" />
                  <div className="text-xs text-[var(--text-secondary)] bg-[var(--surface-sub)] px-4 py-1.5 rounded-lg font-medium">
                    {TICKERS[ticker] || 'Quant Instrument'}
                  </div>
                </div>
                <div className="text-xs text-[var(--text-muted)] flex items-center gap-2.5 bg-[var(--surface)] px-4 py-3 rounded-xl shadow-card">
                  <span className="w-2 h-2 rounded-full bg-bull shadow-bull-glow" />
                  <span className="font-medium">Updated: {lastSync}</span>
                </div>
              </div>

              {/* Time period options */}
              {activeTab === 'overview' && (
                <div className="flex items-center gap-4 bg-[var(--surface)] px-5 py-3 rounded-2xl shadow-card">
                  <span className="text-xs text-[var(--text-muted)] font-semibold uppercase tracking-wide">Period</span>
                  <select
                    value={overviewPeriod}
                    onChange={e => setOverviewPeriod(e.target.value)}
                    className="input-base"
                  >
                    {['1M','3M','6M','1Y','2Y','5Y'].map(p => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>
              )}
              {activeTab === 'ta' && (
                <div className="flex items-center gap-4 bg-[var(--surface)] px-5 py-3 rounded-2xl shadow-card">
                  <span className="text-xs text-[var(--text-muted)] font-semibold uppercase tracking-wide">Period</span>
                  <select
                    value={taPeriod}
                    onChange={e => setTaPeriod(e.target.value)}
                    className="input-base"
                  >
                    {['3M','6M','1Y','2Y'].map(p => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {/* ════════════════════════════════════════════════════════════
                TAB 1 — MARKET OVERVIEW
            ════════════════════════════════════════════════════════════ */}
            {activeTab === 'overview' && (
              <div className="space-y-8 animate-fadeIn">
                {overviewLoading || !overviewData ? (
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
                    {[1, 2, 3, 4].map(i => <Sk key={i} h="h-28" />)}
                  </div>
                ) : (
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
                    <Metric
                      label="Assets Tracked"
                      value={overviewData.kpi.assetsTracked}
                      sub="Diversified portfolio"
                      accent
                    />
                    <Metric
                      label="Top Gainer"
                      value={overviewData.kpi.bestPerformer}
                      sub={`+${overviewData.kpi.bestReturn}% return`}
                      up={true}
                      bull
                    />
                    <Metric
                      label="Max Drawdown"
                      value={`${overviewData.kpi.worstReturn}%`}
                      sub={overviewData.kpi.worstPerformer}
                      up={false}
                      bear
                    />
                    <Metric
                      label="Portfolio Vol"
                      value={`${overviewData.kpi.avgVolatility}%`}
                      sub="Rolling annualized"
                    />
                  </div>
                )}

                {/* Returns Cumulative area and sector bar */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <Panel
                    className="lg:col-span-2"
                    title={`Asset Growth Performance — ${overviewPeriod}`}
                    subtitle="Normalized investment cumulative returns comparison"
                  >
                    {overviewLoading || !overviewData ? <Sk h="h-72" /> : (
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart margin={{ top: 5, right: 5, left: -22, bottom: 0 }}>
                            <defs>
                              {Object.keys(overviewData.cumulative).map((tk, i) => (
                                <linearGradient key={tk} id={`g-${tk}`} x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%"  stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.3} />
                                  <stop offset="95%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.05} />
                                </linearGradient>
                              ))}
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                            <XAxis 
                              dataKey="date" 
                              allowDuplicatedCategory={false} 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                              label={{ value: 'Date', position: 'insideBottom', offset: -5, fill: 'var(--text-secondary)', fontSize: 11 }}
                            />
                            <YAxis 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} 
                              domain={['auto','auto']}
                              label={{ value: 'Cumulative Return (%)', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)', fontSize: 11 }}
                            />
                            <Tooltip content={<ChartTip />} />
                            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
                            {Object.entries(overviewData.cumulative).map(([tk, series], i) => {
                              const col = CHART_COLORS[i % CHART_COLORS.length];
                              const pts = series.dates.map((d, j) => ({ date: d, [tk]: series.values[j] }));
                              return (
                                <Area 
                                  key={tk} 
                                  type="monotone" 
                                  dataKey={tk} 
                                  data={pts} 
                                  stroke={col} 
                                  strokeWidth={2} 
                                  fillOpacity={1} 
                                  fill={`url(#g-${tk})`}
                                  isAnimationActive={false}
                                />
                              );
                            })}
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </Panel>

                  <Panel title="Sector Return Allocations" subtitle="Average % return performance by market sector">
                    {overviewLoading || !overviewData ? <Sk h="h-72" /> : (
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={overviewData.sectors} margin={{ top: 5, right: 5, left: -25, bottom: 50 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                            <XAxis 
                              dataKey="sector" 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 10 }}
                              angle={0}
                              textAnchor="middle"
                              height={60}
                              interval={0}
                            />
                            <YAxis 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                              label={{ value: 'Return (%)', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)', fontSize: 11 }}
                            />
                            <Tooltip content={<ChartTip />} />
                            <Bar dataKey="return" radius={[6,6,0,0]} isAnimationActive={false}>
                              {overviewData.sectors.map((e, i) => (
                                <Cell key={i} fill={e.return >= 0 ? '#00C896' : '#EF4444'} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </Panel>
                </div>

                {/* Grid correlation and return probability histograms */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Panel title="Correlation Grid Matrix" subtitle="Calculated pairwise daily return correlation coefficients" noPad>
                    {overviewLoading || !overviewData ? <div className="p-4"><Sk h="h-72" /></div> : (
                      <div className="overflow-x-auto p-5">
                        <table className="text-2xs border-collapse w-full">
                          <thead>
                            <tr>
                              <th className="p-2 text-left text-muted font-bold" />
                              {overviewData.correlation.columns.map(col => (
                                <th key={col} className="p-2 text-center text-muted font-bold text-3xs">{col}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {overviewData.correlation.index.map((label, rIdx) => (
                              <tr key={label}>
                                <td className="p-2 font-bold text-secondary text-left">{label}</td>
                                {overviewData.correlation.data[rIdx].map((val, cIdx) => {
                                  const absVal = Math.abs(val);
                                  const bg = val >= 0
                                    ? `rgba(91,79,232,${(absVal * 0.6).toFixed(2)})`
                                    : `rgba(255,92,92,${(absVal * 0.6).toFixed(2)})`;
                                  const textCol = absVal > 0.5 ? '#FFFFFF' : '#1A1D29';
                                  return (
                                    <td
                                      key={cIdx}
                                      style={{ backgroundColor: bg, color: textCol }}
                                      className="p-2 text-center font-bold border border-border/30 rounded-md"
                                    >
                                      {val.toFixed(2)}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </Panel>

                  <Panel title="Daily Distributions" subtitle="Return value probabilities (%)">
                    {overviewLoading || !overviewData ? <Sk h="h-72" /> : (
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                            <XAxis 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                              label={{ value: 'Return Range', position: 'insideBottom', offset: -5, fill: 'var(--text-secondary)', fontSize: 11 }}
                            />
                            <YAxis 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                              label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)', fontSize: 11 }}
                            />
                            <Tooltip content={<ChartTip />} />
                            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
                            {Object.entries(overviewData.distributions).map(([tk, valList], i) => {
                              const col = CHART_COLORS[i % CHART_COLORS.length];
                              const minV = Math.min(...valList), maxV = Math.max(...valList);
                              const bw = (maxV - minV) / 30;
                              const bins = Array(30).fill(0);
                              valList.forEach(v => bins[Math.min((v - minV) / bw | 0, 29)]++);
                              const pts = bins.map((c, j) => ({ bin: (minV + j * bw).toFixed(2), count: c }));
                              return (
                                <Area 
                                  key={tk} 
                                  type="monotone" 
                                  dataKey="count" 
                                  name={tk} 
                                  data={pts} 
                                  stroke={col} 
                                  fill={col} 
                                  fillOpacity={0.1} 
                                  strokeWidth={2}
                                  isAnimationActive={false}
                                />
                              );
                            })}
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </Panel>
                </div>
              </div>
            )}

            {/* ════════════════════════════════════════════════════════════
                TAB 2 — TECHNICAL ANALYSIS
            ════════════════════════════════════════════════════════════ */}
            {activeTab === 'ta' && (
              <div className="space-y-4 animate-fadeIn">
                {/* Indicator overlay selections */}
                <div className="flex items-center justify-between bg-[var(--surface)] border border-[var(--border)] rounded-xl p-4 shadow-soft">
                  <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
                    <Settings className="w-4 h-4 text-accent" />
                    INDICATOR OVERLAYS:
                  </div>
                  <div className="flex gap-2">
                    {[
                      { id: 'SMA20', name: 'SMA 20' },
                      { id: 'SMA50', name: 'SMA 50' },
                      { id: 'EMA9',  name: 'EMA 9' },
                      { id: 'BB',    name: 'Bollinger Bands' }
                    ].map(ind => {
                      const sel = taIndicators.includes(ind.id);
                      return (
                        <button
                          key={ind.id}
                          onClick={() => {
                            setTaIndicators(prev => prev.includes(ind.id) ? prev.filter(x => x !== ind.id) : [...prev, ind.id]);
                          }}
                          className={`text-xs font-bold px-4 py-2 rounded-lg transition-all duration-200 ${
                            sel ? 'text-white shadow-glow' : 'bg-[var(--surface-sub)] text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                          }`}
                          style={sel ? { background: 'linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%)' } : {}}
                        >
                          {ind.name}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Main Candlestick */}
                <Panel title={`${ticker} — High Fidelity Candlestick`} subtitle={`${taPeriod} chart interval`}>
                  {taLoading || !taData ? <Sk h="h-[500px]" /> : (
                    <Chart
                      options={{
                        chart: {
                          id: 'candlestick-chart',
                          background: 'transparent',
                          foreColor: 'var(--text-secondary)',
                          toolbar: { 
                            show: true,
                            tools: {
                              download: true,
                              selection: true,
                              zoom: true,
                              zoomin: true,
                              zoomout: true,
                              pan: true,
                              reset: true
                            },
                            autoSelected: 'zoom'
                          },
                          animations: { enabled: false },
                          zoom: {
                            enabled: true,
                            type: 'x',
                            autoScaleYaxis: true
                          },
                          events: {
                            zoomed: function(chartContext, { xaxis }) {
                              setChartZoom({ min: xaxis.min, max: xaxis.max });
                            },
                            beforeResetZoom: function() {
                              setChartZoom({ min: undefined, max: undefined });
                            }
                          }
                        },
                        theme: { mode: theme },
                        grid: { borderColor: 'var(--border)', strokeDashArray: 4 },
                        xaxis: {
                          type: 'datetime',
                          min: chartZoom.min,
                          max: chartZoom.max,
                          axisBorder: { color: 'var(--border)' },
                          axisTicks: { color: 'var(--border)' },
                          labels: { style: { colors: 'var(--text-secondary)', fontFamily: 'Inter', fontSize: '11px' } }
                        },
                        yaxis: {
                          tooltip: {
                            enabled: true
                          },
                          labels: {
                            style: { colors: 'var(--text-secondary)', fontFamily: 'Inter', fontSize: '11px' },
                            formatter: v => `$${v.toFixed(2)}`
                          }
                        },
                        stroke: { width: [1.5, 2, 2, 2, 1.5, 1.5], curve: 'smooth' },
                        colors: ['#2563EB', '#F59E0B', '#00C896', '#06B6D4', '#EF4444', '#10B981'],
                        plotOptions: {
                          candlestick: {
                            colors: { upward: '#00C896', downward: '#EF4444' },
                            wick: {
                              useFillColor: true
                            }
                          }
                        },
                        tooltip: { theme: theme, style: { fontFamily: 'Inter' } }
                      }}
                      series={getApexSeries()}
                      type="line"
                      height={500}
                    />
                  )}
                </Panel>

                {/* Sub oscillators */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Panel title="RSI (14)" subtitle="Relative Strength Index">
                    {taLoading || !taData ? <Sk h="h-44" /> : (
                      <div className="h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={taData.dates.map((d, i) => ({ date: d, rsi: taData.rsi[i] }))} margin={{ left: -25, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                            <XAxis 
                              dataKey="date" 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                              label={{ value: 'Date', position: 'insideBottom', offset: -5, fill: 'var(--text-secondary)', fontSize: 11 }}
                            />
                            <YAxis 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} 
                              domain={[0, 100]}
                              label={{ value: 'RSI Value', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)', fontSize: 11 }}
                            />
                            <Tooltip content={<ChartTip />} />
                            <ReferenceLine y={70} stroke="#EF4444" strokeDasharray="4 4" label={{ value: '70', fill: '#EF4444', fontSize: 11 }} />
                            <ReferenceLine y={30} stroke="#00C896" strokeDasharray="4 4" label={{ value: '30', fill: '#00C896', fontSize: 11 }} />
                            <Line type="monotone" dataKey="rsi" stroke="#06B6D4" strokeWidth={2} dot={false} name="RSI" isAnimationActive={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </Panel>

                  <Panel title="MACD (12, 26, 9)" subtitle="Moving Average Convergence Divergence">
                    {taLoading || !taData ? <Sk h="h-44" /> : (
                      <div className="h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={taData.dates.map((d, i) => ({
                              date: d,
                              macd: taData.macd.macd[i],
                              signal: taData.macd.signal[i]
                            }))}
                            margin={{ left: -25, bottom: 0 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                            <XAxis 
                              dataKey="date" 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                              label={{ value: 'Date', position: 'insideBottom', offset: -5, fill: 'var(--text-secondary)', fontSize: 11 }}
                            />
                            <YAxis 
                              stroke="var(--border)" 
                              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                              label={{ value: 'MACD Value', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)', fontSize: 11 }}
                            />
                            <Tooltip content={<ChartTip />} />
                            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
                            <ReferenceLine y={0} stroke="var(--border)" opacity={0.4} />
                            <Line type="monotone" dataKey="macd" name="MACD" stroke="#2563EB" strokeWidth={2} dot={false} isAnimationActive={false} />
                            <Line type="monotone" dataKey="signal" name="Signal" stroke="#F59E0B" strokeWidth={1.5} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </Panel>
                </div>
              </div>
            )}

            {/* ════════════════════════════════════════════════════════════
                TAB 3 — RISK METRICS
            ════════════════════════════════════════════════════════════ */}
            {activeTab === 'risk' && (
              <div className="space-y-4 animate-fadeIn">
                {/* Risk configs */}
                <div className="bg-[var(--surface)] border border-[var(--border)] rounded-xl p-4 flex flex-wrap gap-4 items-center justify-between shadow-soft">
                  <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
                    <Shield className="w-4 h-4 text-accent" />
                    RISK PARAMETERS:
                  </div>
                  <div className="flex gap-4 items-center">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-[var(--text-muted)]">Confidence Level</span>
                      <select
                        value={riskConf}
                        onChange={e => setRiskConf(Number(e.target.value))}
                        className="input-base"
                      >
                        {[90, 95, 99].map(c => <option key={c} value={c}>{c}%</option>)}
                      </select>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-[var(--text-muted)]">Capital Size ($)</span>
                      <input
                        type="number"
                        value={riskPortVal}
                        onChange={e => setRiskPortVal(Number(e.target.value))}
                        className="input-base w-32"
                      />
                    </div>
                  </div>
                </div>

                {riskLoading || !riskData ? (
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                    {[1,2,3,4,5,6].map(i => <Sk key={i} h="h-20" />)}
                  </div>
                ) : (
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                    <Metric
                      label="Ann. Return"
                      value={`${riskData.kpis.annualizedReturn}%`}
                      bull={riskData.kpis.annualizedReturn >= 0}
                      bear={riskData.kpis.annualizedReturn < 0}
                    />
                    <Metric
                      label="Ann. Volatility"
                      value={`${riskData.kpis.annualizedVolatility}%`}
                    />
                    <Metric
                      label="Sharpe Ratio"
                      value={riskData.kpis.sharpeRatio}
                      accent
                    />
                    <Metric
                      label="Max Drawdown"
                      value={`${riskData.kpis.maxDrawdown}%`}
                      bear
                    />
                    <Metric
                      label="Value at Risk"
                      value={`-${riskData.kpis.varHistorical}%`}
                      bear
                    />
                    <Metric
                      label="Expected Shortfall"
                      value={`-${riskData.kpis.cvar}%`}
                      bear
                    />
                  </div>
                )}

                {/* Drawdowns area */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  <Panel className="lg:col-span-2" title="Historical Equity & Drawdown Curve" subtitle="Displays return levels against historical drawdowns">
                    {riskLoading || !riskData ? <Sk h="h-72" /> : (
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart
                            data={riskData.dates.map((d, i) => ({ date: d, cum: riskData.cumulative[i], dd: riskData.drawdown[i] }))}
                            margin={{ left: -25, bottom: 0 }}
                          >
                            <defs>
                              <linearGradient id="g-cum2" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%"  stopColor="#3B82F6" stopOpacity={0.12} />
                                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                              </linearGradient>
                              <linearGradient id="g-dd2" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%"  stopColor="#EF4444" stopOpacity={0.18} />
                                <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,0.5)" />
                            <XAxis dataKey="date" stroke="#1E293B" tick={{ fill: '#64748B', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                            <YAxis stroke="#1E293B" tick={{ fill: '#64748B', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                            <Tooltip content={<ChartTip />} />
                            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 10, fontFamily: 'JetBrains Mono' }} />
                            <Area type="monotone" dataKey="cum" name="Cum. Return" stroke="#3B82F6" strokeWidth={1.5} fillOpacity={1} fill="url(#g-cum2)" />
                            <Area type="monotone" dataKey="dd" name="Drawdown" stroke="#EF4444" strokeWidth={1.5} fillOpacity={1} fill="url(#g-dd2)" />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </Panel>

                  <Panel title="Dollar Risk Exposure" subtitle="Calculates potential downside in portfolio dollars">
                    {riskLoading || !riskData ? <Sk h="h-72" /> : (
                      <div className="space-y-4">
                        <p className="text-2xs text-secondary leading-relaxed">
                          For a capital size of <span className="text-primary font-bold">${riskPortVal.toLocaleString()}</span>, there is a <span className="text-bear font-bold">{100 - riskConf}%</span> probability of a single-day loss exceeding:
                        </p>
                        <div className="space-y-3">
                          <div className="bg-bear-dim border border-bear/20 rounded p-3.5">
                            <div className="text-[10px] text-muted uppercase font-bold tracking-wider mb-1">
                              Parametric Value-at-Risk
                            </div>
                            <div className="font-mono text-xl font-bold text-bear">
                              -${((riskPortVal * Math.abs(riskData.kpis.varParametric)) / 100).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                            </div>
                          </div>

                          <div className="bg-accent-dim border border-accent/20 rounded p-3.5">
                            <div className="text-[10px] text-muted uppercase font-bold tracking-wider mb-1">
                              Expected Shortfall (CVaR)
                            </div>
                            <div className="font-mono text-xl font-bold text-accent">
                              -${riskData.kpis.cvarDollar.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </Panel>
                </div>
              </div>
            )}

            {/* ════════════════════════════════════════════════════════════
                TAB 4 — PORTFOLIO OPTIMIZATION
            ════════════════════════════════════════════════════════════ */}
            {activeTab === 'portfolio' && (
              <div className="space-y-4 animate-fadeIn">
                {/* Optimizer Actions Panel */}
                <div className="bg-[var(--surface)] border border-[var(--border)] rounded-xl p-4 shadow-soft">
                  <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)] mb-4">
                    <Layers className="w-4 h-4" style={{ color: '#1E40AF' }} />
                    <span className="font-bold">PORTFOLIO OPTIMIZER</span>
                    <span className="text-2xs text-[var(--text-muted)]">Select assets and optimize allocation</span>
                  </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_auto] gap-4 items-start">
                    {/* Stock Selection Buttons */}
                    <div className="flex flex-wrap gap-2">
                      {Object.keys(TICKERS).map(tk => {
                        const isSel = portTickers.includes(tk);
                        return (
                          <button
                            key={tk}
                            onClick={() => {
                              if (isSel) {
                                  if (portTickers.length > 2) setPortTickers(portTickers.filter(x => x !== tk));
                              } else {
                                  setPortTickers([...portTickers, tk]);
                              }
                            }}
                            className={`text-xs font-bold px-3 py-1.5 rounded-lg border-2 transition-all ${
                              isSel ? 'text-white border-transparent shadow-glow' : 'border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--text-secondary)]'
                            }`}
                            style={isSel ? { background: 'linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%)' } : {}}
                          >
                            {tk}
                          </button>
                        );
                      })}
                    </div>

                    {/* Risk-Free Rate Slider */}
                    <div className="flex items-center gap-3 bg-[var(--surface-sub)] px-4 py-2 rounded-lg">
                      <span className="text-xs text-[var(--text-secondary)] font-semibold whitespace-nowrap">Risk-Free Rate: {portRf}%</span>
                      <input
                        type="range" 
                        min="0" 
                        max="8" 
                        step="0.25" 
                        value={portRf}
                        onChange={e => setPortRf(Number(e.target.value))}
                        className="w-32"
                      />
                    </div>

                    {/* Optimize Button */}
                    <button
                      onClick={() => setPortTrigger(prev => prev + 1)}
                      className="btn-primary whitespace-nowrap"
                    >
                      <Zap className="w-4 h-4 text-white" />
                      OPTIMIZE
                    </button>
                  </div>
                </div>

                {portLoading || !portData ? (
                  portTrigger === 0 ? (
                    // Initial state - waiting for user to click OPTIMIZE
                    <div className="min-h-[400px] flex flex-col items-center justify-center gap-6 bg-[var(--surface-sub)]/30 rounded-2xl border-2 border-dashed border-[var(--border)]">
                      <div className="w-20 h-20 rounded-full bg-gradient-accent flex items-center justify-center shadow-glow">
                        <Zap className="w-10 h-10 text-white" />
                      </div>
                      <div className="text-center space-y-3">
                        <p className="text-lg font-bold text-[var(--text-primary)]">Ready to Optimize</p>
                        <p className="text-sm text-[var(--text-secondary)] max-w-lg">
                          Select your preferred stocks above, adjust the risk-free rate if needed, 
                          then click <span className="font-bold text-[var(--text-primary)]">OPTIMIZE</span> to find the best portfolio allocation.
                        </p>
                        <div className="grid grid-cols-3 gap-4 mt-6 max-w-md mx-auto text-xs">
                          <div className="bg-[var(--surface)] p-3 rounded-xl border border-[var(--border)]">
                            <div className="font-bold text-[var(--text-primary)] mb-1">Max Sharpe</div>
                            <div className="text-2xs text-[var(--text-muted)]">Best risk-adjusted returns</div>
                          </div>
                          <div className="bg-[var(--surface)] p-3 rounded-xl border border-[var(--border)]">
                            <div className="font-bold text-[var(--text-primary)] mb-1">Min Variance</div>
                            <div className="text-2xs text-[var(--text-muted)]">Lowest risk portfolio</div>
                          </div>
                          <div className="bg-[var(--surface)] p-3 rounded-xl border border-[var(--border)]">
                            <div className="font-bold text-[var(--text-primary)] mb-1">Frontier</div>
                            <div className="text-2xs text-[var(--text-muted)]">Efficient combinations</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    // Loading state - optimization in progress
                    <div className="min-h-[400px] flex flex-col items-center justify-center gap-4">
                      <div className="relative">
                        <RefreshCw className="w-10 h-10 animate-spin" style={{ color: '#1E40AF' }} />
                        <div className="absolute inset-0 rounded-full border-2 border-[#1E40AF]/20"></div>
                      </div>
                      <div className="text-center space-y-2">
                        <p className="text-base font-bold text-[var(--text-primary)]">Optimizing Portfolio...</p>
                        <p className="text-xs text-[var(--text-secondary)] max-w-md">
                          Computing optimal asset allocation using Modern Portfolio Theory. 
                          This usually takes <span className="font-bold text-[var(--text-primary)]">10-15 seconds</span>.
                        </p>
                        <div className="flex gap-2 justify-center items-center text-2xs text-[var(--text-muted)] mt-3">
                          <span>• Simulating 500 portfolios</span>
                          <span>• Finding Max Sharpe & Min Variance</span>
                          <span>• Computing Efficient Frontier</span>
                        </div>
                        <div className="mt-4 text-xs text-[var(--text-secondary)]">
                          💡 <span className="italic">Hang tight! We're crunching the numbers...</span>
                        </div>
                      </div>
                    </div>
                  )
                ) : (
                  <>
                    {/* Frontier and weights allocation */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                      <Panel className="lg:col-span-2" title="Markowitz Efficient Frontier" subtitle="Volatility vs Annualized Return plot for 10,000 simulated portfolios">
                        <div className="h-72">
                          <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,0.5)" />
                              <XAxis type="number" dataKey="vol" name="Volatility" unit="%" stroke="#1E293B" tick={{ fill: '#64748B', fontSize: 9 }} />
                              <YAxis type="number" dataKey="ret" name="Return" unit="%" stroke="#1E293B" tick={{ fill: '#64748B', fontSize: 9 }} />
                              <Tooltip content={<ChartTip />} />
                              <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                              <Scatter
                                name="Portfolios"
                                data={portData.monteCarlo.volatilities.map((v, i) => ({
                                  vol: Number((v * 100).toFixed(2)),
                                  ret: Number((portData.monteCarlo.returns[i] * 100).toFixed(2)),
                                }))}
                                fill="#3B82F6" opacity={0.1}
                              />
                              <Scatter name="Max Sharpe" data={[{ vol: portData.maxSharpe.volatility, ret: portData.maxSharpe.return }]} fill="#10B981" size={130} />
                              <Scatter name="Min Variance" data={[{ vol: portData.minVariance.volatility, ret: portData.minVariance.return }]} fill="#06B6D4" size={110} />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      </Panel>

                      <Panel
                        title="Optimal Allocation"
                        action={
                          <div className="flex gap-0.5 bg-[#0B0F19] border border-border p-0.5 rounded">
                            {['maxSharpe', 'minVariance'].map(m => (
                              <button
                                key={m}
                                onClick={() => setPortMode(m)}
                                className={`text-3xs font-mono font-bold px-2 py-0.5 rounded transition-all ${
                                  portMode === m ? 'bg-accent text-white' : 'text-muted hover:text-secondary'
                                }`}
                              >
                                {m === 'maxSharpe' ? 'SHARPE' : 'VAR'}
                              </button>
                            ))}
                          </div>
                        }
                      >
                        <div className="h-44">
                          <ResponsiveContainer width="100%" height="100%">
                            <RePieChart>
                              <Pie
                                data={portData.tickers.map((tk, idx) => ({
                                  name: tk,
                                  value: portMode === 'maxSharpe' ? portData.maxSharpe.weights[idx] : portData.minVariance.weights[idx]
                                }))}
                                cx="50%" cy="50%"
                                innerRadius={40} outerRadius={62}
                                paddingAngle={3} dataKey="value"
                              >
                                {portData.tickers.map((_, i) => (
                                  <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                                ))}
                              </Pie>
                              <Tooltip formatter={w => `${(w*100).toFixed(1)}%`} />
                            </RePieChart>
                          </ResponsiveContainer>
                        </div>

                        <div className="space-y-1.5 mt-2.5 max-h-36 overflow-y-auto pr-1">
                          {portData.tickers.map((tk, idx) => {
                            const weight = portMode === 'maxSharpe' ? portData.maxSharpe.weights[idx] : portData.minVariance.weights[idx];
                            if (weight < 0.005) return null;
                            return (
                              <div key={tk} className="flex justify-between items-center text-3xs font-mono">
                                <div className="flex items-center gap-1.5">
                                  <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: CHART_COLORS[idx % CHART_COLORS.length] }} />
                                  <span className="text-secondary font-bold">{tk}</span>
                                </div>
                                <span className="text-primary font-semibold">{(weight * 100).toFixed(1)}%</span>
                              </div>
                            );
                          })}
                        </div>
                      </Panel>
                    </div>

                    {/* Backtest row */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                      <Panel className="lg:col-span-2" title="Historical Portfolio Performance Backtest" subtitle="Optimized portfolios vs benchmarks">
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                              data={portData.backtest.dates.map((d, i) => ({
                                date: d,
                                maxSharpe: portData.backtest.maxSharpe[i],
                                minVar: portData.backtest.minVariance[i],
                                equal: portData.backtest.equalWeight[i],
                                spy: portData.backtest.spy[i]
                              }))}
                              margin={{ left: -25, bottom: 0 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,0.5)" />
                              <XAxis dataKey="date" stroke="#1E293B" tick={{ fill: '#64748B', fontSize: 9 }} />
                              <YAxis stroke="#1E293B" tick={{ fill: '#64748B', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                              <Tooltip content={<ChartTip />} />
                              <Legend wrapperStyle={{ fontSize: 10, paddingTop: 10, fontFamily: 'JetBrains Mono' }} />
                              <Line type="monotone" dataKey="maxSharpe" name="Max Sharpe Portfolio" stroke="#10B981" strokeWidth={1.8} dot={false} />
                              <Line type="monotone" dataKey="minVar" name="Min Variance Portfolio" stroke="#3B82F6" strokeWidth={1.5} dot={false} />
                              <Line type="monotone" dataKey="equal" name="Equal Allocation" stroke="#F59E0B" strokeWidth={1.2} strokeDasharray="4 4" dot={false} />
                              <Line type="monotone" dataKey="spy" name="SPY Index" stroke="#EF4444" strokeWidth={1.2} strokeDasharray="4 4" dot={false} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </Panel>

                      <Panel title="Quantitative Comparisons" subtitle="SLSQP optimized results">
                        <div className="border border-border rounded overflow-hidden text-3xs font-mono">
                          <div className="grid grid-cols-4 bg-[#121826] p-2 text-muted font-bold tracking-wider">
                            <span>Strategy</span>
                            <span>Return</span>
                            <span>Vol</span>
                            <span>Sharpe</span>
                          </div>
                          <div className="grid grid-cols-4 p-2 border-t border-border">
                            <span className="text-bull font-bold">Max Sharpe</span>
                            <span>{portData.maxSharpe.return}%</span>
                            <span>{portData.maxSharpe.volatility}%</span>
                            <span className="text-primary font-bold">{portData.maxSharpe.sharpe}</span>
                          </div>
                          <div className="grid grid-cols-4 p-2 border-t border-border">
                            <span className="text-accent font-bold">Min Var</span>
                            <span>{portData.minVariance.return}%</span>
                            <span>{portData.minVariance.volatility}%</span>
                            <span className="text-primary font-bold">{portData.minVariance.sharpe}</span>
                          </div>
                        </div>
                      </Panel>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* ════════════════════════════════════════════════════════════
                TAB 5 — ML PREDICTIONS
            ════════════════════════════════════════════════════════════ */}
            {activeTab === 'ml' && (
              <div className="space-y-4 animate-fadeIn">
                <div className="flex items-center justify-between gap-4 flex-wrap">
                  <div>
                    <h2 className="text-[15px] font-bold text-primary">Machine Learning Predictions</h2>
                    <p className="text-2xs text-muted mt-0.5">XGBoost prediction regressor for directional trading signals</p>
                  </div>
                  <button
                    onClick={runMlModel}
                    disabled={mlTraining}
                    className="btn-primary"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${mlTraining ? 'animate-spin' : ''}`} />
                    {mlTraining ? 'Training…' : 'Re-train XGBoost'}
                  </button>
                </div>

                {mlTraining || !mlData ? (
                  <div className="h-80 flex flex-col items-center justify-center gap-3">
                    <div className="relative w-8 h-8">
                      <div className="absolute inset-0 border-2 border-border rounded-full" />
                      <div className="absolute inset-0 border-2 border-t-accent border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin" />
                    </div>
                    <div className="text-center font-mono text-3xs">
                      <div className="text-secondary">Fitting XGBoost Strategy Model</div>
                      <div className="text-muted mt-1">Generating features & signals for {ticker}...</div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                      <Panel className="lg:col-span-2" title="XGBoost Strategy returns vs Asset returns" subtitle="Cumulative strategy performance overlay">
                        <div className="h-72">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                              data={mlData.dates.map((d, i) => ({
                                date: d,
                                actual: mlData.actual[i],
                                predicted: mlData.predicted[i]
                              }))}
                              margin={{ left: -25, bottom: 0 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,0.5)" />
                              <XAxis dataKey="date" stroke="#1E293B" tick={{ fill: '#64748B', fontSize: 9 }} />
                              <YAxis stroke="#1E293B" tick={{ fill: '#64748B', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                              <Tooltip content={<ChartTip />} />
                              <Legend wrapperStyle={{ fontSize: 10, paddingTop: 10, fontFamily: 'JetBrains Mono' }} />
                              <Line type="monotone" dataKey="predicted" name="XGBoost Strategy" stroke="#10B981" strokeWidth={1.8} dot={false} />
                              <Line type="monotone" dataKey="actual" name="Buy & Hold (Asset)" stroke="#3B82F6" strokeWidth={1.5} dot={false} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </Panel>

                      {/* Predictor stats list */}
                      <Panel title="Validation Metrics" subtitle="XGBoost Model Classifier Summary">
                        <div className="space-y-0.5 divide-y divide-border">
                          {[
                            { label: 'Model', val: mlData.metrics['Model'], color: 'text-primary font-bold' },
                            { label: 'RMSE', val: mlData.metrics['RMSE'], color: 'text-accent' },
                            { label: 'MAE', val: mlData.metrics['MAE'], color: 'text-warning' },
                            { label: 'Dir. Accuracy', val: `${mlData.metrics['Dir. Acc. (%)']}%`, color: 'text-bull font-bold' },
                            { label: 'R² Metric', val: mlData.metrics['R²'], color: 'text-secondary' }
                          ].map(m => (
                            <div key={m.label} className="flex justify-between items-center py-2.5 text-3xs font-mono">
                              <span className="text-muted uppercase font-bold tracking-wider">{m.label}</span>
                              <span className={m.color}>{m.val}</span>
                            </div>
                          ))}
                        </div>
                      </Panel>
                    </div>

                    {/* Feature rank chart */}
                    <Panel title="Predictor Feature Importance Ranks" subtitle="Ranking relative significance values (F-score)">
                      <div className="h-60">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={mlData.importances} layout="vertical" margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,0.5)" />
                            <XAxis type="number" stroke="#1E293B" tick={{ fill: '#64748B', fontSize: 9 }} />
                            <YAxis dataKey="Feature" type="category" stroke="#1E293B" tick={{ fill: '#94A3B8', fontSize: 9 }} width={90} />
                            <Tooltip content={<ChartTip />} />
                            <Bar dataKey="Importance" fill="#3B82F6" radius={[0, 3, 3, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </Panel>
                  </div>
                )}
              </div>
            )}

          </div>
        </main>

        {/* ─── WATCHLIST (Right Sidebar Panel) ──────────────────────────── */}
        <aside className="w-60 bg-[var(--surface)] border-l border-[var(--border)] p-4 flex flex-col justify-between shrink-0 z-20 overflow-y-auto shadow-card">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold tracking-wider text-[var(--text-primary)] uppercase">MARKET WATCHLIST</h3>
              <span className="text-2xs font-semibold" style={{ color: '#1E40AF' }}>{Object.keys(TICKERS).length} METRICS</span>
            </div>
            
            <div className="space-y-1.5">
              {Object.keys(TICKERS).map(sym => {
                const t = liveTickers.find(x => x.ticker === sym);
                return (
                  <WatchlistRow
                    key={sym}
                    symbol={sym}
                    name={TICKERS[sym]}
                    price={t?.price}
                    pctChange={t?.pctChange}
                    active={ticker === sym}
                    onClick={() => setTicker(sym)}
                  />
                );
              })}
            </div>
          </div>

          {/* Terminal statistics panel */}
          <div className="border-t border-[var(--border)] pt-3 mt-4 space-y-2 select-none">
            <div className="bg-[var(--surface-sub)] border border-[var(--border)] rounded-lg p-3 text-xs text-[var(--text-muted)] space-y-2">
              <div className="flex justify-between">
                <span>Feed Status</span>
                <span className="text-accent font-bold">SIMULATION</span>
              </div>
              <div className="flex justify-between">
                <span>Update Interval</span>
                <span className="text-secondary">3s</span>
              </div>
              <div className="flex justify-between">
                <span>Data Source</span>
                <span className="text-secondary text-[10px]">Yahoo Finance</span>
              </div>
            </div>
            
            {/* Data Disclaimer */}
            <div className="bg-[var(--bg)] border border-[var(--border)] rounded-lg p-3 text-[10px] text-[var(--text-muted)] leading-relaxed">
              <div className="font-bold text-[var(--text-secondary)] mb-1">⚠️ DEMO PROJECT</div>
              <div>Historical market data with simulated price movements. Not for trading decisions.</div>
            </div>
          </div>
        </aside>

      </div>
    </div>
  );
}
