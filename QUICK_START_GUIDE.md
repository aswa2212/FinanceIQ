# 🚀 Quick Start Guide - FinanceIQ Dashboard

## Getting Your Dashboard Running in 3 Steps

---

## Step 1: Setup Environment (One-Time)

### Open PowerShell in Project Folder

```powershell
# Activate virtual environment
venv\Scripts\activate

# Install dependencies (takes 3-5 minutes)
pip install -r requirements.txt
```

**Wait for installation to complete.** You should see:
```
Successfully installed numpy-1.26.4 pandas-2.2.2 ... [many packages]
```

---

## Step 2: Start Dashboard

### Single Command Launch

```powershell
python run_dashboard.py
```

**What happens:**
1. ✅ Checks npm dependencies (auto-installs if needed)
2. ✅ Starts FastAPI backend on http://localhost:8000
3. ✅ Starts React frontend on http://localhost:5173
4. ✅ Opens browser automatically

**Console output should show:**
```
[SETUP] Frontend dependencies verified.
[LAUNCH] Starting FastAPI backend on http://localhost:8000 ...
[LAUNCH] Starting Vite React dev server on http://localhost:5173 ...
[INFO] Waiting for servers to initialize ...
[LAUNCH] Opening dashboard at http://localhost:5173/ ...

============================================================
  [*] Dashboard is running! Press Ctrl+C to terminate.
============================================================
```

---

## Step 3: Explore Dashboard

### You'll see 5 main tabs:

| Tab | What You Can Do |
|-----|-----------------|
| **📊 Overview** | View cumulative returns, correlation matrix, sector performance |
| **📈 Technical** | See candlestick charts with indicators (RSI, MACD, Bollinger Bands) |
| **⚠️ Risk** | Check VaR gauges, drawdown charts, rolling volatility |
| **💼 Portfolio** | Optimize allocations, view Efficient Frontier, backtest strategies |
| **🤖 Predict** | Train ML models, see predictions, analyze feature importance |

---

## 📄 Downloading PDF Reports

### How to Generate a Report:

1. **Look in the top-right corner** of the dashboard
2. **Click the green "Download Report" button**
3. **Read the popup alert** (explains the 60-90 second wait)
4. **Wait patiently** - new tab will show loading spinner
5. **PDF downloads automatically** when ready

### What Happens During Generation:

```
⏱️ 0-15s:  Loading cached market data
⏱️ 15-25s: Calculating risk metrics (VaR, CVaR, Sharpe)
⏱️ 25-45s: Running portfolio optimization (500 simulations)
⏱️ 45-85s: Training XGBoost model & generating predictions
⏱️ 85-90s: Creating charts and finalizing PDF
✅ 90s:    Download starts automatically!
```

### Backend Console Shows Progress:

```
============================================================
🚀 PDF REPORT GENERATION STARTED
============================================================
📊 Tickers: AAPL, MSFT, NVDA, JPM
⏱️  Estimated time: 60-90 seconds
============================================================

📄 Creating title page...
📈 Generating market overview charts...
⚠️  Computing risk analytics...
💼 Running portfolio optimization...
🤖 Training ML model and generating predictions...
✅ Finalizing PDF...

============================================================
✅ PDF REPORT GENERATION COMPLETE!
📦 Size: 847.3 KB
============================================================
```

---

## ⚡ Troubleshooting

### Port 8000 Already in Use

**Error:** `Address already in use: 0.0.0.0:8000`

**Fix:**
```powershell
# Find the process
netstat -ano | findstr :8000

# Kill it (replace 12345 with actual PID)
taskkill /F /PID 12345

# Then restart dashboard
python run_dashboard.py
```

---

### Missing Dependencies Error

**Error:** `ModuleNotFoundError: No module named 'numpy'`

**Fix:**
```powershell
# Make sure venv is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

### PDF Takes Too Long (Over 2 Minutes)

**Possible Reasons:**
1. **First-time generation** - downloading fresh data from Yahoo Finance
2. **Slow CPU** - ML training is CPU-intensive
3. **Network issues** - if cache is empty, needs to download data

**Solutions:**
- ✅ Let it finish once - subsequent PDFs will be faster
- ✅ Check backend console for error messages
- ✅ Pre-populate cache by running: `python generate_reports.py`

---

### Frontend Won't Load

**Error:** Blank page or "Cannot connect to server"

**Fix:**
1. **Check backend is running** - should see FastAPI logs in console
2. **Check port 8000** - backend must be accessible
3. **Clear browser cache** - Ctrl+F5 to hard refresh
4. **Check firewall** - allow localhost connections

---

## 🎯 Expected Dashboard Appearance

### Header:
```
┌─────────────────────────────────────────────────────────┐
│ 🔵 FinanceIQ     [Overview][Technical][Risk][Port.][ML] │
│                            🟢 LIVE  [📥 Download Report] │
└─────────────────────────────────────────────────────────┘
```

### Live Ticker Ribbon:
```
🟢 LIVE MARKET  |  AAPL $185.32 ↑ +1.24%  |  MSFT $401.45 ↓ -0.52%  ...
```

### Main Workspace:
- Left side: Charts and analytics
- Right side: Watchlist with clickable stocks

---

## 🛑 Stopping the Dashboard

**To stop all servers:**
```
Press Ctrl+C in the terminal where run_dashboard.py is running
```

**Console will show:**
```
[STOP] Stopping servers ...
[STOP] Servers stopped. Goodbye!
```

---

## 📚 What's Next?

### Option A: Explore Jupyter Notebook
```powershell
jupyter notebook notebooks/finance_analysis.ipynb
```

### Option B: Generate Static Reports
```powershell
python generate_reports.py
```
Creates 15-20 PNG charts in `reports/figures/`

### Option C: Customize the Dashboard
- Edit `frontend/src/App.jsx` for UI changes
- Edit `backend/main.py` to add new API endpoints
- Edit `src/*.py` modules to change analytics logic

---

## 📞 Need Help?

**Check these files:**
- `README.md` - Full project documentation
- `PDF_REPORT_INFO.md` - PDF generation technical details
- `PROJECT_REPORT.md` - Project methodology and findings

**Common Issues:**
- Port conflicts → Kill process on port 8000/5173
- Missing packages → Reinstall requirements.txt
- Slow PDF → Normal on first run, faster after cache builds

---

*Last updated: 2025-01-28*
