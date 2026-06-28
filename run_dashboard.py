import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"

def check_npm_install():
    """Verify if node_modules exist, otherwise run npm install."""
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("[SETUP] Frontend node_modules not found. Installing npm dependencies ...")
        # Run npm install with legacy peer deps to support React 19 libraries cleanly
        try:
            subprocess.run("npm install --legacy-peer-deps", shell=True, cwd=str(FRONTEND_DIR), check=True)
            print("  [OK]  Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error installing npm packages: {e}")
            sys.exit(1)
    else:
        print("[SETUP] Frontend dependencies verified.")

def run():
    check_npm_install()
    
    processes = []
    try:
        # 1. Start FastAPI Backend on port 8000
        print("[LAUNCH] Starting FastAPI backend on http://localhost:8000 ...")
        backend_cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000", "--host", "127.0.0.1"]
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(backend_proc)
        
        # 2. Start Vite Frontend on port 5173
        print("[LAUNCH] Starting Vite React dev server on http://localhost:5173 ...")
        frontend_proc = subprocess.Popen(
            "npm run dev",
            shell=True,
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(frontend_proc)
        
        # Give servers a few seconds to warm up
        print("[INFO] Waiting for servers to initialize ...")
        time.sleep(3.5)
        
        # 3. Open Web Browser
        print("[LAUNCH] Opening dashboard at http://localhost:5173/ ...")
        webbrowser.open("http://localhost:5173/")
        
        # 4. Stream outputs & monitor
        print("\n============================================================")
        print("  [*] Dashboard is running! Press Ctrl+C to terminate.")
        print("============================================================\n")
        
        while True:
            # Simple keep alive monitor
            for p in processes:
                if p.poll() is not None:
                    print(f"[WARNING] Server process exited prematurely with code {p.returncode}")
                    return
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[STOP] Stopping servers ...")
    finally:
        for p in processes:
            try:
                # Terminate subprocesses cleanly
                if os.name == 'nt':
                    # Windows taskkill for subprocess trees (especially npm shell wrapper)
                    subprocess.run(f"taskkill /F /T /PID {p.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    p.terminate()
            except Exception:
                pass
        print("[STOP] Servers stopped. Goodbye!")

if __name__ == "__main__":
    run()
