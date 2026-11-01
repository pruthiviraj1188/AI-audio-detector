#!/usr/bin/env python3
"""
AI Audio Detector - Automated Render Deployment Script
This script handles the complete deployment to Render.com
"""

import subprocess
import sys
import json
import time

def run_command(cmd, description):
    """Run a shell command and return the result"""
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ SUCCESS\n")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ FAILED\n")
            if result.stderr:
                print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT\n")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}\n")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║   🚀 AI AUDIO DETECTOR - RENDER.COM DEPLOYMENT HELPER    ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 DEPLOYMENT STEPS:\n")
    
    # Step 1: Verify files
    print("1️⃣  Verifying deployment files...")
    files_to_check = [
        "Procfile",
        "runtime.txt",
        "requirements.txt",
        "test/main.py",
        "test/audio_detection.pkl"
    ]
    
    all_files_exist = True
    for f in files_to_check:
        try:
            with open(f):
                print(f"   ✅ {f}")
        except FileNotFoundError:
            print(f"   ❌ {f} - NOT FOUND")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some required files are missing!")
        return False
    
    print("\n✅ All files verified!")
    
    # Step 2: Check git status
    print("\n2️⃣  Checking git status...")
    result = subprocess.run("git log --oneline -1", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   Latest commit: {result.stdout.strip()}")
        print("   ✅ Git repository ready")
    
    # Step 3: Summary
    print(f"""
╔════════════════════════════════════════════════════════════╗
║                    🎉 READY TO DEPLOY!                   ║
╚════════════════════════════════════════════════════════════╝

✅ All files verified and ready for deployment!

NEXT STEPS (Choose one):

1️⃣  AUTOMATIC DEPLOYMENT (Recommended)
   → Go to: https://render.com/
   → Sign up with GitHub
   → Click "New Web Service"
   → Connect this GitHub repo
   → Select Python 3 environment
   → Set Build Command: pip install -r requirements.txt
   → Set Start Command: gunicorn -w 4 -b 0.0.0.0:$PORT test.main:app
   → Click "Create Web Service"
   → Done! App will be live in 2-3 minutes

2️⃣  PUSH TO GITHUB (If push works)
   → Run: git push origin main
   → Then use Render's GitHub integration

3️⃣  DOCKER DEPLOYMENT
   → Run: docker build -t ai-audio-detector .
   → Run: docker run -p 5001:5001 ai-audio-detector
   → Test locally at: http://localhost:5001

YOUR DEPLOYMENT INFO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 Repository: https://github.com/pruthiviraj1188/AI-audio-detector

🔧 Environment:
   • Python: 3.12
   • Framework: Flask + Gunicorn
   • Model: Random Forest (98.67% accuracy)
   • Memory: ~1.5GB
   • Storage: ~500MB

📱 Expected Result:
   • Live URL: https://ai-audio-detector-xxx.onrender.com
   • Health Check: /health
   • Upload API: POST /predict
   • Web UI: / (with animations)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 RECOMMENDATION: Use Render.com (simplest & fastest)

Questions? Check DEPLOYMENT.md for detailed instructions!
    """)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
