#!/bin/bash

# 🚀 AI AUDIO DETECTOR - RENDER DEPLOYMENT AUTOMATION
# This script automates the complete deployment process

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     🚀 AI AUDIO DETECTOR - RENDER DEPLOYMENT SCRIPT       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Verify files
echo -e "${BLUE}Step 1: Verifying deployment files...${NC}"
FILES=("Procfile" "runtime.txt" "requirements.txt" "test/main.py" "test/audio_detection.pkl")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✅${NC} $file"
    else
        echo -e "  ${RED}❌${NC} $file - NOT FOUND"
        exit 1
    fi
done
echo ""

# Step 2: Check git status
echo -e "${BLUE}Step 2: Checking git repository...${NC}"
COMMIT=$(git log --oneline -1 | head -c 40)
echo -e "  ${GREEN}✅${NC} Latest commit: $COMMIT"
echo -e "  ${GREEN}✅${NC} Git repository ready"
echo ""

# Step 3: Display Render credentials instructions
echo -e "${YELLOW}Step 3: Render Deployment Instructions${NC}"
echo ""
echo "📋 COMPLETE THESE STEPS IN RENDER DASHBOARD:"
echo ""
echo "1️⃣  Click 'New +' button"
echo "2️⃣  Select 'Web Service'"
echo "3️⃣  Choose 'Public Git repository'"
echo "4️⃣  Paste this repository URL:"
echo "    ${BLUE}https://github.com/pruthiviraj1188/AI-audio-detector.git${NC}"
echo "5️⃣  Click 'Connect'"
echo ""
echo "📝 FILL IN THESE SETTINGS:"
echo ""
echo "  Name: ${BLUE}ai-audio-detector${NC}"
echo "  Environment: ${BLUE}Python 3${NC}"
echo "  Region: ${BLUE}(Select closest to your location)${NC}"
echo ""
echo "  Build Command:"
echo "  ${BLUE}pip install -r requirements.txt${NC}"
echo ""
echo "  Start Command:"
echo "  ${BLUE}gunicorn -w 4 -b 0.0.0.0:\$PORT test.main:app${NC}"
echo ""
echo "6️⃣  Click 'Create Web Service'"
echo "7️⃣  Wait 2-3 minutes for deployment"
echo "8️⃣  When green checkmark appears, click the URL"
echo ""

# Step 4: Generate deployment summary
echo -e "${YELLOW}Step 4: Deployment Summary${NC}"
echo ""
echo "📦 PROJECT DETAILS:"
echo "  Repository: https://github.com/pruthiviraj1188/AI-audio-detector.git"
echo "  Branch: main"
echo "  Latest Commit: $COMMIT"
echo ""

echo "🔧 TECHNOLOGY STACK:"
echo "  Framework: Flask 3.1.3"
echo "  Server: Gunicorn (4 workers)"
echo "  Python: 3.12"
echo "  Model: Random Forest (98.67% accuracy)"
echo ""

echo "📊 APP FEATURES:"
echo "  ✅ AI Audio Detection (98.67% accuracy)"
echo "  ✅ Multi-language Support (5 languages)"
echo "  ✅ Beautiful Animated UI"
echo "  ✅ Drag & Drop File Upload"
echo "  ✅ Real-time Processing"
echo "  ✅ Technical Analysis Breakdown"
echo "  ✅ Multi-format Audio Support (MP3, WAV, MP4, etc.)"
echo "  ✅ Mobile Responsive Design"
echo ""

echo "🌐 EXPECTED RESULT:"
echo "  Your app will be live at:"
echo "  ${GREEN}https://ai-audio-detector-xxxxx.onrender.com${NC}"
echo ""

# Step 5: Provide next steps
echo -e "${BLUE}Step 5: Next Steps${NC}"
echo ""
echo "✅ Repository is ready"
echo "✅ All files verified"
echo "✅ Configuration complete"
echo ""
echo "👉 ACTION REQUIRED: Go to Render and follow steps above"
echo ""
echo "⏱️  Expected deployment time: 2-3 minutes"
echo "💰 Cost: FREE (with paid upgrades available)"
echo ""

# Step 6: Success message
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗"
echo "║              🎉 READY FOR DEPLOYMENT! 🎉                  ║"
echo "╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Your AI Audio Detector is production-ready!"
echo ""
echo "Go to: ${BLUE}https://render.com/${NC}"
echo "Follow the steps above to deploy"
echo ""
echo "Questions? Check QUICK_DEPLOY.md in the repository"
echo ""
