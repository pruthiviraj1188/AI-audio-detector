# 🚀 AI Audio Detector - Deployment Guide

Complete guide to deploy AI Audio Detector to the cloud.

## ⚡ Quick Deploy (5 Minutes)

### Railway.app (Recommended)

1. **Push code to GitHub**
```bash
git add .
git commit -m "Deployment ready"
git push origin main
```

2. **Go to https://railway.app/**
   - Sign up with GitHub
   - Click "New Project"
   - Select "Deploy from GitHub"
   - Choose this repository
   - Railway auto-deploys!

3. **Get your URL**
   - Railway provides: `https://your-app.railway.app`
   - Your app is live! 🎉

---

## 🌍 Deployment Options

| Platform | Free Tier | Setup Time | Performance | Scaling |
|----------|-----------|-----------|-------------|---------|
| **Railway** | 500GB-hrs/mo | 2 min | ⭐⭐⭐⭐⭐ | Auto |
| **Render** | 750hrs/mo | 2 min | ⭐⭐⭐⭐ | Auto |
| **Heroku** | Limited | 5 min | ⭐⭐⭐ | Auto |
| **Vercel** | Yes (serverless) | 3 min | ⭐⭐⭐⭐ | Auto |
| **AWS** | No | 30 min | ⭐⭐⭐⭐⭐ | Manual |

---

## 📋 Pre-Deployment Checklist

- [x] `requirements.txt` - All dependencies listed
- [x] `Procfile` - Server startup command
- [x] `runtime.txt` - Python 3.12
- [x] `main.py` - Production settings
- [x] `audio_detection.pkl` - Model file included
- [x] `.gitignore` - Secrets excluded

---

## 🚀 Option 1: Railway (Best Choice)

### Step 1: Prepare Code
```bash
cd /Users/pruthiviraj/AI-audio-detector
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Step 2: Create Railway Project
1. Go to https://railway.app/
2. Sign up (connect GitHub)
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose `AI-audio-detector`

### Step 3: Configure
Railway automatically detects:
- Python project
- Installs from `requirements.txt`
- Uses `Procfile` for startup
- Sets `PORT` environment variable

### Step 4: Deploy
- Railway auto-deploys on GitHub push
- Get URL like: `https://ai-audio-detector-production.up.railway.app`
- App is live!

### Step 5: Custom Domain (Optional)
1. Go to Project Settings
2. Click "Add Custom Domain"
3. Enter your domain (e.g., audiodetector.com)
4. Railway provides DNS instructions
5. Update your domain registrar DNS

---

## 🚀 Option 2: Render.com

### Step 1: Connect GitHub
1. Go to https://render.com/
2. Sign up with GitHub
3. Authorize Render to access repos

### Step 2: Create Web Service
1. Click "New +"
2. Select "Web Service"
3. Connect GitHub repo: `AI-audio-detector`
4. Name: `ai-audio-detector`

### Step 3: Configure
- **Environment:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn -w 4 -b 0.0.0.0:$PORT test.main:app`
- **Branch:** main

### Step 4: Deploy
1. Click "Create Web Service"
2. Render deploys automatically
3. Get URL: `https://ai-audio-detector.onrender.com`

---

## 🚀 Option 3: Heroku

### Step 1: Install Heroku CLI
```bash
brew tap heroku/brew && brew install heroku
```

### Step 2: Login
```bash
heroku login
# Opens browser for authentication
```

### Step 3: Create App
```bash
cd /Users/pruthiviraj/AI-audio-detector
heroku create your-app-name
# Creates app and adds remote
```

### Step 4: Deploy
```bash
git push heroku main
# Deploys code to Heroku
```

### Step 5: View App
```bash
heroku open
# Opens your app in browser
```

### Step 6: Monitor
```bash
heroku logs --tail
# View real-time logs
```

---

## 🐳 Docker Deployment

### Step 1: Create Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "test/main.py"]
```

### Step 2: Create .dockerignore
```
.git
.gitignore
__pycache__
*.pyc
.venv
.DS_Store
*.log
```

### Step 3: Build & Run Locally
```bash
docker build -t ai-audio-detector .
docker run -p 5001:5001 ai-audio-detector
```

### Step 4: Push to Docker Hub
```bash
docker login
docker tag ai-audio-detector:latest USERNAME/ai-audio-detector:latest
docker push USERNAME/ai-audio-detector:latest
```

### Step 5: Deploy on Railway/Render with Docker
- Both support deploying from Docker images
- Much faster deployment

---

## 🌐 Setup Custom Domain

### After Deployment on Railway/Render

#### 1. Update DNS (GoDaddy, Namecheap, etc.)
```
Type: CNAME
Name: www
Value: your-app.railway.app
```

Or for Render:
```
Type: CNAME
Name: www
Value: your-app.onrender.com
```

#### 2. Wait for DNS Propagation
- Usually 5-30 minutes
- Check: https://mxtoolbox.com/

#### 3. Verify
```bash
curl https://yourdomain.com/health
# Should return JSON with status: "healthy"
```

---

## 🔒 Environment Variables

Set these in your deployment platform:

```
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=your-secret-key
MAX_FILE_SIZE=52428800
```

### Railway
1. Go to Project
2. Click "Variables"
3. Add each variable
4. Auto-redeploys

### Heroku
```bash
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=your-secret-key
```

### Render
1. Go to Environment
2. Add variables
3. Save and redeploy

---

## 📊 Monitoring

### Railway Logs
```bash
railway logs
```

### Heroku Logs
```bash
heroku logs --tail
```

### Render Logs
- Dashboard → Logs tab

### Check Health Endpoint
```bash
curl https://your-app-url/health

# Response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "version": "1.0.0"
# }
```

---

## 🐛 Troubleshooting

### App Won't Start
```bash
# Check logs
railway logs  # or heroku logs --tail

# Common issues:
# 1. Model file missing: git add test/audio_detection.pkl
# 2. Dependencies missing: Update requirements.txt
# 3. Port not set: Already configured in main.py
```

### Model File Not Found
```bash
# Add model to git
git add test/audio_detection.pkl
git commit -m "Add model file"
git push origin main

# Redeploy
# (Platform auto-redeploys on push)
```

### File Upload Fails
```python
# In main.py, increase max file size
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB
```

### Slow Performance
1. Check CPU/Memory usage in dashboard
2. Upgrade to paid tier if needed
3. Add more instances (auto-scaling)

---

## 💰 Cost Estimates

### Free Usage
- **Railway:** 500 GB-hours/month
- **Render:** 750 hours/month
- **Heroku:** Very limited

### After Free Tier
- **Railway:** ~$10-30/month for continuous usage
- **Render:** ~$7-20/month
- **Heroku:** ~$15-50/month

### Cost Optimization
1. Use Railway or Render (best free tier)
2. Scale down during low traffic
3. Use auto-scaling features
4. Monitor resource usage

---

## 🔄 Continuous Deployment

All platforms auto-deploy on git push:

```bash
# Make changes locally
git add .
git commit -m "Feature: improve detection accuracy"
git push origin main

# Platform automatically:
# 1. Pulls latest code
# 2. Installs dependencies
# 3. Runs build process
# 4. Deploys new version
# Time: 1-5 minutes
```

---

## ✅ Verification Checklist

After deployment:

- [ ] App loads: `https://your-url`
- [ ] Health check works: `https://your-url/health`
- [ ] Can upload audio file
- [ ] Gets AI/Human prediction
- [ ] Confidence shows correctly
- [ ] Technical details display
- [ ] Multi-language works (language selector)
- [ ] Responsive on mobile

---

## 🆘 Need Help?

### Railway Support
- Docs: https://docs.railway.app/
- Discord: https://discord.gg/railway

### Render Support
- Docs: https://render.com/docs
- Chat: https://render.com/support

### Heroku Support
- Docs: https://devcenter.heroku.com/
- Forum: https://help.heroku.com/

---

**Your app will be live globally in minutes!** 🌍🚀

