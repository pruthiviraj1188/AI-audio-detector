# ⚡ FASTEST DEPLOYMENT - 60 SECONDS

## 🚀 Step-by-Step (Copy-Paste Ready)

### Step 1: Create Render Account (30 seconds)
```
1. Open: https://render.com/
2. Click "Sign up"
3. Choose "Continue with GitHub"
4. Click "Authorize render"
5. Done! Account created
```

### Step 2: Create Web Service (20 seconds)
```
1. Click "New +"
2. Click "Web Service"
3. Click "Public Git repository"
4. Paste this URL:
   https://github.com/pruthiviraj1188/AI-audio-detector.git
5. Click "Connect"
```

### Step 3: Configure (10 seconds)
```
Fill in these fields:

Name: ai-audio-detector
Environment: Python 3
Region: (pick closest to you)

Build Command:
pip install -r requirements.txt

Start Command:
gunicorn -w 4 -b 0.0.0.0:$PORT test.main:app
```

### Step 4: Deploy (Click & Wait)
```
1. Click "Create Web Service"
2. Wait 2-3 minutes for deployment
3. Green checkmark = LIVE! ✅
```

---

## ✅ YOUR LIVE URL

After deployment completes, you'll see:
```
https://ai-audio-detector-xxxxx.onrender.com
```

Click it and your app is LIVE! 🎉

---

## 📱 Test Your App

```
1. Visit: https://your-app-url/
2. Upload any audio file (MP3, WAV, MP4, etc.)
3. Click "Analyze Audio"
4. Get instant AI/Human detection! 🎵
```

---

## 🎯 DEPLOYMENT CHECKLIST

- [x] Code ready ✅
- [x] Model included ✅
- [x] Configuration done ✅
- [x] All files verified ✅
- [ ] Go to render.com
- [ ] Sign up with GitHub
- [ ] Create Web Service
- [ ] Connect GitHub repo
- [ ] Enter build/start commands
- [ ] Click Deploy
- [ ] Wait 2-3 minutes
- [ ] ✅ LIVE!

---

## 💡 IF YOU GET STUCK

**"Build failed"**
→ Check that requirements.txt exists

**"App won't start"**
→ Check Start Command is exactly:
`gunicorn -w 4 -b 0.0.0.0:$PORT test.main:app`

**"File upload doesn't work"**
→ Check that model file (audio_detection.pkl) is included

---

## 🎉 SUCCESS SIGNS

✅ Green checkmark on Render  
✅ Page loads at your URL  
✅ Can upload audio files  
✅ Gets AI/Human predictions  
✅ All features working  

---

**That's it! Your app is now LIVE globally! 🌍🚀**

For custom domain, see: DEPLOYMENT.md
