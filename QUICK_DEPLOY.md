# 🚀 Quick Deployment Guide

## 📋 Prerequisites Checklist
- [ ] GitHub account
- [ ] Vercel account (free - https://vercel.com)
- [ ] Render account (free - https://render.com)
- [ ] Code pushed to GitHub

---

## ⚡ Quick Steps (10 minutes)

### 1️⃣ Push to GitHub (If not done)
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2️⃣ Deploy Backend (Render)
1. **Go to**: https://dashboard.render.com
2. **Click**: New + → Web Service
3. **Connect**: Your GitHub repository `Word-Weaver-Quest`
4. **Configure**:
   - Name: `word-weaver-backend`
   - Root Directory: `backend`
   - Runtime: `Python 3`
   - Build Command: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Plan: `Free`
5. **Environment Variables** (click "Advanced"):
   ```
   GOOGLE_API_KEY = your_new_google_api_key_here
   SUPABASE_URL = your_supabase_url_here
   SUPABASE_KEY = your_supabase_key_here
   ```
   ⚠️ **NEVER commit real API keys to Git!** Get your keys from `.env` file.
6. **Click**: Create Web Service
7. **Wait**: 5-10 minutes for deployment
8. **Copy**: Your backend URL (e.g., `https://word-weaver-backend.onrender.com`)

### 3️⃣ Deploy Frontend (Vercel)
1. **Go to**: https://vercel.com/new
2. **Import**: `Word-Weaver-Quest` repository
3. **Configure**:
   - Framework: `Create React App`
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `build`
4. **Environment Variables**:
   ```
   REACT_APP_API_URL = https://your-backend-url.onrender.com
   ```
   ⚠️ Replace with YOUR actual Render backend URL from step 2
5. **Click**: Deploy
6. **Wait**: 2-3 minutes
7. **Done**: Your app is live! 🎉

---

## ✅ Verify Deployment

### Backend Check
Visit: `https://your-backend.onrender.com/docs`
✓ Should show FastAPI Swagger UI

### Frontend Check
Visit: `https://your-app.vercel.app`
✓ Should load the React app

---

## 🔄 Auto-Deploy (Already Set Up!)
Every time you push to GitHub:
- Frontend → Auto-deploys on Vercel
- Backend → Auto-deploys on Render

---

## ⚠️ Important Notes

### Free Tier Limits
- **Render**: Sleeps after 15 min inactivity (30-60s wake-up on first request)
- **Vercel**: Unlimited deployments, 100GB bandwidth/month

### First Request Slow?
This is normal! Render's free tier "wakes up" the server. Subsequent requests are fast.

---

## 🆘 Troubleshooting

**Backend won't start?**
- Check Render logs for errors
- Verify environment variables are set correctly

**Frontend can't connect?**
- Check `REACT_APP_API_URL` in Vercel settings
- Redeploy frontend after changing env vars

**CORS errors?**
- Backend already configured to allow all origins
- If issues persist, check browser console

---

## 📱 Share Your App!

Once deployed, share:
- **Live App**: https://your-app.vercel.app
- **API Docs**: https://your-backend.onrender.com/docs

---

**Need help?** Check the full guide: `DEPLOYMENT.md`
