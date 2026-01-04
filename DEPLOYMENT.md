# 🚀 Deployment Guide - Word Weaver Quest

## Prerequisites
- GitHub account
- Vercel account (sign up at https://vercel.com)
- Render account (sign up at https://render.com)

---

## 📦 Part 1: Deploy Backend to Render (FREE)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add deployment configuration"
git push origin main
```

### Step 2: Deploy on Render
1. Go to https://render.com and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `Word-Weaver-Quest`
4. Configure the service:
   - **Name**: `word-weaver-backend`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`

### Step 3: Add Environment Variables
In Render dashboard, go to **Environment** tab and add:
- `GOOGLE_API_KEY` = `your_google_api_key_here`
- `SUPABASE_URL` = `your_supabase_url_here`
- `SUPABASE_KEY` = `your_supabase_anon_key_here`

⚠️ **Security Note**: Get these values from your local `.env` file. Never commit real keys to Git!

### Step 4: Get Backend URL
After deployment, copy your backend URL (e.g., `https://word-weaver-backend.onrender.com`)

---

## 🎨 Part 2: Deploy Frontend to Vercel (FREE)

### Step 1: Update API URL in Frontend
Before deploying, you need to update the backend API URL in your React app.

Open `frontend/src/App.js` (or wherever you make API calls) and replace:
- `http://localhost:8000` → `https://your-backend-url.onrender.com`

### Step 2: Deploy on Vercel
1. Go to https://vercel.com and sign in with GitHub
2. Click **"Add New..."** → **"Project"**
3. Import `Word-Weaver-Quest` repository
4. Configure:
   - **Framework Preset**: `Create React App`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`

### Step 3: Add Environment Variables (if needed)
If your frontend needs environment variables:
- Go to **Settings** → **Environment Variables**
- Add: `REACT_APP_API_URL` = `https://your-backend-url.onrender.com`

### Step 4: Deploy
Click **"Deploy"** and wait 2-3 minutes.

---

## ✅ Verification

### Test Backend
Visit: `https://your-backend-url.onrender.com/docs`
You should see the FastAPI Swagger documentation.

### Test Frontend
Visit: `https://your-app.vercel.app`
Your React app should load and connect to the backend.

---

## 🔧 Important Notes

### Render Free Tier Limitations
- ⚠️ **Spins down after 15 minutes of inactivity**
- First request after inactivity takes 30-60 seconds (cold start)
- 750 hours/month free (enough for personal projects)

### Solutions for Cold Starts
1. Use a free uptime monitor (e.g., UptimeRobot) to ping every 14 minutes
2. Upgrade to paid tier ($7/month) for always-on service

### CORS Configuration
Make sure your backend allows requests from Vercel domain. Check `backend/main.py` for CORS settings.

---

## 🆘 Troubleshooting

### Backend won't start
- Check Render logs for errors
- Verify all environment variables are set
- Ensure `requirements.txt` has all dependencies

### Frontend can't connect to backend
- Check API URL is correct (no trailing slash)
- Verify CORS is configured properly
- Check browser console for errors

### Database connection issues
- Verify Supabase credentials are correct
- Check Supabase project is active

---

## 📱 Auto-Deployment

Both platforms support automatic deployment:
- Push to GitHub → Automatically deploys to Vercel & Render
- No manual steps needed after initial setup

---

## 🎉 Your App is Live!

**Frontend**: https://your-app.vercel.app  
**Backend API**: https://your-backend.onrender.com

Share your app with the world! 🌍
