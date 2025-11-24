# 🚀 Deploy Your AI Investment Bot to Streamlit Cloud (24/7 Live)

This guide will help you deploy your AI Investment Bot dashboard to Streamlit Cloud so it's accessible 24/7 to anyone with the link.

## Option 1: Streamlit Cloud (Recommended - FREE)

### Step 1: Push to GitHub

1. **Create a GitHub account** (if you don't have one): https://github.com

2. **Create a new repository**:
   - Go to https://github.com/new
   - Name it: `ai-investment-bot`
   - Make it **Public** (required for free Streamlit Cloud)
   - Click "Create repository"

3. **Push your code to GitHub**:
   ```bash
   # In your project directory
   git init
   git add .
   git commit -m "Initial commit - AI Investment Bot"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ai-investment-bot.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io/

2. **Sign in with GitHub** (use your GitHub account)

3. **Click "New app"**

4. **Fill in the details**:
   - **Repository**: Select `YOUR_USERNAME/ai-investment-bot`
   - **Branch**: `main`
   - **Main file path**: `gui/dashboard.py`
   - **App URL**: Choose a custom name (e.g., `ai-investment-bot`)

5. **Click "Deploy"**

6. **Wait 2-3 minutes** for deployment

7. **Your app will be live at**: `https://YOUR_APP_NAME.streamlit.app`

### Step 3: Share Your Link

Once deployed, share this link with anyone:
```
https://YOUR_APP_NAME.streamlit.app
```

The app will:
- ✅ Run 24/7 automatically
- ✅ Auto-update when you push to GitHub
- ✅ Be accessible to anyone with the link
- ✅ Free for public repositories

---

## Option 2: Railway (Alternative - FREE tier available)

1. **Go to**: https://railway.app
2. **Sign up with GitHub**
3. **New Project** → **Deploy from GitHub repo**
4. **Select your repository**
5. **Add a Start Command**:
   ```
   streamlit run gui/dashboard.py --server.port $PORT
   ```
6. **Deploy** - Railway will give you a public URL

---

## Option 3: Render (Alternative - FREE tier)

1. **Go to**: https://render.com
2. **Sign up with GitHub**
3. **New** → **Web Service**
4. **Connect your repository**
5. **Settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run gui/dashboard.py --server.port $PORT --server.address 0.0.0.0`
6. **Deploy** - Render will give you a public URL

---

## Important Notes

### For Streamlit Cloud:
- ✅ **Free** for public repositories
- ✅ **Automatic deployments** on git push
- ✅ **24/7 uptime**
- ✅ **No credit card required**

### Requirements:
- Your repository must be **public** (for free tier)
- Make sure `requirements.txt` is up to date
- The main file should be `gui/dashboard.py`

### Troubleshooting:

If deployment fails:
1. Check that all dependencies are in `requirements.txt`
2. Ensure `gui/dashboard.py` exists and is the main entry point
3. Check Streamlit Cloud logs for errors

### Updating Your App:

Simply push changes to GitHub:
```bash
git add .
git commit -m "Update dashboard"
git push
```

Streamlit Cloud will automatically redeploy!

---

## Quick Start Commands

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Deploy AI Investment Bot"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ai-investment-bot.git

# Push to GitHub
git push -u origin main
```

Then follow Step 2 above to deploy to Streamlit Cloud!

---

## Your Live URL Will Be:

Once deployed, your dashboard will be accessible at:
```
https://YOUR_APP_NAME.streamlit.app
```

Share this link with anyone - they can view your AI Investment Bot dashboard 24/7! 🚀

