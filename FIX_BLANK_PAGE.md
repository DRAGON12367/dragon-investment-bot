# 🔧 Fix Blank Page Issue

## The Problem
Your website is showing a blank page. This is usually caused by:
1. Wrong main file path in Streamlit Cloud
2. Import errors
3. Missing dependencies
4. Runtime errors

## ✅ Solution

### Step 1: Update Streamlit Cloud Settings

1. Go to: https://share.streamlit.io/
2. Find your app: `ai-investment-bot`
3. Click the **"⋮" (three dots)** → **"Settings"**
4. Change **"Main file path"** to:
   ```
   streamlit_app.py
   ```
   (Instead of `gui/dashboard.py`)

5. Click **"Save"**
6. The app will automatically redeploy

### Step 2: Check Logs

If still blank:
1. In Streamlit Cloud, click **"Manage app"**
2. Click **"Logs"** tab
3. Look for error messages
4. Share the errors if you need help

### Step 3: Alternative - Use Direct Path

If `streamlit_app.py` doesn't work, try:
- **Main file path**: `gui/dashboard.py`
- Make sure it's exactly: `gui/dashboard.py` (not `./gui/dashboard.py`)

## 🚀 Quick Fix Commands

If you need to update the code:
```bash
git add streamlit_app.py
git commit -m "Add Streamlit Cloud entry point"
git push dragon main
```

## 📋 Checklist

- [ ] Main file path is `streamlit_app.py` OR `gui/dashboard.py`
- [ ] Branch is `main`
- [ ] Repository is `DRAGON12367/dragon-investment-bot`
- [ ] Check logs for errors
- [ ] Wait 2-3 minutes for redeploy

## 💡 Most Common Fix

**Change main file path to**: `streamlit_app.py`

This is the new entry point I created that handles errors better.

