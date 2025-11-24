# 🚀 Deploy to Public URL - Quick Steps

Your code is ready! Follow these steps to get a public URL:

## Step 1: Create GitHub Repository (2 minutes)

1. **Go to**: https://github.com/new
2. **Repository name**: `ai-investment-bot`
3. **Make it PUBLIC** (required for free Streamlit Cloud)
4. **DO NOT** initialize with README, .gitignore, or license
5. **Click "Create repository"**

## Step 2: Push to GitHub

**Copy and run these commands** (replace YOUR_USERNAME with your GitHub username):

```bash
git remote add origin https://github.com/YOUR_USERNAME/ai-investment-bot.git
git push -u origin main
```

You'll be asked to login to GitHub - use your GitHub username and a Personal Access Token (not password).

**To create a Personal Access Token:**
- Go to: https://github.com/settings/tokens
- Click "Generate new token (classic)"
- Give it a name like "Deploy Bot"
- Check "repo" permission
- Click "Generate token"
- Copy the token and use it as your password when pushing

## Step 3: Deploy to Streamlit Cloud (3 minutes)

1. **Go to**: https://share.streamlit.io/
2. **Click "Sign in"** → Use your GitHub account
3. **Click "New app"**
4. **Fill in**:
   - **Repository**: `YOUR_USERNAME/ai-investment-bot`
   - **Branch**: `main`
   - **Main file path**: `gui/dashboard.py`
   - **App URL**: Choose a name (e.g., `ai-investment-bot`)
5. **Click "Deploy"** ⚡

## Step 4: Get Your Public URL! 🎉

**Your live dashboard will be at:**
```
https://YOUR_APP_NAME.streamlit.app
```

**Share this link with anyone - it's live 24/7!** 🚀

---

## Quick Commands (After creating GitHub repo):

```bash
# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ai-investment-bot.git

# Push to GitHub
git push -u origin main
```

Then deploy on Streamlit Cloud!

