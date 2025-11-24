# PowerShell script to help deploy to public URL
Write-Host "🚀 AI Investment Bot - Public Deployment Helper" -ForegroundColor Green
Write-Host "=" * 50

$username = Read-Host "Enter your GitHub username"
if ([string]::IsNullOrWhiteSpace($username)) {
    Write-Host "❌ Username required!" -ForegroundColor Red
    exit
}

Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Create a new repository on GitHub: https://github.com/new" -ForegroundColor Cyan
Write-Host "   - Name: ai-investment-bot" -ForegroundColor Cyan
Write-Host "   - Make it PUBLIC" -ForegroundColor Cyan
Write-Host "   - DO NOT initialize with README" -ForegroundColor Cyan
Write-Host "`n2. After creating the repo, press Enter to continue..." -ForegroundColor Yellow
Read-Host

Write-Host "`n🔗 Adding GitHub remote..." -ForegroundColor Green
$remoteUrl = "https://github.com/$username/ai-investment-bot.git"
git remote add origin $remoteUrl 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Remote added!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Remote might already exist, continuing..." -ForegroundColor Yellow
    git remote set-url origin $remoteUrl
}

Write-Host "`n📤 Pushing to GitHub..." -ForegroundColor Green
Write-Host "   (You'll need to enter your GitHub username and Personal Access Token)" -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "`n🚀 Now deploy to Streamlit Cloud:" -ForegroundColor Yellow
    Write-Host "   1. Go to: https://share.streamlit.io/" -ForegroundColor Cyan
    Write-Host "   2. Sign in with GitHub" -ForegroundColor Cyan
    Write-Host "   3. Click 'New app'" -ForegroundColor Cyan
    Write-Host "   4. Select: $username/ai-investment-bot" -ForegroundColor Cyan
    Write-Host "   5. Main file: gui/dashboard.py" -ForegroundColor Cyan
    Write-Host "   6. Click 'Deploy'!" -ForegroundColor Cyan
    Write-Host "`n🎉 Your public URL will be: https://YOUR_APP_NAME.streamlit.app" -ForegroundColor Green
} else {
    Write-Host "`n❌ Push failed. Make sure:" -ForegroundColor Red
    Write-Host "   - Repository exists on GitHub" -ForegroundColor Yellow
    Write-Host "   - You have a Personal Access Token (not password)" -ForegroundColor Yellow
    Write-Host "   - Repository is PUBLIC" -ForegroundColor Yellow
}

