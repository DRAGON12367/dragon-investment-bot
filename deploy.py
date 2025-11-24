"""
Quick deployment preparation script.
This helps prepare your project for deployment to Streamlit Cloud.
"""
import os
import subprocess
import sys

def check_git():
    """Check if git is initialized."""
    if not os.path.exists('.git'):
        print("📦 Initializing git repository...")
        subprocess.run(['git', 'init'], check=True)
        print("✅ Git initialized!")
    else:
        print("✅ Git already initialized")

def check_requirements():
    """Check if requirements.txt exists."""
    if os.path.exists('requirements.txt'):
        print("✅ requirements.txt found")
        return True
    else:
        print("❌ requirements.txt not found!")
        return False

def check_dashboard():
    """Check if dashboard.py exists."""
    if os.path.exists('gui/dashboard.py'):
        print("✅ Dashboard file found at gui/dashboard.py")
        return True
    else:
        print("❌ Dashboard file not found!")
        return False

def main():
    print("🚀 AI Investment Bot - Deployment Preparation")
    print("=" * 50)
    
    # Check prerequisites
    check_git()
    
    if not check_requirements():
        print("\n❌ Please ensure requirements.txt exists!")
        return
    
    if not check_dashboard():
        print("\n❌ Please ensure gui/dashboard.py exists!")
        return
    
    print("\n" + "=" * 50)
    print("✅ Your project is ready for deployment!")
    print("\n📋 Next Steps:")
    print("1. Create a GitHub account (if needed): https://github.com")
    print("2. Create a new repository on GitHub")
    print("3. Run these commands:")
    print("   git add .")
    print("   git commit -m 'Deploy AI Investment Bot'")
    print("   git remote add origin https://github.com/YOUR_USERNAME/ai-investment-bot.git")
    print("   git push -u origin main")
    print("4. Go to https://share.streamlit.io/ and deploy!")
    print("\n📖 See DEPLOYMENT_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    main()

