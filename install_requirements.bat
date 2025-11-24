@echo off
REM Install requirements without pyarrow first
echo Installing core requirements...
pip install aiohttp yfinance pandas numpy scikit-learn streamlit plotly nest-asyncio python-dotenv

REM Try to install pyarrow from pre-built wheel (optional)
echo.
echo Attempting to install pyarrow (optional, may skip if build fails)...
pip install pyarrow || echo PyArrow installation skipped - not required for core functionality

echo.
echo Installation complete!
echo If pyarrow failed, the dashboard will still work without it.

