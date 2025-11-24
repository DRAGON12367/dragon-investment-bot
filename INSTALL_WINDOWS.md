# Windows Installation Guide

## Quick Install (Recommended)

Use the provided batch script:

```bash
install_requirements.bat
```

## Manual Install

If the batch script doesn't work, install dependencies step by step:

### Step 1: Install Core Dependencies

```bash
pip install aiohttp yfinance pandas numpy scikit-learn streamlit plotly nest-asyncio python-dotenv
```

### Step 2: Install PyArrow (Optional)

PyArrow is optional and may fail to build on Windows. If it fails, you can skip it - the dashboard will still work.

**Option A: Try pre-built wheel**
```bash
pip install pyarrow
```

**Option B: Skip PyArrow**
The application will work without it. PyArrow is only needed for some advanced pandas features that we don't use.

### Step 3: Verify Installation

```bash
python -c "import streamlit; import pandas; import yfinance; print('All core dependencies installed!')"
```

## Troubleshooting

### PyArrow Build Error

If you see CMake/Visual Studio errors when installing pyarrow:

1. **Skip it** - PyArrow is not required for this project
2. **Or install Visual Studio Build Tools**:
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Install "Desktop development with C++" workload
   - Then retry: `pip install pyarrow`

### Alternative: Use Conda

If pip continues to have issues:

```bash
conda create -n trading-bot python=3.10
conda activate trading-bot
conda install pandas numpy scikit-learn
pip install aiohttp yfinance streamlit plotly nest-asyncio python-dotenv
```

## Running the Dashboard

Once installed, run:

```bash
streamlit run gui/dashboard.py
```

The dashboard will work perfectly without pyarrow!

