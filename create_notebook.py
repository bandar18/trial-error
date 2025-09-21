#!/usr/bin/env python3
"""
Script to create the Upstox Trade Monitor Jupyter notebook
"""

import json

# Create the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Upstox-Integrated Trade-Position Monitor + IV–RV Context Dashboard\n",
                "\n",
                "## Mission\n",
                "Create a comprehensive trading dashboard that provides:\n",
                "1. **IV-RV Context Dashboard**: Real-time implied volatility analysis with IVR/IVP, realized volatility, VRP, and regime classification\n",
                "2. **Trade-Position Monitor**: Live portfolio P&L tracking with Greeks analysis and stress testing\n",
                "\n",
                "## Features\n",
                "- Live Upstox v3 API integration (read-only)\n",
                "- Real-time market data via WebSocket\n",
                "- Advanced volatility analytics (IVR, IVP, VRP, term structure)\n",
                "- Portfolio Greeks monitoring and stress testing\n",
                "- Action prompts for delta hedging and position management\n",
                "- India VIX regime overlay\n",
                "\n",
                "## Scale\n",
                "- ~60-90 liquid single-stock underlyings (NSE only)\n",
                "- No order placement - read-only decision support\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. CONFIG - API Credentials\n",
                "\n",
                "**Instructions**: Paste your Upstox API credentials in the cell below. Create a `.env` file in the same directory with:\n",
                "```\n",
                "UPSTOX_API_KEY=your_api_key\n",
                "UPSTOX_API_SECRET=your_api_secret\n",
                "UPSTOX_REDIRECT_URI=your_redirect_uri\n",
                "UPSTOX_ACCESS_TOKEN=your_access_token\n",
                "```"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CONFIG SECTION - Paste your Upstox API credentials here\n",
                "import os\n",
                "from dotenv import load_dotenv\n",
                "from pydantic import BaseSettings\n",
                "from typing import Optional\n",
                "\n",
                "# Load environment variables\n",
                "load_dotenv()\n",
                "\n",
                "class UpstoxConfig(BaseSettings):\n",
                "    api_key: str\n",
                "    api_secret: str\n",
                "    redirect_uri: str\n",
                "    access_token: Optional[str] = None\n",
                "    \n",
                "    class Config:\n",
                "        env_file = \".env\"\n",
                "\n",
                "# Initialize config\n",
                "config = UpstoxConfig(\n",
                "    api_key=os.getenv('UPSTOX_API_KEY', 'YOUR_API_KEY_HERE'),\n",
                "    api_secret=os.getenv('UPSTOX_API_SECRET', 'YOUR_API_SECRET_HERE'),\n",
                "    redirect_uri=os.getenv('UPSTOX_REDIRECT_URI', 'YOUR_REDIRECT_URI_HERE'),\n",
                "    access_token=os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_ACCESS_TOKEN_HERE')\n",
                ")\n",
                "\n",
                "print(f\"API Key configured: {config.api_key[:8]}...\")\n",
                "print(f\"Access Token configured: {bool(config.access_token)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Imports and Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Core libraries\n",
                "import asyncio\n",
                "import json\n",
                "import time\n",
                "import warnings\n",
                "from datetime import datetime, timedelta\n",
                "from typing import Dict, List, Optional, Tuple, Any\n",
                "from dataclasses import dataclass\n",
                "from pathlib import Path\n",
                "\n",
                "# Data processing\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from scipy import stats\n",
                "import statsmodels.api as sm\n",
                "from statsmodels.tsa.holtwinters import ExponentialSmoothing\n",
                "\n",
                "# HTTP and async\n",
                "import httpx\n",
                "from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type\n",
                "\n",
                "# Visualization\n",
                "import plotly.graph_objects as go\n",
                "import plotly.express as px\n",
                "from plotly.subplots import make_subplots\n",
                "import ipywidgets as widgets\n",
                "from IPython.display import display, clear_output\n",
                "\n",
                "# Logging\n",
                "from loguru import logger\n",
                "\n",
                "# Suppress warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Configure logging\n",
                "logger.remove()\n",
                "logger.add(lambda msg: print(msg, end=\"\"), level=\"INFO\")\n",
                "logger.add(\"upstox_monitor.log\", level=\"DEBUG\", rotation=\"1 day\")\n",
                "\n",
                "print(\"✅ All dependencies imported successfully\")\n",
                "print(f\"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook
with open('/workspace/upstox_trade_monitor.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook created successfully!")