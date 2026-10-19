"""全局配置。产品名只在这里改，界面和报告都从这里读。"""
import os
from pathlib import Path

from dotenv import load_dotenv


# 本地开发从仓库根目录读取 .env；进程中已设置的环境变量优先。
load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)

APP_NAME = "量见"          # 候选：验策 / 策镜 / 策验室
APP_VERSION = "0.2.0-dev"
APP_TAGLINE = "把策略想法变成可复现、可追溯的历史实验"
DISCLAIMER = "历史回测不代表未来收益；测试未发现问题不等于策略已被证明有效。仅供学习研究，不构成投资建议。"

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_REPORT_MODEL = os.environ.get("LLM_REPORT_MODEL", "") or LLM_MODEL
LLM_TIMEOUT_SECONDS = max(1.0, float(os.environ.get("LLM_TIMEOUT_SECONDS", "90")))
LLM_AVAILABLE = bool(LLM_API_KEY and LLM_BASE_URL and LLM_MODEL)
