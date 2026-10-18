"""全局配置。产品名只在这里改，界面和报告都从这里读。"""
import os

APP_NAME = "量见"          # 候选：验策 / 策镜 / 策验室
APP_VERSION = "0.1.0"
APP_TAGLINE = "把策略想法变成可复现、可追溯的历史实验"
DISCLAIMER = "历史回测不代表未来收益；测试未发现问题不等于策略已被证明有效。仅供学习研究，不构成投资建议。"

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_AVAILABLE = bool(LLM_API_KEY and LLM_BASE_URL and LLM_MODEL)
