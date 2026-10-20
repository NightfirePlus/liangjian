from core.agent import route_agent_message


def test_messages_build_strategy_before_report_exists():
    assert route_agent_message("为什么最大回撤这么大？", report_ready=False) == "strategy"


def test_report_questions_do_not_change_strategy_after_report():
    assert route_agent_message("为什么最大回撤这么大？", report_ready=True) == "report_question"
    assert route_agent_message("参数热力图说明了什么？", report_ready=True) == "report_question"


def test_explicit_strategy_changes_leave_report_qa_mode():
    assert route_agent_message("把快线改为10日", report_ready=True) == "strategy"
    assert route_agent_message("用创业板ETF重新测试这个策略", report_ready=True) == "strategy"
