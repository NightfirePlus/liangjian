from core.finance import answer_finance_question


def test_local_finance_answer_can_suggest_supported_strategy():
    result = answer_finance_question("双均线策略是什么？", use_llm=False)

    assert "平均价格" in result["answer"]
    assert result["suggested_strategy"] == "ma_cross"
    assert result["source"] == "local"


def test_finance_answer_validates_model_output():
    result = answer_finance_question(
        "什么是回测？",
        use_llm=True,
        complete=lambda *_args: {
            "answer": "回测是在历史数据上执行规则。",
            "related_concepts": ["样本外验证", "滑点", "回撤", "多余概念"],
            "suggested_strategy": None,
        },
    )

    assert result["source"] == "ai"
    assert len(result["related_concepts"]) == 3


def test_finance_answer_falls_back_on_invalid_model_output():
    result = answer_finance_question(
        "最大回撤是什么意思？",
        use_llm=True,
        complete=lambda *_args: {"unexpected": True},
    )

    assert "最大跌幅" in result["answer"]
    assert result["source"] == "local"


def test_local_finance_answer_can_suggest_rsi():
    result = answer_finance_question("RSI 超买超卖是什么意思？", use_llm=False)

    assert "均值回归" in result["answer"]
    assert result["suggested_strategy"] == "rsi_reversion"
