import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from gregg_limper.response.steps.relevancy import RelevancyStep
from gregg_limper.response.engine import PipelineContext
from gregg_limper.config import core

@pytest.mark.asyncio
async def test_relevancy_step_loop():
    # Mock context
    payload = MagicMock()
    payload.messages = [{"role": "user", "content": "Hello"}]
    payload.history.messages = []
    
    context = PipelineContext(discord_message=MagicMock(), payload=payload)
    context.response_text = "Bad Response 1"
    context.messages = list(payload.messages)
    context.messages.append({"role": "assistant", "content": "Bad Response 1"})
    
    step = RelevancyStep()
    
    # Mock check_relevancy and oai.chat
    with patch("gregg_limper.response.steps.relevancy.check_relevancy", new_callable=AsyncMock) as mock_check, \
         patch("gregg_limper.response.steps.relevancy.oai.chat", new_callable=AsyncMock) as mock_chat:
        
        # check_relevancy called:
        # 1. "Bad Response 1" -> FAIL
        # 2. "Bad Response 2" -> FAIL
        # 3. "Good Response" -> PASS
        mock_check.side_effect = [
            {"decision": "FAIL", "missing": ["items"], "issues": []},
            {"decision": "FAIL", "missing": ["more detail"], "issues": []},
            {"decision": "PASS", "missing": [], "issues": []},
        ]
        
        # oai.chat called for regeneration:
        # 1. Returns "Bad Response 2"
        # 2. Returns "Good Response"
        mock_chat.side_effect = ["Bad Response 2", "Good Response"]
        
        core.RELEVANCY_CHECK_MODEL_ID = "test-check-model"
        core.MSG_MODEL_ID = "test-msg-model"
        core.RELEVANCY_CHECK_MAX_LOOPS = 3
        
        final_context = await step.run(context)
        
        # Expect the FINAL response
        assert final_context.response_text == "Good Response"
        assert mock_check.call_count == 3
        assert mock_chat.call_count == 2
        
        # Verify temperature scaling
        # Loop 1: 0.7 + (1/3)*0.3 ~= 0.8
        # Loop 2: 0.7 + (2/3)*0.3 ~= 0.9
        
        # Check calls to chat
        # Call 1
        args1, kwargs1 = mock_chat.call_args_list[0]
        assert kwargs1["temperature"] == pytest.approx(0.8)
        
        # Call 2
        args2, kwargs2 = mock_chat.call_args_list[1]
        assert kwargs2["temperature"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_relevancy_step_no_config():
    core.RELEVANCY_CHECK_MODEL_ID = None
    context = MagicMock()
    
    step = RelevancyStep()
    final_context = await step.run(context)
    
    assert final_context == context


@pytest.mark.asyncio
async def test_relevancy_step_history_growth():
    # Verify that history grows with critiques
    payload = MagicMock()
    payload.messages = [{"role": "user", "content": "Hello"}]
    payload.history.messages = []
    
    context = PipelineContext(discord_message=MagicMock(), payload=payload)
    context.response_text = "Bad 1"
    context.messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Bad 1"}]
    
    step = RelevancyStep()
    
    with patch("gregg_limper.response.steps.relevancy.check_relevancy", new_callable=AsyncMock) as mock_check, \
         patch("gregg_limper.response.steps.relevancy.oai.chat", new_callable=AsyncMock) as mock_chat:

        # 1. "Bad 1" -> Fail
        # 2. "Good" -> Pass
        mock_check.side_effect = [
            {"decision": "FAIL", "missing": ["details"], "issues": []},
            {"decision": "PASS", "missing": [], "issues": []},
        ]
        mock_chat.return_value = "Good"

        core.RELEVANCY_CHECK_MODEL_ID = "test-model"
        core.RELEVANCY_CHECK_MAX_LOOPS = 3
        
        final_context = await step.run(context)

    assert final_context.response_text == "Good"
    
    # Verify messages passed to chat
    # Should be: [User, Assistant(Bad 1), User(Instruction derived from critique)]
    # Because we use sliding window, we construct it from base ([User]) + [Bad 1] + [Instruction]
    args, kwargs = mock_chat.call_args
    messages_passed = args[0]
    
    # The list passed to chat is mutable, but in our code we create a NEW list each loop:
    # relevancy_messages = list(base_messages) ...
    # So messages_passed should be exactly what was sent.
    
    # We expect:
    # 1. User "Hello" (from base)
    # 2. Assistant "Bad 1" (the failed attempt)
    # 3. User "Instruction" (the critique)
    
    assert len(messages_passed) == 3
    assert messages_passed[0]["content"] == "Hello"
    assert messages_passed[1]["content"] == "Bad 1"
    assert "Include" in messages_passed[2]["content"]
    
    # Final context messages should be CLEAN:
    # [User, Assistant(Good)]
    # It should NOT contain Bad 1 or the Instruction.
    assert len(final_context.messages) == 2
    assert final_context.messages[0]["content"] == "Hello"
    assert final_context.messages[1]["content"] == "Good"
