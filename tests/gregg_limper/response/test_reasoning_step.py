import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gregg_limper.response.steps.reasoning import ReasoningStep
from gregg_limper.response.engine import PipelineContext
from gregg_limper.config import core

@pytest.mark.asyncio
async def test_reasoning_step_no_config():
    # If no REASONING_MODEL_ID, step should do nothing
    core.REASONING_MODEL_ID = None
    context = MagicMock()
    step = ReasoningStep()
    
    result = await step.run(context)
    assert result == context

@pytest.mark.asyncio
async def test_reasoning_step_execution():
    core.REASONING_MODEL_ID = "test-reasoning-model"
    
    # Mock context
    context = PipelineContext(discord_message=MagicMock(), payload=MagicMock())
    context.messages = [
        {"role": "system", "content": "Original System Prompt"},
        {"role": "user", "content": "Help me with X"}
    ]
    
    # Mock OpenAI response
    mock_trace = "User wants X. I should provide Y."
    
    with patch("gregg_limper.response.steps.reasoning.oai.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_trace
        
        step = ReasoningStep()
        await step.run(context)
        
        # Verify oai.chat called with correct model and prompt
        mock_chat.assert_called_once()
        args, kwargs = mock_chat.call_args
        messages = args[0]
        model = kwargs["model"]
        
        assert model == "test-reasoning-model"
        # Should have replaced system prompt
        assert messages[0]["role"] == "system"
        assert "internal monologue" in messages[0]["content"]
        # Should include user message
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Help me with X"
        
        # Verify trace injected into context
        assert len(context.messages) == 3
        last_msg = context.messages[-1]
        assert last_msg["role"] == "system"
        assert "Internal Reasoning Plan" in last_msg["content"]
        assert mock_trace in last_msg["content"]

@pytest.mark.asyncio
async def test_reasoning_step_failure_handling():
    core.REASONING_MODEL_ID = "test-model"
    context = PipelineContext(discord_message=MagicMock(), payload=MagicMock())
    context.messages = [{"role": "user", "content": "Hi"}]
    
    # Simulate API failure
    with patch("gregg_limper.response.steps.reasoning.oai.chat", side_effect=Exception("API Error")):
        step = ReasoningStep()
        # Should not raise, just return context unmodified
        result = await step.run(context)
        assert result == context
        assert len(context.messages) == 1
