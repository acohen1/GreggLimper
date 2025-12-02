import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gregg_limper.response.steps.tools import ToolExecutionStep
from gregg_limper.response.engine import PipelineContext
from gregg_limper.config import core
from gregg_limper.tools import ToolSpec

@pytest.mark.asyncio
async def test_tool_step_no_config():
    # If no TOOL_CHECK_MODEL_ID, step should do nothing
    core.TOOL_CHECK_MODEL_ID = None
    context = MagicMock()
    step = ToolExecutionStep()
    
    result = await step.run(context)
    assert result == context

@pytest.mark.asyncio
async def test_tool_step_no_tools_registered():
    # If no tools registered, step should do nothing
    core.TOOL_CHECK_MODEL_ID = "test-model"
    context = MagicMock()
    
    with patch("gregg_limper.response.steps.tools.get_registered_tool_specs", return_value=[]):
        step = ToolExecutionStep()
        result = await step.run(context)
        assert result == context

@pytest.mark.asyncio
async def test_tool_step_execution():
    core.TOOL_CHECK_MODEL_ID = "test-model"
    
    # Mock context
    context = PipelineContext(discord_message=MagicMock(), payload=MagicMock())
    context.messages = [{"role": "user", "content": "Check server"}]
    context.tool_context = MagicMock()
    
    # Mock tool specs
    mock_spec = ToolSpec(name="test_tool", description="desc", parameters={})
    
    # Mock OpenAI response with tool call
    mock_msg = MagicMock()
    mock_msg.content = "Thinking..."
    mock_call = MagicMock()
    mock_call.id = "call_123"
    mock_call.function.name = "test_tool"
    mock_call.function.arguments = "{}"
    mock_msg.tool_calls = [mock_call]
    
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    
    # Mock OpenAI response with NO tool call (stop condition)
    mock_msg_stop = MagicMock()
    mock_msg_stop.tool_calls = []
    mock_choice_stop = MagicMock()
    mock_choice_stop.message = mock_msg_stop
    mock_resp_stop = MagicMock()
    mock_resp_stop.choices = [mock_choice_stop]

    with patch("gregg_limper.response.steps.tools.get_registered_tool_specs", return_value=[mock_spec]), \
         patch("gregg_limper.response.steps.tools.oai.chat_full", side_effect=[mock_resp, mock_resp_stop]) as mock_chat, \
         patch("gregg_limper.response.steps.tools.execute_tool", new_callable=AsyncMock) as mock_exec:
        
        mock_exec.return_value.context_content = "Tool Output"
        mock_exec.return_value.response_content = "http://example.com/image.gif"
        
        step = ToolExecutionStep()
        await step.run(context)
        
        # Verify tool was executed
        mock_exec.assert_called_once_with("test_tool", "{}", context=context.tool_context)
        
        # Verify result injected into context
        # We expect: User, System(Tool Result)
        assert len(context.messages) == 2
        assert context.messages[1]["role"] == "system"
        assert "Tool 'test_tool' returned" in context.messages[1]["content"]
        assert "Tool Output" in context.messages[1]["content"]
        
        # Verify artifact captured
        assert len(context.response_fragments) == 1
        assert context.response_fragments[0] == "http://example.com/image.gif"
