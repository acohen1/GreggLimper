import pytest
import json
from unittest.mock import MagicMock
from pathlib import Path
from gregg_limper.response.tracer import PipelineTracer
from gregg_limper.response.engine import PipelineContext

@pytest.mark.asyncio
async def test_tracer_capture_and_write(tmp_path):
    # Patch TRACE_FILE to use tmp_path
    trace_file = tmp_path / "pipeline_trace.json"
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr("gregg_limper.response.tracer.TRACE_FILE", trace_file)
        
        tracer = PipelineTracer()
        
        # Mock context
        context = PipelineContext(discord_message=MagicMock(), payload=MagicMock())
        context.response_text = "Test Response"
        context.messages = [{"role": "user", "content": "Hi"}]
        context.response_fragments = ["http://example.com"]
        
        # Capture a step
        tracer.capture("TestStep", context)
        
        # Verify file written
        assert trace_file.exists()
        
        # Verify content
        with trace_file.open() as f:
            data = json.load(f)
            
        assert data["trace_id"].startswith("trace_")
        assert "total_latency_ms" in data
        assert data["final_response"] == "Test Response"
        
        assert len(data["steps"]) == 1
        step = data["steps"][0]
        assert step["step"] == "TestStep"
        assert step["response_text"] == "Test Response"
        assert step["response_fragments"] == ["http://example.com"]
        assert step["message_count"] == 1
        
        # Check new fields
        assert "latency_ms" in step
        assert "elapsed_ms" in step
        assert len(step["new_messages"]) == 1
        assert step["new_messages"][0]["content"] == "Hi"

@pytest.mark.asyncio
async def test_tracer_integration_mock():
    # Verify integration with ResponsePipeline via mocks
    # This is a lightweight check to ensure wiring is correct
    from gregg_limper.response.engine import ResponsePipeline
    
    step_mock = MagicMock()
    step_mock.run = MagicMock(side_effect=lambda ctx: ctx) # async mock handled below
    
    # Need to make run async
    async def run_mock(ctx):
        return ctx
    step_mock.run = run_mock
    step_mock.__class__.__name__ = "MockStep"
    
    tracer_mock = MagicMock()
    
    pipeline = ResponsePipeline([step_mock], tracer=tracer_mock)
    context = PipelineContext(discord_message=MagicMock(), payload=MagicMock())
    
    await pipeline.run(context)
    
    # Should call capture twice: Start and After MockStep
    assert tracer_mock.capture.call_count == 2
    tracer_mock.capture.assert_any_call("Start", context)
    tracer_mock.capture.assert_any_call("MockStep", context)
