import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gregg_limper.response.steps.refinement import RefinementStep
from gregg_limper.response.engine import PipelineContext
from gregg_limper.config import core

@pytest.mark.asyncio
async def test_refinement_step_loop():
    # Mock context
    payload = MagicMock()
    payload.messages = [{"role": "user", "content": "Hello"}]
    payload.history.messages = []
    
    # Context now requires discord_message, payload is optional but we set it for this test
    context = PipelineContext(discord_message=MagicMock(), payload=payload)
    context.response_text = "Part 1"
    
    # Mock generation step
    gen_step = AsyncMock()
    # Loop 1: returns "Part 2" (Rewrite 1)
    # Loop 2: returns "Part 3" (Rewrite 2 - Final)
    
    # We need to simulate the generation step updating the context
    async def side_effect(ctx):
        if ctx.response_text == "Part 1":
            ctx.response_text = "Part 2"
        elif ctx.response_text == "Part 2":
            ctx.response_text = "Part 3"
        return ctx
        
    gen_step.run.side_effect = side_effect
    
    step = RefinementStep(gen_step)
    
    with patch("gregg_limper.response.steps.refinement.oai.check_completeness", new_callable=AsyncMock) as mock_check:
        # check_completeness called:
        # 1. Initial response "Part 1" -> False
        # 2. Rewrite 1 "Part 2" -> False
        # 3. Rewrite 2 "Part 3" -> True
        mock_check.side_effect = [False, False, True]
        
        core.DETAIL_CHECK_MODEL_ID = "test-model"
        core.DETAIL_CHECK_MAX_LOOPS = 3
        
        final_context = await step.run(context)
        
        # Expect the FINAL rewrite
        assert final_context.response_text == "Part 3"
        assert gen_step.run.call_count == 2
        assert mock_check.call_count == 3


@pytest.mark.asyncio
async def test_refinement_step_no_config():
    core.DETAIL_CHECK_MODEL_ID = None
    context = MagicMock()
    gen_step = AsyncMock()
    
    step = RefinementStep(gen_step)
    final_context = await step.run(context)
    
    assert final_context == context
    gen_step.run.assert_not_called()


@pytest.mark.asyncio
async def test_refinement_step_history_reset():
    # Verify that the history is reset in each loop and instruction is passed
    payload = MagicMock()
    # 1 system, 1 user
    payload.messages = [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hello"}]
    payload.history.messages = []
    
    context = PipelineContext(discord_message=MagicMock(), payload=payload)
    context.response_text = "Part 1"
    
    gen_step = AsyncMock()
    
    async def side_effect(ctx):
        ctx.response_text = "Part 2"
        return ctx
        
    gen_step.run.side_effect = side_effect
    
    step = RefinementStep(gen_step)
    
    with patch("gregg_limper.response.steps.refinement.oai.check_completeness", new_callable=AsyncMock) as mock_check:
        # 1. Initial "Part 1" -> Incomplete
        # 2. "Part 2" -> Complete
        mock_check.side_effect = [False, True]
        
        core.DETAIL_CHECK_MODEL_ID = "test-model"
        core.DETAIL_CHECK_MAX_LOOPS = 3
        
        final_context = await step.run(context)
        
        assert final_context.response_text == "Part 2"
        
        # Verify generation step calls
        # Call 1 (Loop 1): Should have [Sys, User, Assistant(Part 1), User(Instruction)]
        args, _ = gen_step.run.call_args
        ctx_passed = args[0]
        messages_passed = ctx_passed.messages
        
        assert len(messages_passed) == 4
        assert messages_passed[2]["role"] == "assistant"
        assert messages_passed[2]["content"] == "Part 1"
        assert messages_passed[3]["role"] == "user"
        assert "REWRITE" in messages_passed[3]["content"]
