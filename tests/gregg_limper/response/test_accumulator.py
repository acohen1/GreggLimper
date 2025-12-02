import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gregg_limper.response.accumulator import accumulate_response, check_completeness
from gregg_limper.config import core

@pytest.mark.asyncio
async def test_check_completeness_complete():
    with patch("gregg_limper.response.accumulator.oai.check_completeness", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True
        core.DETAIL_CHECK_MODEL_ID = "test-model"
        
        result = await check_completeness("Some response", [])
        assert result is True
        mock_check.assert_called_once()

@pytest.mark.asyncio
async def test_check_completeness_continue():
    with patch("gregg_limper.response.accumulator.oai.check_completeness", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = False
        core.DETAIL_CHECK_MODEL_ID = "test-model"
        
        result = await check_completeness("Incomplete response", [])
        assert result is False

@pytest.mark.asyncio
async def test_accumulate_response_loop():
    # Mock payload
    payload = MagicMock()
    payload.messages = [{"role": "user", "content": "Hello"}]
    payload.history.messages = []
    
    # Mock runner
    runner = AsyncMock()
    # First call (initial) is done outside, so runner is called for loop 1 and loop 2
    # Loop 1: returns " part 2", incomplete
    # Loop 2: returns " part 3", complete
    runner.side_effect = [
        ("Part 2", []),
        ("Part 3", [])
    ]
    
    with patch("gregg_limper.response.accumulator.check_completeness", new_callable=AsyncMock) as mock_check:
        # check_completeness called:
        # 1. Initial response -> False
        # 2. Initial + part 2 -> False
        # 3. Initial + part 2 + part 3 -> True
        mock_check.side_effect = [False, False, True]
        
        core.DETAIL_CHECK_MODEL_ID = "test-model"
        core.DETAIL_CHECK_MAX_LOOPS = 3
        
        final_response = await accumulate_response("Part 1", payload, runner)
        
        # Expect the FINAL rewrite, not a concatenation
        assert final_response == "Part 3"
        assert runner.call_count == 2
        assert mock_check.call_count == 3

@pytest.mark.asyncio
async def test_accumulate_response_no_config():
    core.DETAIL_CHECK_MODEL_ID = None
    payload = MagicMock()
    runner = AsyncMock()
    
    final_response = await accumulate_response("Part 1", payload, runner)
    
    assert final_response == "Part 1"
@pytest.mark.asyncio
async def test_accumulate_response_history_reset():
    # Verify that the history is reset in each loop
    payload = MagicMock()
    # 1 system, 1 user
    payload.messages = [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hello"}]
    payload.history.messages = []
    
    runner = AsyncMock()
    # Initial: "Part 1"
    # Loop 1: "Part 2"
    runner.side_effect = [
        ("Part 2", [])
    ]
    
    with patch("gregg_limper.response.accumulator.oai.check_completeness", new_callable=AsyncMock) as mock_check:
        # 1. Initial "Part 1" -> Incomplete
        # 2. "Part 1 Part 2" -> Complete
        mock_check.side_effect = [False, True]
        
        core.DETAIL_CHECK_MODEL_ID = "test-model"
        core.DETAIL_CHECK_MAX_LOOPS = 3
        
        final_response = await accumulate_response("Part 1", payload, runner)
        
        assert final_response == "Part 2"
        
        # Verify runner calls
        # Call 1 (Loop 1): Should have [Sys, User, Assistant(Part 1), User(Continue)]
        # Length should be 2 (original) + 2 (appended) = 4
        args, _ = runner.call_args
        messages_passed = args[0]
        assert len(messages_passed) == 4
        assert messages_passed[2]["role"] == "assistant"
        assert messages_passed[2]["content"] == "Part 1"
        assert messages_passed[3]["role"] == "user"
        assert "REWRITE" in messages_passed[3]["content"]
