"""
Debug tracer for the response pipeline.
Captures state snapshots after each step and writes them to disk.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from gregg_limper.response.engine import PipelineContext

logger = logging.getLogger(__name__)

TRACE_FILE = Path("data/runtime/pipeline_trace.json")


class PipelineTracer:
    """
    Records the evolution of the pipeline state.
    """

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.last_message_count = 0
        self.trace_id = f"trace_{int(self.start_time)}"

    def capture(self, step_name: str, context: PipelineContext) -> None:
        """
        Capture a snapshot of the context after a step execution.
        """
        now = time.time()
        latency_ms = (now - self.last_step_time) * 1000
        elapsed_ms = (now - self.start_time) * 1000
        
        # Calculate new messages added in this step
        current_count = len(context.messages)
        new_messages = context.messages[self.last_message_count:]
        
        snapshot = {
            "step": step_name,
            "latency_ms": round(latency_ms, 2),
            "elapsed_ms": round(elapsed_ms, 2),
            "message_count": current_count,
            "new_messages": new_messages,
            "metadata": dict(context.step_metadata),
        }
        
        # Only include response fields if populated
        if context.response_text:
            snapshot["response_text"] = context.response_text
            
        if context.response_fragments:
            snapshot["response_fragments"] = list(context.response_fragments)
        
        self.steps.append(snapshot)
        self.last_step_time = now
        self.last_message_count = current_count
        
        self._write_trace(context)

    def _write_trace(self, context: PipelineContext | None = None) -> None:
        """
        Write the current trace to disk.
        """
        try:
            TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            now = time.time()
            total_latency_ms = (now - self.start_time) * 1000
            
            data = {
                "trace_id": self.trace_id,
                "start_time": self.start_time,
                "total_latency_ms": round(total_latency_ms, 2),
                "final_response": context.response_text if context else "",
                "steps": self.steps
            }
            
            with TRACE_FILE.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error("Failed to write pipeline trace: %s", e)
