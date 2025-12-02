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
        self.trace_id = f"trace_{int(self.start_time)}"

    def capture(self, step_name: str, context: PipelineContext) -> None:
        """
        Capture a snapshot of the context after a step execution.
        """
        snapshot = {
            "step": step_name,
            "timestamp": time.time(),
            "response_text": context.response_text,
            "response_fragments": list(context.response_fragments),
            "message_count": len(context.messages),
            # We capture the last message to see what was added
            "last_message": context.messages[-1] if context.messages else None,
        }
        
        self.steps.append(snapshot)
        self._write_trace()

    def _write_trace(self) -> None:
        """
        Write the current trace to disk.
        """
        try:
            TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "trace_id": self.trace_id,
                "start_time": self.start_time,
                "steps": self.steps
            }
            
            with TRACE_FILE.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error("Failed to write pipeline trace: %s", e)
