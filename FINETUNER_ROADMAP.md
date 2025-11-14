# Finetuner Roadmap

## Vision
Mirror the production Gregg Limper prompt stack 1:1 while building a high-signal finetuning dataset that only includes whitelisted speakers, respects an earliest-date cutoff, and injects synthetic tool calls without touching live RAG/memo stores.

## Milestones
1. **Data Intake**
   - ✅ History collector scaffolding created with earliest-date and whitelist filters.
   - ✅ Raw transcripts persisted per channel (`finetune-data/raw/`) for auditing.

2. **Conversation Segmentation**
   - ✅ Deterministic gap-based chunker + test coverage.
   - ✅ Enhanced LLM boundary prompt + injectable decider hook.

3. **Relabeling & Augmentation**
   - ✅ Relabeling uses formatter output to mirror prod history.
   - ✅ Synthetic `retrieve_context` insertion with tests.

4. **Prompt-Shaped Formatting**
   - ✅ Prompt header + metadata export scaffolding.
   - ✅ Context blocks now mirror `gather_context` output for each segment.

5. **Quality Gates**
   - Score tone fidelity, coverage, and duplication before writing.
   - Maintain audit logs linking samples back to original Discord message IDs.

## Open Questions
- What heuristics should trigger synthetic tool calls beyond obvious "remember when" lines?
- Which LLM/model do we trust for boundary detection at scale?
- How should we score conversation quality before exporting (e.g., LLM critic vs heuristics)?

## Next Actions
- Integrate real Discord pulls through the collector + cache so formatter sees authentic messages.
- Wire the LLM-backed segment decider with richer prompts and fallbacks.
- Expand tuner tests (collector mocks, CLI dry runs, runner integration) as features land.
- Define how to capture the target assistant reply per segment for supervised finetuning.
- Implement runner dry-run integration tests that mock Discord + LLM paths end-to-end.
- Design quality filters/critics to reject low-signal or off-tone segments before export.
- Improve synthetic tool heuristics so `retrieve_context` only fires on concrete lore references.
- Enrich the CLI with dry-run stats/logging flags for auditing LLM decisions.
