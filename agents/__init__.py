"""Expert agents (Block 5a). Each agent = an Agent Card + read-only tools + a reasoner.

  tools.py      evidence tools; an agent may call only the tools on its own card
  reasoners/    rules (deterministic, evidence-driven) and ollama (local LLM); same interface
  specialist.py one agent: run its tools, ask its reasoner, return a Proposal

Information barrier: nothing here may import the lab (`oncall`); agents see incidents and
their own evidence slice, never ground truth.
"""
