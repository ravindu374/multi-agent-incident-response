"""The intelligent controller: classify, rank and delegate incidents to specialist agents.

Built block by block:
  1. intake      - incident -> features                                   (done)
  2. classifier  - P(owner domain), P(involved domains) -> domain team
                   2a rules baseline (done), 2b logistic regression (done)
  3. registry    - A2A Agent Cards + live agent state + hard filter                 (done)
  4. router      - conformal team -> per-domain agent by expected utility -> fallbacks  (done)
  5. orchestrator- 5a agents as A2A servers + wire format (done),
                   5b investigate in parallel, fallbacks, pointers, escalation, cross-check (done),
                   5c optimizer: minimum-expected-loss staged plan, second look, policy agent (done)
  6. learner     - updates from verified outcomes only

Information barrier: nothing in this package may import the lab (`oncall`), which holds
ground truth. The controller sees only what an on-call engineer would see.
"""

DOMAINS = ("network", "data", "infra", "security")
