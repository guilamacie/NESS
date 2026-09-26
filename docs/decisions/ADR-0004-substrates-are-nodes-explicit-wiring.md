# ADR-0004: Substrates are composition nodes with explicit input wiring

**Decision** There is no separate `substrates:` section; a substrate is a node of kind
`substrate` and must wire its inputs (`history: observation://target_history`) like every other
node. **Why** every read is an edge checked against the task access policy (addendum §3.4, §17.1);
hidden default reads would bypass the compiler. **Consequence** configs are slightly longer; the
resolved wiring printed by `ness validate` is complete.
