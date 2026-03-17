# Choosing between XPC, UDS, and shared rings

**Item:** Choosing between XPC, UDS, and shared rings  
**Task:** Plan how to choose and document IPC: XPC vs UDS vs shared rings.  
**Secondary tasks:** Relate to local-service boundaries and DAW/plugin safety.  
**Context:** Apple silicon / IPC; local-service boundaries.  
**Verification status:** UNVERIFIED  
**Verification basis:** source_title_only; no XPC/UDS/rings in repo; no pasted source.  
**Known facts:**
- No XPC, UDS, or shared rings refs in repo
- apple-silicon-low-latency.md does not specify IPC  
**Unknowns:** Use case; latency/throughput; platform; "shared rings" Apple vs generic.  
**Assumptions:** None  
**Constraints:** Human approval before implementation; plugin/DAW stability.  
**Ambiguities:** Which boundary (UI↔engine, Python↔C++, etc.).  
**Source text / data:** UNKNOWN (full source block not present)  
**Output format:** Design doc (options, tradeoffs, decision); no code until decision.
