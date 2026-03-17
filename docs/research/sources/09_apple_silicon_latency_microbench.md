# Microbench for one-way latency on Apple silicon

**Item:** Microbench for one-way latency on Apple silicon  
**Task:** Plan microbench for one-way latency on Apple silicon.  
**Secondary tasks:** Tie to apple-silicon-low-latency.md; inform buffer and QoS.  
**Context:** Apple silicon latency; docs/apple-silicon-low-latency.md.  
**Verification status:** PARTIALLY_VERIFIED  
**Verification basis:** repo_scan_only + source_title_only; doc exists; no microbench script; no pasted source.  
**Known facts:**
- Doc exists; sub-10 ms target
- 64/128 samples at 48 kHz; Instruments/Xcode
- No dedicated microbench script in repo  
**Unknowns:** "One-way" definition; target process; script vs Xcode vs doc procedure.  
**Assumptions:** None  
**Constraints:** Realtime safety per doc.  
**Ambiguities:** Scope of "one-way"; which component.  
**Source text / data:** UNKNOWN (full source block not present)  
**Output format:** Doc (procedure/spec); optional script; no new binaries in repo.
