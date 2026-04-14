# noexcept-lie RT Audit — 2026Q2

Scan produced by `scripts/audit/noexcept_rt_audit.py`. 0 findings across 2 source roots.

Each entry is a function declared `noexcept` whose body allocates, locks, or throws. In an RT callback context each is one failure away from `std::terminate`. AU/VST3 validation rejects plugins whose `processBlock` reaches any of these.

**Flag legend**:
- `new` / `lock` / `throw` → CRIT (program will terminate on failure)
- `string-alloc` / `container-grow` / `to_string` → HIGH (implicit heap alloc via libc++; bad_alloc escapes as terminate)

## Fix patterns

- **`container-grow`**: pre-allocate `std::vector::reserve(max_n)` in `prepareToPlay`; write via index assignment, never `push_back`. Or switch to `boost::container::static_vector<T, N>` / `std::array<T, N>` with a size counter.
- **`to_string` / `string-alloc`**: replace log lines with `juce::Logger::writeToLog` gated by `#ifdef JUCE_DEBUG`, or move the log to the UI thread via a lock-free SPSC queue.
- **`new`**: allocate in `prepareToPlay`; the RT path only writes into preallocated buffers.
- **`lock`**: replace with `std::atomic<T>` or a lock-free SPSC queue (`readerwriterqueue` is already linked).
- **`throw`**: return an error code or `std::optional`. RT code cannot throw across audio callbacks.

## CI gate suggestion

Once CRIT count is zero, add to the CI workflow:
```yaml
- name: noexcept RT audit (regression gate)
  run: python3 scripts/audit/noexcept_rt_audit.py > /dev/null
```

The script exits non-zero if any CRIT finding exists.
