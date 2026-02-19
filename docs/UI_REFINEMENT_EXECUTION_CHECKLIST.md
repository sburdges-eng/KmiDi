# UI Refinement Execution Checklist

## React / TypeScript

- [x] Keep semantic token usage for UI primitives.
- [ ] Remove remaining hardcoded colors in component styles.
- [ ] Add focus-visible and high-contrast variants.
- [ ] Add accessibility checks for keyboard navigation and labels.

## JUCE / C++

- [ ] Validate visual parity between desktop and plugin surfaces.
- [ ] Standardize spacing and typography scales in plugin controls.
- [ ] Add UI performance counters for redraw and interaction latency.

## Shared UX

- [ ] Ensure 44px minimum touch/click targets across controls.
- [ ] Define interaction states (hover, active, disabled, loading).
- [ ] Align Side A/Side B navigation behavior with spec docs.
