# Research AI Evaluation and Guardrails

## Evaluation Tracks

1. **Retrieval Quality**
   - Precision@k for known question set.
   - Citation coverage rate.
2. **Answer Quality**
   - Factuality checks against golden references.
   - Hallucination rate under adversarial prompts.
3. **Mapping Quality**
   - Emotion-to-music parameter consistency.
   - Blend interpolation sanity checks.

## Guardrail Tests

- Medical advice refusal test suite.
- Harmful instruction refusal suite.
- Citation absence detection suite.
- Confidence calibration checks.

## Success Criteria

- Citation coverage >= 95% for answerable questions.
- Hallucination rate <= 2% on curated eval set.
- 100% pass rate on hard safety policy checks.
