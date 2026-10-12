# Prompt-interface decision

## Why the main post-training comparison uses a shared template

SFT, DPO and RLVR checkpoints share the Tülu instruction interface. Applying
the same SFT chat template controls prompt serialization and prevents a
checkpoint-specific interface mismatch from being mistaken for a training
effect. The main RLVR − DPO conclusion therefore uses the shared template.

## Diagnostic evidence

On the fixed stratified 100-question subset:

| Interface | DPO pass@1 | RLVR pass@1 | RLVR − DPO |
|---|---:|---:|---:|
| no chat template | 34.25% | 25.75% | -8.50 pp |
| shared Tülu template | 42.25% | 43.25% | +1.00 pp |

The prompt-interface interaction is +9.5 pp, with a question-bootstrap 95% CI
of approximately [2.25, 16.75] pp. The preregistered rule stopped the diagnostic
after seed 1 because RLVR had caught up.

## Base interpretation

Forcing the shared SFT template onto Base controls serialized prompt form but
is behaviorally unnatural for the pretrained checkpoint:

| Base interface | pass@1 | pass@4 | empty parsed answers |
|---|---:|---:|---:|
| native `plain-boxed`, seed 1 | 8.35% | 22.60% | 16.75% |
| shared SFT template, seed 1 | 4.15% | 13.60% | 57.15% |

Therefore:

- shared-template Base is an **interface-controlled sensitivity condition**;
- native Base is a **capability-adapted behavioral condition**;
- neither should be silently substituted for the other;
- causal attribution of RLVR remains anchored on DPO versus RLVR under the
  same shared template.

For future representation comparisons, all checkpoints must be teacher-forced
on identical token sequences even if their natural generation interfaces differ.

