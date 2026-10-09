# Third-party notices

This repository consolidates independently developed experiments and their evidence. It does not apply a new blanket license to every component.

- **Early MATH500 pilot:** the original [MIT license](math500/pilot/LICENSE) is retained, copyright 2026 Clara Chen.
- **GSM8K:** copied dataset content is covered by the upstream [MIT license](gsm8k/licenses/GSM8K-MIT.txt). The frozen [RLVR dataset card](gsm8k/licenses/RLVR-DATASET-CARD.md) records the mixed collection's ODC-BY-1.0 designation; component datasets retain their respective terms. Preparation provenance in the experiment manifests records pinned sources.
- **AllenAI Open Instruct:** prompt/preparation references and derived material retain the upstream [Apache 2.0 notice](gsm8k/licenses/OPEN-INSTRUCT-LICENSE.txt).
- **Syntax / llmpp:** the pinned submodule `syntax/third_party/llmpp` retains its upstream MIT license. UD English EWT is not redistributed in the syntax directory; obtain it from the pinned source and follow its upstream license.
- **MATH500 evaluator:** the frozen source, upstream identity and local patch are retained under `math500/provenance/`. Consult those files and the referenced upstream project for applicable source and dataset terms.
- **Model weights:** not redistributed. Meta Llama and AllenAI checkpoint access/use requirements apply independently.

Source provenance is not a replacement for a license grant. Where no explicit project license was present, this migration does not invent one.
