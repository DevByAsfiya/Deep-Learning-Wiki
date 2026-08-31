# Roadmap

What is planned, in rough order. This is a working document, not a commitment.

---

## Near term

| Item | Section | Why |
| --- | --- | --- |
| Backpropagation from scratch in NumPy | `implementations/from-scratch/` | The single best way to make Chapter 1 stick |
| Transformer architecture note | `04-natural-language-processing/transformers/` | Prerequisite for almost everything current |
| Attention mechanism note | `05-sequence-models/attention/` | Feeds the transformer note |
| Paper note: Attention Is All You Need | `papers/notes/` | Anchor paper for the NLP section |
| Optimizers note (SGD to AdamW) | `02-training-and-optimization/optimizers/` | Referenced constantly elsewhere |

## Medium term

- Object detection: the path from R-CNN to YOLO to DETR
- Diffusion models, from the forward process through to sampling
- Quantization and distillation, with measured latency numbers
- A worked end to end project: dataset, training, evaluation, deployment

## Long term

- Graph neural networks as section `11`
- Interpretability, with actual probing experiments rather than summaries
- A reading log with dates, so the sequence of ideas is visible over time

## Maintenance

| Task | Cadence |
| --- | --- |
| Verify all internal links resolve | monthly |
| Review `#draft` notes and promote or delete | monthly |
| Prune `resources/` links that have died | quarterly |
| Re-read one older note and update it | weekly |

The last one matters more than it looks. A note you never revisit is a note you
have stopped learning from.
