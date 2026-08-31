
# Deep Learning Knowledge Base

A personal, growing wiki for everything deep learning: structured course notes,
research paper summaries, from-scratch implementations, and the practical
knowledge needed to take a paper and turn it into working code.

Built to be read by a complete beginner from the first page, and to stay useful
as a reference years later.

---

## What this repository is for

Three things, in order of importance.

1. **Learn.** A beginner should be able to open `00-start-here/` and walk a
   path from "what is a neuron" through to implementing a transformer, without
   ever needing to guess what to read next.
2. **Remember.** Every paper read, every technique learned, and every hard-won
   debugging lesson gets written down in a consistent format so it is findable
   two years later.
3. **Build.** When a project starts, this repository should answer "has someone
   already solved this, and what did they learn" from my own notes rather than
   from a cold search.

---

## Quick navigation

| I want to... | Go to |
| --- | --- |
| Start learning from zero | [00-start-here/](00-start-here/INDEX.md) |
| Find a specific topic | [INDEX.md](INDEX.md) (master index) |
| Look up a term | [_meta/glossary.md](_meta/glossary.md) |
| Read paper summaries | [papers/](papers/INDEX.md) |
| Find working code | [implementations/](implementations/INDEX.md) |
| Add a new note | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Find courses, books, datasets | [resources/](resources/INDEX.md) |

---

## Repository structure

```text
Deep-Learning-Wiki/
├── README.md                     you are here
├── INDEX.md                      master index of every document
├── CONTRIBUTING.md               how to add and format content
├── LICENSE.md
│
├── _meta/                        conventions that keep the repo consistent
│   ├── conventions.md            naming, formatting, file layout rules
│   ├── taxonomy.md               the tag vocabulary
│   ├── adding-content.md         how to add each content type
│   ├── llm-prompt.md             prompts for AI-assisted writing
│   ├── glossary.md               every term, one place
│   └── roadmap.md                what is planned next
│
├── _templates/                   copy these when adding content
│   ├── TOPIC_TEMPLATE.md
│   ├── PAPER_NOTE_TEMPLATE.md
│   ├── IMPLEMENTATION_TEMPLATE.md
│   └── INDEX_TEMPLATE.md
│
├── 00-start-here/                orientation and learning paths
├── 01-foundations/               maths, ML basics, neural networks
├── 02-training-and-optimization/ optimizers, regularization, tuning
├── 03-computer-vision/           CNNs, detection, segmentation, ViT
├── 04-natural-language-processing/
├── 05-sequence-models/           RNNs, attention, forecasting
├── 06-generative-models/         autoencoders, GANs, diffusion
├── 07-reinforcement-learning/
├── 08-multimodal/
├── 09-efficiency-and-deployment/ quantization, distillation, serving
├── 10-responsible-ai/            interpretability, fairness, robustness
│
├── papers/                       paper summaries and reading log
├── implementations/              from-scratch code and notebooks
├── resources/                    courses, books, datasets, tools
└── assets/                       diagrams and images
```

### Why numbered folders

The numbers encode a **suggested reading order**, not a strict hierarchy. A
beginner reads roughly `00 → 01 → 02` and then branches into whichever domain
folder matters to them. The numbers also keep the folder listing stable in every
file browser and IDE, which plain alphabetical names do not.

Numbers `11` onward are deliberately free. New domains (graph neural networks,
neuro-symbolic methods, whatever comes next) get appended without renumbering
anything, so no existing link ever breaks.

---

## How every document is structured

Consistency is what makes a wiki usable at scale. Every topic file follows the
same shape:

1. **Front matter block** stating source, level, prerequisites, coverage, and
   related documents.
2. **A linked index** at the top, one entry per heading, so any section is one
   click away.
3. **The body**, where each concept moves through the same four stages: plain
   English, then the maths, then the intuition, then working code.
4. **Where it is used in practice**, because a technique you cannot place in a
   real system is a technique you have not really learned.

See [_templates/TOPIC_TEMPLATE.md](_templates/TOPIC_TEMPLATE.md) for the exact
skeleton and [_meta/conventions.md](_meta/conventions.md) for the formatting
rules.

---

## Conventions worth knowing before you contribute

- **Filenames** are lowercase, hyphen separated, and descriptive:
  `residual-networks.md`, not `ResNets.md` or `notes3.md`.
- **Every folder has an `INDEX.md`** listing its contents with a one line
  summary and a status. A folder without an index is considered broken.
- **Every document has an internal linked index.** Long documents without one
  are unusable on a phone.
- **Links between documents are relative**, so the repository works when cloned,
  browsed on GitHub, or opened in Obsidian.
- **No em dashes.** Plain, direct prose. Commas and colons do the same work
  without the affectation.
- **Claims carry their source.** Where a note asserts something non-obvious, it
  says whether that came from a paper, from experience, or from inference. See
  [_meta/conventions.md](_meta/conventions.md#claim-tagging).

---

## Status

| Section | State |
| --- | --- |
| 00 Start here | Scaffolded |
| 01 Foundations | One complete document |
| 02 Training and optimization | Scaffolded |
| 03 Computer vision | One complete document |
| 04 to 10 | Scaffolded, awaiting content |
| Papers | Scaffolded |
| Implementations | Scaffolded |

Sections marked "scaffolded" have their folder structure, index, and template in
place. They are ready to receive notes and will not need restructuring later.

---

## Adding your first note in three steps

```bash
# 1. copy the template into the right folder
cp _templates/TOPIC_TEMPLATE.md 04-natural-language-processing/transformers/attention-mechanism.md

# 2. write it, then register it in that folder's index
$EDITOR 04-natural-language-processing/INDEX.md

# 3. add it to the master index
$EDITOR INDEX.md
```

Full detail in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Notes and prose are CC BY-SA 4.0. Code is MIT. See [LICENSE.md](LICENSE.md).

