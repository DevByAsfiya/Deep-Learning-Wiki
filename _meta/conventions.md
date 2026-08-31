# Conventions

The rules that keep this repository navigable as it grows. If a rule here
conflicts with something you want to do, change the rule deliberately rather
than quietly making an exception.

---

## Index

- [File and folder naming](#file-and-folder-naming)
- [Document structure](#document-structure)
- [Internal indexes and anchors](#internal-indexes-and-anchors)
- [Linking](#linking)
- [Tables](#tables)
- [Code blocks](#code-blocks)
- [Maths notation](#maths-notation)
- [Diagrams](#diagrams)
- [Prose style](#prose-style)
- [Claim tagging](#claim-tagging)

---

## File and folder naming

<a id="file-and-folder-naming"></a>

| Rule | Example |
| --- | --- |
| Lowercase, hyphen separated | `residual-networks.md` |
| Describe the topic, never the source | `batch-normalization.md`, not `lecture-7.md` |
| Paper notes are year first | `2015-he-deep-residual-learning.md` |
| Folder indexes are always `INDEX.md` | uppercase, so it sorts to the top |
| Meta and template folders are underscore prefixed | `_meta/`, `_templates/` |

The underscore prefix keeps machinery folders visually separate from content
folders and sorts them above the numbered sections in most file browsers.

## Document structure

<a id="document-structure"></a>

Every topic note has, in this order:

1. An H1 title
2. A front matter blockquote: source, level, prerequisites, coverage, related
3. A one paragraph orientation
4. A horizontal rule
5. The internal index
6. A horizontal rule
7. The body

The front matter is a blockquote rather than YAML because it renders as readable
text on GitHub. If tooling later needs machine readable metadata, YAML front
matter can be added above it without breaking anything.

## Internal indexes and anchors

<a id="internal-indexes-and-anchors"></a>

Any document longer than about two screens needs an index. Three levels:

```markdown
### [Part or chapter](#part-or-chapter)

- **[Section](#section)**
  - [Subsection](#subsection)
```

GitHub generates anchors automatically by lowercasing the heading, stripping
punctuation, and replacing spaces with hyphens. That breaks in two cases:

- **Repeated headings.** Five sections called "Implementation" produce
  `#implementation`, `#implementation-1`, and so on. Fragile.
- **Headings with unusual characters.** Anchor generation differs between
  GitHub, Obsidian, and static site generators.

So write an explicit anchor immediately after each heading:

```markdown
### The maths

<a id="resnet-the-maths"></a>
```

Explicit anchors work identically everywhere and never shift when a heading is
reworded.

## Linking

<a id="linking"></a>

Always relative, never absolute:

```markdown
[Deep Learning Fundamentals](../01-foundations/neural-networks/deep-learning-fundamentals.md)
```

Relative links work when the repository is cloned, browsed on GitHub, or opened
in a local markdown editor. Absolute GitHub URLs break in all but one of those.

**Cross links go both ways.** If A links to B, B links back to A. One way links
degrade into dead ends.

## Tables

<a id="tables"></a>

Use a table when there are three or more parallel items with the same
attributes. Below three, prose is clearer.

Always include a header row and always pad with spaces so the raw markdown is
readable:

```markdown
| Term | Meaning |
| --- | --- |
| Neuron | One tiny function that weights its inputs and squashes the result |
```

Keep cells to roughly one sentence. Anything longer belongs in prose beneath the
table.

## Code blocks

<a id="code-blocks"></a>

Always tag the language. Untagged blocks get no syntax highlighting and no
tooling support.

| Content | Tag |
| --- | --- |
| Python | `python` |
| Shell | `bash` |
| Maths, ASCII art, directory trees, program output | `text` |
| JSON, YAML config | `json`, `yaml` |

Use `text` rather than leaving it blank, so it is clear the choice was
deliberate.

Code in notes should run. If it cannot run standalone, say so explicitly in a
comment rather than leaving the reader to discover it.

## Maths notation

<a id="maths-notation"></a>

**Inline:** backticks, for short expressions. `z = Wx + b`.

**Display:** a fenced `text` block.

````markdown
```text
a[l+2] = g(z[l+2] + a[l])
```
````

LaTeX is not used, because GitHub renders it inconsistently depending on
context, and this repository should read correctly in a plain text editor. Unicode
superscripts and Greek letters are fine and often clearer than LaTeX for short
expressions.

Define every symbol in a table immediately after the equation.

## Diagrams

<a id="diagrams"></a>

Preference order:

1. **ASCII in a `text` block.** Diffs cleanly in git, readable everywhere, no
   build step.
2. **Mermaid**, when the structure genuinely needs graph layout.
3. **An image in `assets/diagrams/`**, only when neither of the above works.

An ASCII diagram that survives a copy paste into a terminal is worth more than a
prettier image that rots.

## Prose style

<a id="prose-style"></a>

- **No em dashes.** Use a comma, a colon, or start a new sentence.
- Plain, direct language. Explain it the way you would to a colleague.
- Short sentences. Break long ones.
- First person when recording your own experience or judgement.
- Define jargon at first use.
- Prefer the concrete example over the abstract statement, then generalise.

## Claim tagging

<a id="claim-tagging"></a>

Non-obvious claims carry their provenance:

| Tag | Means | Example |
| --- | --- | --- |
| `[SOURCE]` | From a specific paper, book, or documentation page. Cite it. | `[SOURCE]` ResNet-152 reached 3.57% top-5 error on ImageNet (He et al., 2015). |
| `[FIRST-HAND]` | Observed while building something. | `[FIRST-HAND]` Fine tuning at lr 1e-3 destroyed the pretrained features within two batches. |
| `[INFERENCE]` | Reasoned from other facts, could be wrong. | `[INFERENCE]` This likely means the bottleneck is dataloader bound, not GPU bound. |
| `[ASSUMPTION]` | Working assumption, not yet verified. | `[ASSUMPTION]` Assuming the label noise is roughly uniform across classes. |

Only tag claims where being wrong would matter. Tagging everything makes the
document unreadable and the tags meaningless.
