# Contributing

This is a personal knowledge base, so "contributing" mostly means future me
adding notes without breaking the conventions past me set up. The rules below
exist to keep the repository navigable once it holds several hundred documents.

---

## The three kinds of content

| Kind | Lives in | Template |
| --- | --- | --- |
| **Topic note** | A numbered domain folder | [TOPIC_TEMPLATE.md](_templates/TOPIC_TEMPLATE.md) |
| **Paper note** | `papers/notes/` | [PAPER_NOTE_TEMPLATE.md](_templates/PAPER_NOTE_TEMPLATE.md) |
| **Implementation** | `implementations/` | [IMPLEMENTATION_TEMPLATE.md](_templates/IMPLEMENTATION_TEMPLATE.md) |

If a piece of writing does not fit one of these three, it probably belongs in
`_meta/` or `resources/`.

---

## Adding a topic note

1. **Find the right folder.** Use the master [INDEX.md](INDEX.md). If nothing
   fits, that is a signal to add a subfolder, not to dump the file at the root.
2. **Copy the template.** Do not start from a blank file. The template exists so
   every note has front matter, an index, and the four stage structure.
3. **Name the file** in lowercase with hyphens, describing the topic and not the
   source: `batch-normalization.md`, not `week4-lecture-notes.md`.
4. **Write the front matter** honestly. The `Prerequisites` line is the one that
   makes the repository usable by a beginner, so fill it in properly.
5. **Generate the internal index** after the content is written, so it matches
   the final headings.
6. **Register the file** in the folder's `INDEX.md` and in the root
   `INDEX.md`.
7. **Add cross links.** If the new note relates to an existing one, link both
   directions. One way links rot.

---

## Adding a paper note

Paper notes are not summaries of the abstract. The point is to capture what you
would need to reimplement or argue about the paper a year from now.

1. Copy `_templates/PAPER_NOTE_TEMPLATE.md` into `papers/notes/`.
2. Name it `year-first-author-short-title.md`, for example
   `2015-he-deep-residual-learning.md`. Year first keeps the folder sorted
   chronologically, which is how the field actually makes sense.
3. Fill in the **one line contribution** before anything else. If you cannot
   write it in one line, you have not understood the paper yet.
4. Record it in `papers/INDEX.md` and add the appropriate topic links.

---

## Writing standards

**Structure every concept in four stages.**

1. Plain English. What is the problem, what is the idea. No notation.
2. The maths. The actual equations, with every symbol defined.
3. The intuition. Why the maths works, in words.
4. Code. Something that runs, not pseudocode.

Then finish with where it is used in a real system.

**Prose rules.**

- No em dashes. Use commas, colons, or a full stop.
- Plain, direct language. Write the way you would explain it to a colleague.
- First person where you are recording your own experience or opinion.
- Short sentences over clever ones.
- Define jargon on first use, in a table where there are several terms.

<a id="claim-tagging"></a>
**Claim tagging.** When a note asserts something that a reader might want to
verify, mark where it came from:

| Tag | Means |
| --- | --- |
| `[SOURCE]` | Stated in a specific paper, book, or docs page. Cite it. |
| `[FIRST-HAND]` | I observed this myself while building something. |
| `[INFERENCE]` | I reasoned this from other facts. Could be wrong. |
| `[ASSUMPTION]` | Working assumption, not yet verified. |

Use these sparingly, only on claims that matter. Tagging every sentence makes
the document unreadable.

---

## Formatting rules

**Tables.** Use them for jargon definitions, comparisons, and any two column
lookup. Always include a header row. Keep cells short, and put long explanations
in prose instead.

**Code blocks.** Always tag the language:

````markdown
```python
import torch.nn as nn
```
````

Use `text` for maths, ASCII diagrams, shell output, and directory trees, since
that prevents markdown from mangling asterisks and underscores.

**Maths.** Inline maths uses backticks for simple expressions like `z = Wx + b`.
Display maths uses a fenced `text` block so it renders identically everywhere,
including on GitHub, which does not render LaTeX consistently in all contexts.

**Diagrams.** ASCII diagrams in fenced `text` blocks are preferred, because they
survive in a plain text editor and diff cleanly in git. Use Mermaid where a
diagram genuinely needs to be graphical. Put image files in
`assets/diagrams/` and link them relatively.

**Internal indexes.** Every document longer than roughly two screens needs one.
Format:

```markdown
## Index

### [Part heading](#part-heading)

- **[Section heading](#section-heading)**
  - [Subsection](#subsection)
```

Anchors are generated from the heading text: lowercase, punctuation removed,
spaces replaced with hyphens. Where a heading repeats within one document, add
an explicit `<a id="..."></a>` anchor immediately after it.

---

## Before you commit

- [ ] Front matter is complete and honest
- [ ] Internal index matches the actual headings
- [ ] Every internal link resolves
- [ ] Code blocks are language tagged and the code runs
- [ ] Registered in the folder `INDEX.md` and the root `INDEX.md`
- [ ] Cross links added in both directions
- [ ] No em dashes
