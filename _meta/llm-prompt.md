# LLM Prompt Templates

> **Use these when asking an AI assistant to write content for this repository.**
> Copy the relevant block, fill in the bracketed parts, and attach the two files
> named in it.

An LLM has no knowledge of this repository's conventions unless you hand them
over. Attaching [adding-content.md](adding-content.md) and
[conventions.md](conventions.md) is the step that makes the difference between
output you can paste straight in and output you have to rewrite.

---

## Index

- **[Before you start](#before-you-start)**
- **[The full prompt](#the-full-prompt)**
- **[Short prompt for quick additions](#short-prompt-for-quick-additions)**
- **[Prompt: topic note](#prompt-topic-note)**
- **[Prompt: paper note](#prompt-paper-note)**
- **[Prompt: notebook or implementation](#prompt-notebook-or-implementation)**
- **[Prompt: converting existing notes](#prompt-converting-existing-notes)**
- **[Prompt: reviewing what you already have](#prompt-reviewing-what-you-already-have)**
- **[After the LLM responds](#after-the-llm-responds)**
- **[What LLMs get wrong here](#what-llms-get-wrong-here)**

---

## Before you start

<a id="before-you-start"></a>

Every prompt below assumes you attach these two files from this repository:

| File | Why |
| --- | --- |
| `_meta/adding-content.md` | The workflow, placement rules, and templates |
| `_meta/conventions.md` | Naming, anchors, tables, maths notation, claim tagging |

If the assistant cannot accept file attachments, paste the Formatting rules and
Interlinking rules sections of `adding-content.md` into the prompt instead.

Also useful to attach, when relevant:

- The folder `INDEX.md` for wherever the content is going, so the model can write
  the exact index row and see what already exists
- An existing note like `03-computer-vision/architectures/cnn-classic-architectures.md`
  as a worked example of the house style

---

## The full prompt

<a id="the-full-prompt"></a>

Use this for anything substantial.

```text
I maintain a deep learning knowledge base repo at
github.com/DevByAsfiya/Deep-Learning-Wiki. I'm adding new content and it
must follow the repo's existing conventions exactly.

I've attached _meta/adding-content.md and _meta/conventions.md. Read them
first and follow them precisely.

WHAT I'M ADDING:
[a topic note on X / a paper note for Y / a notebook that does Z]

SOURCE MATERIAL:
[paste your notes, attach a PDF, describe what to cover, or say
"write it from scratch"]

WHERE I THINK IT GOES:
[your guess at the folder path, or "you decide" if unsure]

WHAT I NEED BACK:
1. The complete file, ready to save, using the correct template
2. The exact Windows Command Prompt commands to place it
3. The exact index rows to add, and which files to add them to
4. Cross links I should add to existing notes, in both directions
5. The commit message

CONSTRAINTS:
- No em dashes anywhere
- Explicit <a id="..."></a> anchor under every heading
- Relative paths only, never GitHub URLs
- Maths in fenced text blocks, not LaTeX
- Language tag on every code block
- Every concept follows: plain English, then the maths, then the
  intuition, then runnable code, then where it is used in production
- Front matter block with source, level, prerequisites, covers, related
- Internal linked index at the top if the file runs longer than two screens
```

---

## Short prompt for quick additions

<a id="short-prompt-for-quick-additions"></a>

For a small note where you do not want to write the long version.

```text
[attach _meta/adding-content.md]

Add a topic note on [X] to my Deep-Learning-Wiki repo. Follow the attached
conventions exactly. Give me the file, the placement commands, the index
rows, and the commit message.
```

---

## Prompt: topic note

<a id="prompt-topic-note"></a>

```text
[attach _meta/adding-content.md, _meta/conventions.md, and the target
folder's INDEX.md]

Write a topic note on [CONCEPT] for my Deep-Learning-Wiki repo, following
the attached conventions.

Target folder: [e.g. 04-natural-language-processing/transformers/]
Level: [beginner / intermediate / advanced]
Assumed prerequisites: [what the reader already knows]

Cover: [list the specific subtopics you want included]

Structure every concept as: plain English with no notation, then the maths
with every symbol defined in a table and one worked numeric example, then
the intuition in words, then runnable PyTorch code, then a section naming
concrete production systems where it is used, then common mistakes.

Return the complete file, the placement commands for Windows Command
Prompt, the index rows for both the folder INDEX.md and the root INDEX.md,
the topic lookup table row, and the commit message.
```

---

## Prompt: paper note

<a id="prompt-paper-note"></a>

```text
[attach _meta/adding-content.md and the paper PDF or link]

Write a paper note for [PAPER TITLE] following the PAPER_NOTE_TEMPLATE
format described in the attached conventions.

Filename: papers/notes/[year-firstauthor-short-title].md

Write the one line contribution FIRST. If the paper's contribution cannot
be stated in one line, tell me that rather than writing something vague.

I care most about these three sections, so give them real substance:
- What I would question: weak ablations, results that may not hold at
  other scales, comparisons that were not apples to apples
- What I would use: which parts transfer to my own work and under what
  conditions
- Implementation notes: hyperparameters buried in appendices, things the
  official code does that the paper never mentions

Do not reproduce figures or substantial text from the paper. Paraphrase
and cite.

Return the file, the placement command, the reading log row for
papers/INDEX.md, the by-topic row, which existing topic notes should link
to this paper, and the commit message.
```

---

## Prompt: notebook or implementation

<a id="prompt-notebook-or-implementation"></a>

```text
[attach _meta/adding-content.md]

Write a from-scratch implementation of [THING] for my Deep-Learning-Wiki
repo, following the attached conventions.

Framework: [NumPy only / PyTorch]
Target: [implementations/notebooks/ or implementations/from-scratch/name/]
Runtime target: should train in under [N] minutes on [CPU / a T4 GPU]

Requirements:
- A README.md following IMPLEMENTATION_TEMPLATE.md, with the walkthrough
  structure: explain the step, show the code, explain what just happened
- Code that actually runs, with pinned versions in requirements.txt
- An expected output block so a reader can tell whether their run worked
- A "Things that went wrong" section with real debugging notes, not
  placeholders
- Links to the topic note this implements, and tell me what to add to that
  note to link back

If this is a notebook, include the markdown cells: an opening cell stating
what it builds and its prerequisites, a cell before each section, and a
closing cell with expected output.

Return the files, the placement commands for Windows Command Prompt, the
index rows for implementations/INDEX.md, and the commit message.
```

---

## Prompt: converting existing notes

<a id="prompt-converting-existing-notes"></a>

For turning a PDF, a Word document, or messy notes into a repo-shaped file.

```text
[attach _meta/adding-content.md and the source document]

Convert the attached [PDF / notes / document] into a topic note for my
Deep-Learning-Wiki repo.

Preserve all the content. Do not summarise or drop sections.

Requirements:
- Remove every em dash, replacing with commas or colons as the sentence
  needs
- Build a linked index at the top covering every heading, three levels deep
- Explicit <a id="..."></a> anchor under every heading
- Reconstruct all tables as clean markdown tables
- Put maths and ASCII diagrams in fenced text blocks so nothing gets
  mangled
- Language tag every code block
- Add the front matter block: source, level, prerequisites, covers, related

Before you finish, verify: every index link resolves to an anchor that
exists, no table has ragged rows, and no em dashes remain. Tell me if any
content in the source was unclear or unreadable rather than guessing.

Target folder: [path]

Return the file, placement commands, index rows, and commit message.
```

---

## Prompt: reviewing what you already have

<a id="prompt-reviewing-what-you-already-have"></a>

Useful every few months as maintenance.

```text
[attach _meta/adding-content.md and the note you want reviewed]

Review this note against the attached repo conventions. Check for:

- Em dashes
- Headings missing explicit anchors
- Index entries that do not match the actual headings
- Absolute GitHub URLs where relative paths belong
- Code blocks without a language tag
- Maths written as LaTeX instead of fenced text blocks
- Concepts that skip a stage: missing plain English, missing worked
  example, missing code, or missing the production use case
- Claims that should carry a [SOURCE] or [FIRST-HAND] tag

List what needs fixing, with the exact replacement text for each. Do not
rewrite the whole file unless I ask.
```

---

## After the LLM responds

<a id="after-the-llm-responds"></a>

Do not push straight away. Run the checks from
[adding-content.md](adding-content.md#verification-commands):

```cmd
cd /d "C:\Users\Asfiya\Downloads\Deep-Learning-Wiki"
findstr /s /m /c:"—" *.md
git status
git diff --stat
```

Then read the file yourself, checking three things specifically:

1. **Do the relative paths have the right number of `../`?** This is the most
   common LLM error and it is invisible until you click the link on GitHub.
2. **Does the internal index match the actual headings?** Models sometimes write
   the index from the outline they planned rather than the file they produced.
3. **Does the code actually run?** Run it. Do not trust it because it looks
   right.

Then register it in the indexes and commit.

---

## What LLMs get wrong here

<a id="what-llms-get-wrong-here"></a>

Knowing the failure modes tells you where to look.

| Failure | Why it happens | How to catch it |
| --- | --- | --- |
| Wrong number of `../` in a path | The model does not know the file's real depth | Click every link after pushing, or run a link check |
| Index does not match headings | Written from a plan, not the finished file | Compare the index against `findstr /r "^### "` output |
| Em dashes reappear | Strong default habit, survives one instruction | `findstr /s /m /c:"—" *.md` |
| Forgets the index rows entirely | Not asked for them explicitly | Always include point 3 in the prompt |
| Invents plausible but wrong numbers | Filling gaps rather than admitting uncertainty | Verify any benchmark figure or parameter count against the source |
| Absolute GitHub URLs | Common pattern in training data | Search the diff for `github.com` |
| Code that looks right and does not run | No execution feedback | Run it |
| Summarises when asked to convert | Default to compression | State "preserve all content, do not summarise" |

The pattern: models are good at following format rules you state explicitly, and
poor at anything requiring knowledge of your actual file tree. State the format
rules, verify the paths yourself.

---

## Related documents

- [adding-content.md](adding-content.md), the full workflow this supports
- [conventions.md](conventions.md), the formatting rules to attach
- [CONTRIBUTING.md](../CONTRIBUTING.md), the short version
- [Master index](../INDEX.md)
