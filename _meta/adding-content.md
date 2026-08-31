# Adding Content

> **Read this before adding anything to the repository.**
> It covers every content type, the exact commands, the structure rules, and the
> checks that stop the repository from decaying as it grows.

The repository is only useful if everything in it follows the same shape. This
document is what keeps that true when you are adding a note at 1am six months
from now and cannot remember the rules.

---

## Index

- **[The five minute version](#the-five-minute-version)**
- **[Before you start: the git workflow](#before-you-start-the-git-workflow)**
  - [Every session starts and ends the same way](#every-session-starts-and-ends-the-same-way)
  - [Working on a branch](#working-on-a-branch)
  - [Commit message format](#commit-message-format)
- **[Deciding where content goes](#deciding-where-content-goes)**
  - [The decision table](#the-decision-table)
  - [When nothing fits](#when-nothing-fits)
  - [Creating a new section](#creating-a-new-section)
- **[Type 1: A topic note](#type-1-a-topic-note)**
- **[Type 2: A research paper note](#type-2-a-research-paper-note)**
- **[Type 3: A Jupyter notebook](#type-3-a-jupyter-notebook)**
- **[Type 4: Code and from-scratch implementations](#type-4-code-and-from-scratch-implementations)**
- **[Type 5: Linking to another repository](#type-5-linking-to-another-repository)**
- **[Type 6: Images and diagrams](#type-6-images-and-diagrams)**
- **[Type 7: A resource link](#type-7-a-resource-link)**
- **[Interlinking rules](#interlinking-rules)**
  - [Relative paths](#relative-paths)
  - [Anchors](#anchors)
  - [Bidirectional linking](#bidirectional-linking)
- **[Registering content in the indexes](#registering-content-in-the-indexes)**
- **[Formatting rules](#formatting-rules)**
- **[The pre-commit checklist](#the-pre-commit-checklist)**
- **[Verification commands](#verification-commands)**
- **[Common mistakes](#common-mistakes)**
- **[Quick command reference](#quick-command-reference)**

---

## The five minute version

<a id="the-five-minute-version"></a>

Every single addition, regardless of type, follows these six steps:

```text
1. PULL      get the latest state
2. PLACE     copy the right template into the right folder
3. WRITE     fill it in, following the four stage structure
4. REGISTER  add it to the folder INDEX.md and the root INDEX.md
5. LINK      add cross links in both directions
6. VERIFY    run the checks, then commit and push
```

Step 4 is the one people skip. A note that exists but is not registered in an
index is a note you will never find again. If you are short on time, write less
but always register it.

---

## Before you start: the git workflow

<a id="before-you-start-the-git-workflow"></a>

All commands below are for Windows Command Prompt, since that is the setup in
use. On Mac or Linux the git commands are identical; only the file commands
(`copy`, `mkdir`, `dir`) change to `cp`, `mkdir -p`, `ls`.

### Every session starts and ends the same way

<a id="every-session-starts-and-ends-the-same-way"></a>

Open Command Prompt and navigate to the repository:

```cmd
cd /d "C:\Users\Asfiya\Downloads\Deep-Learning-Wiki"
```

Always pull before you write anything:

```cmd
git pull origin main
```

Do your work. Then, at the end:

```cmd
git status
git add .
git commit -m "Add note on batch normalization"
git push origin main
```

Run `git status` before `git add .` every time. It shows exactly what you are
about to commit. Ten seconds of reading it prevents committing a 2GB model
checkpoint or a half-finished draft.

### Working on a branch

<a id="working-on-a-branch"></a>

For anything larger than a single note, work on a branch so `main` always stays
in a clean state:

```cmd
git checkout -b add-transformer-note
```

Write, commit as often as you like, then merge back:

```cmd
git checkout main
git pull origin main
git merge add-transformer-note
git push origin main
git branch -d add-transformer-note
```

For a single small note, committing straight to `main` is fine. Do not
overengineer a personal repository.

### Commit message format

<a id="commit-message-format"></a>

One line, present tense, saying what changed:

| Situation | Message |
| --- | --- |
| New topic note | `Add note on batch normalization` |
| New paper note | `Add paper note: Attention Is All You Need` |
| New notebook | `Add backprop-from-scratch notebook` |
| Editing existing | `Expand ResNet section with bottleneck maths` |
| Fixing something | `Fix broken anchor links in fundamentals index` |
| Index updates only | `Update indexes for new NLP notes` |

Avoid `update`, `changes`, `wip`, and `asdf`. Six months from now the git log is
how you reconstruct what you were doing.

---

## Deciding where content goes

<a id="deciding-where-content-goes"></a>

### The decision table

<a id="the-decision-table"></a>

| What you have | Where it goes | Template |
| --- | --- | --- |
| Explanation of a concept | `NN-section/subfolder/topic-name.md` | `TOPIC_TEMPLATE.md` |
| Summary of a paper you read | `papers/notes/year-author-title.md` | `PAPER_NOTE_TEMPLATE.md` |
| A Jupyter notebook | `implementations/notebooks/` | `IMPLEMENTATION_TEMPLATE.md` for its README |
| Python scripts implementing something | `implementations/from-scratch/name/` | `IMPLEMENTATION_TEMPLATE.md` |
| A link to someone else's repo | Inside the relevant topic note, plus `resources/tools.md` | none |
| A link to your own other repo | The relevant topic note, plus root `INDEX.md` | none |
| A diagram or screenshot | `assets/diagrams/` or `assets/images/` | none |
| A course, book, or dataset | `resources/courses.md`, `books.md`, `datasets.md` | none |
| A rule about how the repo works | `_meta/conventions.md` | none |

### When nothing fits

<a id="when-nothing-fits"></a>

If a piece of content does not fit any row above, that is a signal, not a
problem. Work through these in order:

1. **Can it be split?** "Notes on my CV project" is usually three things: a
   topic note on the technique, a notebook, and a paper note. Split it.
2. **Does it need a new subfolder?** If you have three or more notes that belong
   together inside an existing section, make a subfolder.
3. **Does it need a new section?** Only if it is a whole domain that does not
   exist yet, like graph neural networks.

Never drop a file at the repository root. The root holds only `README.md`,
`INDEX.md`, `CONTRIBUTING.md`, and `LICENSE.md`. Anything else there is a sign
the decision was skipped.

### Creating a new section

<a id="creating-a-new-section"></a>

New domains get the next free number, starting at `11`. Never renumber existing
folders, because every link in the repository would break.

```cmd
mkdir 11-graph-neural-networks
mkdir 11-graph-neural-networks\fundamentals
mkdir 11-graph-neural-networks\architectures
copy _templates\INDEX_TEMPLATE.md 11-graph-neural-networks\INDEX.md
```

Then add a `.gitkeep` to each empty subfolder, or git will not track them:

```cmd
type nul > 11-graph-neural-networks\fundamentals\.gitkeep
type nul > 11-graph-neural-networks\architectures\.gitkeep
```

Fill in the `INDEX.md`, then register the new section in the root `INDEX.md` and
in the structure diagram in `README.md`.

---

## Type 1: A topic note

<a id="type-1-a-topic-note"></a>

The most common addition. A topic note explains one concept properly.

**Step 1. Create the subfolder if it does not exist.**

```cmd
mkdir 04-natural-language-processing\transformers
```

**Step 2. Copy the template.**

```cmd
copy _templates\TOPIC_TEMPLATE.md 04-natural-language-processing\transformers\attention-mechanism.md
```

Filename rules: lowercase, hyphens, describes the topic not the source.
`attention-mechanism.md` is correct. `Attention.md`, `week5notes.md`, and
`attention_mechanism.md` are all wrong.

**Step 3. Fill in the front matter.**

```markdown
> **Source:** Vaswani et al. 2017, plus CS224n lecture 8
> **Level:** intermediate
> **Prerequisites:** [RNNs](../../05-sequence-models/rnn-lstm-gru/rnn-basics.md), matrix multiplication
> **Covers:** scaled dot-product attention, multi-head attention, self-attention, masking
> **Related:** [Transformer architecture](transformer-architecture.md)
```

The `Prerequisites` line is the one that makes the repository usable by a
beginner. Fill it in honestly, and link to the notes that cover each item.

**Step 4. Write the body in four stages.**

Every concept in the note moves through the same sequence:

| Stage | What goes in it |
| --- | --- |
| **Plain English** | The problem and the idea, with zero notation. If a beginner cannot follow this part, the rest is wasted |
| **The maths** | The actual equations, every symbol defined in a table, and at least one worked numeric example |
| **The intuition** | Why the maths works, in words. This is the part people remember |
| **Implementation** | Code that runs. Not pseudocode |

Then close with **In practice**, naming concrete production systems where the
technique is used, and **Common mistakes**.

**Step 5. Add the internal index.**

Write it after the content is finished, so it matches the real headings. Put an
explicit anchor under every heading:

```markdown
## Index

- **[Scaled dot-product attention](#scaled-dot-product-attention)**
  - [The maths](#scaled-dot-product-the-maths)
- **[Multi-head attention](#multi-head-attention)**

---

## Scaled dot-product attention

<a id="scaled-dot-product-attention"></a>
```

Anchors are mandatory on any document longer than two screens. Details in
[Anchors](#anchors) below.

**Step 6. Register and commit.**

```cmd
git add .
git commit -m "Add note on attention mechanism"
git push origin main
```

---

## Type 2: A research paper note

<a id="type-2-a-research-paper-note"></a>

A paper note is not a summary of the abstract. It records what you would need to
reimplement or argue about the paper a year from now.

**Step 1. Copy the template with the correct filename.**

Format: `year-firstauthor-short-title.md`, all lowercase.

```cmd
copy _templates\PAPER_NOTE_TEMPLATE.md papers\notes\2017-vaswani-attention-is-all-you-need.md
```

Year first keeps the folder chronological, which is how a subfield actually
makes sense when you read back through it.

**Step 2. Write the one line contribution first.**

Before anything else in the file, write the single idea the paper contributed.
If you cannot write it in one line, you have not understood the paper yet. Go
read it again rather than writing a vague note you will not trust later.

**Step 3. Fill in the rest.**

The template sections exist for a reason:

| Section | Why it matters |
| --- | --- |
| The problem | What was broken before. Without this the contribution is meaningless |
| The method in detail | Enough to reimplement without reopening the paper |
| Results worth remembering | Which numbers matter and which are noise |
| What I would question | Weak ablations, results that may not hold at other scales |
| What I would use | Which parts transfer to your own work, and under what conditions |
| Implementation notes | Gotchas, buried appendix hyperparameters, things the official code does that the paper never mentions |

The last two are what make the note worth having. Anyone can summarise a paper;
only you can record what you would actually do with it.

**Step 4. Do not commit the PDF.**

Link to the arXiv or publisher page instead. The `.gitignore` already excludes
`papers/pdfs/`, so you can keep local copies without them entering version
control. PDFs bloat the repository and are usually not yours to redistribute.

```cmd
mkdir papers\pdfs
```

Anything you put there stays local.

**Step 5. Register it in three places.**

1. The reading log table in `papers/INDEX.md`
2. The "By topic" table in the same file
3. The related topic note, under Further reading

**Step 6. Commit.**

```cmd
git add .
git commit -m "Add paper note: Attention Is All You Need"
git push origin main
```

---

## Type 3: A Jupyter notebook

<a id="type-3-a-jupyter-notebook"></a>

Notebooks need more care than markdown, because they store output as well as
code and can silently become enormous.

**Step 1. Clear all output before committing.**

This is the single most important rule for notebooks. In Jupyter:
`Kernel > Restart Kernel and Clear All Outputs`. Then save.

Why it matters:

- A notebook with plots and images embedded can be 50MB. The same notebook
  cleared is 30KB
- Output is stored as base64 blobs, so every rerun creates a huge, unreadable
  diff even when you changed one line
- Cleared notebooks are reviewable in a git diff; uncleared ones are not

**Step 2. Place it.**

```cmd
copy "C:\path\to\your\notebook.ipynb" implementations\notebooks\backprop-from-scratch.ipynb
```

Same filename rules: lowercase, hyphens, descriptive.

**Step 3. Give it a README if it is more than one notebook.**

For a multi-file implementation, create a folder:

```cmd
mkdir implementations\from-scratch\backprop
copy _templates\IMPLEMENTATION_TEMPLATE.md implementations\from-scratch\backprop\README.md
```

**Step 4. Make the notebook self-documenting.**

A notebook that is just code cells is not a note. It needs:

- A markdown cell at the top: what this builds, what you should understand after
  working through it, and its prerequisites with links
- A markdown cell before each section explaining what happens next and why
- A cell at the end stating the expected output, so a reader knows whether their
  run is correct

**Step 5. Pin the dependencies.**

```cmd
type nul > implementations\notebooks\requirements.txt
```

List the packages with versions. A notebook that only ran on the exact library
versions you had that week is not reproducible.

**Step 6. Link it to a topic note.**

A notebook without an explanation is a snippet. In the related topic note, under
Implementation, add:

```markdown
Runnable version: [backprop from scratch](../../implementations/notebooks/backprop-from-scratch.ipynb)
```

And in the notebook's opening cell, link back to the topic note. Both directions,
always.

**Step 7. Check the size, then commit.**

```cmd
dir implementations\notebooks\*.ipynb
```

If any notebook is over about 1MB, you did not clear the output. Go back to
step 1.

```cmd
git add .
git commit -m "Add backprop-from-scratch notebook"
git push origin main
```

---

## Type 4: Code and from-scratch implementations

<a id="type-4-code-and-from-scratch-implementations"></a>

For Python scripts rather than notebooks.

**Step 1. One folder per implementation.**

```cmd
mkdir implementations\from-scratch\resnet-cifar10
cd implementations\from-scratch\resnet-cifar10
```

**Step 2. Standard layout.**

```text
resnet-cifar10/
├── README.md          from IMPLEMENTATION_TEMPLATE.md
├── requirements.txt   pinned versions
├── model.py           the architecture
├── data.py            loading and augmentation
└── train.py           the training loop
```

**Step 3. Write the README before the code is finished.**

The template asks for a walkthrough: explain the step, show the code, explain
what just happened. That structure forces you to justify each design decision,
and it is what makes the implementation teach rather than just run.

The **Things that went wrong** section at the end is the most valuable part six
months later. Record the real debugging: the shape error that took an hour, the
learning rate that silently destroyed the pretrained weights, the augmentation
that broke the labels.

**Step 4. Never commit model weights or data.**

The `.gitignore` already blocks `*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.onnx`,
`*.safetensors`, `data/`, `datasets/`, `runs/`, and `wandb/`.

Verify before committing:

```cmd
git status
```

If you see a `.pt` file listed, something is wrong with the ignore rules. Stop
and fix it rather than committing a 500MB file, which is very painful to remove
from git history afterwards.

**Step 5. State the expected output.**

```markdown
## Expected output

```text
epoch 1/10  train_loss 2.301  val_acc 0.112
epoch 10/10 train_loss 0.284  val_acc 0.891
```

Runtime: about 12 minutes on a T4 GPU, about 90 minutes on CPU.
```

Without this, a reader cannot tell whether their run succeeded.

---

## Type 5: Linking to another repository

<a id="type-5-linking-to-another-repository"></a>

Three different cases, handled three different ways.

### Case A: Referencing someone else's repo

Just link to it in prose, inside the relevant topic note:

```markdown
The reference implementation is at
[huggingface/transformers](https://github.com/huggingface/transformers).
```

If it is a tool you would recommend generally, also add a row to
`resources/tools.md` with a note on what it is for. Do not add every repo you
come across; the resources files are curated, not comprehensive.

### Case B: Linking to your own other repository

Add it to the root `INDEX.md` under a "Related repositories" section, and link
it from the topic note it relates to:

```markdown
## Related repositories

| Repository | What it holds |
| --- | --- |
| [DevByAsfiya/some-project](https://github.com/DevByAsfiya/some-project) | Production implementation of the CV pipeline described in [03-computer-vision](03-computer-vision/INDEX.md) |
```

Always say what the other repo contains and how it relates. A bare link is not
useful.

### Case C: Actually including another repository

Two options, and the choice matters.

**Option 1, a plain link.** Almost always the right answer. No coupling, nothing
to maintain, works forever.

**Option 2, a git submodule.** Only if you genuinely need the code present in
your working tree at a pinned commit:

```cmd
git submodule add https://github.com/someone/some-repo.git external/some-repo
git commit -m "Add some-repo as submodule"
git push origin main
```

Anyone cloning then needs:

```cmd
git clone --recurse-submodules https://github.com/DevByAsfiya/Deep-Learning-Wiki.git
```

Submodules are a genuine maintenance burden: they confuse clones, complicate
pulls, and break when the upstream repo moves. For a knowledge base, a link is
nearly always better. Only reach for a submodule when you need reproducibility
against a specific commit.

**Never copy someone else's repository into yours.** It breaks their license
attribution, bloats your history, and goes stale immediately.

---

## Type 6: Images and diagrams

<a id="type-6-images-and-diagrams"></a>

**Preference order, best first:**

1. **ASCII diagram in a fenced `text` block.** Diffs cleanly in git, readable in
   any editor, no files to manage, never breaks.
2. **Mermaid**, when the structure genuinely needs graph layout. GitHub renders
   it natively.
3. **An image file**, only when neither of the above works.

An ASCII diagram that survives a copy paste into a terminal is worth more than a
prettier image that rots.

**For image files:**

```cmd
copy "C:\path\to\diagram.png" assets\diagrams\resnet-skip-connection.png
```

Reference it with a relative path from the note:

```markdown
![ResNet skip connection](../../assets/diagrams/resnet-skip-connection.png)
```

Rules:

| Rule | Reason |
| --- | --- |
| PNG for diagrams, JPG for photos | PNG keeps text sharp |
| Keep files under about 500KB | Repository size compounds |
| Descriptive filenames, lowercase and hyphenated | `resnet-skip-connection.png`, not `Screenshot 2026-08-31.png` |
| Always write alt text in the brackets | Accessibility, and it shows when the image fails to load |
| Never screenshot a figure from a paper | Copyright. Redraw it or describe it |

---

## Type 7: A resource link

<a id="type-7-a-resource-link"></a>

Courses, books, datasets, and tools go in the matching file in `resources/`.

Add a row to the existing table, keeping the same columns:

```markdown
| CS285 (Berkeley) | Deep reinforcement learning | advanced | Assignments are the value. Lectures assume solid probability |
```

The last column is the point. A list of links is worthless; a list of links with
your judgement attached is worth keeping. Say who it is for, what it assumes,
and whether it was worth your time.

---

## Interlinking rules

<a id="interlinking-rules"></a>

### Relative paths

<a id="relative-paths"></a>

Always relative, never absolute GitHub URLs:

```markdown
GOOD: [Fundamentals](../01-foundations/neural-networks/deep-learning-fundamentals.md)
BAD:  [Fundamentals](https://github.com/DevByAsfiya/Deep-Learning-Wiki/blob/main/...)
```

Relative links work when the repository is cloned, browsed on GitHub, or opened
in Obsidian or VS Code. Absolute URLs work in exactly one of those, and break if
you ever rename the repository.

Counting the `../` correctly:

| From | To | Path |
| --- | --- | --- |
| Root `INDEX.md` | `03-computer-vision/INDEX.md` | `03-computer-vision/INDEX.md` |
| `03-computer-vision/INDEX.md` | Root `INDEX.md` | `../INDEX.md` |
| `03-computer-vision/architectures/note.md` | Root `INDEX.md` | `../../INDEX.md` |
| `03-computer-vision/architectures/note.md` | `01-foundations/neural-networks/other.md` | `../../01-foundations/neural-networks/other.md` |

One `../` per folder level you need to climb.

### Anchors

<a id="anchors"></a>

Always write explicit anchors. Never rely on auto-generated ones:

```markdown
### The maths

<a id="attention-the-maths"></a>
```

Three reasons this matters:

1. **Repeated headings break.** Five sections called "Implementation" generate
   `#implementation`, `#implementation-1`, `#implementation-2`. Reorder the
   document and every link silently points somewhere wrong.
2. **Renderers disagree.** GitHub, Obsidian, and static site generators slug
   headings differently, especially with punctuation or non-ASCII characters.
3. **Rewording a heading breaks auto anchors.** An explicit anchor survives.

Prefix anchors with the topic when a heading name is generic, so
`attention-the-maths` rather than `the-maths`.

Linking to an anchor in another file:

```markdown
[the bottleneck calculation](../03-computer-vision/architectures/cnn-classic-architectures.md#8-the-computational-cost-problem-and-the-bottleneck-fix)
```

### Bidirectional linking

<a id="bidirectional-linking"></a>

**If A links to B, B links back to A.** One way links rot into dead ends and
make the repository feel like a pile rather than a web.

When you add a note, ask: which existing notes should mention this one? Then go
and edit them. It takes two minutes and it is the difference between a wiki and
a folder of files.

Use the `Related` line in the front matter for the strongest one or two
connections, and inline prose links for everything else.

---

## Registering content in the indexes

<a id="registering-content-in-the-indexes"></a>

Every new file gets registered in **two or three** places. This is not optional.

**1. The folder `INDEX.md`.** Add a row to the Contents table:

```markdown
| [Attention mechanism](transformers/attention-mechanism.md) | Scaled dot-product and multi-head attention | intermediate | Complete |
```

Also update the Subfolders table status from `Stub` to `1 document` or a count.

**2. The root `INDEX.md`.** Add a row under the matching section:

```markdown
| [Attention mechanism](04-natural-language-processing/transformers/attention-mechanism.md) | Scaled dot-product and multi-head attention | Complete |
```

And change the section line from "Scaffolded" once it has content.

**3. The topic lookup table at the bottom of root `INDEX.md`.** One row per major
concept, alphabetical:

```markdown
| Attention | [Attention mechanism](04-natural-language-processing/transformers/attention-mechanism.md) |
```

This table is the fastest way to find anything. Keep it current or it stops
being trusted, and once it stops being trusted nobody uses it.

**Status values:** `Complete` (written and reviewed), `Draft` (written, not
reviewed), `Stub` (placeholder).

---

## Formatting rules

<a id="formatting-rules"></a>

| Element | Rule |
| --- | --- |
| **Em dashes** | Never. Use a comma, a colon, or start a new sentence |
| **Code blocks** | Always language tagged: `python`, `bash`, `cmd`, `json`, `yaml`, or `text` |
| **Maths, ASCII art, directory trees, program output** | Fenced `text` blocks, never LaTeX |
| **Inline maths** | Backticks: `z = Wx + b` |
| **Tables** | Header row always. One sentence per cell. Longer explanations go in prose |
| **Filenames** | Lowercase, hyphens, descriptive of the topic not the source |
| **Prose** | Plain and direct. Short sentences. Define jargon at first use |

**Why no LaTeX:** GitHub renders it inconsistently depending on context, and the
notes should read correctly in a plain text editor. Fenced `text` blocks also
stop markdown from mangling expressions like `w**2` or `L = -log(y_hat)`.

**Claim tagging.** On claims a reader might want to verify:

| Tag | Means |
| --- | --- |
| `[SOURCE]` | Stated in a specific paper, book, or docs page. Cite it |
| `[FIRST-HAND]` | You observed this while building something |
| `[INFERENCE]` | You reasoned it from other facts. Could be wrong |
| `[ASSUMPTION]` | Working assumption, not yet verified |

Use these only where being wrong would matter. Tagging every sentence makes the
document unreadable and the tags meaningless.

---

## The pre-commit checklist

<a id="the-pre-commit-checklist"></a>

Run through this before every push:

- [ ] Front matter is complete and the prerequisites are honest
- [ ] Internal index matches the actual headings in the file
- [ ] Every heading has an explicit `<a id="..."></a>` anchor
- [ ] Every internal link resolves
- [ ] Code blocks are language tagged and the code runs
- [ ] Notebooks have cleared output and are under 1MB
- [ ] No model weights, datasets, or PDFs staged
- [ ] Registered in the folder `INDEX.md`
- [ ] Registered in the root `INDEX.md`
- [ ] Added to the topic lookup table
- [ ] Cross links added in both directions
- [ ] No em dashes
- [ ] `git status` reviewed before `git add .`

---

## Verification commands

<a id="verification-commands"></a>

**Count what is tracked:**

```cmd
git ls-files | find /c /v ""
```

**Find notes that exist but may not be registered:**

```cmd
dir /s /b *.md | findstr /v /i "INDEX.md README.md CONTRIBUTING.md LICENSE.md _templates _meta resources"
```

Compare that list against the tables in root `INDEX.md`. Anything present in one
and missing from the other needs fixing.

**Check for em dashes:**

```cmd
findstr /s /m /c:"—" *.md
```

Should print nothing. Any filename it prints needs fixing.

**Check for large files before committing:**

```cmd
forfiles /s /m *.* /c "cmd /c if @fsize GTR 1000000 echo @path @fsize"
```

Anything over 1MB should be deliberate. Notebooks over 1MB mean uncleared
output.

**See the headings inside a note:**

```cmd
findstr /r "^#### " 01-foundations\neural-networks\deep-learning-fundamentals.md
```

**Check what changed before committing:**

```cmd
git status
git diff --stat
```

---

## Common mistakes

<a id="common-mistakes"></a>

| Mistake | What happens | Fix |
| --- | --- | --- |
| Writing a note but not registering it | You never find it again | Always do step 4 |
| Committing a notebook with output | Repo bloats, diffs become unreadable | Restart kernel and clear all outputs first |
| `git add *` instead of `git add .` | Every `.gitkeep` and `.gitignore` is skipped, empty folders vanish | Always use `git add .` |
| Absolute GitHub URLs for internal links | Break on clone and on rename | Relative paths only |
| Relying on auto-generated anchors | Silently point to the wrong section after a reorder | Explicit `<a id>` tags |
| One way links | The repo becomes a pile of files, not a wiki | Link both directions |
| Creating a new folder without `.gitkeep` | The empty folder disappears on push | `type nul > folder\.gitkeep` |
| Renumbering sections | Every link in the repo breaks | Append `11`, `12`, never renumber |
| Committing a model checkpoint | Very painful to remove from history later | Check `git status` before every add |
| Filenames like `week5-notes.md` | Unfindable in six months | Name the topic, not the source |
| Letting status columns go stale | The indexes stop being trusted | Update indexes in the same commit as the content |

---

## Quick command reference

<a id="quick-command-reference"></a>

```cmd
REM Start a session
cd /d "C:\Users\Asfiya\Downloads\Deep-Learning-Wiki"
git pull origin main

REM New topic note
mkdir SECTION\SUBFOLDER
copy _templates\TOPIC_TEMPLATE.md SECTION\SUBFOLDER\topic-name.md

REM New paper note
copy _templates\PAPER_NOTE_TEMPLATE.md papers\notes\2017-vaswani-attention.md

REM New implementation
mkdir implementations\from-scratch\name
copy _templates\IMPLEMENTATION_TEMPLATE.md implementations\from-scratch\name\README.md

REM New section (next free number, never renumber)
mkdir 11-new-domain
mkdir 11-new-domain\subfolder
type nul > 11-new-domain\subfolder\.gitkeep
copy _templates\INDEX_TEMPLATE.md 11-new-domain\INDEX.md

REM Verify
git status
findstr /s /m /c:"—" *.md
git ls-files | find /c /v ""

REM Commit and push
git add .
git commit -m "Add note on TOPIC"
git push origin main
```

---

## Related documents

- [CONTRIBUTING.md](../CONTRIBUTING.md), the shorter version of these rules
- [conventions.md](conventions.md), formatting and naming in detail
- [taxonomy.md](taxonomy.md), the controlled tag vocabulary
- [roadmap.md](roadmap.md), what to write next
- [Master index](../INDEX.md)
