# How to Use This Repository

> **Level:** everyone
> **Prerequisites:** none

If you are new here, read this page first. It takes about five minutes and will
save you from wandering.

---

## Index

- [What this is](#what-this-is)
- [How documents are organised](#how-documents-are-organised)
- [Three ways to read](#three-ways-to-read)
- [How to actually learn from this](#how-to-actually-learn-from-this)

---

## What this is

<a id="what-this-is"></a>

A structured wiki covering deep learning from first principles through to
production systems. It holds four kinds of material:

| Kind | Where | What it gives you |
| --- | --- | --- |
| Topic notes | numbered folders `01` to `10` | The concepts, explained and worked through |
| Paper notes | `papers/` | What a paper contributed, and whether it still matters |
| Implementations | `implementations/` | Code you can run and modify |
| Resources | `resources/` | Courses, books, datasets, tooling |

## How documents are organised

<a id="how-documents-are-organised"></a>

Numbered folders suggest a reading order. `00` orients you, `01` and `02` build
foundations everyone needs, and `03` onward are domain specialisations you can
enter in any order once the foundations are in place.

Every folder has an `INDEX.md` listing its contents. Every document longer than
a couple of screens has its own index at the top, so you can jump straight to a
subsection instead of scrolling.

Every topic note opens with a front matter block:

```text
> Source:        where this came from
> Level:         beginner, intermediate, or advanced
> Prerequisites: what you need to know first
> Covers:        the concepts inside
> Related:       other documents worth reading alongside
```

Read the prerequisites line before starting. Most confusion comes from opening a
document two steps ahead of where you are.

## Three ways to read

<a id="three-ways-to-read"></a>

**Sequentially,** if you are learning from scratch. Follow
[learning-paths.md](learning-paths.md).

**By search,** if you have a specific question. Use the
[master index](../INDEX.md), or your editor's search across the repository. Every
term is defined in [_meta/glossary.md](../_meta/glossary.md).

**By problem,** if you are building something. Start at the domain folder that
matches your task, read its `INDEX.md`, and follow links from there.

## How to actually learn from this

<a id="how-to-actually-learn-from-this"></a>

Reading notes produces the feeling of understanding without the substance of it.
Four habits that close that gap:

1. **Predict before you run.** Before executing any code block, write down what
   you expect the output shape or the loss curve to look like. Being wrong is
   where the learning happens.
2. **Implement the smallest version.** Every architecture note has an
   implementation section. Type it out rather than copying it. Typing forces you
   through every line.
3. **Break it on purpose.** Remove the skip connection, set the learning rate to
   1.0, initialise the weights to zero. Watching a known failure mode occur
   teaches you more than reading about it.
4. **Write your own note.** When you learn something new, add it here in the
   template format. Writing for a future reader is the strongest test of whether
   you understood it.
