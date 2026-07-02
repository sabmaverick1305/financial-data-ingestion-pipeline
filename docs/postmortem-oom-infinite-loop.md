# Post-Mortem: Three Weeks of OOM Crashes Traced to Three Lines of Python

**Severity:** P0 — pipeline completely non-functional  
**Duration:** Multiple weeks across local and cloud environments  
**Root cause:** Infinite loop in `chunk_text()` caused unbounded memory growth  
**Fix size:** 1 line added, 1 line moved  

---

## The Short Version

Our document processing pipeline was killed by the operating system on every
single run — on local machines, on ECS Fargate at 7 GB, at 8 GB, and at 16 GB.
Every investigation pointed at Docling, the ML library we used for PDF table
extraction. We spent weeks scaling memory, rebuilding Docker images, switching
runtimes, and reading Docling internals.

The real cause was in `chunker.py` — a 50-line utility file with no ML
dependencies. A `while` loop that could never terminate. Three lines of Python
that consumed every byte of RAM given to them, no matter how much that was.

---

## Background

We built a pipeline to ingest AMFI India's financial research documents — 447
PDFs and spreadsheets covering monthly mutual fund reports from 2009 to 2025.
The pipeline had four stages:

1. **Text extraction** — PyMuPDF reads page text from PDFs, detects whether the
   document has a selectable text layer
2. **Table extraction** — Docling (a CPU-only torch model) extracts tables,
   figures, and structured markdown
3. **Chunking** — a pure-Python function splits the extracted text into
   overlapping 1000-character windows for downstream RAG indexing
4. **Upload** — artifacts written to S3, status updated in RDS PostgreSQL

The pipeline ran as a single ECS Fargate task. On every document it processed,
the task was killed with **exit code 137** — the Linux signal for an OOM kill.

---

## Timeline

```
Week 1, Day 1   First OOM kill in local Docker (4 GB container)
Week 1, Day 2   Increased to 8 GB. Still killed.
Week 1, Day 3   Rebuilt image without CUDA. Still killed.
Week 1, Day 4   Pinned Docling version. Still killed.
Week 1, Day 5   Tried pdfplumber instead of Docling. Still killed.
Week 2, Day 1   Deployed to ECS Fargate at 7 GB. Still killed.
Week 2, Day 2   Deployed to ECS Fargate at 8 GB. Still killed.
Week 2, Day 3   Deployed to ECS Fargate at 16 GB. Still killed.
Week 2, Day 4   Investigated MPIRE (multiprocessing) as culprit. No signal.
Week 2, Day 5   Split into separate workers by memory profile.
Week 3, Day 1   Lightweight chunk-worker OOMs on its first document.
Week 3, Day 1   Investigate chunk_text(). Find infinite loop. Fix in 60 seconds.
Week 3, Day 1   Pipeline processes all 447 documents without a single crash.
```

---

## The Investigation — What We Got Wrong

### Theory 1: Docling is leaking memory

This seemed reasonable. Docling loads large PyTorch models — layout detection,
table structure recognition, optional OCR. At startup it consumes ~2–3 GB of
RAM just loading weights. The first thing we measured was peak RSS after
conversion:

```
docling_extractor.before_convert  rss_mb=553
docling_extractor.after_convert   rss_mb=1615
```

That 1 GB growth per document looked like a leak. We tried:
- Calling `gc.collect()` after each conversion
- Deleting the result object explicitly (`del result, doc, stream`)
- Using a class-level cached converter (`_converter`) so models load once
- Setting `DOCLING_THREADS=1` to eliminate multiprocessing overhead
- Switching Docling's accelerator to CPU-only mode

RSS stabilised. Docling wasn't leaking. The OOM still happened.

### Theory 2: The Docker image includes CUDA libraries

The original `torch` installation pulled in CUDA libraries — over 6 GB of
unnecessary GPU libraries in a CPU-only container. We rebuilt with:

```dockerfile
RUN pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
```

Image dropped from ~8 GB to ~2.3 GB. The OOM still happened.

### Theory 3: MPIRE (multiprocessing) is spawning untracked child processes

Docling's dependency tree includes `mpire`, a multiprocessing library. Child
processes don't inherit the parent's RSS measurement. We searched for any
code path that spawned subprocesses during conversion. We found none that were
active in our CPU-only configuration. The OOM still happened.

### Theory 4: The container needs more memory

We escalated from 4 GB → 8 GB → 16 GB. ECS Fargate allows up to 120 GB.
At each threshold, the task was killed before completing a single document. The
kill happened at roughly the same wall-clock time regardless of how much memory
was available. That was the first real clue — but we missed it.

**What we should have asked:** "Why does the kill take the same amount of time
regardless of memory?" The answer, in retrospect, is obvious: an infinite loop
consuming memory at a fixed rate will exhaust any finite supply — it just takes
proportionally longer.

### Theory 5: olmOCR as an alternative

At this point, morale was low and the hypothesis was "Docling is fundamentally
broken in our environment." We attempted to replace it with olmOCR, a 7B
vision-language model from Allen AI. We spent a week:
- Researching the GitHub repo and its inference API
- Testing DeepInfra (model removed from their catalog)
- Attempting to self-host on an AWS GPU instance (blocked by account-level
  Free Tier restriction)
- Testing Cirrascale (TCP handshake timeout — network unreachable)
- Testing Parasail (no olmOCR in their live model catalog)

All four paths were dead ends, verified live. We reverted to Docling.

---

## The Architectural Change That Found the Bug

After the olmOCR detour, we redesigned the pipeline into four separate workers,
each running in its own ECS task definition with an appropriate memory budget:

| Worker | Memory | Runtime |
|---|---|---|
| text-worker | 2 GB | PyMuPDF + pandas |
| table-worker | 16 GB | Docling (CPU) |
| ocr-worker | 16 GB | Docling (CPU + OCR) |
| chunk-worker | 2 GB | Pure Python |

The motivation was not debugging. It was operational: right-size the memory for
each stage so the expensive Docling stage doesn't hold up the cheap text
extraction stage in a shared container.

We deployed the four workers. The text-worker ran cleanly on all 447 documents.
The table-worker processed several batches without crashing.

Then the chunk-worker ran on its first document. **It was killed with exit 137.**

A 2 GB container running **pure Python** — no PyTorch, no Docling, no
subprocesses, no C extensions beyond the Python runtime itself — was killed for
running out of memory.

This was the diagnostic breakthrough. Not because we immediately knew the cause,
but because it made every previous theory impossible.

- Docling can't cause this — the chunk-worker doesn't import Docling.
- CUDA libraries can't cause this — they're not in this image.
- MPIRE can't cause this — there is no multiprocessing in chunking.
- Memory limits can't cause this — 2 GB is enough for any reasonable text
  chunking operation.

The only code running in the chunk-worker that could consume unbounded memory
was `chunk_text()`.

---

## The Root Cause

Here is the complete original implementation of `chunk_text()`:

```python
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    if not text.strip():
        return []

    chunks: list[dict] = []
    start = 0
    chunk_id = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1

        chunks.append({
            "chunk_id": chunk_id,
            "text": text[start:end].strip(),
            "start": start,
            "end": end,
        })
        chunk_id += 1

        start = end - overlap   # ← THE BUG
```

The loop's exit condition is `while start < text_len`. The loop's advance
is `start = end - overlap`.

Consider what happens when we process the **last chunk** of a document:

1. `start` is somewhere near the end. Say `start = 25700`, `text_len = 25789`,
   `overlap = 200`.
2. `end = min(25700 + 1000, 25789) = 25789`. We hit the end of the text.
3. The boundary search condition is `if end < text_len` — this is `False`
   because `end == text_len`. So no boundary adjustment.
4. We append the final chunk (correct).
5. `start = end - overlap = 25789 - 200 = 25589`.

Now `start = 25589`, which is less than `text_len = 25789`. **The loop
continues.**

Next iteration:
1. `end = min(25589 + 1000, 25789) = 25789`. Same end.
2. Boundary condition still `False`. No adjustment.
3. We append **the same chunk again** — character positions 25589 to 25789.
4. `start = 25789 - 200 = 25589`. Same start.

The loop runs forever. `chunks` grows by one element per iteration, consuming
approximately `chunk_size * sizeof(dict)` bytes per iteration — roughly 1 KB
per cycle. At several thousand iterations per second, a 16 GB container
exhausts its memory in minutes. A 2 GB container exhausts its memory faster.
A 120 GB container would take proportionally longer and then also be killed.

The loop is not quadratic. It is not even polynomial. It is **literally
infinite** — it terminates only when the operating system sends SIGKILL.

### Why the symptom matched Docling

Before the split-worker architecture, the pipeline ran all stages sequentially
in one process. The execution order was:

1. Download PDF from S3 (~1 second)
2. Run Docling table extraction (~30–90 seconds, peak RSS ~4 GB)
3. Upload tables to S3 (~2 seconds)
4. Run `chunk_text()` — **enters infinite loop**
5. OOM kill

From a monitoring perspective, the process died immediately after Docling
finished. Docling consumed ~4 GB of memory before dying. Every symptom pointed
at Docling.

The chunk step was invisible — it appeared to be "part of post-processing" and
its memory growth was masked by Docling's already-high baseline RSS. By the
time the infinite loop had run for a few minutes, we were already at 6–7 GB
and the OOM kill arrived.

### Reproduction

Once we knew where to look, reproducing was trivial:

```python
import time
from financial_pipeline.processing.chunker import chunk_text

# A document-sized string: just over one overlap length past a chunk boundary
text = "word " * 5158  # ~25789 characters
start = time.time()
result = chunk_text(text)  # hangs here forever in the old code
print(f"Done in {time.time()-start:.1f}s, {len(result)} chunks")
```

With the old code, this call never returns. With the fix, it returns in under
a millisecond with 32 chunks.

---

## The Fix

```python
        chunk_id += 1

        if end >= text_len:   # ← ADDED: exit before recomputing start
            break

        start = end - overlap
```

One `if` statement. One `break`. The loop now exits as soon as it has consumed
the last character of the text, instead of recomputing a `start` that can fail
to advance.

The complete fix in diff form:

```diff
         chunks.append({
             "chunk_id": chunk_id,
             "text": text[start:end].strip(),
             "start": start,
             "end": end,
         })
         chunk_id += 1
 
+        if end >= text_len:
+            break
+
         start = end - overlap
```

---

## Verification

We verified the fix with a parameterised edge-case suite before deploying:

```
text_len=0     → []          (empty — early return)
text_len=1     → 1 chunk
text_len=199   → 1 chunk     (less than overlap)
text_len=200   → 1 chunk     (exactly overlap)
text_len=201   → 1 chunk     (overlap + 1)
text_len=999   → 1 chunk     (less than chunk_size)
text_len=1000  → 1 chunk     (exactly chunk_size)
text_len=1001  → 2 chunks    (chunk_size + 1)
text_len=1199  → 2 chunks    (chunk_size + overlap - 1)
text_len=1200  → 2 chunks    (chunk_size + overlap)
text_len=1201  → 2 chunks    (chunk_size + overlap + 1)
text_len=25789 → 32 chunks   (real document)
```

Every case terminates in under 1 ms. None produce duplicate chunk boundaries.

After deploying, the pipeline processed all 447 documents — 5 waves of 5
concurrent table-worker tasks — with zero OOM kills and zero exit-137 events.
Processing time: ~6 hours end-to-end for the full backlog.

---

## What Made This Bug Hard to Find

**1. The symptom was a perfect frame-up.**

The process consumed gigabytes of memory and was killed immediately after
loading a large ML model. Every observable fact pointed at the ML library.
The true cause was upstream and invisible.

**2. Scaling the wrong variable revealed nothing.**

When an OOM kill happens, the natural response is "add more memory." We
escalated four times. Each time, the process was killed later — but still
killed. We interpreted "killed later" as "getting closer." We were not. We
were just watching an infinite loop run for longer.

**3. The bug required a specific input to trigger.**

The infinite loop only activates when the remaining text at the end of the
document is shorter than `overlap` (200 characters). Short documents, or
documents whose total length happens to divide cleanly by `chunk_size - overlap`
(800), never trigger it. Our single test document during development happened
to avoid the condition. 447 production documents could not.

**4. The architectural boundary was the missing diagnostic tool.**

Without the split-worker architecture, it was physically impossible to observe
the chunk stage independently. Chunking ran inside the same process as Docling
— it had no separate log line, no separate memory measurement, no separate exit
code. The monolith hid the bug by aggregating its symptoms with a nearby heavy
operation.

The split-worker architecture did not fix the bug. It made the bug visible for
the first time.

---

## Lessons

### 1. When a fix doesn't work, question the diagnosis — not the fix

We applied four reasonable fixes to Docling-based theories. None worked. The
correct lesson from "fixing Docling didn't help" is "Docling is not the cause"
— not "we need a bigger fix for Docling." We stayed in the wrong hypothesis
tree for weeks because each failed fix felt like progress.

### 2. OOM kills are symptoms of unbounded allocation, not capacity limits

Exit 137 means "a process consumed all available memory." It does not mean
"the process needed more memory than it had." Infinite loops, quadratic
algorithms, and accumulating data structures can trigger OOM on arbitrarily
large machines. The first question when you see exit 137 should be: "Is there
any code path that could grow unboundedly?" — not "how do I get a bigger box?"

### 3. Monolithic processes aggregate their symptoms

A single-process pipeline has one exit code, one RSS measurement, one log
stream. When it crashes, you know the process died — you do not know which
stage of work caused it. Splitting stages into separate processes (or separate
tasks) gives you independent exit codes, independent memory measurements, and
an independent verdict on each stage. This is worth doing even when you have
no bugs, because the next bug will be invisible until you do.

### 4. Test with your production data volume before you ship

One test document cannot reveal a bug that only activates on documents whose
text length has a specific relationship to your chunking parameters. Real data
is diverse in ways that synthetic tests are not. Run your full dataset — even a
sample — before declaring a feature complete.

### 5. The most boring file is the most dangerous

The file we never questioned was the one that caused three weeks of damage.
`chunker.py` had no dependencies, no configuration, no IO. It was fifty lines.
We reviewed Docling internals, Docker layer configurations, ECS task definitions,
and AWS service limits. We did not review `chunk_text()`. Trivial utility
functions deserve the same scrutiny as complex library integrations — more so,
because they are scrutinized less.

---

## The Fix in Production

After the single-line fix was deployed:

```
table-worker: wave 1/3/4 — all exit code 0
ocr-worker:   all exit code 0
chunk-worker: all exit code 0
Final state:  processed = 440 / 440
```

Every document that had been stuck for weeks processed cleanly in its first
attempt. No document was stuck, stalled, or killed. The infinite loop had been
the only obstacle.

---

## Appendix: The Bug, Annotated

```python
while start < text_len:            # (A) loop condition
    end = min(start + chunk_size, text_len)

    if end < text_len:             # (B) only adjusts boundary mid-document
        boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + overlap:
            end = boundary + 1

    chunks.append({ ... })         # (C) appends chunk — always executes
    chunk_id += 1

    start = end - overlap          # (D) ← BUG: can fail to advance start
                                   #     when end == text_len:
                                   #       start = text_len - overlap
                                   #     if text_len - overlap < current start:
                                   #       start does not advance → (A) True forever
```

The fix breaks the feedback loop between (C) and (D) by exiting as soon as the
last character has been consumed:

```python
    chunks.append({ ... })
    chunk_id += 1

    if end >= text_len:            # ← exits before (D) can regress start
        break

    start = end - overlap
```

The invariant `start` must always increase on each iteration. The old code
violated this invariant at the document boundary. The new code enforces it by
exiting instead of recomputing.
