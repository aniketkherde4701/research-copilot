# Academic Research Co-Pilot

Ask a research question. The agent searches arXiv, picks the single most relevant
paper, reads it, and answers from its contents — with a citation to the passage it
used, or an explicit refusal if the paper doesn't cover the question.

Built on the Gemini and Pinecone SDKs directly, without a RAG framework.

**[Try it live](https://REPLACE-WITH-YOUR-URL.streamlit.app)** - free-tier demo, quota
shared across visitors and reset daily. Clone it below to run without limits.

![Demo](docs/demo.gif)

---

## What it does

```
question → rewrite as arXiv queries → search → select ONE paper
         → download & semantically chunk → embed & index
         → retrieve top-8 passages → synthesise a cited answer
```

**Example.** *"How do diffusion models compare to GANs for image generation?"* →
finds a comparative-analysis paper, reads it, and returns a short brief with each
claim numbered back to the passage it came from.

## Running it

Requires Python 3.11+, a [Gemini API key](https://aistudio.google.com/apikey) and a
[Pinecone key](https://app.pinecone.io) — both free tier.

```bash
git clone <this repo> && cd rag-copilot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paste your two keys
python test_setup.py          # preflight: keys, models, index, arXiv
streamlit run app.py
```

`test_setup.py` checks every dependency before you run anything, so a bad key
fails in ten seconds rather than halfway through the pipeline.

## Design decisions

**Query rewriting before search.** arXiv matches keywords, not meaning, so a
natural-language question retrieves badly. An LLM step generates three *different
search strategies* — current jargon, the mechanism described in plainer words, and
landmark model names — with the rule that no term may be mandatory across all three.
Results are interleaved round-robin so no single strategy fills every slot.

**Semantic chunking, not fixed-size.** Sentence windows are embedded, cosine distance
is measured between consecutive windows, and the text is cut where that distance
exceeds the 95th percentile — i.e. where the topic actually changes. A percentile
rather than a fixed threshold, because what counts as a large jump differs between a
terse maths paper and a discursive survey.

**Asymmetric embeddings.** Documents are embedded with `RETRIEVAL_DOCUMENT` and
questions with `RETRIEVAL_QUERY`. A question and the passage answering it are not
paraphrases, and the task type tells the model to place them near each other anyway.
This is why `gemini-embedding-001` is used rather than the newer model, which dropped
the parameter.

**Two-level caching.** Questions map to the paper they resolved to; papers get their
own Pinecone namespace. A repeat question skips search, selection and indexing
entirely — **58s → 14s, measured.**

**Reusing vectors at index time.** Boundary detection already embeds every sentence
window, so chunk vectors are the mean of their windows rather than a second embedding
pass. Cuts a paper from ~600 embedding requests to 25, which is what keeps ingestion
inside the free tier's 100/minute ceiling.

**A deliberate refusal path.** If the retrieved passages don't answer the question,
the model says so instead of improvising. An assistant that always produces a
confident answer is one that sometimes lies.

**No LangChain.** Calling the SDKs directly meant the chunking strategy and embedding
task types were explicit choices rather than framework defaults. On a five-stage
deterministic pipeline the abstraction costs more than it saves.

## Known limitations

Stated plainly, because they're real:

- **arXiv's ranking is the ceiling.** It matches titles, abstracts and metadata but
  never the paper body, and has no citation or impact signal — so relevance is
  term-frequency matching. A perfectly matching paper can sit at rank 400. Three
  rounds of query tuning could not surface CLIP for a question about contrastive
  pre-training, because the string "CLIP" appears nowhere in its title or abstract.
  The fix is Semantic Scholar for discovery, arXiv for retrieval.
- **Coverage.** arXiv is strong in CS, maths, physics and statistics; thin to absent
  in medicine, biology and the social sciences.
- **One paper per question.** Deliberate scoping. Multi-paper synthesis means
  reconciling conflicting findings, which is a harder problem.
- **Chunking is coarser than the design intends.** Sentence *windows* rather than
  sentences, and a 40-chunk cap, both to fit the free-tier quota. Two constants in
  `config.py` restore full granularity on a paid key.
- **No quantitative retrieval evaluation.** `eval/run_eval.py` supports manual
  judgement of paper selection and citation faithfulness; a labelled question–passage
  set with recall@k is the first thing I'd add.

## Layout

```
app.py            Streamlit UI — no business logic, calls run_pipeline only
agent_core.py     orchestration: query rewrite, selection, synthesis, caching
rag_pipeline.py   semantic chunking, Pinecone indexing and retrieval
tools.py          arXiv search, PDF download, text extraction, reference stripping
llm.py            Gemini wrapper: retries, rate limiting, JSON mode
config.py         every tunable constant and model ID
prompts/          prompt templates, kept out of the code
eval/             manual evaluation harness
tests/            chunking and reference-stripping tests
```

## Stack

Python 3.11 · Gemini (`google-genai`) · Pinecone · Streamlit · PyMuPDF · NumPy · arxiv
