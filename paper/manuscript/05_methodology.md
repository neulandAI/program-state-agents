---
title: The architecture
status: draft-3
---

# 2. Architecture

We describe three agents, each building on the one before it (Fig. 1). The reference agent keeps everything in the prompt. The snippet REPL agent moves the work into a REPL but still prints what its searches return. The document REPL agent, our architecture, also moves the retrieved documents out of the prompt and into the program.


![One question of the 150 for three agents. Each bar is the prompt the model received at one turn (one model call; turns numbered along the x axis), on one scale; below, documents held in program state, or for the snippet REPL agent the snippets printed so far. The reference agent with the document tool (27 searches, 94k-token prompt) and the snippet REPL agent (175 snippets) answered incorrectly; the document REPL agent held 103 documents after six searches and answered correctly with an 11.7k-token final prompt. None of the three returned a labelled gold document for this question.](paper/figures/fig2_example.pdf){#fig:example width=2col}

## 2.1 Reference Agent

The *reference agent* is the search agent shipped with BrowseComp-Plus, run unmodified [@chen2026browsecompplus]. It uses function calling through the OpenAI Responses API and one `search` tool that returns the five best-ranked documents, each cut to its first 512 tokens (a *snippet*). Five hits of 512 tokens are the benchmark's documented settings; we keep them for every snippet-based agent, so all baselines see the same search results. The agent runs for up to 100 turns and keeps every tool result in the conversation. We also run variants that tackle the trade-off inside the prompt: adding the benchmark's `get_document` tool so the agent can read whole documents in its prompt; holding that agent to our cap of 10 searches, with 5 or 25 snippets per search (it learns the cap when a search is refused and gets no initial search); and keeping the conversation under 30,000 tokens, either by trimming the oldest tool results or by folding them into a summary written by the same model.

## 2.2 Snippet REPL Agent

The snippet REPL agent and our document REPL agent (Section III-C) both run in the REPL of the RLM library [@zhang2025rlm; @zhang2025rlmcode]: in each turn (one model call) the model writes Python, the REPL executes it, and only the printed output is appended to the conversation, cut at 20,000 characters per code block, the library's default, which we did not tune. The model ends by assigning its answer to a reserved variable.

The *snippet REPL agent* reads through a sub-model, as recursive language models do: each search prints the same five snippets the reference agent receives, `get_document` fetches a whole document, and the model passes fetched documents to a sub-model instead of printing them. Its snippets enter the prompt, so its prompt still grows with every search; it has no search cap (we also run it with one) and a turn limit of 100.

## 2.3 Document REPL Agent

Our architecture keeps the REPL and changes what a search returns and where it goes. `search(query, k)` returns the complete text of the top *k* documents and merges them into a dictionary `DOCS`, keyed by document id, that persists across turns; it prints one summary line of about 25 tokens and returns the hits to the calling code. The model chooses *k*, the *width* of the search, from 1 to 50. `grep_docs` matches a regular expression against every held document and prints the matches in context, following Sen et al., who found literal matching often better than vector retrieval [@sen2026grep]; `unread_docs` lists documents not yet looked at. Neither counts as a search.

A *search cap*, announced in the prompt and enforced outside the model's code, limits searches per question (10, or 20 in the tuned setting). Held documents do not enter the prompt, so how much the agent retrieves is limited by the cap rather than by the context window. Because only printed output enters the prompt, the prompt does not grow with the documents held: across all 830 questions its median was 21k tokens and its maximum 107k, against the model's 272k-token input limit. If it did approach the limit, the history management studied in Section V (trimming or summarising) could be added; on the reference agent, neither helped. An *initial search* with the question, counted against the cap, gives every run the same start. The turn limit is 30, and a question still running after 90 minutes counts as unanswered. Fig. 2 shows one question end to end.

So that no agent can look answers up on the web, egress is closed: the REPL refuses network modules, file access, `eval` and `exec`, and the process resolves only the model endpoint and the local retrieval server.


## 2.4 Tool-Store Agent

For the ablations, the *tool-store agent* keeps our store but replaces code with tools. It is a function-calling agent with our retriever, search cap and initial search. Its `search` tool puts whole documents into a store outside the prompt and returns one line per hit; the agent reads the store only through a `read` tool (a slice of a document) and a `grep` tool. It chooses its own width, or is held to 25 documents per search.
