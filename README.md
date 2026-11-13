# Do repeated results tell a search agent when to stop?

Code for our IR course project (group 23, Radboud University, 2026). We run Search-R1 (Qwen2.5-7B, PPO
checkpoint) with BM25 on the Total Recall QA subsets Wiki1 and Ecommerce, and change only its search tool:

| System | What the search tool returns for every search | Result folder |
|---|---|---|
| `search_r1` | The 3 highest-ranked BM25 passages for the query, whether or not the agent has already seen them in this question. | `search_r1_bm25` |
| `search_r1_notice` | The same 3 passages as `search_r1`. If none of them is new (all were already shown earlier in this question), the line *"No more new results for this query."* is added after them. | `search_r1_bm25_notice` |
| `search_r1_novel` | The 3 highest-ranked passages among the BM25 top 30 that were not shown earlier in this question. If fewer than 3 such passages are left, the list is topped up to 3 with already-shown passages from the same top 30, highest-ranked first, without telling the agent. We call these top-ups *fill-ups*. | `search_r1_bm25_novel_passage_d30` |
| `search_r1_novel_notice` | Like `search_r1_novel`, but without fill-ups: if fewer than 3 unseen passages are left in the top 30, it returns only those (2, 1 or none), followed by the line *"No more new results for this query."* | `search_r1_bm25_novel_passage_d30_nofill_notice` |

In both notice variants, a search for which BM25 finds nothing at all returns only the line.
