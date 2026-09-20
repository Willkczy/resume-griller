# M0 historical artifact review

Current disposition: the dated sections below preserve the audit/remediation sequence.
Local and remote branch sanitation is complete; GitHub PR-reference cleanup is
explicitly deferred. The owner authorized local synthetic-data M1 work and then local
implementation commits. See [the current handoff](REFACTOR_HANDOFF.md) for superseding
status; old pending-authorization statements below are historical, not current gates.

Reviewed 2026-09-15. Scope: the eight previously removed artifacts across all
locally reachable refs. No network retrieval, payload publication, history rewriting,
or runtime deletion was performed. This review does not establish provenance.

## Findings

Traversed 71 reachable commits with `git rev-list --all` and
`git ls-tree -r -z`; selected every blob version at the eight known paths, then
searched all traversed trees for identical blobs under other paths. Inspected blob
bytes locally with `git cat-file blob`; PDF text was extracted in memory with
PyMuPDF. Only identifiers and pattern counts were emitted. No raw content was saved.

The eight paths resolve to 7 unique blobs. No additional
identical-blob paths were found beyond the known duplicate PDF. Each selected path
had one unique blob version in the traversed history.

| Artifact | Blob | Email patterns | Phone-like patterns | Profile URL patterns |
|---|---|---:|---:|---:|
| `data/exported_prompts.json` | `0040991ca853044d7120e44e24fb15ab97dfba3a` | 0 | 0 | 0 |
| `data/sample_resumes/Resume4.txt` | `02ad63fe7900f88cbeb8aa8a4a759cc27c008d36` | 1 | 0 | 2 |
| `LLM_Inference.ipynb` | `44b3de02e6b9516f93249f3b10575487b49c93cb` | 1 | 1 | 0 |
| `data/sample_resumes/Resume3.txt` | `473f6d9985bafa693e95223341d8345a75e0109d` | 1 | 1 | 0 |
| `data/sample_resumes/Resume2.txt` | `4907906857b68fb3716504672f6854a9814a2951` | 1 | 1 | 1 |
| `data/sample_resumes/resume_sp.pdf`; `frontend/data/uploads/resume_sp_20251217_144220_fd7dbd5e.pdf` | `559f4d8d4d5d37bbfa0a25a694d7b231b8de6678` | 1 | 1 | 1 |
| `data/sample_resumes/Resume1.txt` | `eed33b2e9b2368d7decc3b8b147b3b086736b487` | 1 | 1 | 2 |

Counts are unique regex matches per blob, not identified people or verified PII.
The notebook contains four cells and zero saved outputs, but its source still
contains contact-like text. The derived prompt export has no matches for these
three contact patterns; that does not clear its resume-derived narrative.

No selected blob matched the limited private-key-header or recognizable
OpenAI/Groq/Hugging Face/AWS token patterns used by this check. This is not a
credential audit: arbitrary secrets, other token formats, embedded PDF objects,
image-only content and contextual personal information may not match.

All seven blobs are reachable through each of these locally recorded refs:

- `refs/heads/codex/stabilize-phase-1`
- `refs/heads/feature/full-resume-context`
- `refs/heads/feature/langgraph-interview-agent`
- `refs/heads/main`
- `refs/heads/rag-pipeline`
- `refs/remotes/origin/HEAD`
- `refs/remotes/origin/codex/stabilize-phase-1`
- `refs/remotes/origin/feature/full-resume-context`
- `refs/remotes/origin/feature/langgraph-interview-agent`
- `refs/remotes/origin/main`
- `refs/remotes/origin/rag-pipeline`
- `refs/stash`

Remote-tracking refs are local snapshots. The remote was not fetched; current
server refs, forks, collaborators’ clones, reflog-only/unreachable objects and
external backups were not audited. Modified copies at unknown paths are outside
this exact-path/exact-blob review. Git author metadata was not treated as fixture data.

## Disposition and concrete next action

Historical distribution is not cleared. These artifacts have contact-like data and
unknown origin; the owner requested removal rather than asserting synthetic origin.
The working-tree deletions address future file contents but do not erase these blobs.

Recommended remediation scope, requiring separate authorization:

1. Remove all historical versions of the eight listed paths from the intended
   distributable repository refs, and verify that the seven blob IDs above are no
   longer reachable from those refs. Before executing, account for any additional
   renamed/modified copies discovered by a broader content review.
2. Perform and verify the rewrite in an isolated repository first. Preserve the
   current dirty worktree and unrelated runtime data. Any retained recovery copy
   still contains the artifacts and must be excluded from distribution.
3. Agree which remote branches/tags are affected and coordinate collaborator
   recovery before any force push. Existing local branches, remote-tracking refs
   and stash retain old history until deliberately handled; do not delete them
   implicitly. Remote/server cleanup and external copies need separate follow-through.
4. Recheck the intended distributable tree/history for the removed data and known
   secret patterns. Record scope and limits; do not claim all clones were erased.

No rewrite or push has been authorized or performed. A verified synthetic provenance
determination could change the historical disposition, but this review cannot infer
that human fact. M0 remains open pending that determination or scoped remediation.
Pending owner learning is separate and does not block technical work.

## 2026-09-20 remediation update

The owner authorized isolated local removal and reports being the only user.
The isolated rewrite is verified; see [HISTORY_REMEDIATION.md](HISTORY_REMEDIATION.md).
This supersedes earlier statements that no rewrite was authorized: only the isolated
copy was rewritten. Original checkout, remote and unrelated runtime data are unchanged.
Adoption/remote cleanup remain pending; synthetic provenance was not inferred.

## 2026-09-20 adoption and remote verification update

The owner authorized local adoption and five remote branch updates; both are complete.
Active local history (including three stashes) and the five remote branch histories
are sanitized. A fresh mirror still retrieves the seven blobs through GitHub PR refs
1–5, all closed/merged. M0 remains open for that host-side issue, not branch adoption.
See [HISTORY_REMEDIATION.md](HISTORY_REMEDIATION.md) and the prepared
[GitHub request](GITHUB_HISTORY_CLEANUP_REQUEST.md). No request has been sent.
