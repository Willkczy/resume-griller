# M0 isolated history remediation

Current disposition: the dated sections below preserve the audit/remediation sequence.
Local and remote branch sanitation is complete; GitHub PR-reference cleanup is
explicitly deferred. The owner authorized local synthetic-data M1 work and then local
implementation commits. See [the current handoff](REFACTOR_HANDOFF.md) for superseding
status; old pending-authorization statements below are historical, not current gates.

Date: 2026-09-20. Status: local adoption and five remote branch updates complete;
GitHub pull-request refs still retain the removed blobs.

## Authorization and boundary

The owner chose removal of all eight artifacts, authorized an isolated local rewrite
and verification, and reported being the only repository user. This does not establish
synthetic provenance. The owner subsequently authorized the five remote branch
updates and local adoption; both are complete. No working-tree implementation commit
was made. The owner statement does not prove no external copies exist.

## Verified candidate

Bare repository: `/var/folders/l0/nt1smsf16v947rzfdtppqlx00000gn/T/rg-m0-sanitized-v11drz5g/verified.git`.
Verification evidence and commit map: `/var/folders/l0/nt1smsf16v947rzfdtppqlx00000gn/T/rg-m0-sanitized-v11drz5g/verification.json` and
`/var/folders/l0/nt1smsf16v947rzfdtppqlx00000gn/T/rg-m0-sanitized-v11drz5g/verified.git/filter-repo/commit-map`.
This is temporary local storage; preserve or relocate the candidate before relying
on it long-term. The earlier `repository.git` in the same temporary directory is
superseded and must not be used for adoption.

The candidate contains committed history only. Current uncommitted implementation
and documentation changes remain in the original worktree. Do not switch/reset that
worktree or assume those changes exist in this candidate.

## Method and verification

- Made independent local mirror copies; no shared/hardlinked object store was used.
- Filtered all eight paths and the seven known blob IDs using `git filter-repo`.
  Set `--prune-empty never --prune-degenerate never` to preserve commit topology.
- Initial fresh-clone safety check rejected the no-hardlinks local clone. Force was
  used only inside the newly created isolated copy, never the source. Ref coverage
  then exposed automatic removal of a unique remote-main commit in that candidate.
  Rebuilt the final candidate with `git clone --mirror --no-local`, moved remote
  snapshots into `refs/archive/origin/*`, and removed only its local origin config
  before filtering. This preserves the distinct local and remote-main histories.
- All 71 source commits appear in the final commit map; all 71 rewritten commits
  were checked. Compared each original tree minus selected paths/blobs with its
  rewritten tree: zero unrelated file/mode/blob changes.
- All 14 source refs are accounted for. Remote snapshots were renamed to archive
  refs; branch/stash refs and two Codex tree refs were preserved. Filter-repo warned
  that the tree refs were skipped; their unchanged object IDs and trees were checked.
- No selected path remains in any historical commit or ref tree. None of the seven
  blobs is reachable through any ref, and `git cat-file -e` fails for each removed
  blob: they are absent from this candidate object database, not merely hidden.
- `git fsck --full --no-reflogs`: passed. Candidate has no configured remote.
- Source tracked/nonignored-file hash manifest matched after verification, including
  existing dirty edits. Source branch history was not modified; runtime stores were
  not used. No application behavior changed, so application tests were not rerun.

## Prepared remote update (not executed)

Read-only `git ls-remote --heads --tags origin` found five branches and no tags.
Each remote tip matches a source commit represented in the verified map. The exact
old/new IDs below allow explicit lease checks before any separately approved update.
Do not use a mirror push: archive, stash and Codex refs are local preservation refs.

| Remote branch | Expected old tip | Verified replacement tip |
|---|---|---|
| `refs/heads/codex/stabilize-phase-1` | `631cc55cd498f320f733d93db6d4f848d9bfbaa6` | `87b114eb9f114476cf8b42dfade4f2e783372a4e` |
| `refs/heads/feature/full-resume-context` | `8af589e6fbdd655f89ffd2b3b21ff7037460e8fc` | `5bc67787117877cbd6a1ed8041fe5d94eb4d34c9` |
| `refs/heads/feature/langgraph-interview-agent` | `3d502b33f527bed3d0c449b5aca32e31ad48b19f` | `39e1da7537720d2241399fbbc4402c482bd718e2` |
| `refs/heads/main` | `e73e5d6006f0ba3ad3d2a71887ba75a1ab8f9160` | `c99de99b462762265cac470fc2ca76e231e3b107` |
| `refs/heads/rag-pipeline` | `dd70f27d92563fc9426438a0c9d0d427fe5922d4` | `a83e01c7b1f77e1f15ab170c5ee74f3bde77dd5f` |

Remaining decision: authorize promotion of these five verified replacements using
explicit old-tip leases. A changed remote tip must stop the update for review.
Afterward verify remote refs; repository-host cached objects, hidden PR refs and
external copies may require separate cleanup. Do not claim global erasure from a
branch push. The existing local source clone still retains the old data and must
be kept out of distribution until deliberately replaced or remediated.

Separately preserve and carry the uncommitted M0 changes onto sanitized history.
Committing those changes or replacing the current checkout requires authorization;
exclude machine-local `.claude/settings.local.json` from any implementation commit.

## M0 disposition

The isolated-copy remediation is complete for the eight paths/seven known blobs.
M0 is not declared complete for the current repository: adoption/remote remediation
remain pending. Clean Python/frontend setup and text-only checks retain their dated
evidence in the handoff. No broad PII audit, real-user processing approval, or M1
authorization is implied. Owner learning remains pending/not assessed.


## 2026-09-20 promotion, adoption and remaining GitHub refs

The owner authorized the prepared remote replacements and local adoption. Completed:

- `git push --atomic` with five explicit `--force-with-lease=<ref>:<old>` guards
  and only the five verified branch refspecs succeeded. Read-back verified each tip;
  no tags were advertised. Archive/stash/Codex refs were not pushed.
- Before local adoption, found three stash reflog entries. Two older entries were
  outside the earlier reachable-ref review. Included all three in a rebuilt candidate,
  bringing validation to 75 commits. Every retained tree matches its original minus
  selected artifacts; all three stashes have sanitized IDs and preserved reflog
  entries. The same seven blobs cover the removed paths in these extra snapshots.
- Swapped only Git metadata on the same filesystem, preserving original configuration,
  hooks, excludes and working files. The index was clean before adoption; rebuilt it
  from sanitized HEAD without updating working files. All pre-swap tracked/nonignored
  file hashes matched afterward; no implementation files were committed or staged.
- Active branch HEAD is `87b114eb9f114476cf8b42dfade4f2e783372a4e`. The full adoption
  map is now in `.git/filter-repo/commit-map`. The temporary `adoption.git` was moved
  into the active `.git`; it is no longer a separate candidate directory.
- Active local verification, including refs and reflogs: 75 commits checked, all
  eight paths absent, all seven blobs physically absent, `git fsck --full` passed.
- Original metadata retained at
  `/Users/willkczy/Projects/resume-griller-history-recovery-20260920/original-history.DO-NOT-DISTRIBUTE.git`.
  The private recovery directory has a warning marker and is outside the distributable
  project. It intentionally retains original data; no deletion is authorized.

A fresh remote mirror exposed five additional GitHub-managed `refs/pull/1/head`
through `refs/pull/5/head`. All retain the seven blobs, despite all five branch
histories being sanitized. The fresh mirror contains 117 reachable commits and passed
`git fsck --full`; absence verification correctly failed. PR API metadata confirms
all five are closed and merged. Repository metadata reports zero forks; this is a
point-in-time API result, not a guarantee about external copies.

The earlier `ls-remote --heads --tags` checks were scoped to branches/tags and did
not cover these refs. A full remote-cleanup claim would therefore be incorrect.
GitHub documents that PR refs are read-only and that Support handles eligible
PR-reference/cache cleanup. It will not remove non-sensitive data through that process.
See [GitHub's procedure](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

Prepared [GITHUB_HISTORY_CLEANUP_REQUEST.md](GITHUB_HISTORY_CLEANUP_REQUEST.md)
with the five PRs, first changed commit, fork count and an explicit unknown-provenance
statement. It has not been submitted; sending it requires owner authorization.
M0 remains open only for this verified server-side retention issue. Do not substitute
PR archiving or another branch push for proof of removal. Owner learning remains
pending/not assessed; M1 is not started.

The temporary `remote-verification.git` under the earlier candidate directory contains
these PR copies and is marked DO NOT DISTRIBUTE with private permissions. Both it and
the original-metadata backup are recovery/evidence copies, not sanitized deliverables.
