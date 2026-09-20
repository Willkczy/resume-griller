# GitHub cleanup request — draft, not submitted

Prepared 2026-09-20. This is a reviewable message, not a claim that GitHub will
accept the request. The owner must confirm its factual description before submission.

## Message

Repository: Willkczy/resume-griller

I own this repository and am its only user. I have removed eight legacy artifacts
(four TXT resumes, two PDF copies, an inference notebook and a derived prompt export)
from local history and force-updated all five remote branches with explicit leases.
The resume files and notebook source contain contact-like email/telephone/profile
information. I cannot establish their original provenance or permission to distribute
that information; I removed them as a precaution. I am not asserting confirmed
credential exposure or verified real-person identity. Please assess whether this
qualifies for your sensitive-data removal assistance.

A fresh mirror still retrieves the seven removed blobs through these five refs:

- refs/pull/1/head
- refs/pull/2/head
- refs/pull/3/head
- refs/pull/4/head
- refs/pull/5/head

All five pull requests are closed and merged. I understand that removing their
historical references can affect old PR diff views. Please advise on dereferencing
these PR refs and, if eligible, removing cached views and running server-side garbage
collection after confirming no other server-side references retain the removed data.

First changed commit, computed from the verified filter-repo commit map:
old 8adc190aa77c86d9458cb937d18d078e9907a50d
replacement 97240db8f4b6718db43fbe22243e9cb849494d86

The repository API reported zero forks at the time of review. All eight removed
artifacts were ordinary Git blobs, not LFS pointers. My active local repository
and all five advertised branch histories are verified clean for these eight paths
and seven blobs. No sensitive file contents are attached to this request.

## Submission and verification

Do not submit until the owner explicitly approves sending this message. Support may
require confirmation about the data's sensitivity; do not invent provenance to meet
that requirement. GitHub's published policy says it does not remove non-sensitive
data through this process.

After GitHub acts, use a new temporary mirror to verify that the five PR refs no
longer retain the seven blob IDs in `HISTORICAL_DATA_REVIEW.md`, and that none of the
eight paths remains in its reachable commit trees. Cached/unreachable server objects
also require GitHub's confirmation; branch checks alone cannot prove global erasure.

Official reference: [Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
Submit through the [GitHub Support portal](https://support.github.com/).
