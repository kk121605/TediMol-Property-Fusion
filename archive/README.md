# Historical implementation references

`branch_entrypoints/` retains the six later A1/A2/A3 convenience wrappers. They are excluded from the live fusion directories because the original trained receipt did not include them. Their original sibling-path assumptions do not hold at this archived location. Use the shared entrypoints with an explicit mode instead.

`continuation/` retains selected 1.2M-step continuation, resume-contract and NumPy seed-recovery source and original tests. These refer to the historical project tree, reports and deployment receipts. They are not a complete current cluster package and are excluded from supported tests/CI.

The original A30 handoff had a stop-use notice; subsequent fixes occurred on the cluster. Current remote files and experiment state were not queried during this publication. See [the reproduction guide](../docs/reproducibility.md).
