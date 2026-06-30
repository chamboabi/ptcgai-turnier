---
name: submission-no-numpy
description: Submission/arena env has torch but NOT numpy; bundled cg is older than parent cg
metadata:
  type: project
---

The tournament submission environment (env/ at rl_mcts/, and presumably the arena) ships **torch but not numpy**. `import numpy` fails; torch runs fine without it (prints a harmless "Failed to initialize NumPy" warning).

**Why:** anything bundled into `submission_builder/` must be numpy-free. The slim opponent-deck predictor `deck_predict_lite.py` is therefore pure Python (no numpy), unlike the training-side `deck_predict.py` which uses numpy + sklearn.

**How to apply:** when adding code to the submission, use only torch + stdlib + bundled cg. Don't assume numpy ships with torch.

Also: the bundled `cg/` (submission_builder/cg, copied into dist) is an OLDER version than the parent `rl_mcts/cg/` — e.g. parent `cg.sim` has `set_seed`, bundled does not, and parent has `cg/recorder.py` while the bundle does not. For local end-to-end testing of the bundled agent, build a test dir using the parent `cg/` (which has a matching recorder), not the dist cg.
