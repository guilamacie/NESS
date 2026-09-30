# ADR-0013: Optimizer parameter groups and learning-rate schedules

**Decision**
* `optimizer.param_groups`: an ordered list of `{match, lr | lr_scale, weight_decay,
  grad_multiplier, clip_norm}`. `match` is a shell glob over owned group ids (`node/group`,
  edge-parameter ids); the first matching rule wins; a rule matching no owned group fails closed.
  Unknown keys in `optimizer`, its rules or its schedule fail closed.
* `optimizer.schedule`: `{kind: constant | linear | cosine, warmup_steps, total_steps,
  min_lr_ratio}`, linear warm-up then the chosen decay, step-based (1-based updates), floor after
  `total_steps`. The position is the optimizer `step`, already in the learner snapshot.
* Order of operations per update: `grad_multiplier` -> per-group `clip_norm` -> global
  `clip_norm` (over all groups) -> decoupled weight decay `p * (1 - lr_g * wd_g)` before the step
  -> SGD / Adam step with `lr_g = (rule lr, or arm lr * lr_scale) * schedule(step)`.
* Logged per update and group: effective lr, grad multiplier, raw and applied gradient norm,
  update norm. The rules, the resolved group assignment and the schedule go to the manifest's
  `algorithm_specs.optimizer` and to learner checkpoints; restoring a checkpoint whose optimizer
  configuration differs from the arm's is refused.
* Without `param_groups` and `schedule` the update is the v0.2 arithmetic, bitwise.

**Why** Joint co-training needs the adapted module to learn materially slower than new heads.
A per-module parameter reparametrisation only emulates this under Adam without decay or clipping.

**Tests** `tests/test_optimizer_groups.py` (v0.2 bitwise reference; B1 SGD exact and Adam first
step ratio; B2 closed forms; clipping order; B3 mid-schedule restore continues bitwise).
