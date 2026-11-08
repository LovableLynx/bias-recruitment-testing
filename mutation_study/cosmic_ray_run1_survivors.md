# cosmic-ray run 1: surviving mutants on the security and compliance gates

Run 1 used the tests as first written, before any mutation result was seen.
Raw results: `cosmic_ray_dump_security.jsonl`, `cosmic_ray_dump_compliance.jsonl`.

| Gate | Mutants | Killed | Survived |
|---|---|---|---|
| security (`src/security_gate.py`) | 52 | 42 | 10 |
| compliance (`src/compliance_gate.py`) | 58 | 49 | 9 |

Each survivor was classified by hand by reading its diff against the gate's
policy:

- **Equivalent**: no input the gate can receive changes the decision.
- **Date-coincidental**: equivalent only on the day the run happened.
- **Gap**: a real behaviour change that no test exercised.

## Security gate (10 survivors)

| Mutation | Class | Why it survived |
|---|---|---|
| `severity == "CRITICAL"` -> `<= "CRITICAL"` | equivalent | No severity label sorts before "CRITICAL" (labels are CRITICAL, HIGH, MODERATE, LOW, UNKNOWN). |
| `severity == "HIGH"` -> `<= "HIGH"` | equivalent | This also counts CRITICAL findings as HIGH, but any CRITICAL finding already blocks. |
| `critical > 0` -> `critical != 0` | equivalent | A count is never negative. |
| `sum(1 ...)` -> `sum(2 ...)` for critical | equivalent | Only `> 0` is tested on this count. |
| `today or date.today()` -> `today and date.today()` | date-coincidental | The tests' evaluation date was the day of the run. |
| `severity == "CRITICAL"` -> `is "CRITICAL"` | gap (fixture confound) | Identity fails for strings loaded from JSON. The only real fixture with CRITICAL findings also has HIGH findings, which block anyway. |
| `high > max_high` -> `high != max_high` | gap (untested branch) | No test has a HIGH count below a non-zero limit. |
| `high > max_high` -> `high is not max_high` | gap (untested branch) | Same as above. |
| waiver `expiry >= today` -> `expiry == today` | gap (untested branch) | No test has a waiver that expires after the evaluation date. |
| `MAX_HIGH = 0` -> `1` | gap (test reads the constant) | The over-limit test built its input as `MAX_HIGH + 1` findings, so the input moved with the mutant. |

## Compliance gate (9 survivors)

| Mutation | Class | Why it survived |
|---|---|---|
| `max(selected) == 0` -> `<= 0` | equivalent | A count is never negative. |
| `selected / total` -> `selected * total` | gap (untested branch) | Every boundary fixture has groups of 100, and the real groups are nearly equal in size, so rates and counts never disagreed. |
| `selected / total` -> `selected % total` | gap (untested branch) | Same as above. |
| `len(groups) < 2` -> `!= 2` | gap (untested branch) | Both three-group fixtures expect BLOCK. No three-group fixture expects DEPLOY. |
| `total < min` -> `total == min` | gap (fixture confound) | The small-group fixture also fails the four-fifths ratio. |
| `total < min` -> `total is min` | gap (fixture confound) | Same as above. |
| `total < min` -> `total <= min` | gap (untested edge) | No fixture has a group of exactly the minimum size. |
| `MIN_GROUP_SIZE = 30` -> `31` | gap (test reads the constant, no edge) | The small-group test built its input as `MIN_GROUP_SIZE - 1`. |
| `MIN_GROUP_SIZE = 30` -> `29` | gap (test reads the constant, no edge) | Same as above. |

The hand-selected X2 mutant on the compliance gate (`any` -> `all` in the
group-size check) survived for the same reason as `total == min`.

## Totals

Across the 19 survivors:

- 6 are not real faults: 5 equivalent and 1 date-coincidental.
- 13 are real gaps: 5 in the security gate and 8 in the compliance gate.

The tests marked "Added after the first mutation run" in
`tests/test_security_gate_boundary_cases.py` and
`tests/test_compliance_gate_boundary_cases.py` close the gaps. The run 2
results are in `cosmic_ray_dump_<gate>_run2.jsonl`.

## Run 3 (full rerun from scratch, `scripts/rerun_all_mutation_checks.sh`)

Run 3 reproduced runs 1 and 2 with two exceptions. Both were checked and
found to be glitches in the run itself, not results:

- cosmic-ray on `src/deployment_gate.py` recorded 4 of 20 jobs as `no-test`.
  In cosmic-ray that outcome means the mutation could not be applied, so no
  test ran. Two fresh reruns (`cosmic_ray_dump_fairness_run3a.jsonl`,
  `..._run3b.jsonl`) killed 20 of 20, matching the original study.
- One delete-test candidate on the compliance gate
  (`test_gate_deploys_three_groups_within_four_fifths`) was recorded as
  caught. The same mutant passed in run 2 and in five reruns. The harness now
  re-runs any caught delete-test or vacuous-assertion candidate once, and
  records it as survived (marked transient) if the re-run passes.
