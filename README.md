# Sokoban solver — make.fun sokoban-solver-01-run-20261121

One program, `main.ts` (Node 22, TypeScript, no dependencies). It reads
levels from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes
`$MAKEFUN_OUTPUT_DIR/<id>/solution.txt` for each instance.

## Method

Generated levels are a 1-wide hallway with 1–4 rooms off it, each room
opening through a single door; a box pushed out of a door can never come
back, so rooms are independent problems. The solver:

1. Cuts the hallway row's dead squares out of the floor and takes each
   remaining component holding boxes as a room.
2. Solves every room push-optimally, in scan order, with the other rooms'
   boxes treated as walls, stitching rooms with BFS walks along the hallway.
3. Per room: A* over pushes. A state is the sorted box squares plus the
   player's reachable region (named by its lowest square). The heuristic is
   the minimum-cost matching of boxes to goals under push distances
   (reverse BFS from each goal over the push graph, ignoring other boxes),
   which is admissible — so the first solved state popped is push-optimal.
   Dead squares and frozen 2x2 blocks are pruned. A push-BFS retry and a
   whole-board attempt are fallbacks; an unsolvable level yields an empty
   (valid) answer rather than an illegal move sequence.
4. Deterministic throughout: expansion-count budgets only, no clock, fixed
   direction order and heap tie-breaks, so two runs write identical bytes.

## Local test

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
MAKEFUN_OUTPUT_DIR=out MAKEFUN_REFERENCE_DIR=<kit>/practice/reference \
  MAKEFUN_SCORES_DIR=scores node --experimental-strip-types \
  <kit>/program/scorer/score.ts
```

Practice set: 1,000,000 / 1,000,000 (every instance solved in P* pushes).
