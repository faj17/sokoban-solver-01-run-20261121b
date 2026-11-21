// Sokoban solver: room-decomposed push-optimal A*.
//
// Strategy: a generated level is a 1-wide hallway with 1-4 rooms off it, each
// room opening through a single door, and a box pushed out of a door can never
// come back — so rooms are independent. The solver cuts the hallway row's dead
// squares out of the floor, takes each remaining component holding boxes as a
// room, and solves every room push-optimally on its own (other rooms' boxes
// treated as walls), stitching the rooms with BFS walks along the hallway.
// Per room the search is A* over pushes: a state is the sorted box squares
// plus the player's reachable region (named by its lowest square). The
// heuristic is the minimum-cost matching of boxes to goals under push
// distances (reverse BFS from each goal over the push graph, ignoring other
// boxes), which is admissible, so the first solved state popped is
// push-optimal. Dead squares (reverse-reachable from no goal) and frozen
// 2x2 blocks are pruned. If a room search ever fails, a push-BFS (same code,
// zero heuristic) is retried; if that fails too, the whole board is tried as
// a last resort. Nothing here reads the clock or the environment: every
// budget is an expansion count and every tie-break is fixed, so two runs
// write byte-identical output. A level that cannot be solved yields an empty
// (valid, zero-point) answer rather than an illegal move sequence.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Direction order is fixed everywhere (determinism): up, down, left, right.
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a box alone can be pushed onto some goal: pull back
// from the goals over open floor (a box at x can be pulled from x-d when the
// player can stand at x-2d... in reverse: from x, a box could have come from
// y = x-d if the pusher could stand at y-d = x-2d).
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Push distances ignoring other boxes: dist[g][cell] = fewest pushes to move
// a box from cell onto goal g, or Infinity. Reverse BFS from g over the push
// graph: a box at y can be pushed to x = y+d when the pusher can stand at
// y-d and x is floor.
function pushDistances(level: Level): number[][] {
  const { w, wall, goals } = level;
  const size = wall.length;
  const out: number[][] = [];
  for (const g of goals) {
    const dist = new Array<number>(size).fill(Infinity);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const y = x - d; // box position that pushes to x
        if (y < 0 || y >= size || wall[y] || dist[y] !== Infinity) continue;
        const stand = y - d; // where the pusher stands
        if (stand < 0 || stand >= size || wall[stand]) continue;
        dist[y] = dist[x]! + 1;
        queue.push(y);
      }
    }
    out.push(dist);
  }
  return out;
}

// Minimum-cost assignment of boxes to goals under dist (brute-force
// permutations for <= 4 boxes; sum of per-box minima above that). Infinity
// when no assignment exists.
function heuristic(boxes: number[], dist: number[][], goals: number[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 4) {
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (let j = 0; j < goals.length; j++) {
        const d = dist[j]![b]!;
        if (d < best) best = d;
      }
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  }
  // Brute-force permutations of goal indices.
  const perm = goals.map((_, j) => j);
  let best = Infinity;
  const rec = (k: number, sum: number): void => {
    if (sum >= best) return;
    if (k === n) {
      best = sum;
      return;
    }
    for (let j = k; j < n; j++) {
      const tmp = perm[k]!;
      perm[k] = perm[j]!;
      perm[j] = tmp;
      const d = dist[perm[k]!]![boxes[k]!]!;
      if (d !== Infinity) rec(k + 1, sum + d);
      const tmp2 = perm[k]!;
      perm[k] = perm[j]!;
      perm[j] = tmp2;
    }
  };
  rec(0, 0);
  return best;
}

interface SearchCtx {
  level: Level;
  live: Uint8Array;
  dist: number[][];
  goalSet: Uint8Array;
  seen: Int32Array;
  mark: Int32Array;
  stack: Int32Array;
  occupied: Uint8Array;
  stamp: number;
}

// Lowest square of the player's reachable region from `from` (boxes in
// occupied block). Marks the region in mark with a fresh stamp — a different
// array from the expansion region's seen, so computing a child's region never
// clobbers the stamp the parent's push generation is checking against.
function regionLow(ctx: SearchCtx, from: number): number {
  const { level, mark, stack } = ctx;
  const { w, wall } = level;
  const stamp = ++ctx.stamp;
  let low = from;
  let n = 0;
  stack[n++] = from;
  mark[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && ctx.occupied[q] === 0 && mark[q] !== stamp) {
        mark[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// Whether the box just pushed to t sits in a frozen 2x2 block of walls and
// boxes holding a box off its goal.
function frozenBlock(ctx: SearchCtx, t: number): boolean {
  const { level, goalSet } = ctx;
  const { w, wall } = level;
  const occ = ctx.occupied;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goalSet[a]) || (occ[b] && !goalSet[b]) || (occ[c] && !goalSet[c]) || (occ[d] && !goalSet[d])) return true;
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes (occupied), lower
// LURD letters, first direction first on ties. Throws if unreachable.
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const { w, wall } = level;
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface Found {
  pushes: number;
  moves: string;
}

// Push-optimal solve of one level (a room sub-level or a whole board).
// A* over pushes when useHeuristic, else push-BFS (zero heuristic).
// Returns null when the search gives up (expansion cap) or has no solution.
function solvePush(level: Level, maxExpansions: number, useHeuristic: boolean): Found | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const dist = useHeuristic ? pushDistances(level) : [];
  const ctx: SearchCtx = {
    level,
    live,
    dist,
    goalSet,
    seen: new Int32Array(size),
    mark: new Int32Array(size),
    stack: new Int32Array(size),
    occupied: new Uint8Array(size),
    stamp: 0,
  };
  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet[b]);
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  const keyOf = (boxes: number[], low: number): string => low + ':' + boxes.join(',');
  // Heap entries [f, g, seq, state]. Order: lower f, then deeper g, then
  // earlier seq — all deterministic.
  const heap: [number, number, number, number][] = [];
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  const stBoxes: number[][] = [level.boxes.slice()];
  const stPlayer: number[] = [level.player];
  const stG: number[] = [0];
  const stKey: string[] = [];
  const parent: number[] = [-1];
  const best = new Map<string, number>();
  for (const b of level.boxes) ctx.occupied[b] = 1;
  const startLow = regionLow(ctx, level.player);
  for (const b of level.boxes) ctx.occupied[b] = 0;
  const startH = useHeuristic ? heuristic(level.boxes, dist, goals) : 0;
  if (startH === Infinity) return null;
  const startKey = keyOf(level.boxes, startLow);
  best.set(startKey, 0);
  stKey.push(startKey);
  heapPush([startH, 0, 0, 0]);
  let seq = 1;
  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, g, , s] = heapPop();
    // Stale entry: a better path to this state's key was found since.
    if (best.get(stKey[s]!) !== g) continue;
    const boxes = stBoxes[s]!;
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expanded >= maxExpansions) return null;
    expanded++;
    for (const b of boxes) ctx.occupied[b] = 1;
    const here = ++ctx.stamp;
    // Mark the region from the stored player square.
    {
      const { stack, seen } = ctx;
      let n = 0;
      stack[n++] = stPlayer[s]!;
      seen[stPlayer[s]!] = here;
      while (n > 0) {
        const p = stack[--n]!;
        for (let k = 0; k < 4; k++) {
          const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
          if (q < 0 || q >= size) continue;
          if (wall[q] === 0 && ctx.occupied[q] === 0 && seen[q] !== here) {
            seen[q] = here;
            stack[n++] = q;
          }
        }
      }
    }
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        const stand = b - d;
        if (stand < 0 || stand >= size || t < 0 || t >= size) continue;
        if (ctx.seen[stand] !== here || wall[t] || ctx.occupied[t] || !live[t]) continue;
        ctx.occupied[b] = 0;
        ctx.occupied[t] = 1;
        const frozen = frozenBlock(ctx, t);
        let key = '';
        let next: number[] | null = null;
        let low = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          low = regionLow(ctx, b);
          key = keyOf(next, low);
        }
        ctx.occupied[t] = 0;
        ctx.occupied[b] = 1;
        if (frozen || next === null) continue;
        const prevG = best.get(key);
        if (prevG !== undefined && prevG <= g + 1) continue;
        const h = useHeuristic ? heuristic(next, dist, goals) : 0;
        if (h === Infinity) continue;
        best.set(key, g + 1);
        const ns = stBoxes.length;
        stBoxes.push(next);
        stPlayer.push(b);
        stG.push(g + 1);
        stKey.push(key);
        parent.push(s);
        heapPush([g + 1 + h, g + 1, seq++, ns]);
      }
    }
    for (const b of boxes) ctx.occupied[b] = 0;
  }
  if (found < 0) return null;
  // Reconstruct: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const nextBoxes = stBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) ctx.occupied[b] = 1;
    moves += walkTo(level, ctx.occupied, player, from - d);
    for (const b of boxes) ctx.occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

// Split a level into rooms: cut the dead squares of the player's row (the
// hallway: a box there reaches no goal) out of the floor; each remaining
// component holding boxes is a room. Null when a box starts in the hallway,
// a goal lies in the hallway, or a room's box/goal counts differ.
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    rooms[part[b]!]!.boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Replay moves on a level; true only when every step is legal and every box
// ends on a goal.
function verify(level: Level, moves: string): boolean {
  const { w, wall } = level;
  const goalSet = new Uint8Array(wall.length);
  for (const g of level.goals) goalSet[g] = 1;
  const box = new Uint8Array(wall.length);
  for (const b of level.boxes) box[b] = 1;
  let player = level.player;
  for (const ch of moves) {
    const lower = ch.toLowerCase();
    const dir = DIRS.find(([, , l]) => l === lower);
    if (!dir) return false;
    const d = dir[1] * w + dir[0];
    const next = player + d;
    if (next < 0 || next >= wall.length || wall[next]) return false;
    if (box[next]) {
      const beyond = next + d;
      if (beyond < 0 || beyond >= wall.length || wall[beyond] || box[beyond]) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < wall.length; i++) {
    if (box[i] && !goalSet[i]) return false;
  }
  return true;
}

function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      let solved = solvePush(sub, 1_000_000, true);
      if (solved === null) solved = solvePush(sub, 600_000, false);
      if (solved === null) {
        ok = false;
        break;
      }
      moves += solved.moves;
      // Advance the player to where the room's moves leave it.
      let p = player;
      for (const ch of solved.moves.toLowerCase()) {
        p += ch === 'u' ? -level.w : ch === 'd' ? level.w : ch === 'l' ? -1 : 1;
      }
      player = p;
      room.done = true;
    }
    if (ok && verify(level, moves)) return moves;
  }
  // Fallback: solve the whole board at once (also covers levels that are no
  // set of rooms, e.g. boxes in the hallway).
  let solved = solvePush(level, 1_000_000, true);
  if (solved === null) solved = solvePush(level, 600_000, false);
  if (solved !== null && verify(level, solved.moves)) return solved.moves;
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
