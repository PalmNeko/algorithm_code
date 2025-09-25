from typing import List, Tuple
# from local_driver import Alg3D, Board # ローカル検証用
from local_driver import Alg3D # ローカル検証用
import math
import copy
# from framework import Alg3D # 本番用

from typing import List, Tuple, Dict, Optional
import time
import math
import random

# --- 設定 ---
MAX_TIME = 1.5  # 秒。思考時間の上限（必要なら増やす）
MAX_NODES = 2000000  # 探索ノード上限（安全装置）
# --------------------------------

# 4x4x4 の全座標を index に変換
def idx(x,y,z): return z*16 + y*4 + x

# 全勝利ラインを生成（プログラムにて網羅）
def generate_lines():
    lines = []
    # 直線: x-direction, y-direction, z-direction
    for z in range(4):
        for y in range(4):
            lines.append([idx(x,y,z) for x in range(4)])  # x-line
    for z in range(4):
        for x in range(4):
            lines.append([idx(x,y,z) for y in range(4)])  # y-line
    for y in range(4):
        for x in range(4):
            lines.append([idx(x,y,z) for z in range(4)])  # z-line

    # 各層の対角（z 固定）
    for z in range(4):
        lines.append([idx(i,i,z) for i in range(4)])
        lines.append([idx(3-i,i,z) for i in range(4)])

    # 各 x 固定の対角（x 固定、y-z 平面）
    for x in range(4):
        lines.append([idx(x,i,i) for i in range(4)])
        lines.append([idx(x,3-i,i) for i in range(4)])

    # 各 y 固定の対角（y 固定、x-z 平面）
    for y in range(4):
        lines.append([idx(i,y,i) for i in range(4)])
        lines.append([idx(3-i,y,i) for i in range(4)])

    # 4 つの立方体対角（完全な空間対角）
    lines.append([idx(i,i,i) for i in range(4)])
    lines.append([idx(3-i,i,i) for i in range(4)])
    lines.append([idx(i,3-i,i) for i in range(4)])
    lines.append([idx(3-i,3-i,i) for i in range(4)])

    # 重複除去（念のため）
    uniq = []
    seen = set()
    for line in lines:
        t = tuple(line)
        if t not in seen:
            seen.add(t)
            uniq.append(line)
    return uniq

LINES = generate_lines()

# ボードをフラット配列(64)に変換
def board_to_flat(board: List[List[List[int]]]) -> List[int]:
    flat = [0]*64
    for z in range(4):
        for y in range(4):
            for x in range(4):
                flat[idx(x,y,z)] = board[z][y][x]
    return flat

# 指定列(x,y)に置けるか
def can_play(flat, x,y):
    return flat[idx(x,y,3)] == 0  # top が空いていればその列は置ける

# その列に置くとどの z に入るか（最も低い空き）
def drop_z(flat, x,y):
    for z in range(4):
        if flat[idx(x,y,z)] == 0:
            return z
    return None

# 置いて新しいフラットボードを返す（変更はコピー）
def play_move(flat, x,y, player):
    z = drop_z(flat,x,y)
    if z is None:
        raise ValueError("invalid move")
    new = list(flat)
    new[idx(x,y,z)] = player
    return new, z

# 勝ち判定（ある player が任意のラインを埋めているか）
def is_win_flat(flat, player):
    for line in LINES:
        ok = True
        for p in line:
            if flat[p] != player:
                ok = False
                break
        if ok:
            return True
    return False

# 終局かどうか
def is_full(flat):
    return all(v != 0 for v in flat)

# シンプルな評価関数（非終局用）
# 各ラインについて opponent がいなければ player のポイントが増える（個数で重み付け）
LINE_WEIGHTS = [0, 1, 10, 50, 1000]  # 0..4 個の石に対する重み（4は実際の勝ち）
def evaluate_flat(flat, player):
    opp = 3 - player
    score = 0
    for line in LINES:
        cnt_p = 0
        blocked = False
        for p in line:
            if flat[p] == opp:
                blocked = True
                break
            elif flat[p] == player:
                cnt_p += 1
        if not blocked:
            score += LINE_WEIGHTS[cnt_p]
    return score

# negamax with alpha-beta, transposition table, node/time limit
class Searcher:
    def __init__(self, max_time=MAX_TIME, max_nodes=MAX_NODES):
        self.start_time = 0.0
        self.max_time = max_time
        self.max_nodes = max_nodes
        self.nodes = 0
        self.tt: Dict[Tuple[Tuple[int,...], int], int] = {}  # (board_tuple, player) -> score

    def out_of_time(self):
        return (time.time() - self.start_time) > self.max_time or self.nodes > self.max_nodes

    def negamax(self, flat, player, alpha, beta):
        # time/node cutoff
        if self.out_of_time():
            raise TimeoutError

        key = (tuple(flat), player)
        if key in self.tt:
            return self.tt[key]

        self.nodes += 1

        # terminal?
        if is_win_flat(flat, 3 - player):
            # 相手が直前に勝っていた -> 負け
            return -1000000 - 1
        if is_full(flat):
            return 0

        # quick heuristic bound: immediate winning moves
        moves = []
        for y in range(4):
            for x in range(4):
                if can_play(flat,x,y):
                    moves.append((x,y))
        # Move ordering: check immediate win first
        win_moves = []
        block_moves = []
        other_moves = []
        for (x,y) in moves:
            new, z = play_move(flat, x,y, player)
            if is_win_flat(new, player):
                win_moves.append((x,y))
                continue
            # check if opponent has immediate win next -> it's a blocking candidate
            opp_wins = False
            for (ox,oy) in moves:
                if ox == x and oy == y:
                    continue
                if can_play(new, ox,oy):
                    new2, _ = play_move(new, ox,oy, 3-player)
                    if is_win_flat(new2, 3-player):
                        opp_wins = True
                        break
            if opp_wins:
                block_moves.append((x,y))
            else:
                other_moves.append((x,y))
        ordered = win_moves + block_moves + other_moves

        # if immediate win exists, return best score quickly
        if win_moves:
            self.tt[key] = 1000000
            return 1000000

        best = -10**9
        for (x,y) in ordered:
            new, z = play_move(flat, x,y, player)
            val = -self.negamax(new, 3-player, -beta, -alpha)
            if val > best:
                best = val
            alpha = max(alpha, val)
            if alpha >= beta:
                break

        # store heuristic if not decisive
        if best == -10**9:
            # no moves? draw
            best = 0
        # store in TT
        self.tt[key] = best
        return best

    # wrapper: iterative deepening / time safe selection of best move
    def find_best(self, flat, player):
        self.start_time = time.time()
        self.nodes = 0
        self.tt.clear()

        # Legal moves
        moves = [(x,y) for y in range(4) for x in range(4) if can_play(flat,x,y)]
        if not moves:
            return (0,0)

        # Immediate win?
        for (x,y) in moves:
            new, z = play_move(flat, x,y, player)
            if is_win_flat(new, player):
                return (x,y)

        # Immediate block?
        for (x,y) in moves:
            new, z = play_move(flat, x,y, player)
            # If opponent has immediate win next, this move blocks?
            opp_can_win = False
            opp_moves = [(ox,oy) for oy in range(4) for ox in range(4) if can_play(new,ox,oy)]
            for (ox,oy) in opp_moves:
                new2, _ = play_move(new, ox,oy, 3-player)
                if is_win_flat(new2, 3-player):
                    opp_can_win = True
                    break
            if opp_can_win:
                return (x,y)

        # Order moves by heuristic to improve alpha-beta
        scored = []
        for (x,y) in moves:
            new, z = play_move(flat, x,y, player)
            sc = evaluate_flat(new, player) - evaluate_flat(new, 3-player)
            scored.append(((x,y), sc))
        scored.sort(key=lambda t: t[1], reverse=True)
        ordered_moves = [t[0] for t in scored]

        best_move = ordered_moves[0]
        best_score = -10**9

        # We will try to search; but limit by time/nodes. Use try/except to cut off.
        try:
            for (x,y) in ordered_moves:
                if self.out_of_time():
                    break
                new, z = play_move(flat, x,y, player)
                score = -self.negamax(new, 3-player, -1000001, 1000001)
                if score > best_score or (score == best_score and random.random() < 0.1):
                    best_score = score
                    best_move = (x,y)
        except TimeoutError:
            # time out: return current best
            pass

        return best_move

class MyAI(Alg3D):

    # --- ここが外部向けの get_move ---
    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報 board[z][y][x]
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        """
        強い Qubic (4x4x4) AI
        - get_move は (x,y) を返す。
        - board[z][y][x] が 0/1/2 で与えられる前提。
        """
        flat = board_to_flat(board)
        searcher = Searcher(max_time=MAX_TIME, max_nodes=MAX_NODES)
        # 早期勝ち/ブロックのチェックと最善手を探す
        move = searcher.find_best(flat, player)
        return move
