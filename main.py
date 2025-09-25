from typing import List, Tuple
# from local_driver import Alg3D, Board # ローカル検証用
# from local_driver import Alg3D # ローカル検証用
import math
import copy
from framework import Alg3D # 本番用


class Board:
    def __init__(self, board: List[List[List[int]]]):
        self.board = board
        self.mask_lines = self.create_mask_lines()
        self.black_board, self.white_board = self.board_to_bitboards()

     # ビットボード変換ヘルパー
    def board_to_bitboards(self):
        p1 = 0
        p2 = 0
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    idx = z*16 + y*4 + x
                    if self.board[z][y][x] == 1:
                        p1 |= 1 << idx
                    elif self.board[z][y][x] == 2:
                        p2 |= 1 << idx
        return p1, p2

    # 有効手列挙
    def valid_moves(self) -> List[Tuple[int,int]]:
        moves = []
        for x in range(4):
            for y in range(4):
                if self.board[3][y][x] == 0:
                    moves.append((x, y))
        # print(moves)
        return moves
    
    def print_board(self, board: List[List[List[int]]]):
        for z in range(4):
            print(f"Layer {z}:")
            for y in range(4):
                print(" ".join(str(board[z][y][x]) for x in range(4)))
            print()
    
    def put_player_bit(self, x: int, y: int, player: int) -> List[List[List[int]]]:
        for z in range(4):
            if self.board[z][y][x] == 0:
                self.board[z][y][x] = player
                # ビットボード更新
                idx = z*16 + y*4 + x
                if player == 1:
                    self.black_board |= 1 << idx
                elif player == 2:
                    self.white_board |= 1 << idx
                break
    
    def rollback_player_bit(self, x: int, y: int) -> List[List[List[int]]]:
        for z in range(4):
            if self.board[3 - z][y][x] != 0:
                self.board[3 - z][y][x] = 0
                # ビットボード更新
                idx = (3 - z)*16 + y*4 + x
                self.black_board &= ~(1 << idx)
                self.white_board &= ~(1 << idx)
                break

    # 黒勝利判定
    def is_win_black(self) -> bool:
        return self.is_win(self.black_board)

    # 白勝利判定
    def is_win_white(self) -> bool:
        return self.is_win(self.white_board)

    # 勝利判定
    def is_win(self, bitboard: int) -> bool:
        # 斜め種類
        for mask in self.mask_lines:
            if (bitboard & mask) == mask:
                return True
        return False

    # 勝利判定ライン
    def create_mask_lines(self):
        lines = []

        # x方向
        for z in range(4):
            for y in range(4):
                mask = 0
                for x in range(4):
                    mask |= 1 << (z*16 + y*4 + x)
                lines.append(mask)
        # y方向
        for z in range(4):
            for x in range(4):
                mask = 0
                for y in range(4):
                    mask |= 1 << (z*16 + y*4 + x)
                lines.append(mask)
        # z方向
        for y in range(4):
            for x in range(4):
                mask = 0
                for z in range(4):
                    mask |= 1 << (z*16 + y*4 + x)
                lines.append(mask)

        # 斜め
        # x方向
        for x in range(4):
            mask = 0
            # 斜め1
            for w in range(4):
                z = w
                y = w
                mask |= 1<< (z*16 + y*4 + x)
            lines.append(mask)
            mask = 0
            for w in range(4):
                z = w
                y = 3 - w
                mask |= 1<< (z*16 + y*4 + x)
            lines.append(mask)
        # y方向　# 斜め1
        for y in range(4):
            mask = 0
            
            for w in range(4):
                z = w
                x = w
                mask |= 1<< (z*16 + y*4 + x)
            lines.append(mask)
            mask = 0
            for w in range(4):
                z = w
                x = 3 - w
                mask |= 1<< (z*16 + y*4 + x)
            lines.append(mask)
        
        # z方向　# 斜め1
        for z in range(4):
            mask = 0
            for w in range(4):
                y = w
                x = w
                mask |= 1<< (z*16 + y*4 + x)
            lines.append(mask)
            mask = 0
            for w in range(4):
                y = w
                x = 3 - w
                mask |= 1<< (z*16 + y*4 + x)
            lines.append(mask)
            # 立体対角（4本）
        
        mask = 0
        for i in range(4):
            mask |= 1 << (i*16 + i*4 + i)
        lines.append(mask)

        mask = 0
        for i in range(4):
            mask |= 1 << (i*16 + i*4 + (3-i))
        lines.append(mask)

        mask = 0
        for i in range(4):
            mask |= 1 << (i*16 + (3-i)*4 + i)
        lines.append(mask)

        mask = 0
        for i in range(4):
            mask |= 1 << (i*16 + (3-i)*4 + (3-i))
        lines.append(mask)

        return lines

class MyAI(Alg3D):
    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]: # 置く場所(x, y)
        # ここにアルゴリズムを書く
        board_instance = Board(board)
        _, move = self.negamax(board_instance, depth=3, alpha=-math.inf, beta=math.inf, player=player)
        return move

    # 評価関数（簡易: 3連を狙う）
    def evaluate(self, board: Board, player: int) -> int:
        # 勝ち判定最優先
        if board.is_win_black():
            return 1000 if player == 1 else -1000
        if board.is_win_white():
            return 1000 if player == 2 else -1000
        return 0  # ここを拡張して3連/2連加点可能

    # αβ探索
    def negamax(self, board: Board, depth: int, alpha: int, beta: int, player: int, color: int = 1):
        moves = board.valid_moves()
        if depth == 0 or not moves:
            return color * self.evaluate(board, player), None  # 手がない場合は引き分け

        best_move = moves[0]
        max_eval = -math.inf

        for x, y in moves:
            z = board.put_player_bit(x, y, player)

            # 勝利判定を手ごとに実施
            if (player == 1 and board.is_win_black()) or (player == 2 and board.is_win_white()):
                eval_score = 1000 + depth
                board.rollback_player_bit(x, y)
                return eval_score, (x, y)  # 勝ち手なら即返す

            eval_score, _ = self.negamax(board, depth - 1, -beta, -alpha, 3 - player, -color)
            eval_score = -eval_score
            board.rollback_player_bit(x, y)

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = (x, y)

            alpha = max(alpha, eval_score)
            if alpha >= beta:
                break  # βカット

        return max_eval, best_move
