import sys
import bisect

from typing import List
from Structure import ListNode


class Solution_2:
    # 51 https://leetcode.com/problems/n-queens/
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        board = [["."] * n for _ in range(n)]
        row = [False] * n
        column = [False] * n
        left = [False] * (2 * n - 1)
        right = [False] * (2 * n - 1)
        self.__solveNQueensHelper(n, res, 0, board, row, column, left, right)
        return res

    def __solveNQueensHelper(self, n: int, res: List[List[str]], i: int,
                             board: List[List[str]], row: List[bool],
                             column: List[bool], left: List[bool],
                             right: List[bool]) -> None:
        if i == n:
            res.append(["".join(b) for b in board])
            return

        for j in range(n):
            if row[i] or column[j] or left[n - i + j - 1] or right[j + i]:
                continue
            row[i] = True
            column[j] = True
            left[n - i + j - 1] = True
            right[j + i] = True
            board[i][j] = "Q"
            self.__solveNQueensHelper(n, res, i + 1, board, row, column, left,
                                      right)
            row[i] = False
            column[j] = False
            left[n - i + j - 1] = False
            right[j + i] = False
            board[i][j] = "."

    # 52 https://leetcode.com/problems/n-queens-ii/
    def totalNQueens(self, n: int) -> int:
        res = []
        board = [["."] * n for _ in range(n)]
        row = [False] * n
        column = [False] * n
        left = [False] * (2 * n - 1)
        right = [False] * (2 * n - 1)
        self.__totalNQueensHelper(n, res, 0, board, row, column, left, right)
        return len(res)

    def __totalNQueensHelper(self, n: int, res: List[List[str]], i: int,
                             board: List[List[str]], row: List[bool],
                             column: List[bool], left: List[bool],
                             right: List[bool]) -> None:
        if i == n:
            res.append(["".join(b) for b in board])
            return

        for j in range(n):
            if row[i] or column[j] or left[n - i + j - 1] or right[j + i]:
                continue
            row[i] = True
            column[j] = True
            left[n - i + j - 1] = True
            right[j + i] = True
            board[i][j] = "Q"
            self.__totalNQueensHelper(n, res, i + 1, board, row, column, left,
                                      right)
            row[i] = False
            column[j] = False
            left[n - i + j - 1] = False
            right[j + i] = False
            board[i][j] = "."

    # 53 https://leetcode.com/problems/maximum-subarray/
    def maxSubArray(self, nums: List[int]) -> int:
        res = -2**31
        i = j = 0
        curSum = 0

        while j < len(nums):
            curSum += nums[j]
            res = max(res, curSum)
            while curSum < 0 and i <= j:
                curSum -= nums[i]
                i += 1
                if i <= j:
                    res = max(res, curSum)
            j += 1

        return res


if __name__ == "__main__":
    solution = Solution_2()
    # nums = [7, 0, 9, 6, 9, 6, 1, 7, 9, 0, 1, 2, 9, 0, 3]
    # target = 18
    # s = "()()"
    words = ["foo", "bar", "oof"]

    res = solution.solveNQueens(3)
    print(res)