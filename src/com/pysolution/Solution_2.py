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

    # 54 https://leetcode.com/problems/spiral-matrix/
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if len(matrix) == 0:
            return []

        m, n = len(matrix), len(matrix[0])
        res = []

        step = 0
        p, q = 0, 0
        while step < n:
            for i in range(q, q + n - step):
                res.append(matrix[p][i])

            if m - step - 1 <= 0:
                return res
            for i in range(p + 1, p + m - step):
                res.append(matrix[i][q + n - step - 1])

            if n - step - 1 <= 0:
                return res
            for i in range(q + n - step - 2, q - 1, -1):
                res.append(matrix[p + m - step - 1][i])

            if m - step - 2 <= 0:
                return res
            for i in range(p + m - step - 2, p, -1):
                res.append(matrix[i][q])

            p += 1
            q += 1
            step += 2

        return res

    # 55 https://leetcode.com/problems/jump-game/
    def canJump(self, nums: List[int]) -> bool:
        maxRange = 0

        i, n = 0, len(nums) - 1
        while i <= maxRange:
            maxRange = max(maxRange, i + nums[i])
            if maxRange >= n:
                return True
            i += 1

        return maxRange + 1 >= len(nums)

    # 56 https://leetcode.com/problems/merge-intervals/
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 0:
            return []

        intervals.sort()
        res = [intervals[0]]

        for i in range(1, len(intervals)):
            if intervals[i][0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                res.append(intervals[i])

        return res


if __name__ == "__main__":
    solution = Solution_2()
    # nums = [0, 1]
    matrix = [[1, 3], [4, 5]]
    # target = 18
    # s = "()()"
    words = ["foo", "bar", "oof"]

    res = solution.merge(matrix)
    print(res)