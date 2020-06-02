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

    # 57 https://leetcode.com/problems/insert-interval/
    def insert(self, intervals: List[List[int]],
               newInterval: List[int]) -> List[List[int]]:
        i = 0
        while i < len(intervals):
            if newInterval[0] <= intervals[i][0]:
                break
            i += 1

        intervals.insert(i, newInterval)

        res = [intervals[0]]
        for i in range(1, len(intervals)):
            if intervals[i][0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                res.append(intervals[i])

        return res

    # 58 https://leetcode.com/problems/length-of-last-word/
    def lengthOfLastWord(self, s: str) -> int:
        s = s.strip()
        words = s.split(" ")
        return len(words[-1])

    # 59 https://leetcode.com/problems/spiral-matrix-ii/
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0] * n for _ in range(n)]
        direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = set()

        i, j = 0, 0
        t = n * n
        d = 0
        while len(visited) != t:
            visited.add((i, j))
            matrix[i][j] = len(visited)
            next_i = i + direction[d][0]
            next_j = j + direction[d][1]
            if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n or (
                    next_i, next_j) in visited:
                d = (d + 1) % 4
            i += direction[d][0]
            j += direction[d][1]

        return matrix

    # 60 https://leetcode.com/problems/permutation-sequence/
    def getPermutation(self, n: int, k: int) -> str:
        factorials, nums = [1], ['1']
        for i in range(1, n):
            factorials.append(factorials[i - 1] * i)
            nums.append(str(i + 1))

        k -= 1

        res = []
        for i in range(n - 1, -1, -1):
            idx = k // factorials[i]
            k -= idx * factorials[i]

            res.append(nums[idx])
            del nums[idx]

        return ''.join(res)

    # 61 https://leetcode.com/problems/rotate-list/
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return head

        cur = head
        length = 1
        while cur.next:
            length += 1
            cur = cur.next

        k = k % length
        if k == 0:
            return head

        pre = None
        new_head = head
        for i in range(length - k):
            pre = new_head
            new_head = new_head.next

        pre.next = None
        cur = new_head
        while cur.next:
            cur = cur.next

        cur.next = head

    # 62 https://leetcode.com/problems/unique-paths/
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n for _ in range(m)]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]

    # 63 https://leetcode.com/problems/unique-paths-ii/
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        dp = [[0] * n for _ in range(m)]

        if obstacleGrid[0][0] == 0:
            dp[0][0] = 1

        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    continue
                if i > 0:
                    dp[i][j] += dp[i - 1][j]
                if j > 0:
                    dp[i][j] += dp[i][j - 1]

        return dp[m - 1][n - 1]

    # 64 https://leetcode.com/problems/minimum-path-sum/
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        dp = [[0] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                dp[i][j] = grid[i][j]
                if i == 0 and j == 0:
                    continue
                if i == 0:
                    dp[i][j] += dp[i][j - 1]
                elif j == 0:
                    dp[i][j] += dp[i - 1][j]
                else:
                    dp[i][j] += min(dp[i][j - 1], dp[i - 1][j])

        return dp[m - 1][n - 1]

    # 65 https://leetcode.com/problems/valid-number/
    def isNumber(self, s: str) -> bool:
        # https://leetcode.com/problems/valid-number/discuss/348874/Python-3-Regex-with-example
        import re
        #Example:               +-     1 or 1. or 1.2 or .2   e +- 1
        engine = re.compile(r"^[+-]?((\d+\.?\d*)|(\d*\.?\d+))(e[+-]?\d+)?$")
        return engine.match(s.strip(" "))

    # 66 https://leetcode.com/problems/plus-one/
    def plusOne(self, digits: List[int]) -> List[int]:
        carrier = 1
        n = len(digits) - 1

        for i in range(n, -1, -1):
            if carrier == 0:
                break
            sum_val = carrier + digits[i]
            carrier = sum_val // 10
            remainder = sum_val % 10
            digits[i] = remainder

        if carrier != 0:
            digits.insert(0, carrier)

        return digits

    # 67 https://leetcode.com/problems/add-binary/
    def addBinary(self, a: str, b: str) -> str:
        res = []

        if len(a) < len(b):
            a, b = b, a

        a = list(reversed(a))
        b = list(reversed(b))

        carrier = 0
        for i in range(len(b)):
            sum_val = int(a[i]) + int(b[i]) + carrier
            remainder = sum_val % 2
            carrier = sum_val // 2
            res.insert(0, str(remainder))

        for i in range(len(b), len(a)):
            sum_val = int(a[i]) + carrier
            remainder = sum_val % 2
            carrier = sum_val // 2
            res.insert(0, str(remainder))

        if carrier != 0:
            res.insert(0, "1")

        return "".join(res)

    # 68 https://leetcode.com/problems/text-justification/
    def fullJustify(self, words, maxWidth):
        res, cur, num_of_letters = [], [], 0
        for w in words:
            if num_of_letters + len(w) + len(cur) > maxWidth:
                for i in range(maxWidth - num_of_letters):
                    cur[i % (len(cur) - 1 or 1)] += ' '
                res.append(''.join(cur))
                cur, num_of_letters = [], 0
            cur += [w]
            num_of_letters += len(w)
        return res + [' '.join(cur).ljust(maxWidth)]

    # 69 https://leetcode.com/problems/sqrtx/
    def mySqrt(self, x: int) -> int:
        i, j = 1, x

        while i <= j:
            mid = (i + j) // 2

            small_sqr = mid**2
            large_sqr = (mid + 1)**2

            if small_sqr > x:
                j = mid - 1
            elif large_sqr <= x:
                i = mid + 1
            else:
                return mid

        return j

    # 70 https://leetcode.com/problems/climbing-stairs/
    def climbStairs(self, n: int) -> int:
        pass


if __name__ == "__main__":
    solution = Solution_2()
    # nums = [0, 1]
    # matrix = [[1, 3], [4, 5]]
    # target = 18
    # s = "()()"
    # words = ["foo", "bar", "oof"]

    res = solution.generateMatrix(2)
    print(res)