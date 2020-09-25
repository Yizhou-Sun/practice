import sys
import bisect

from typing import List
from .Structure import ListNode
from .Structure import TreeNode


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
        if n == 1:
            return 1
        if n == 2:
            return 2

        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2

        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n]

    # 71 https://leetcode.com/problems/simplify-path/
    def simplifyPath(self, path: str) -> str:
        stack = []

        dirs = path.split("/")

        for p in dirs:
            if p == "" or p == ".":
                continue

            if p == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(p)

        return "/" + "/".join(stack)

    # 72 https://leetcode.com/problems/edit-distance/
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(n + 1):
            dp[0][i] = i
        for i in range(m + 1):
            dp[i][0] = i

        for i in range(m):
            for j in range(n):
                w1Char = word1[i]
                w2Char = word2[j]

                if w1Char == w2Char:
                    dp[i + 1][j + 1] = min(dp[i][j], dp[i + 1][j] + 1,
                                           dp[i][j + 1] + 1)
                else:
                    dp[i +
                       1][j +
                          1] = min(dp[i][j], dp[i + 1][j], dp[i][j + 1]) + 1

        return dp[m][n]

    # 73 https://leetcode.com/problems/set-matrix-zeroes/
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        flag = 0
        col = False
        row = False

        for i in range(m):
            if matrix[i][0] == 0:
                col = True

        for j in range(n):
            if matrix[0][j] == 0:
                row = True

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    matrix[i][0] = flag
                    matrix[0][j] = flag

        for i in range(1, m):
            for j in range(1, n):
                if matrix[0][j] == flag or matrix[i][0] == flag:
                    matrix[i][j] = 0

        if col:
            for i in range(m):
                matrix[i][0] = 0

        if row:
            for j in range(n):
                matrix[0][j] = 0

    # 74 https://leetcode.com/problems/search-a-2d-matrix/
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        if m == 0:
            return False
        n = len(matrix[0])
        if n == 0:
            return False

        i, j = 0, m - 1

        while i <= j:
            mid = (i + j) // 2
            if matrix[mid][0] == target:
                return True
            elif matrix[mid][0] > target:
                j = mid - 1
            else:
                i = mid + 1

        row = j
        i, j = 0, n - 1

        while i <= j:
            mid = (i + j) // 2
            if matrix[row][mid] == target:
                return True
            elif matrix[row][mid] > target:
                j = mid - 1
            else:
                i = mid + 1

        return False

    # 75 https://leetcode.com/problems/sort-colors/
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = -1
        mid = 0
        right = len(nums)

        while mid < right:
            if nums[mid] == 1:
                mid += 1
            elif nums[mid] == 2:
                nums[mid], nums[right - 1] = nums[right - 1], nums[mid]
                right -= 1
            else:
                nums[mid], nums[left + 1] = nums[left + 1], nums[mid]
                left += 1
                mid += 1

    # 76 https://leetcode.com/problems/minimum-window-substring/
    def minWindow(self, s: str, t: str) -> str:
        minLen = len(s) + 1
        missing = len(t)
        charCount = {}
        resL, resR = 0, 0

        for c in t:
            charCount[c] = charCount.get(c, 0) + 1

        left, right = 0, 0
        while right < len(s):
            rChar = s[right]
            if rChar in charCount:
                charCount[rChar] -= 1

                if charCount[rChar] >= 0:
                    missing -= 1
            right += 1

            while not missing:
                if right - left < minLen:
                    minLen = right - left
                    resL, resR = left, right
                lChar = s[left]
                if lChar in charCount:
                    charCount[lChar] += 1
                    if charCount[lChar] > 0:
                        missing += 1
                left += 1

        return s[resL:resR]

    # 77 https://leetcode.com/problems/combinations/
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        self.__combineHelper(res, [], 1, n, k)
        return res

    def __combineHelper(self, res: List[List[int]], cur: List[int], start: int,
                        n: int, k: int) -> None:
        if len(cur) == k:
            res.append(list(cur))

        for i in range(start, n + 1):
            cur.append(i)
            self.__combineHelper(res, cur, i + 1, n, k)
            del cur[-1]

    # 78 https://leetcode.com/problems/subsets/
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)

        for i in range(n + 1):
            self.__subsetsHelper(res, [], 1, nums, i)

        return res

    def __subsetsHelper(self, res: List[List[int]], cur: List[int], start: int,
                        nums: List[int], k: int) -> None:
        if len(cur) == k:
            res.append(list(cur))

        for i in range(start, len(nums) + 1):
            cur.append(nums[i - 1])
            self.__subsetsHelper(res, cur, i + 1, nums, k)
            del cur[-1]

    # 79 https://leetcode.com/problems/word-search/
    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])
        visited = [[False] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if self.__existDfsSearch(board, i, j, word, 0, visited):
                    return True

        return False

    def __existDfsSearch(self, board: List[List[int]], x: int, y: int,
                         word: str, p: int, visited: List[List[bool]]) -> bool:
        m = len(board)
        n = len(board[0])

        if visited[x][y] or board[x][y] != word[p]:
            return False

        visited[x][y] = True

        p += 1
        if p == len(word):
            return True

        if x > 0 and self.__existDfsSearch(board, x - 1, y, word, p, visited):
            return True

        if y + 1 < n and self.__existDfsSearch(board, x, y + 1, word, p,
                                               visited):
            return True

        if x + 1 < m and self.__existDfsSearch(board, x + 1, y, word, p,
                                               visited):
            return True

        if y > 0 and self.__existDfsSearch(board, x, y - 1, word, p, visited):
            return True

        visited[x][y] = False

        return False

    # 80 https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return len(nums)

        left, right = 1, 2
        pre = None

        while right < len(nums):
            if nums[right] == nums[left] and nums[right] == nums[left - 1]:
                right += 1
            else:
                nums[left + 1] = nums[right]
                left += 1
                right += 1

        return left + 1

    # 81 https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
    def search(self, nums: List[int], target: int) -> bool:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True

            if nums[left] == nums[mid] and nums[right] == nums[mid]:
                left += 1
                right -= 1
            elif nums[left] <= nums[mid]:
                if target > nums[mid] or target < nums[left]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if target > nums[right] or target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1

        return False

    # 82 https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummyHead = ListNode(-1)
        curr = dummyHead
        dummyHead.next = head

        while head:
            if head.next and head.val == head.next.val:
                while head.next and head.val == head.next.val:
                    head = head.next
                head = head.next
                curr.next = head
            else:
                curr = curr.next
                head = head.next

        return dummyHead.next

    # 83 https://leetcode.com/problems/remove-duplicates-from-sorted-list/
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        curr = head

        while curr and curr.next:
            if curr.val == curr.next.val:
                curr.next = curr.next.next
            else:
                curr = curr.next

        return head

    # 84 https://leetcode.com/problems/largest-rectangle-in-histogram/
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        stack = [-1]
        res = 0

        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                top = stack.pop()
                res = max(res, heights[top] * (i - stack[-1] - 1))
            stack.append(i)

        while stack[-1] != -1:
            top = stack.pop()
            res = max(res, heights[top] * (n - stack[-1] - 1))

        return res

    # 85 https://leetcode.com/problems/maximal-rectangle/
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        maxarea = 0

        dp = [[0] * len(matrix[0]) for _ in range(len(matrix))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == '0': continue

                # compute the maximum width and update dp with it
                width = dp[i][j] = dp[i][j - 1] + 1 if j else 1

                # compute the maximum area rectangle with a lower right corner at [i, j]
                for k in range(i, -1, -1):
                    width = min(width, dp[k][j])
                    maxarea = max(maxarea, width * (i - k + 1))
        return maxarea

    # 86 https://leetcode.com/problems/partition-list/
    def partition(self, head: ListNode, x: int) -> ListNode:
        prev = curr = dummyHead = ListNode(-1)
        curr.next = head

        while head and head.val < x:
            prev = head
            head = head.next
            curr = curr.next

        while head:
            if head.val < x:
                prev.next = head.next
                head.next = curr.next
                curr.next = head
                curr = curr.next
                head = prev.next
            else:
                prev = prev.next
                head = head.next

        return dummyHead.next

    # 87 https://leetcode.com/problems/scramble-string/
    def isScramble(self, s1: str, s2: str) -> bool:
        n, m = len(s1), len(s2)

        if n != m or sorted(s1) != sorted(s2):
            return False

        if n < 4 or s1 == s2:
            return True

        f = self.isScramble
        for i in range(1, n):
            if f(s1[:i], s2[:i]) and f(s1[i:], s2[i:]) or \
               f(s1[:i], s2[-i:]) and f(s1[i:], s2[:-i]):
                return True

        return False

    # 88 https://leetcode.com/problems/merge-sorted-array/
    def merge(self, nums1: List[int], m: int, nums2: List[int],
              n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p = m + n - 1

        i = m - 1
        j = n - 1

        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[p] = nums1[i]
                p -= 1
                i -= 1
            else:
                nums1[p] = nums2[j]
                p -= 1
                j -= 1

        while j >= 0:
            nums1[p] = nums2[j]
            p -= 1
            j -= 1

    # 89 https://leetcode.com/problems/gray-code/
    def grayCode(self, n: int) -> List[int]:
        res = [0]
        for i in range(n):
            for j in range(len(res) - 1, -1, -1):
                res.append(res[j] | 1 << i)

        return res

    # 90 https://leetcode.com/problems/subsets-ii/
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        self.__subsetsWithDupHelper(nums, res, [], 0)

        return res

    def __subsetsWithDupHelper(self, nums: List[int], res: List[List[int]],
                               curr: List[int], start: int) -> None:
        res.append(list(curr))

        i = start

        while i < len(nums):
            if i != start and nums[i] == nums[i - 1]:
                i += 1
                continue
            curr.append(nums[i])
            self.__subsetsWithDupHelper(nums, res, curr, i + 1)
            del curr[-1]
            i += 1

    # 89 https://leetcode.com/problems/decode-ways/
    def numDecodings(self, s: str) -> int:
        self.__records = {}

        return self.__numDecodingsHelper(s)

    def __numDecodingsHelper(self, s: str) -> int:
        if s.startswith("0"):
            return 0

        if s in self.__records:
            return self.__records[s]

        if len(s) <= 1:
            self.__records[s] = 1
            return 1

        res = self.__numDecodingsHelper(s[1:])

        if int(s[:2]) < 27:
            res += self.__numDecodingsHelper(s[2:])

        self.__records[s] = res

        return res

    # 92 https://leetcode.com/problems/reverse-linked-list-ii/
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if not head or m == n: return head

        dummyHead = ListNode(-1)

        dummyHead.next = head
        curr = dummyHead

        for i in range(m - 1):
            curr = curr.next

        last = curr.next
        for i in range(n - m):
            node = curr.next
            curr.next = last.next
            last.next = last.next.next
            curr.next.next = node

        return dummyHead.next

    # 93 https://leetcode.com/problems/restore-ip-addresses/
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []

        self.__restoreIpAddrHelper(s, res, [], 0)

        return res

    def __restoreIpAddrHelper(self, s: str, res: List[str], curr: List[str],
                              start: int) -> None:
        if len(curr) == 4:
            if len(s) == start:
                res.append(".".join(curr))
            return

        end = min(len(s), start + 3)
        for i in range(start, end):
            num = s[start:i + 1]
            if num[0] == "0" and len(num) != 1:
                continue
            if int(num) > 255:
                continue

            curr.append(num)
            self.__restoreIpAddrHelper(s, res, curr, i + 1)
            del curr[-1]

    # 94 https://leetcode.com/problems/binary-tree-inorder-traversal/
    def inorderTraversal(self, root):

        res, stack = [], []
        while True:
            while root:
                stack.append(root)
                root = root.left
            if not stack:
                return res
            node = stack.pop()
            res.append(node.val)
            root = node.right

    # 95 https://leetcode.com/problems/unique-binary-search-trees-ii/
    def generateTrees(self, n: int) -> List[TreeNode]:
        if n == 0:
            return []

        return self.__generateTreesHelper(1, n + 1)

    def __generateTreesHelper(self, start: int, end: int) -> List[TreeNode]:
        if start >= end:
            return [None]

        res = []
        for i in range(start, end):
            leftTrees = self.__generateTreesHelper(start, i)
            rightTrees = self.__generateTreesHelper(i + 1, end)
            for lt in leftTrees:
                for rt in rightTrees:
                    root = TreeNode(i, lt, rt)
                    res.append(root)

        return res

    # 96 https://leetcode.com/problems/unique-binary-search-trees/
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(n + 1):
            for j in range(i):
                dp[i] += dp[j] * dp[i - j - 1]

        return dp[n]

    # 97 https://leetcode.com/problems/interleaving-string/
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        cache = [[-1] * len(s2) for _ in range(len(s1))]

        return self.__isInterleaveHelper(s1, 0, s2, 0, s3, 0, cache)

    def __isInterleaveHelper(self, s1: str, i: int, s2: str, j: int, s3: str,
                             k: int, cache: List[List[int]]) -> bool:
        if i == len(s1):
            return s2[j:] == s3[k:]
        if j == len(s2):
            return s1[i:] == s3[k:]

        if cache[i][j] != -1:
            return cache[i][j] == 1

        res = False
        if i < len(s1) and s1[i] == s3[k]:
            res = res or self.__isInterleaveHelper(s1, i + 1, s2, j, s3, k + 1,
                                                   cache)
        if j < len(s2) and s2[j] == s3[k]:
            res = res or self.__isInterleaveHelper(s1, i, s2, j + 1, s3, k + 1,
                                                   cache)

        cache[i][j] = 1 if res else 0

        return res

    # 98 https://leetcode.com/problems/validate-binary-search-tree/
    def isValidBST(self, root: TreeNode) -> bool:
        return self.__isValidBSTHelper(root, None, None)

    def __isValidBSTHelper(self, root: TreeNode, minVal: int,
                           maxVal: int) -> bool:
        if not root:
            return True

        if minVal is not None and root.val <= minVal:
            return False
        if maxVal is not None and root.val >= maxVal:
            return False

        return self.__isValidBSTHelper(root.left, minVal,
                                       root.val) and self.__isValidBSTHelper(
                                           root.right, root.val, maxVal)

    # 99 https://leetcode.com/problems/recover-binary-search-tree/
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        stack = []

        x = y = pred = None

        while stack or root:
            while root:
                stack.append(root)
                root = root.left

            root = stack.pop()

            if pred and root.val < pred.val:
                y = root
                if x is None:
                    x = pred
                else:
                    break

            pred = root
            root = root.right

        x.val, y.val = y.val, x.val

    # 100 https://leetcode.com/problems/same-tree/
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True

        if not p or not q:
            return False

        if p.val != q.val:
            return False

        return self.isSameTree(p.left, q.left) and self.isSameTree(
            p.right, q.right)


if __name__ == "__main__":
    solution = Solution_2()
    # nums = [0, 1]
    # matrix = [[1, 3], [4, 5]]
    # target = 18
    # s = "()()"
    # words = ["foo", "bar", "oof"]

    res = solution.generateTrees(3)
    print(res)