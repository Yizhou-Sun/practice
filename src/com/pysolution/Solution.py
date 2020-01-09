import sys
import bisect

from typing import List
from Structure import ListNode


class SolutionPage1:
    # 1 https://leetcode.com/problems/two-sum/
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        myDict = {}
        for i in range(len(nums)):
            if target - nums[i] in myDict:
                return [myDict[target - nums[i]], i]
            else:
                myDict[nums[i]] = i

    # 2 https://leetcode.com/problems/add-two-numbers/
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        cur = dummy = ListNode(0)
        carry = 0
        dummy.next = cur
        while l1 or l2 or carry != 0:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next

            cur.next = ListNode(carry % 10)
            cur = cur.next
            carry = int(carry / 10)

        return dummy.next

    # 3 https://leetcode.com/problems/longest-substring-without-repeating-characters/
    def lengthOfLongestSubstring(self, s: str) -> int:
        # res = i = j = 0
        # isDuplicate = set()

        # while j < len(s):
        #     if s[j] in isDuplicate:
        #         isDuplicate.remove(s[i])
        #         i += 1
        #     else:
        #         isDuplicate.add(s[j])
        #         j += 1
        #     res = max(res, j - i)

        charMap = {}
        res = j = 0

        for i in range(len(s)):
            if s[i] in charMap and j <= charMap[s[i]]:
                j = charMap[s[i]] + 1
            charMap[s[i]] = i
            res = max(res, i - j + 1)

        return res

    # 4 https://leetcode.com/problems/median-of-two-sorted-arrays/
    def findMedianSortedArrays(self, nums1: List[int],
                               nums2: List[int]) -> float:
        def findNthFromArrays(nums1: List[int], nums2: List[int],
                              i: int) -> int:
            if not nums1:
                return nums2[i]
            if not nums2:
                return nums1[i]

            if i == 0:
                return min(nums1[0], nums2[0])

            mid = int((i - 1) / 2)
            m1 = m2 = sys.maxsize

            if mid < len(nums1):
                m1 = nums1[mid]
            if mid < len(nums2):
                m2 = nums2[mid]

            if m1 < m2:
                return findNthFromArrays(nums1[mid + 1:], nums2, i - mid - 1)
            else:
                return findNthFromArrays(nums1, nums2[mid + 1:], i - mid - 1)

        n = len(nums1) + len(nums2)
        if n % 2 != 0:
            return findNthFromArrays(nums1, nums2, int(n / 2))
        else:
            m1 = findNthFromArrays(nums1, nums2, int(n / 2))
            m2 = findNthFromArrays(nums1, nums2, int(n / 2) - 1)
            return (m1 + m2) / 2

    # 5 https://leetcode.com/problems/longest-palindromic-substring/
    def longestPalindrome(self, s: str) -> str:
        if len(s) == 0:
            return ""
        self.l = 0
        self.r = 1

        for i in range(len(s)):
            self.extendPalindrome(s, i, i)
            self.extendPalindrome(s, i, i + 1)

        return s[self.l:self.r]

    def extendPalindrome(self, s: str, i: int, j: int) -> None:
        while i >= 0 and j < len(s) and s[i] == s[j]:
            i -= 1
            j += 1
        if self.r - self.l < j - i:
            self.l = i + 1
            self.r = j
        return

    # 6 https://leetcode.com/problems/zigzag-conversion/
    def convert(self, s: str, numRows: int) -> str:
        n = len(s)
        if numRows <= 1 or numRows >= n:
            return s

        res = ""
        interval = delta = 2 * numRows - 2

        for i in range(numRows):
            j = i
            while j < n:
                res += s[j]
                if j + delta < n and delta != 0 and delta != interval:
                    res += s[j + delta]
                j += interval
            delta -= 2

        return res

    # 7 https://leetcode.com/problems/reverse-integer/
    def reverse(self, x: int) -> int:
        maxInt = 2**31 - 1
        minInt = 2**31
        sign = 1 if x >= 0 else -1
        num = abs(x)

        res = 0
        while num != 0:
            reminder = num % 10
            num = int(num / 10)
            res = res * 10 + reminder

        if sign == 1:
            return res if res <= maxInt else 0
        else:
            return -res if res <= minInt else 0

    # 8 https://leetcode.com/problems/string-to-integer-atoi/
    def myAtoi(self, s: str) -> int:
        maxInt = 2**31 - 1
        minInt = -2**31
        s = s.strip()
        if len(s) == 0:
            return 0

        i = res = 0
        sign = 1
        if s[i] == "-":
            sign = -1
            i += 1
        elif s[i] == "+":
            sign = 1
            i += 1

        while i < len(s) and s[i].isdigit():
            res = res * 10 + int(s[i])
            i += 1

        return max(minInt, min(sign * res, maxInt))

    # 9 https://leetcode.com/problems/palindrome-number/
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        s = str(x)

        i, j = 0, len(s) - 1

        while i < j and s[i] == s[j]:
            i += 1
            j -= 1

        return i >= j

    # 10 https://leetcode.com/problems/regular-expression-matching/
    def isMatch_Regular_Expression_Matching(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]

        for i in range(0, m + 1):
            for j in range(0, n + 1):
                if i == 0 and j == 0:
                    dp[i][j] = True
                    continue
                schar = "" if i == 0 else s[i - 1]
                pchar = "" if j == 0 else p[j - 1]
                if (pchar == "." and i != 0) or pchar == schar:
                    dp[i][j] = dp[i - 1][j - 1]
                elif pchar == "*":
                    pre = p[j - 2]
                    dp[i][j] = dp[i][j -
                                     2] or ((pre == "." or pre == schar) and
                                            (dp[i - 1][j] or dp[i - 1][j - 1]))

        return dp[m][n]

    # 11 https://leetcode.com/problems/container-with-most-water/
    def maxArea(self, height: List[int]) -> int:
        res = 0
        i, j = 0, len(height) - 1

        while i < j:
            res = max(res, min(height[i], height[j]) * (j - i))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return res

    # 12 https://leetcode.com/problems/integer-to-roman/
    def intToRoman(self, num: int) -> str:
        int2Roman = {
            1000: "M",
            900: "CM",
            500: "D",
            400: "CD",
            100: "C",
            90: "XC",
            50: "L",
            40: "XL",
            10: "X",
            9: "IX",
            5: "V",
            4: "IV",
            1: "I",
        }
        res = ""
        for key, value in int2Roman.items():
            for _ in range(int(num / key)):
                res += value
            num %= key
        return res

    # 13 https://leetcode.com/problems/roman-to-integer/
    def romanToInt(self, s: str) -> int:
        roman2Int = {
            "M": 1000,
            "D": 500,
            "C": 100,
            "L": 50,
            "X": 10,
            "V": 5,
            "I": 1
        }
        res = 0

        for i in range(len(s) - 1):
            if roman2Int[s[i]] < roman2Int[s[i + 1]]:
                res -= roman2Int[s[i]]
            else:
                res += roman2Int[s[i]]
        res += roman2Int[s[len(s) - 1]]

        return res

    # 14 https://leetcode.com/problems/longest-common-prefix/
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ""
        if len(strs) == 0:
            return res

        n = len(strs[0])
        for i in range(n):
            c = strs[0][i]
            for s in strs:
                if i >= len(s) or c != s[i]:
                    return res
            res += c

        return res

    # 15 https://leetcode.com/problems/3sum/
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        for i, v in enumerate(nums):
            if i > 0 and v == nums[i - 1]:
                continue
            if v > 0:
                break
            m, n = i + 1, len(nums) - 1

            while m < n:
                total = nums[m] + nums[n] + v
                if total > 0:
                    n -= 1
                elif total < 0:
                    m += 1
                else:
                    res.append([v, nums[m], nums[n]])
                    m += 1
                    n -= 1
                    while m < len(nums) and nums[m] == nums[m - 1]:
                        m += 1
                    while n >= 0 and nums[n] == nums[n + 1]:
                        n -= 1
                pass

        return res

    # 16 https://leetcode.com/problems/3sum-closest/
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        res = nums[0] + nums[1] + nums[2]

        for i, val in enumerate(nums):
            m, n = i + 1, len(nums) - 1
            while m < n:
                total = val + nums[m] + nums[n]
                diff = total - target
                if abs(diff) < abs(res - target):
                    res = total

                if diff < 0:
                    m += 1
                elif diff > 0:
                    n -= 1
                else:
                    return res

        return res

    # 17 https://leetcode.com/problems/letter-combinations-of-a-phone-number/
    def letterCombinations(self, digits: str) -> List[str]:
        if digits == "":
            return []
        num2letter = [
            "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"
        ]
        res = [""]

        for i in digits:
            tmp = []
            for c in num2letter[int(i)]:
                for s in res:
                    tmp.append(s + c)
            res = tmp

        return res

    # 18 https://leetcode.com/problems/4sum/
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        res = []
        self.__NthSumHelper(nums, 0, target, 4, res, [])
        return res

    def __NthSumHelper(self, nums: List[int], start: int, target: int, N: int,
                       res: List[List[int]], lt: List[int]) -> None:
        L = len(nums)
        if start == L or N * nums[start] > target:
            return

        if N == 2:
            i, j = start, L - 1
            while i < j:
                total = nums[i] + nums[j]
                if total == target:
                    res.append(lt + [nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
                elif total < target:
                    i += 1
                else:
                    j -= 1
        else:
            for i in range(start, L):
                if i == start or nums[i] != nums[i - 1]:
                    self.__NthSumHelper(nums, i + 1, target - nums[i], N - 1,
                                        res, lt + [nums[i]])

    # 19 https://leetcode.com/problems/remove-nth-node-from-end-of-list/
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummyHead = ListNode(-1)
        dummyHead.next = head

        first = dummyHead
        for _ in range(n):
            first = first.next
        delete = head
        pre = dummyHead

        while first.next:
            first = first.next
            delete = delete.next
            pre = pre.next
        pre.next = delete.next

        return dummyHead.next

    # 20 https://leetcode.com/problems/valid-parentheses/
    def isValid(self, s: str) -> bool:
        stack = []

        for c in s:
            if not stack:
                stack.append(c)
            elif c == '(' or c == '{' or c == "[":
                stack.append(c)
            elif c == ')' and stack.pop() != '(':
                return False
            elif c == '}' and stack.pop() != "{":
                return False
            elif c == ']' and stack.pop() != '[':
                return False

        return not stack

    # 21 https://leetcode.com/problems/merge-two-sorted-lists/
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        cur = dummyHead = ListNode(-1)

        while l1 and l2:
            if l1.val >= l2.val:
                cur.next = l2
                l2 = l2.next
            else:
                cur.next = l1
                l1 = l1.next
            cur = cur.next

        if l1:
            cur.next = l1
        else:
            cur.next = l2

        return dummyHead.next

    # 22 https://leetcode.com/problems/generate-parentheses/
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        self.__generateParenthesisHelper(res, "", n, 0)
        return res

    def __generateParenthesisHelper(self, res: List[str], s: str, i: int,
                                    unmatch: int) -> None:
        if i == 0 and unmatch == 0:
            res.append(s)
            return

        if i != 0:
            self.__generateParenthesisHelper(res, s + '(', i - 1, unmatch + 1)
        if unmatch != 0:
            self.__generateParenthesisHelper(res, s + ')', i, unmatch - 1)

    # 23 https://leetcode.com/problems/merge-k-sorted-lists/
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        L = len(lists)
        if L == 1:
            return lists[0]
        if L == 0:
            return None

        mid = L // 2
        left = self.mergeKLists(lists[:mid])
        right = self.mergeKLists(lists[mid:])

        return self.__mergeKListsHelper(left, right)

    def __mergeKListsHelper(self, l: ListNode, r: ListNode) -> ListNode:
        dummy = cur = ListNode(0)
        while l and r:
            if l.val < r.val:
                cur.next = l
                l = l.next
            else:
                cur.next = r
                r = r.next
            cur = cur.next
        cur.next = l or r

        return dummy.next

    # 24 https://leetcode.com/problems/swap-nodes-in-pairs/
    def swapPairs(self, head: ListNode) -> ListNode:
        k = 2
        cur = dummy = ListNode(0)

        while head:
            i = k
            count = head
            while count and i > 0:
                i -= 1
                count = count.next
            if i != 0:
                cur.next = head
                break

            pre, tail = None, head
            i = k
            while head and i > 0:
                i -= 1
                temp = head.next
                head.next = pre
                pre = head
                head = temp
            cur.next = pre
            cur = tail

        return dummy.next

    # 25 https://leetcode.com/problems/reverse-nodes-in-k-group/
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        cur = dummy = ListNode(0)

        while head:
            i = k
            count = head
            while count and i > 0:
                i -= 1
                count = count.next
            if i != 0:
                cur.next = head
                break

            pre, tail = None, head
            i = k
            while head and i > 0:
                i -= 1
                temp = head.next
                head.next = pre
                pre = head
                head = temp
            cur.next = pre
            cur = tail

        return dummy.next

    # 26 https://leetcode.com/problems/remove-duplicates-from-sorted-array/
    def removeDuplicates(self, nums: List[int]) -> int:
        count = 0

        cur, pre = -1, 0
        for i, val in enumerate(nums):
            if i == 0:
                cur, pre = 1, val
                count = 1
                continue
            if val != pre:
                nums[cur] = val
                pre = val
                cur += 1
                count += 1

        return count

    # 27 https://leetcode.com/problems/remove-element/
    def removeElement(self, nums: List[int], val: int) -> int:
        L, i = 0, 0

        while i < len(nums) and nums[i] != val:
            L += 1
            i += 1
        cur = i

        while i < len(nums):
            if nums[i] != val:
                nums[cur] = nums[i]
                cur += 1
                L += 1
            i += 1

        return L

    # 28 https://leetcode.com/problems/implement-strstr/
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0

        for i in range(len(haystack)):
            m, n = 0, i
            if i + len(needle) > len(haystack):
                return -1
            while m < len(needle) and n < len(haystack):
                if needle[m] != haystack[n]:
                    break
                m += 1
                n += 1
            if m == len(needle):
                return i

        return -1

    # TODO: Learn this one
    # 29 https://leetcode.com/problems/divide-two-integers/
    def divide(self, dividend: int, divisor: int) -> int:
        MIN, MAX = -2**31, 2**31 - 1

        res = 0
        sign = 1 if dividend * divisor >= 0 else -1

        dividend = abs(dividend)
        divisor = abs(divisor)

        while dividend >= divisor:
            temp, i = divisor, 1
            while dividend >= temp:
                dividend -= temp
                res += i
                i <<= 1
                temp <<= 1

        res *= sign

        return MAX if res > MAX or res < MIN else res

    # 30 https://leetcode.com/problems/substring-with-concatenation-of-all-words/
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not words:
            return []
        res = []
        l, N = len(words[0]), len(words)
        L = N * l

        wordTable = {}
        for w in words:
            wordTable[w] = wordTable.get(w, 0) + 1

        i, n = 0, len(s) - L
        while i <= n:
            j = i
            curTable = {}

            while j < i + L:
                w = s[j:j + l]
                if w not in wordTable:
                    break
                curTable[w] = curTable.get(w, 0) + 1
                if curTable[w] > wordTable[w]:
                    break
                j += l
            if j == i + L:
                res.append(i)

            i += 1

        return res

    # 31 https://leetcode.com/problems/next-permutation/
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1

        if i == -1:
            nums.reverse()
            return None

        j = len(nums) - 1
        while j >= 0 and nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

        i, j = i + 1, len(nums) - 1
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1

        return None

    # 32 https://leetcode.com/problems/longest-valid-parentheses/
    def longestValidParentheses(self, s: str) -> int:
        # TLE solution
        # res = 0
        # n = len(s)
        # isValid = [[False] * (n + 1) for _ in range(n + 1)]
        # for i in range(2, n + 1):
        #     if s[i - 2] == "(" and s[i - 1] == ")":
        #         isValid[i - 2][i] = True
        #         res = 2
        # for i in range(2, n + 1):
        #     for j in range(n - i + 1):
        #         if s[j] == "(" and s[j + i - 1] == ")":
        #             if i == 2:
        #                 isValid[j][j + i] = True
        #             else:
        #                 isValid[j][j + i] = isValid[j + 1][j + i - 1]
        #                 for k in range(j, j + i, 2):
        #                     isValid[j][j + i] = isValid[j][j + i] or (isValid[j][k] and isValid[k][j + i])
        #             if isValid[j][j + i]:
        #                 res = max(res, i)
        # return res

        # stack solution
        # res = 0
        # m_stack = [-1]

        # for i, char in enumerate(s):
        #     if char == "(":
        #         m_stack.append(i)
        #     else:
        #         m_stack.pop()
        #         if not m_stack:
        #             m_stack.append(i)
        #         else:
        #             res = max(res, i - m_stack[-1])
        # return res
        n = len(s)
        isValid = [0] * n

        for i in range(1, n):
            if s[i] == "(":
                continue
            if isValid[i - 1]:
                if s[i - isValid[i - 1] - 1] == "(":
                    isValid[i] = isValid[i - 1] + 2
            else:
                if s[i - 1] == "(":
                    isValid[i] = 2
                    if i >= 2:
                        isValid[i] += isValid[i - 2]
                pass

        return max(isValid)

    # 33 https://leetcode.com/problems/search-in-rotated-sorted-array/
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1

        return self.__searchHelper(nums, target, 0, len(nums))

    def __searchHelper(self, nums: List[int], target: int, start: int,
                       end: int) -> int:
        if start == end - 1:
            return start if nums[start] == target else -1

        if target < nums[start] and target > nums[end - 1]:
            return -1

        mid = (end + start) // 2
        res = self.__searchHelper(nums, target, start, mid)
        if res != -1:
            return res

        return self.__searchHelper(nums, target, mid, end)

    # 34 https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = bisect.bisect_left(nums, target + 1)
        right = bisect.bisect_right(nums, target - 1)

        return [right, left - 1] if right <= left - 1 else [-1, -1]

    # 35 https://leetcode.com/problems/search-insert-position/
    def searchInsert(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums) - 1

        while low <= high:
            mid = (high + low) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1

        return low

    # 36 https://leetcode.com/problems/valid-sudoku/
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = {}
        column = {}
        box = {}
        for i in range(9):
            row[i] = set()
            column[i] = set()
            box[i] = set()

        for i in range(9):
            for j in range(9):
                c = board[i][j]
                if c == ".":
                    continue
                n = j // 3 + 3 * (i // 3)
                if c in row[i] or c in column[j] or c in box[n]:
                    return False
                row[i].add(c)
                column[j].add(c)
                box[n].add(c)

        return True

    # 37 https://leetcode.com/problems/sudoku-solver/
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = {}
        column = {}
        box = {}
        for i in range(9):
            row[i] = set()
            column[i] = set()
            box[i] = set()

        for i in range(9):
            for j in range(9):
                c = board[i][j]
                if c == ".":
                    continue

                row[i].add(c)
                column[j].add(c)
                n = j // 3 + 3 * (i // 3)
                box[n].add(c)

        self.__fillSudoku(board, row, column, box)

    def __fillSudoku(self, board: List[List[str]], row: set, column: set,
                     box: set) -> bool:
        for i in range(9):
            for j in range(9):
                c = board[i][j]
                if c != ".":
                    continue

                row_set = row[i]
                column_set = column[j]
                n = j // 3 + 3 * (i // 3)
                box_set = box[n]

                for k in range(1, 10):
                    k = str(k)
                    if k in row_set or k in column_set or k in box_set:
                        continue
                    row_set.add(k)
                    column_set.add(k)
                    box_set.add(k)
                    board[i][j] = k
                    if self.__fillSudoku(board, row, column, box):
                        return True
                    row_set.remove(k)
                    column_set.remove(k)
                    box_set.remove(k)
                    board[i][j] = "."
                return False

        return True

    # 38 https://leetcode.com/problems/count-and-say/
    def countAndSay(self, n: int) -> str:
        res = "1"

        for _ in range(1, n):
            temp = ""
            count = 1
            pre = res[0]
            i = 1
            while i < len(res):
                if res[i] == pre:
                    count += 1
                else:
                    temp += str(count) + pre
                    count = 1
                    pre = res[i]
                i += 1
            temp += str(count) + pre
            res = temp

        return res

    # 39 https://leetcode.com/problems/combination-sum/
    def combinationSum(self, candidates: List[int],
                       target: int) -> List[List[int]]:
        res = []
        self.__combinationSumBacktrace(candidates, target, 0, res, [], 0)

        return res

    def __combinationSumBacktrace(self, candidates: List[int], target: int,
                                  start: int, res: List[List[int]],
                                  curList: List[int], curSum: int) -> None:
        if curSum > target:
            return
        if curSum == target:
            res.append(list(curList))
            return

        for i in range(start, len(candidates)):
            curList.append(candidates[i])
            self.__combinationSumBacktrace(candidates, target, i, res, curList,
                                           curSum + candidates[i])
            curList.pop()

    # 40 https://leetcode.com/problems/combination-sum-ii/
    def combinationSum2(self, candidates: List[int],
                        target: int) -> List[List[int]]:
        candidates.sort()
        res = []
        self.__combinationSum2Backtrace(candidates, target, 0, res, [], 0)

        return res

    def __combinationSum2Backtrace(self, candidates: List[int], target: int,
                                   start: int, res: List[List[int]],
                                   curList: List[int], curSum: int) -> None:
        if curSum > target:
            return
        if curSum == target:
            res.append(list(curList))
            return

        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            curList.append(candidates[i])
            self.__combinationSum2Backtrace(candidates, target, i + 1, res,
                                            curList, curSum + candidates[i])
            curList.pop()

    # 41 https://leetcode.com/problems/first-missing-positive/
    def firstMissingPositive(self, nums: List[int]) -> int:
        length = len(nums)
        freq = [0] * length

        for i, n in enumerate(nums):
            if n > 0 and n <= length:
                freq[n - 1] += 1

        for i, n in enumerate(freq):
            if n == 0:
                return i + 1

        return length + 1

    # 42 https://leetcode.com/problems/trapping-rain-water/
    def trap(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        res = 0
        max_left, max_right = 0, 0

        while left < right:
            if height[left] < height[right]:
                if height[left] < max_left:
                    res += max_left - height[left]
                else:
                    max_left = height[left]
                left += 1
            else:
                if height[right] < max_right:
                    res += max_right - height[right]
                else:
                    max_right = height[right]
                right -= 1
                pass

        return res

    # 43 https://leetcode.com/problems/multiply-strings/
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"

        nums = [0] * (len(num1) + len(num2))

        for i in range(len(num1) - 1, -1, -1):
            for j in range(len(num2) - 1, -1, -1):
                m_sum = int(num1[i]) * int(num2[j]) + nums[i + j + 1]
                nums[i + j] += m_sum // 10
                nums[i + j + 1] = m_sum % 10

        while nums[0] == 0:
            del nums[0]

        res = ""
        for n in nums:
            res = res + str(n)
        return res

    # 44 https://leetcode.com/problems/wildcard-matching/
    def isMatch_Wildcard_Matching(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for i in range(m + 1):
            for j in range(1, n + 1):
                if i != 0 and (p[j - 1] == "?" or p[j - 1] == s[i - 1]):
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == "*":
                    dp[i][j] = dp[i - 1][j] or dp[i][j - 1]

        return dp[m][n]

    # 45 https://leetcode.com/problems/jump-game-ii/
    def jump(self, nums: List[int]) -> int:
        # DP solution Time Limit Exceeded
        # n = len(nums)
        # dp = [n] * n
        # dp[n - 1] = 0

        # for i in range(n - 2, -1, -1):
        #     k = 0
        #     while k <= nums[i] and k + i < n:
        #         dp[i] = min(dp[i], 1 + dp[k + i])
        #         k += 1
        # return dp[0]

        if len(nums) == 1:
            return 0

        steps = 0
        preRange = 0
        curRange = 0

        for i, n in enumerate(nums):
            if i == len(nums) - 1:
                break
            if i + n > curRange:
                curRange = i + n
            if i == preRange:
                steps += 1
                preRange = curRange
                curRange = 0

        return steps

if __name__ == "__main__":
    solution = SolutionPage1()
    nums = [7,0,9,6,9,6,1,7,9,0,1,2,9,0,3]
    # target = 18
    # s = "()()"
    # words = ["foo", "bar"]

    res = solution.jump(nums)
    print(res)