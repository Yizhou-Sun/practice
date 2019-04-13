import sys

from typing import List
from Structure import ListNode


class Solution:
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
        n = len(nums1) + len(nums2)
        if n % 2 != 0:
            return self.findNthFromArrays(nums1, nums2, int(n / 2))
        else:
            m1 = self.findNthFromArrays(nums1, nums2, int(n / 2))
            m2 = self.findNthFromArrays(nums1, nums2, int(n / 2) - 1)
            return (m1 + m2) / 2

    def findNthFromArrays(self, nums1: List[int], nums2: List[int],
                          i: int) -> int:
        if not nums1:
            return nums2[i]
        if not nums2:
            return nums1[i]

        if i == 0:
            return min(nums1[0], nums2[0])

        mid = int((i - 1) / 2)
        m1 = m2 = sys.maxsize

        if (mid < len(nums1)):
            m1 = nums1[mid]
        if (mid < len(nums2)):
            m2 = nums2[mid]

        if (m1 < m2):
            return self.findNthFromArrays(nums1[mid + 1:], nums2, i - mid - 1)
        else:
            return self.findNthFromArrays(nums1, nums2[mid + 1:], i - mid - 1)

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
        if len(s) == 0: return 0

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
        if x < 0: return False
        s = str(x)

        i, j = 0, len(s) - 1

        while i < j and s[i] == s[j]:
            i += 1
            j -= 1

        return i >= j

    # 10 https://leetcode.com/problems/regular-expression-matching/
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]

        dp[0][0] = True
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c1 = s[i - 1]
                c2 = p[j - 1]
                if c2 == "." or c1 == c2:
                    dp[i][j] = dp[i - 1][j - 1]
                elif c2 == "*":
                    c3 = p[j - 2]
                    dp[i][j] = (dp[i - 1][j - 1] or dp[i - 1][j]) and (c3 == "." or c1 == c3)

        return dp[m][n]

    # Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.

    # '.' Matches any single character.
    # '*' Matches zero or more of the preceding element.
    # The matching should cover the entire input string (not partial).

    # Note:

    # s could be empty and contains only lowercase letters a-z.
    # p could be empty and contains only lowercase letters a-z, and characters like . or *.
    # Example 1:

    # Input:
    # s = "aa"
    # p = "a"
    # Output: false
    # Explanation: "a" does not match the entire string "aa".
    # Example 2:

    # Input:
    # s = "aa"
    # p = "a*"
    # Output: true
    # Explanation: '*' means zero or more of the precedeng element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
    # Example 3:

    # Input:
    # s = "ab"
    # p = ".*"
    # Output: true
    # Explanation: ".*" means "zero or more (*) of any character (.)".
    # Example 4:

    # Input:
    # s = "aab"
    # p = "c*a*b"
    # Output: true
    # Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore it matches "aab".
    # Example 5:

    # Input:
    # s = "mississippi"
    # p = "mis*is*p*."
    # Output: false