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
                    dp[i][j] = dp[i][j - 2] or (
                        (pre == "." or pre == schar) and
                        (dp[i - 1][j] or dp[i - 1][j - 1]))

        return dp[m][n]

    # 11 https://leetcode.com/problems/container-with-most-water/
    def maxArea(self, height: List[int]) -> int:
        res = 0
        i, j = 0, len(height) - 1

        while i < j:
            res = max(res, min(height[i], height[j]) * (j - i))
            if (height[i] < height[j]):
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
            1: "I"
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
        if len(strs) == 0: return res

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
        nums.sort()
        n = len(nums) - 2
        res = []

        for i in range(n):
            if i != 0 and nums[i] == nums[i - 1]:
                continue
            ans = [nums[i]]
            self.threeSumBackTrace(nums, res, i + 1, 0, ans)

        return res

    def threeSumBackTrace(self, nums: List[int], res: List[List[int]],
                          index: int, _sum: int, ans: List[int]) -> None:
        if len(ans) == 3:
            if sum(ans) == 0:
                res.append(list(ans))
            ans = ans[:-1]
            return

        for i in range(index, len(nums)):
            if i != index and nums[i] == nums[i - 1]:
                continue
            self.threeSumBackTrace(nums, res, i, _sum, ans)

        ans = ans[:-1]