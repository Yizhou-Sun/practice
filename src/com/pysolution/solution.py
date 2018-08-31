from structure import ListNode


class Solution:
    # 5 https://leetcode.com/problems/longest-palindromic-substring/description/
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        # DP?
        n = len(s)
        dp = [[False] * n for i in range(n)]
        maxLen = 1
        start = 0
        end = 1

        for i in range(n):
            for j in range(n - i):
                if i == 0:
                    dp[j][j] = True
                elif i == 1 and s[j] == s[j + i]:
                    dp[j][j + 1] = True
                elif s[j] == s[j + i] and dp[j + 1][j + i - 1]:
                    dp[j][j + i] = True

                if dp[j][j + i] and maxLen < i + 1:
                        start = j
                        end = j + i + 1
                        maxLen = i + 1
        return s[start:end]

    # 4 https://leetcode.com/problems/median-of-two-sorted-arrays/description/
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        length = len(nums1) + len(nums2)
        mid = length / 2
        isTwo = length % 2

    # 3 https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        """
        Faster solution: use a map to find the tail position
        head = res = 0
        tail = -1
        mymap = {}
        while head < len(s):
            if s[head] not in mymap or tail > mymap[s[head]]:
                mymap[s[head]] = head
                res = max(res, head - tail)
                head += 1
            else:
                tail = mymap[s[head]]
        return res
        """
        res = head = 0
        tail = -1
        letterSet = set()
        while head < len(s):
            if s[head] not in letterSet:
                letterSet.add(s[head])
                res = max(res, head - tail)
                head += 1
            else:
                tail += 1
                letterSet.remove(s[tail])
        return res

    # 2 https://leetcode.com/problems/add-two-numbers/description/
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        cur = head = ListNode(0)
        carry = 0

        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            cur.next = ListNode(carry % 10)
            carry = int(carry / 10)
            cur = cur.next

        return head.next

    # 1 https://leetcode.com/problems/two-sum/description/
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        myDict = {}
        for i in range(len(nums)):
            if nums[i] in myDict:
                return [myDict[nums[i]], i]
            else:
                myDict[target - nums[i]] = i