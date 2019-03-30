from typing import List
from Structure import ListNode


class Solution:
    # 1 https://leetcode.com/problems/two-sum/description/
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

            return 0