import sys
import bisect

from typing import List
from Structure import ListNode
from Structure import TreeNode


class Solution_3:
    # 101 https://leetcode.com/problems/symmetric-tree/
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True

        return self.__isSymmetricHelper(root.left, root.right)

    def __isSymmetricHelper(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False

        return self.__isSymmetricHelper(p.left, q.right) and \
            self.__isSymmetricHelper(p.right, q.left)

    def generateAbbreviations(self, word: str) -> List[str]:
        backtrace = []

        def helper(num: int, chars: List[str], index: int):
            if num == 0:
                backtrace.append(list(chars))
                return

            for i in range(index, len(chars)):
                temp = chars[i]
                chars[i] = "1"
                helper(num - 1, chars, index + 1)
                chars[i] = temp

        chars = list(word)

        # for i in range(len(chars) + 1):
        #     self.test(i, chars, 0, backtrace)

        self.test(2, chars, 0, backtrace)
        res = ["".join(i) for i in backtrace]

        return res

    def test(self, num: int, chars: List[str], index: int, backtrace: List[List[str]]) -> None:
        if num == 0:
            print(chars, num, index)
            backtrace.append(list(chars))
            return

        for i in range(index, len(chars)):
            temp = chars[i]
            chars[i] = "1"
            self.test(num - 1, chars, i + 1, backtrace)
            chars[i] = temp

if __name__ == "__main__":
    solution = Solution_3()
    # nums = [0, 1]
    # matrix = [[1, 3], [4, 5]]
    # target = 18
    # s = "()()"
    # words = ["foo", "bar", "oof"]
    res = solution.generateAbbreviations("word")
    print(res)