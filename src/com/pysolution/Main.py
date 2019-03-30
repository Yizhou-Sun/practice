import sys
import Solution


def main():
    sol = Solution.Solution()
    nums = [2, 7, 11, 15]
    target = 18
    res = sol.twoSum(nums, target)
    print(res)


if __name__ == "__main__":
    main()
