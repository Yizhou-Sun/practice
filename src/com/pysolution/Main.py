import sys
import Solution


def main():
    sol = Solution.Solution()
    nums = [2, 7, 11, 15]
    target = 18
    string = "PAYPALISHIRING"
    res = sol.threeSumClosest([-1, 2, 1, -4], 1)
    print(res)


if __name__ == "__main__":
    main()
