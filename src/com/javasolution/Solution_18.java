package com.javasolution;

import java.util.*;
import java.lang.*;

import com.javasolution.util.*;
import com.javasolution.structdesign.*;

public class Solution_18 {
    // 857 https://leetcode.com/problems/minimum-cost-to-hire-k-workers/description/
    public double mincostToHireWorkers(int[] quality, int[] wage, int K) {
        double[][] workerRatio = new double[quality.length][2];
        for (int i = 0; i < quality.length; i++) {
            workerRatio[i][0] = quality[i];
            workerRatio[i][1] = wage[i] / (double) quality[i];
        }

        Arrays.sort(workerRatio, (a, b) -> (Double.compare(a[1], b[1])));
        double res = Double.MAX_VALUE, sum = 0;
        PriorityQueue<Double> pq = new PriorityQueue<>((a, b) -> Double.compare(b, a));

        for (double[] worker : workerRatio) {
            sum += worker[0];
            pq.add(worker[0]);
            if (pq.size() > K)
                sum -= pq.poll();
            if (pq.size() == K)
                res = Math.min(res, sum * worker[1]);
        }
        return res;
    }

    // 858 https://leetcode.com/problems/mirror-reflection/description/
    public int mirrorReflection(int p, int q) {
        int comDivisor = largestCommonDivisor(p, q);
        p = p / comDivisor;
        q = q / comDivisor;

        if (q % 2 == 0) {
            if (p % 2 == 0)
                return 3;
            else
                return 0;
        } else {
            if (p % 2 == 0)
                return 2;
            else
                return 1;
        }
    }

    private int largestCommonDivisor(int i, int j) {
        // i must larger than j
        int r = -1;
        while (r != 0) {
            r = i % j;
            if (r == 0)
                break;
            i = j;
            j = r;
        }
        return j;
    }

    // 859 https://leetcode.com/problems/buddy-strings/description/
    public boolean buddyStrings(String A, String B) {
        if (A.length() != B.length())
            return false;
        int[] index = new int[2];
        int n = A.length();
        int count = 0;
        int[] cArr = new int[26];

        for (int i = 0; i < n; i++) {
            if (A.charAt(i) != B.charAt(i)) {
                if (count == 2) {
                    return false;
                }
                index[count] = i;
                count++;
            } else {
                cArr[A.charAt(i) - 'a']++;
            }
        }
        if (count == 1)
            return false;
        if (count == 2 && A.charAt(index[0]) == B.charAt(index[1]) && A.charAt(index[1]) == B.charAt(index[0]))
            return true;
        for (int i : cArr) {
            if (i >= 2)
                return true;
        }
        return false;
    }

    // 860 https://leetcode.com/problems/lemonade-change/description/
    public boolean lemonadeChange(int[] bills) {
        int[] changes = new int[2];
        for (int bill : bills) {
            if (bill == 5) {
                changes[0]++;
            } else if (bill == 10) {
                changes[1]++;
                if (changes[0]-- == 0)
                    return false;
            } else {
                if (changes[1] > 0 && changes[0] > 0) {
                    changes[1]--;
                    changes[0]--;
                } else if (changes[0] > 2) {
                    changes[0] -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    // 861 https://leetcode.com/problems/score-after-flipping-matrix/description/
    // A A A A !
    public int matrixScore(int[][] A) {
        int M = A.length, N = A[0].length, res = (1 << (N - 1)) * M;
        for (int j = 1; j < N; j++) {
            int cur = 0;
            for (int i = 0; i < M; i++)
                cur += A[i][j] == A[i][0] ? 1 : 0;
            res += Math.max(cur, M - cur) * (1 << (N - j - 1));
        }
        return res;
    }

    // 862
    // https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/
    // TODO: Emmmmmm......
    public int shortestSubarray(int[] A, int K) {
        int n = A.length, res = n + 1;
        int[] prefixSum = new int[n + 1];
        Deque<Integer> d = new ArrayDeque<>();

        for (int i = 0; i < n; i++) {
            prefixSum[i + 1] = prefixSum[i] + A[i];
        }
        for (int i = 0; i <= n; i++) {
            while (d.size() > 0 && prefixSum[i] - prefixSum[d.getFirst()] >= K) {
                res = Math.min(res, i - d.pollFirst());
            }
            while (d.size() > 0 && prefixSum[i] <= prefixSum[d.getLast()]) {
                d.pollLast();
            }
            d.addLast(i);
        }
        if (res <= n) {
            return res;
        }
        return -1;
    }

    // 863
    // https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/description/
    public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {
        List<TreeNode> waitList = new ArrayList<>();
    }

    // 864 https://leetcode.com/problems/shortest-path-to-get-all-keys/description/

    // 865
    // https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/description/

    // 866 https://leetcode.com/problems/prime-palindrome/description/

    // 867 https://leetcode.com/problems/transpose-matrix/description/

    // 902
    // https://leetcode.com/problems/numbers-at-most-n-given-digit-set/description/
    // TODO: Review!!!
    public int atMostNGivenDigitSet(String[] D, int N) {
        int result = 0;
        String s = String.valueOf(N);
        for (int i = 1; i <= s.length(); i++) {
            result += helper(D, i, Integer.toString(N));
        }
        return result;
    }

    private int helper(String[] D, int K, String s) {
        if (s.equals("0")) {
            return 0;
        }

        if (s.length() > K) {
            return (int) Math.pow(D.length, K);
        }

        int count = 0;
        int char0 = s.charAt(0) - '0';

        for (int i = 0; i < D.length; i++) {

            if (Integer.parseInt(D[i]) < char0) {
                count += helper(D, K - 1, s);
            } else if (Integer.parseInt(D[i]) == char0) {
                if (s.length() > 1) {
                    int charRem = Integer.parseInt(s.substring(1, 2)) == 0 ? 0 : Integer.parseInt(s.substring(1));
                    count += helper(D, K - 1, Integer.toString(charRem));
                } else {
                    count++;
                }
            }
        }
        return count;
    }

    // 903
    // https://leetcode.com/problems/valid-permutations-for-di-sequence/description/
    // TODO: A very good question!
    public int numPermsDISequence(String S) {
        int MOD = 1000000007;
        int n = S.length();

        int[][] dp = new int[n + 1][n + 1];
        for (int i = 0; i <= n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= i; j++) {
                if (S.charAt(i - 1) == 'I') {
                    for (int k = 0; k < j; ++k) {
                        dp[i][j] += dp[i - 1][k];
                        dp[i][j] %= MOD;
                    }
                } else {
                    for (int k = j; k < i; ++k) {
                        dp[i][j] += dp[i - 1][k];
                        dp[i][j] %= MOD;
                    }
                }
            }
        }

        long ans = 0;
        for (int x : dp[n]) {
            ans += x;
            ans %= MOD;
        }
        return (int) ans;
    }

    // 904 https://leetcode.com/problems/fruit-into-baskets/description/
    public int totalFruit(int[] tree) {
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        ArrayList<Integer> fruitType = new ArrayList<>();

        int i = 0, j = 0;
        while (j < tree.length) {
            if (fruitType.size() == 0) {
                fruitType.add(tree[j]);
            } else if (fruitType.size() == 1) {
                if (tree[j] != fruitType.get(0)) {
                    fruitType.add(tree[j]);
                }
            } else if (tree[j] != fruitType.get(1)) {
                if (tree[j] == fruitType.get(0)) {
                    fruitType.remove(0);
                    fruitType.add(tree[j]);
                } else {
                    res = Math.max(res, j - i);
                    i = map.remove(fruitType.remove(0)) + 1;
                    fruitType.add(tree[j]);
                }
            }
            map.put(tree[j], j);
            j++;
        }
        res = Math.max(res, j - i);
        return res;
    }

    // 905 https://leetcode.com/problems/sort-array-by-parity/description/
    public int[] sortArrayByParity(int[] A) {
        int i = 0, j = A.length;

        while (i < j) {
            if (A[i] % 2 == 0) {
                i++;
            } else if (A[j - 1] % 2 != 0) {
                j--;
            } else {
                int temp = A[i];
                A[i] = A[j - 1];
                A[j - 1] = temp;
            }
        }
        return A;
    }

    // 906 https://leetcode.com/problems/super-palindromes/description/
    // Note: Bad question ...
    public int superpalindromesInRange(String L, String R) {
        Long l = Long.valueOf(L), r = Long.valueOf(R);

        int result = 0;
        for (long i = (long) Math.sqrt(l); i * i <= r;) {
            long p = nextP(i);
            if (p * p <= r && isP(p * p)) {
                result++;
            }
            i = p + 1;
        }
        return result;
    }

    private long nextP(long l) {
        String s = "" + l;
        int len = s.length();
        List<Long> cands = new LinkedList<>();
        cands.add((long) Math.pow(10, len) - 1);
        String half = s.substring(0, (len + 1) / 2);
        String nextHalf = "" + (Long.valueOf(half) + 1);
        String reverse = new StringBuilder(half.substring(0, len / 2)).reverse().toString();
        String nextReverse = new StringBuilder(nextHalf.substring(0, len / 2)).reverse().toString();
        cands.add(Long.valueOf(half + reverse));
        cands.add(Long.valueOf(nextHalf + nextReverse));
        long result = Long.MAX_VALUE;
        for (long i : cands) {
            if (i >= l) {
                result = Math.min(result, i);
            }
        }
        return result;
    }

    private boolean isP(long l) {
        String s = "" + l;
        int i = 0, j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i++) != s.charAt(j--)) {
                return false;
            }
        }
        return true;
    }

    // 907 https://leetcode.com/problems/sum-of-subarray-minimums/description/
    public int sumSubarrayMins(int[] A) {
        int MOD = (int) (1e9 + 7);
        int res = 0;
        int[] leftLen = new int[A.length];
        Stack<int[]> stl = new Stack<>();
        int[] rightLen = new int[A.length];
        Stack<int[]> str = new Stack<>();
        for (int i = 0; i < A.length; i++) {
            leftLen[i] = 1;
            while (!stl.isEmpty() && stl.peek()[0] > A[i])
                leftLen[i] += stl.pop()[1];
            stl.push(new int[] { A[i], leftLen[i] });
        }
        for (int i = A.length - 1; i >= 0; i--) {
            rightLen[i] = 1;
            while (!str.isEmpty() && str.peek()[0] >= A[i])
                rightLen[i] += str.pop()[1];
            str.push(new int[] { A[i], rightLen[i] });
        }
        for (int i = 0; i < A.length; ++i)
            res = (res + A[i] * leftLen[i] * rightLen[i]) % MOD;
        return res;
    }
}
