package com.javasolution;

import java.util.*;
import java.lang.*;

import com.javasolution.util.*;
import com.javasolution.structdesign.*;

public class Solution_19 {
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

    // 908 https://leetcode.com/problems/smallest-range-i/
    public int smallestRangeI(int[] A, int K) {
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;

        for (int i = 0; i < A.length; i++) {
            max = Math.max(max, A[i] - K);
            min = Math.min(min, A[i] + K);
        }

        if (max > min) {
            return max - min;
        } else {
            return 0;
        }
    }

    // 909 https://leetcode.com/problems/snakes-and-ladders/
    public int snakesAndLadders(int[][] board) {
        int m = board.length;
        int n = board[0].length;
        int[] nBoard = new int[m * n + 1];
        int[] dp = new int[m * n + 1];

        for (int i = m - 1, index = 1; i >= 0; i--) {
            if ((m - 1 - i) % 2 == 0) {
                for (int j = 0; j < n; j++) {
                    dp[index] = Integer.MAX_VALUE;
                    nBoard[index++] = board[i][j];
                }
            } else {
                for (int j = n - 1; j >= 0; j--) {
                    dp[index] = Integer.MAX_VALUE;
                    nBoard[index++] = board[i][j];
                }
            }
        }

        boolean[] visited = new boolean[m * n + 1];
        Queue<Integer> q = new LinkedList<>();
        int start = nBoard[1] > -1 ? nBoard[1] : 1;

        q.offer(start);
        visited[start] = true;
        int step = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                int cur = q.poll();
                if (cur == n * n) {
                    return step;
                }
                for (int next = cur + 1; next <= Math.min(cur + 6, n * n); next++) {
                    int dest = nBoard[next] > -1 ? nBoard[next] : next;
                    if (!visited[dest]) {
                        visited[dest] = true;
                        q.offer(dest);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    // 910 https://leetcode.com/problems/smallest-range-ii/
    // TODO: AAAAAAAA?
    public int smallestRangeII(int[] A, int K) {
        int n = A.length;
        Arrays.sort(A);
        int min = A[0], max = A[n - 1];
        int width = A[n - 1] - A[0];

        for (int i = 0; i < n - 1; i++) {
            max = Math.max(max, A[i] + 2 * K);
            min = Math.min(A[0] + 2 * K, A[i + 1]);
            width = Math.min(width, max - min);
        }

        return Math.min(width, max - min);
    }

    // 910 https://leetcode.com/problems/online-election/
    // javasolution.structdesign

    // 913 https://leetcode.com/problems/cat-and-mouse/
    // TODO: miao miao miao?
    public int catMouseGame(int[][] graph) {
        int n = graph.length;
        // (cat, mouse, mouseMove = 0)
        int[][][] color = new int[n][n][2];
        int[][][] outdegree = new int[n][n][2];
        for (int i = 0; i < n; i++) { // cat
            for (int j = 0; j < n; j++) { // mouse
                outdegree[i][j][0] = graph[j].length;
                outdegree[i][j][1] = graph[i].length;
                for (int k : graph[i]) {
                    if (k == 0) {
                        outdegree[i][j][1]--;
                        break;
                    }
                }
            }
        }
        Queue<int[]> q = new LinkedList<>();
        for (int k = 1; k < n; k++) {
            for (int m = 0; m < 2; m++) {
                color[k][0][m] = 1;
                q.offer(new int[] { k, 0, m, 1 });
                color[k][k][m] = 2;
                q.offer(new int[] { k, k, m, 2 });
            }
        }
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int cat = cur[0], mouse = cur[1], mouseMove = cur[2], c = cur[3];
            if (cat == 2 && mouse == 1 && mouseMove == 0) {
                return c;
            }
            int prevMouseMove = 1 - mouseMove;
            for (int prev : graph[prevMouseMove == 1 ? cat : mouse]) {
                int prevCat = prevMouseMove == 1 ? prev : cat;
                int prevMouse = prevMouseMove == 1 ? mouse : prev;
                if (prevCat == 0) {
                    continue;
                }
                if (color[prevCat][prevMouse][prevMouseMove] > 0) {
                    continue;
                }
                if (prevMouseMove == 1 && c == 2 || prevMouseMove == 0 && c == 1
                        || --outdegree[prevCat][prevMouse][prevMouseMove] == 0) {
                    color[prevCat][prevMouse][prevMouseMove] = c;
                    q.offer(new int[] { prevCat, prevMouse, prevMouseMove, c });
                }
            }
        }
        return color[2][1][0];
    }

    // 914 https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/
    public boolean hasGroupsSizeX(int[] deck) {
        HashMap<Integer, Integer> freqMap = new HashMap<>();
        int res = 0;
        for (int i : deck)
            freqMap.put(i, freqMap.getOrDefault(i, 0) + 1);
        for (int i : freqMap.values())
            res = getGCD(i, res);
        return res > 1;
    }

    private int getGCD(int a, int b) {
        if (b == 0)
            return a;
        else
            return getGCD(b, a % b);
    }

    // 915 https://leetcode.com/problems/partition-array-into-disjoint-intervals/
    public int partitionDisjoint(int[] A) {
        Stack<Integer> minST = new Stack<>();

        for (int i = A.length - 1; i > 0; i--) {
            if (minST.isEmpty())
                minST.push(A[i]);
            else if (minST.peek() < A[i])
                minST.push(minST.peek());
            else
                minST.push(A[i]);
        }

        int max = A[0];
        for (int i = 1; i < A.length; i++) {
            if (max <= minST.peek())
                return i;
            max = Math.max(max, A[i]);
        }
        return -1;
    }

    // 916 https://leetcode.com/problems/word-subsets/
    public List<String> wordSubsets(String[] A, String[] B) {
        int[] uni = new int[26], tmp;
        int i;
        for (String b : B) {
            tmp = counter(b);
            for (i = 0; i < 26; ++i)
                uni[i] = Math.max(uni[i], tmp[i]);
        }
        List<String> res = new ArrayList<>();
        for (String a : A) {
            tmp = counter(a);
            for (i = 0; i < 26; ++i)
                if (tmp[i] < uni[i])
                    break;
            if (i == 26)
                res.add(a);
        }
        return res;
    }

    private int[] counter(String word) {
        int[] count = new int[26];
        for (char c : word.toCharArray())
            count[c - 'a']++;
        return count;
    }

    // 917 https://leetcode.com/problems/reverse-only-letters/submissions/
    public String reverseOnlyLetters(String S) {
        char[] cArr = S.toCharArray();
        int i = 0, j = cArr.length - 1;

        while (i < j) {
            char l = Character.toLowerCase(cArr[i]);
            char r = Character.toLowerCase(cArr[j]);
            if (l < 'a') {
                i++;
                continue;
            }
            if (r < 'a') {
                j--;
                continue;
            }
            char temp = cArr[i];
            cArr[i] = cArr[j];
            cArr[j] = temp;
            i++;
            j--;
        }

        return new String(cArr);
    }
}
