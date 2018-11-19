package com.javasolution;

import java.util.*;

import com.javasolution.util.*;

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

    // 918 https://leetcode.com/problems/maximum-sum-circular-subarray/
    public int maxSubarraySumCircular(int[] A) {
        int sum = 0;
        for (int i : A)
            sum += i;

        int curMin = A[0], min = A[0];
        int curMax = A[0], max = A[0];
        for (int i = 1; i < A.length; i++) {
            curMin = Math.min(A[i], A[i] + curMin);
            curMax = Math.max(A[i], A[i] + curMax);
            min = Math.min(min, curMin);
            max = Math.max(max, curMax);
        }
        return max > 0 ? Math.max(max, sum - min) : max;
    }

    // 919 https://leetcode.com/problems/complete-binary-tree-inserter/
    // javasolution.structdesign

    // 920 https://leetcode.com/problems/number-of-music-playlists
    // TODO: Emmm, patient
    public int numMusicPlaylists(int N, int L, int K) {
        long MOD = (long) 1e9 + 7;
        long[][] dp = new long[N + 1][L + 1];
        for (int i = K + 1; i <= N; ++i)
            for (int j = i; j <= L; ++j)
                if ((i == j) || (i == K + 1))
                    dp[i][j] = factorial(i);
                else
                    dp[i][j] = (dp[i - 1][j - 1] * i + dp[i][j - 1] * (i - K)) % MOD;
        return (int) dp[N][L];
    }

    private long factorial(int n) {
        long MOD = (long) 1e9 + 7;
        return n > 0 ? factorial(n - 1) * n % MOD : 1;
    }

    // 921 https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/
    public int minAddToMakeValid(String S) {
        int res = 0;
        int left = 0;

        for (int i = 0; i < S.length(); i++) {
            if (S.charAt(i) == '(') {
                left += 1;
            } else {
                if (left > 0)
                    left -= 1;
                else
                    res++;
            }
        }
        res += left;
        return res;
    }

    // 922 https://leetcode.com/problems/sort-array-by-parity-ii/
    public int[] sortArrayByParityII(int[] A) {
        int i = 0, j = 1;

        while (i < A.length && j < A.length) {
            if (A[i] % 2 == 0) {
                i += 2;
            } else if (A[j] % 2 == 1) {
                j += 2;
            } else {
                int tmp = A[i];
                A[i] = A[j];
                A[j] = tmp;
            }
        }

        return A;
    }

    // 923 https://leetcode.com/problems/3sum-with-multiplicity/
    public int threeSumMulti(int[] A, int target) {
        int res = 0;
        int MOD = 1000000007;
        int[] freq = new int[101];

        for (int a : A) {
            freq[a] += 1;
        }

        for (int i = 0; i < 101; i++) {
            int n = freq[i];
            if (n == 0)
                continue;

            if (n >= 3 && 3 * i == target) {
                System.out.println(combination(n, 3));
                res = (res + combination(n, 3)) % MOD;
            }
            if (n >= 2 && 2 * i <= target && target - 2 * i < 101) {
                if (target - 2 * i != i)
                    res = (res + combination(n, 2) * freq[target - 2 * i]) % MOD;
            }
            if (n >= 1) {
                for (int j = i + 1; j < 101; j++) {
                    if (freq[j] == 0)
                        continue;
                    for (int k = j + 1; k < 101; k++) {
                        if (freq[k] == 0)
                            continue;
                        if (i + j + k > target)
                            break;
                        if (i + j + k == target) {
                            res = (res + n * freq[j] * freq[k]) % MOD;
                        }
                    }
                }
            }
        }

        return res;
    }

    private int combination(int n, int r) {
        long numerator = 1;
        long denominator = 1;

        for (int i = n; i > (n - r); i--) {
            numerator = (numerator * i);
        }
        for (int i = 1; i <= r; i++) {
            denominator = denominator * i;
        }
        return (int) ((numerator / denominator) % 1000000007);
    }

    // 924 https://leetcode.com/problems/minimize-malware-spread/
    public int minMalwareSpread_1(int[][] graph, int[] initial) {
        Map<Integer, Integer> infected = new HashMap<>();

        for (int i = 0; i < initial.length; i++) {

            Set<Integer> visited = new HashSet<>();

            for (int j = 0; j < initial.length; j++) {
                if (j == i)
                    continue;
                minMalwareSpreadHelper_1(graph, visited, initial[i], initial[j]);
            }
            infected.put(initial[i], visited.size());
        }

        int Minitial = Integer.MAX_VALUE;
        int res = Integer.MAX_VALUE;
        for (Map.Entry<Integer, Integer> entry : infected.entrySet()) {
            if (entry.getValue() < Minitial) {
                Minitial = entry.getValue();
                res = entry.getKey();
            } else if (entry.getValue() == Minitial) {
                res = Math.min(res, entry.getKey());
            }
        }
        return res;
    }

    private void minMalwareSpreadHelper_1(int[][] graph, Set<Integer> visited, int removed, int node) {
        if (visited.contains(node))
            return;
        visited.add(node);

        for (int i = 0; i < graph.length; i++) {
            if (graph[node][i] == 1)
                minMalwareSpreadHelper_1(graph, visited, removed, i);
        }
    }

    // 925 https://leetcode.com/problems/long-pressed-name/
    public boolean isLongPressedName(String name, String typed) {
        int i = 0, j = 0;

        while (i < name.length() && j < typed.length()) {
            if (name.charAt(i) == typed.charAt(j)) {
                i++;
                j++;
                if (i < name.length() && name.charAt(i) == name.charAt(i - 1)) {
                    continue;
                }
                while (j < typed.length() && typed.charAt(j) == typed.charAt(j - 1)) {
                    j++;
                }
            } else {
                return false;
            }
        }
        if (i < name.length() || j < typed.length())
            return false;
        return true;
    }

    // 926 https://leetcode.com/problems/flip-string-to-monotone-increasing/
    public int minFlipsMonoIncr(String S) {
        int n = S.length();
        int leftOne = 0;
        int rightZero = 0;
        int res = 0;

        for (int i = 0; i < n; i++) {
            if (S.charAt(i) == '0')
                rightZero++;
        }
        res = rightZero;
        for (int i = 0; i < n; i++) {
            if (S.charAt(i) == '0') {
                rightZero -= 1;
            } else {
                leftOne += 1;
            }
            res = Math.min(res, rightZero + leftOne);
        }
        return res;
    }

    // 927 https://leetcode.com/problems/three-equal-parts/
    public int[] threeEqualParts(int[] A) {
        int total = 0;
        for (int a : A) {
            if (a == 1)
                total += 1;
        }

        if (total % 3 != 0)
            return new int[] { -1, -1 };

        int expect = total / 3;
        int i = 0, j = A.length - 1;
        if (total == 0)
            return new int[] { 0, A.length - 1 };

        int m = A.length - 1;
        for (int count = 0; count < expect; m--) {
            if (A[m] == 1)
                count += 1;
        }
        m += 1;

        while (A[i] == 0)
            i += 1;
        for (int k = m; k < A.length; k++, i++) {
            if (A[i] != A[k])
                return new int[] { -1, -1 };
        }
        i -= 1;

        j = i + 1;
        while (A[j] == 0)
            j += 1;
        for (int k = m; k < A.length; k++, j++) {
            if (A[j] != A[k])
                return new int[] { -1, -1 };
        }

        return new int[] { i, j };
    }

    // 928 https://leetcode.com/problems/minimize-malware-spread-ii/
    public int minMalwareSpread_2(int[][] graph, int[] initial) {
        Map<Integer, Integer> infected = new HashMap<>();

        for (int i = 0; i < initial.length; i++) {

            Set<Integer> visited = new HashSet<>();

            for (int j = 0; j < initial.length; j++) {
                if (j == i)
                    continue;
                minMalwareSpreadHelper_2(graph, visited, initial[i], initial[j]);
            }
            infected.put(initial[i], visited.size());
        }

        int Minitial = Integer.MAX_VALUE;
        int res = Integer.MAX_VALUE;
        for (Map.Entry<Integer, Integer> entry : infected.entrySet()) {
            if (entry.getValue() < Minitial) {
                Minitial = entry.getValue();
                res = entry.getKey();
            } else if (entry.getValue() == Minitial) {
                res = Math.min(res, entry.getKey());
            }
        }
        return res;
    }

    private void minMalwareSpreadHelper_2(int[][] graph, Set<Integer> visited, int removed, int node) {
        if (visited.contains(node))
            return;
        visited.add(node);

        for (int i = 0; i < graph.length; i++) {
            if (i == removed)
                continue;
            if (graph[node][i] == 1)
                minMalwareSpreadHelper_2(graph, visited, removed, i);
        }
    }

    // 929 https://leetcode.com/problems/unique-email-addresses/
    public int numUniqueEmails(String[] emails) {
        Set<String> uniqueEmail = new HashSet<>();

        for (String email : emails) {
            String[] names = email.split("@");
            String localName = names[0].substring(0, names[0].indexOf('+')).replaceAll("\\.", "");
            uniqueEmail.add(localName + names[1]);
        }

        return uniqueEmail.size();
    }

    // 930 https://leetcode.com/problems/binary-subarrays-with-sum/
    public int numSubarraysWithSum(int[] A, int S) {
        if (A.length == 0)
            return 0;
        int m = 0, n = 1;
        int res = 0;
        int curSum = A[0];

        while (m < A.length && n <= A.length) {
            if (curSum < S) {
                if (n >= A.length)
                    break;
                while (n < A.length && curSum < S) {
                    curSum += A[n];
                    n++;
                }
            } else if (curSum > S) {
                while (m < A.length && curSum > S) {
                    curSum -= A[m];
                    m++;
                }
            } else {
                while (n < A.length && A[n] == 0) {
                    n++;
                }
                if (S == 0) {
                    int len = n - m;
                    res += (1 + len) * len / 2;
                    if (n < A.length) {
                        m = n;
                        n += 1;
                        curSum = A[m];
                    } else {
                        return res;
                    }
                } else {
                    int i = m, j = n;
                    while (A[i] != 1) {
                        i++;
                    }
                    while (A[j - 1] != 1) {
                        j--;
                    }
                    int a = i - m + 1;
                    int b = n - j + 1;
                    res += a * b;
                    m = i + 1;
                    curSum -= 1;
                }
            }
        }
        return res;
    }

    // 931 https://leetcode.com/problems/minimum-falling-path-sum/
    public int minFallingPathSum(int[][] A) {
        int n = A.length;
        int[][] dp = new int[n][n];

        for (int i = 0; i < n; i++) {
            dp[0][i] = A[0][i];
        }

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j > 0)
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][j - 1]);
                if (j < n - 1)
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][j + 1]);
                dp[i][j] += A[i][j];
            }
        }

        int res = Integer.MAX_VALUE;
        for (int i : dp[n - 1]) {
            res = Math.min(res, i);
        }
        return res;
    }

    // 932 https://leetcode.com/problems/beautiful-array/
    public int[] beautifulArray(int N) {
        ArrayList<Integer> list = new ArrayList<>();
        list.add(1);
        while (list.size() < N) {
            ArrayList<Integer> tmp = new ArrayList<>();
            for (int i : list)
                if (i * 2 - 1 <= N)
                    tmp.add(i * 2 - 1);
            for (int i : list)
                if (i * 2 <= N)
                    tmp.add(i * 2);
            list = tmp;
        }

        int[] res = new int[N];
        int i = 0;
        for (int n : list) {
            res[i] = n;
            i++;
        }
        return res;
    }

    // 933 https://leetcode.com/problems/number-of-recent-calls/
    // javasolution.structdesign

    // 934 https://leetcode.com/problems/shortest-bridge/
    public int shortestBridge(int[][] A) {
        int n = A.length;
        boolean[][] visited = new boolean[n][n];
        int[][] dirs = new int[][] { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };
        Queue<int[]> q = new LinkedList<>();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (A[i][j] == 1) {
                    findStart(A, visited, q, i, j, dirs);
                    break;
                }
            }
            if (!q.isEmpty()) {
                break;
            }
        }

        int res = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                int[] cur = q.poll();
                for (int[] dir : dirs) {
                    int i = cur[0] + dir[0];
                    int j = cur[1] + dir[1];
                    if (i >= 0 && j >= 0 && i < n && j < n && !visited[i][j]) {
                        if (A[i][j] == 1) {
                            return res;
                        }
                        q.offer(new int[] { i, j });
                        visited[i][j] = true;
                    }
                }
            }
            res++;
        }
        return -1;
    }

    private void findStart(int[][] A, boolean[][] visited, Queue<int[]> q, int i, int j, int[][] dirs) {
        if (i < 0 || j < 0 || i >= A.length || j >= A[0].length || visited[i][j] || A[i][j] == 0) {
            return;
        }
        visited[i][j] = true;
        q.offer(new int[] { i, j });
        for (int[] dir : dirs) {
            findStart(A, visited, q, i + dir[0], j + dir[1], dirs);
        }
    }

    // 935 https://leetcode.com/problems/knight-dialer/
    public int knightDialer(int N) {
        int mod = (int) 1e9 + 7;
        int[][] dp = new int[N + 1][10];
        int[][] graph = new int[][] { { 4, 6 }, { 6, 8 }, { 7, 9 }, { 4, 8 }, { 3, 9, 0 }, {}, { 1, 7, 0 }, { 2, 6 },
                { 1, 3 }, { 2, 4 } };

        for (int i = 0; i < 10; i++) {
            dp[1][i] = 1;
        }
        for (int n = 2; n <= N; n++) {
            for (int i = 0; i < 10; i++) {
                for (int nextN : graph[i]) {
                    dp[n][i] = (dp[n][i] + dp[n - 1][nextN]) % mod;
                }
            }
        }

        int res = 0;
        for (int i = 0; i < 10; i++) {
            res = (res + dp[N][i]) % mod;
        }
        return res;
    }

    // 936 https://leetcode.com/problems/stamping-the-sequence/
    public int[] movesToStamp(String stamp, String target) {
        char[] tArr = target.toCharArray();
        char[] sArr = stamp.toCharArray();

        Stack<Integer> st = new Stack<>();
        int cnt = 0, total = 10 * tArr.length;
        boolean change = false;

        while (cnt < total) {
            change = false;
            for (int i = 0; i + sArr.length <= tArr.length; i++) {
                if (validate(tArr, sArr, i)) {
                    Arrays.fill(tArr, i, i + sArr.length, '*');
                    st.push(i);
                    change = true;
                }
            }
            if (!change)
                break;
            cnt++;
        }

        for (char c : tArr) {
            if (c != '*')
                return new int[] {};
        }

        int[] res = new int[st.size()];

        for (int i = 0; i < res.length; i++) {
            res[i] = st.pop();
        }
        return res;
    }

    private boolean validate(char[] tArr, char[] sArr, int i) {
        boolean flag = false;
        for (int j = 0; j < sArr.length; i++, j++) {
            if (tArr[i] == '*')
                continue;
            if (tArr[i] != sArr[j])
                return false;
            flag = true;
        }
        return flag;
    }

    // 937 https://leetcode.com/problems/reorder-log-files/
    public String[] reorderLogFiles(String[] logs) {
        String[] res = new String[logs.length];

        PriorityQueue<String> letterLog = new PriorityQueue<>(new Comparator<String>() {
            public int compare(String a, String b) {
                a = a.substring(a.indexOf(" "));
                b = b.substring(b.indexOf(" "));
                return a.compareTo(b);
            }
        });
        Queue<String> digitLog = new ArrayDeque<>();

        for (String s : logs) {
            String[] sArr = s.split(" ");
            if (Character.isDigit(sArr[1].charAt(0))) {
                digitLog.offer(s);
            } else {
                letterLog.offer(s);
            }
        }

        int i = 0;
        while (!letterLog.isEmpty()) {
            res[i] = letterLog.poll();
            i++;
        }
        while (!digitLog.isEmpty()) {
            res[i] = digitLog.poll();
            i++;
        }
        return res;
    }

    // 938 https://leetcode.com/problems/range-sum-of-bst/
    public int rangeSumBST(TreeNode root, int L, int R) {
        if (root == null)
            return 0;
        if (root.val < L)
            return rangeSumBST(root.right, L, R);
        if (root.val > R)
            return rangeSumBST(root.left, L, R);
        return root.val + rangeSumBST(root.left, L, R) + rangeSumBST(root.right, L, R);
    }
}
