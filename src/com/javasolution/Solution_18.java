package com.javasolution;

import java.util.*;

import com.javasolution.util.*;

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
        Stack<TreeNode> path = new Stack<>();
        if (!findPath(root, target, path)) {
            System.out.println("Not Found");
        }

        List<Integer> res = new ArrayList<>();
        TreeNode pre = null;
        while (!path.isEmpty()) {
            TreeNode cur = path.pop();

            if (K == 0) {
                res.add(cur.val);
                break;
            } else if (pre == null) {
                res.addAll(BFSdistanceK(cur, K));
            } else {
                if (cur.left == pre && cur.right != null) {
                    res.addAll(BFSdistanceK(cur.right, K - 1));
                } else if (cur.right == pre && cur.left != null) {
                    res.addAll(BFSdistanceK(cur.left, K - 1));
                }
            }
            pre = cur;
            K--;
        }
        return res;
    }

    private boolean findPath(TreeNode root, TreeNode target, Stack<TreeNode> path) {
        if (root == null)
            return false;

        path.push(root);
        if (root.val == target.val || findPath(root.left, target, path) || findPath(root.right, target, path)) {
            return true;
        }
        path.pop();
        return false;
    }

    private List<Integer> BFSdistanceK(TreeNode root, int K) {
        List<Integer> res = new LinkedList<>();
        if (K < 0)
            return res;

        Deque<TreeNode> dq = new ArrayDeque<>();
        dq.addLast(root);
        while (K != 0 && dq.size() != 0) {
            Deque<TreeNode> nextL = new ArrayDeque<>();
            for (TreeNode n : dq) {
                if (n.left != null) {
                    nextL.addLast(n.left);
                }
                if (n.right != null) {
                    nextL.addLast(n.right);
                }
            }
            K--;
            dq = nextL;
        }
        if (K != 0)
            return res;
        for (TreeNode n : dq) {
            res.add(n.val);
        }
        return res;
    }

    // 864 https://leetcode.com/problems/shortest-path-to-get-all-keys/description/
    // TODO: hard question！ good question！
    class State {
        int keys, i, j;

        State(int keys, int i, int j) {
            this.keys = keys;
            this.i = i;
            this.j = j;
        }
    }

    public int shortestPathAllKeys(String[] grid) {
        int x = -1, y = -1, m = grid.length, n = grid[0].length(), max = -1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                char c = grid[i].charAt(j);
                if (c == '@') {
                    x = i;
                    y = j;
                }
                if (c >= 'a' && c <= 'f') {
                    max = Math.max(c - 'a' + 1, max);
                }
            }
        }
        State start = new State(0, x, y);
        Queue<State> q = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        visited.add(0 + " " + x + " " + y);
        q.offer(start);
        int[][] dirs = new int[][] { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        int step = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                State cur = q.poll();
                if (cur.keys == (1 << max) - 1) {
                    return step;
                }
                for (int[] dir : dirs) {
                    int i = cur.i + dir[0];
                    int j = cur.j + dir[1];
                    int keys = cur.keys;
                    if (i >= 0 && i < m && j >= 0 && j < n) {
                        char c = grid[i].charAt(j);
                        if (c == '#') {
                            continue;
                        }
                        if (c >= 'a' && c <= 'f') {
                            keys |= 1 << (c - 'a');
                        }
                        if (c >= 'A' && c <= 'F' && ((keys >> (c - 'A')) & 1) == 0) {
                            continue;
                        }
                        if (!visited.contains(keys + " " + i + " " + j)) {
                            visited.add(keys + " " + i + " " + j);
                            q.offer(new State(keys, i, j));
                        }
                    }
                }
            }
            step++;
        }
        return -1;
    }

    // 865
    // https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/description/
    // Node there is a brilliant solution using pair
    private int depth;
    private List<Deque<TreeNode>> paths;

    public TreeNode subtreeWithAllDeepest(TreeNode root) {
        if (root == null)
            return root;
        Deque<TreeNode> q = new ArrayDeque<>();
        paths = new LinkedList<>();
        depth = 0;

        DFSFindAllPath(q, root);
        TreeNode pre = null;

        while (paths.get(0).size() != 0) {
            TreeNode n = paths.get(0).peekFirst();
            for (Deque<TreeNode> path : paths) {
                if (n != path.pollFirst()) {
                    return pre;
                }
            }
            pre = n;
        }
        return pre;
    }

    private void DFSFindAllPath(Deque<TreeNode> q, TreeNode root) {
        q.offerLast(root);

        if (root.left == null && root.right == null) {
            if (q.size() == depth) {
                paths.add(new ArrayDeque<>(q));
            } else if (q.size() > depth) {
                depth = q.size();
                paths = new LinkedList<>();
                paths.add(new ArrayDeque<>(q));
            }
        }
        if (root.left != null) {
            DFSFindAllPath(q, root.left);
        }
        if (root.right != null) {
            DFSFindAllPath(q, root.right);
        }

        q.pollLast();
    }

    // 866 https://leetcode.com/problems/prime-palindrome/description/
    public int primePalindrome(int N) {
        if (8 <= N && N <= 11)
            return 11;
        for (int x = 1; x < 100000; x++) {
            String s = Integer.toString(x);
            String r = new StringBuilder(s).reverse().toString().substring(1);
            int y = Integer.parseInt(s + r);
            if (y >= N && isPrime(y))
                return y;
        }
        return -1;
    }

    private boolean isPrime(int x) {
        if (x < 2 || x % 2 == 0)
            return x == 2;
        for (int i = 3; i * i <= x; i += 2)
            if (x % i == 0)
                return false;
        return true;
    }

    // 867 https://leetcode.com/problems/transpose-matrix/description/
    public int[][] transpose(int[][] A) {
        int m = A.length;
        if (m == 0)
            return A;
        int n = A[0].length;
        int[][] B = new int[n][m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                B[i][j] = A[j][i];
            }
        }
        return B;
    }

    // 868 https://leetcode.com/problems/binary-gap/description/
    public int binaryGap(int N) {
        int res = 0;
        int cur = 0;
        int pre = -1;

        while (N != 0) {
            cur++;
            if ((N & 1) == 1) {
                if (pre != -1) {
                    res = Math.max(res, cur - pre);
                }
                pre = cur;
            }
            N = N >> 1;
        }
        return res;
    }

    // 869 https://leetcode.com/problems/reordered-power-of-2/description/
    public boolean reorderedPowerOf2(int N) {
        String strN = String.valueOf(N);
        int len = strN.length();
        int[] numDict = new int[10];
        for (int i = 0; i < len; i++) {
            numDict[strN.charAt(i) - '0'] += 1;
        }
        long val = 1;
        while (String.valueOf(val).length() < len) {
            val = val * 2;
        }

        while (String.valueOf(val).length() == len) {
            String strval = String.valueOf(val);
            int i;
            for (i = 0; i < len; i++) {
                if (numDict[strval.charAt(i) - '0'] == 0) {
                    for (i -= 1; i >= 0; i--) {
                        numDict[strval.charAt(i) - '0'] += 1;
                    }
                    break;
                }
                numDict[strval.charAt(i) - '0'] -= 1;
            }
            if (i == len) {
                return true;
            }
            val = val * 2;
        }

        return false;
    }

    // 870 https://leetcode.com/problems/advantage-shuffle/description/
    public int[] advantageCount(int[] A, int[] B) {
        int[] advanA = new int[A.length];
        Arrays.sort(A);
        PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] a, int[] b) {
                return b[0] - a[0];
            }
        });

        for (int i = 0; i < A.length; i++) {
            pq.offer(new int[] { B[i], i });
        }

        int head = 0;
        int tail = A.length - 1;
        while (!pq.isEmpty()) {
            int[] b = pq.poll();
            if (A[tail] > b[0]) {
                advanA[b[1]] = A[tail];
                tail--;
            } else {
                advanA[b[1]] = A[head];
                head++;
            }
        }
        return advanA;
    }

    // 871
    // https://leetcode.com/problems/minimum-number-of-refueling-stops/description/
    // TODO Reeeeview
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        long[] dp = new long[stations.length + 1];
        dp[0] = startFuel;

        for (int i = 0; i < stations.length; i++) {
            for (int j = i; j >= 0 && dp[j] >= stations[i][0]; j--) {
                dp[j + 1] = Math.max(dp[j + 1], dp[j] + stations[i][1]);
            }
        }
        for (int i = 0; i <= stations.length; i++) {
            if (dp[i] >= target)
                return i;
        }
        return -1;
    }

    // 872 https://leetcode.com/problems/leaf-similar-trees/description/
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> seq1 = new ArrayList<>();
        findLeafSeq(seq1, root1);
        List<Integer> seq2 = new ArrayList<>();
        findLeafSeq(seq2, root2);
        if (seq1.size() != seq2.size())
            return false;

        for (int i = 0; i < seq1.size(); i++) {
            if (seq1.get(i) != seq2.get(i))
                return false;
        }
        return true;
    }

    private void findLeafSeq(List<Integer> seq, TreeNode root) {
        if (root.left == null && root.right == null) {
            seq.add(root.val);
            return;
        }
        if (root.left != null) {
            findLeafSeq(seq, root.left);
        }
        if (root.right != null) {
            findLeafSeq(seq, root.right);
        }
    }

    // 873
    // https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/description/
    public int lenLongestFibSubseq(int[] A) {
        int n = A.length;
        int[][] dp = new int[n][n];
        int res = 2;
        Map<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < n; i++) {
            map.put(A[i], i);
            for (int j = 0; j < i; j++) {
                dp[j][i] = 2;
                int k = map.getOrDefault(A[i] - A[j], -1);
                if (k == -1 || k >= j)
                    continue;
                dp[j][i] = dp[k][j] + 1;
                res = Math.max(res, dp[j][i]);
            }
        }
        return res == 2 ? 0 : res;
    }

    // 874 https://leetcode.com/problems/walking-robot-simulation/description/
    public int robotSim(int[] commands, int[][] obstacles) {
        Set<String> obSet = new HashSet<>();
        int[][] move = new int[][] { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        for (int[] p : obstacles) {
            obSet.add(Arrays.toString(p));
        }
        int[] position = new int[] { 0, 0 };
        int direction = 0, res = 0;
        for (int cmd : commands) {
            if (cmd == -1) {
                direction = (direction + 1) % 4;
            } else if (cmd == -2) {
                direction = (direction + 3) % 4;
            } else {
                while (cmd != 0) {
                    int[] next = new int[] { position[0] + move[direction][0], position[1] + move[direction][1] };
                    if (obSet.contains(Arrays.toString(next))) {
                        break;
                    }
                    position = next;
                    cmd--;
                }
                res = Math.max(res, position[0] * position[0] + position[1] * position[1]);
            }
        }
        return res;
    }

    // 875 https://leetcode.com/problems/koko-eating-bananas/description/
    public int minEatingSpeed(int[] piles, int H) {
        int l = 1, r = 0;
        for (int i : piles) {
            r = Math.max(r, i);
        }
        while (l <= r) {
            int m = (l + r) / 2;
            int sum = 0;
            for (int i : piles) {
                sum += (int) Math.ceil((double) i / m);
            }
            if (sum > H)
                l = m + 1;
            else
                r = m - 1;
        }
        return l;
    }

    // 876 https://leetcode.com/problems/middle-of-the-linked-list/description/
    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    // 877 https://leetcode.com/problems/stone-game/description/
    public boolean stoneGame(int[] piles) {
        int n = piles.length;
        int[][] dp = new int[n][n];

        for (int i = 0; i < n; i++) {
            dp[i][i] = piles[i];
        }
        for (int i = 0; i < n; i++) {
            for (int j = i, k = 0; j < n; j++, k++) {
                if (i == 0) {
                    dp[k][j] = piles[j];
                } else if (i == 1) {
                    dp[k][j] = Math.max(piles[k], piles[j]);
                } else {
                    dp[k][j] = Math.max(piles[k] + Math.min(dp[k + 1][j - 1], dp[k + 2][j]),
                            piles[j] + Math.min(dp[k][j - 2], dp[k + 1][j - 1]));
                }
            }
        }
        return dp[0][n - 1] > dp[1][n - 1] || dp[0][n - 1] > dp[0][n - 2];
    }

    // 878 https://leetcode.com/problems/nth-magical-number/description/
    // Note I don't like this question
    public int nthMagicalNumber(int N, int A, int B) {
        long l = 2, r = (long) 1e14;
        long MOD = (long) 1e9 + 7;
        long lcm = A * B / generateGCD(A, B);
        while (l < r) {
            long m = (l + r) / 2;
            if (m / A + m / B - m / lcm < N)
                l = m + 1;
            else
                r = m - 1;
        }
        return (int) (l % MOD);
    }

    private int generateGCD(int a, int b) {
        if (b == 0)
            return a;
        else
            return generateGCD(b, a % b);
    }

    // 879 https://leetcode.com/problems/profitable-schemes/description/
    // TODO: review!
    public int profitableSchemes(int G, int P, int[] group, int[] profit) {
        int MOD = 1000000000 + 7;
        int[][] dp = new int[G + 1][P + 1];
        int len = group.length;
        dp[0][0] = 1;
        for (int i = 0; i < len; i++) {
            for (int m = P; m >= 0; m--) {
                for (int n = G - group[i]; n >= 0; n--) {
                    int p = Math.min(P, profit[i] + m);
                    dp[n + group[i]][p] = (dp[n + group[i]][p] + dp[n][m]) % MOD;
                }
            }
        }

        int res = 0;
        for (int i = 0; i <= G; i++) {
            res = (res + dp[i][P]) % MOD;
        }
        return res;
    }

    // 880 https://leetcode.com/problems/decoded-string-at-index/description/
    public String decodeAtIndex(String S, int K) {
        long len = 0;
        int i;
        for (i = 0; len < K; i++) {
            char c = S.charAt(i);
            if (Character.isDigit(c)) {
                len = len * (c - '0');
            } else {
                len = len + 1;
            }
        }

        while (i > 0) {
            i--;
            char c = S.charAt(i);
            if (Character.isDigit(c)) {
                len = len / (c - '0');
                K = (int) (K % len);
            } else {
                if (K % len == 0) {
                    return String.valueOf(c);
                }
                len--;
            }
        }
        return "";
    }

    // 881 https://leetcode.com/problems/boats-to-save-people/description/
    public int numRescueBoats(int[] people, int limit) {
        int res = 0;

        Arrays.sort(people);
        int head = 0, tail = people.length - 1;
        while (head <= tail) {
            if (people[tail] + people[head] <= limit) {
                res += 1;
                head += 1;
                tail -= 1;
            } else {
                res += 1;
                tail -= 1;
            }
        }
        return res;
    }

    // 882
    // https://leetcode.com/problems/reachable-nodes-in-subdivided-graph/description/
    // TODO Must redo this question!
    public int reachableNodes(int[][] edges, int M, int N) {
        int[][] graph = new int[N][N];

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                graph[i][j] = -1;
            }
        }
        for (int i = 0; i < edges.length; i++) {
            graph[edges[i][0]][edges[i][1]] = edges[i][2];
            graph[edges[i][1]][edges[i][0]] = edges[i][2];
        }

        int res = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> (b[1] - a[1]));
        boolean[] visited = new boolean[N];
        pq.offer(new int[] { 0, M });

        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            int start = cur[0];
            int move = cur[1];
            if (visited[start]) {
                continue;
            }
            visited[start] = true;
            res++;
            for (int i = 0; i < N; i++) {
                if (graph[start][i] != -1) {
                    if (move > graph[start][i] && !visited[i]) {
                        pq.offer(new int[] { i, move - graph[start][i] - 1 });
                    }
                    graph[i][start] -= Math.min(move, graph[start][i]);
                    res += Math.min(move, graph[start][i]);
                }
            }
        }
        return res;
    }

    // 883 https://leetcode.com/problems/projection-area-of-3d-shapes/description/
    public int projectionArea(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        // xy
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] != 0) {
                    res = res + 1;
                }
            }
        }
        // yz
        for (int i = 0; i < n; i++) {
            int high = 0;
            for (int j = 0; j < m; j++) {
                high = Math.max(high, grid[j][i]);
            }
            res = res + high;
        }
        // zx
        for (int i = 0; i < m; i++) {
            int high = 0;
            for (int j = 0; j < n; j++) {
                high = Math.max(high, grid[i][j]);
            }
            res = res + high;
        }
        return res;
    }

    // 884
    // https://leetcode.com/problems/uncommon-words-from-two-sentences/description/
    public String[] uncommonFromSentences(String A, String B) {
        Set<String> res = new HashSet<>();
        Set<String> wordSet = new HashSet<>();

        for (String s : A.split(" ")) {
            if (wordSet.contains(s)) {
                res.remove(s);
            } else {
                res.add(s);
                wordSet.add(s);
            }
        }
        for (String s : B.split(" ")) {
            if (wordSet.contains(s)) {
                res.remove(s);
            } else {
                res.add(s);
                wordSet.add(s);
            }
        }
        return (String[]) res.toArray();
    }

    // 885 https://leetcode.com/problems/spiral-matrix-iii/description/
    public int[][] spiralMatrixIII(int R, int C, int r0, int c0) {
        int[][] direction = new int[][] { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        List<int[]> res = new LinkedList<>();
        int n = R * C;
        int r = r0;
        int c = c0;
        int len = 1;
        int count = 0;
        int i = 0;
        int d = 0;

        while (res.size() != n) {
            if (r >= 0 && r < R && c >= 0 && c < C)
                res.add(new int[] { r, c });

            r += direction[d][0];
            c += direction[d][1];
            i++;
            if (i == len) {
                i = 0;
                d = (d + 1) % 4;
                count++;
                if (count == 2) {
                    len += 1;
                    count = 0;
                }
            }
        }
        return res.toArray(new int[0][0]);
    }

    // 886 https://leetcode.com/problems/possible-bipartition/description/
    public boolean possibleBipartition(int N, int[][] dislikes) {
        boolean[][] graph = new boolean[N + 1][N + 1];
        boolean[] visited = new boolean[N + 1];
        Set<Integer> group1 = new HashSet<>();
        Set<Integer> group2 = new HashSet<>();

        for (int[] dislike : dislikes) {
            graph[dislike[0]][dislike[1]] = true;
            graph[dislike[1]][dislike[0]] = true;
        }
        for (int i = 1; i <= N; i++) {
            if (!dfsBipartition(graph, i, group1, group2, visited)) {
                return false;
            }
        }
        return true;
    }

    private boolean dfsBipartition(boolean[][] graph, int node, Set<Integer> cur, Set<Integer> next,
            boolean[] visited) {
        if (visited[node]) {
            return true;
        }
        visited[node] = true;
        cur.add(node);
        boolean[] edges = graph[node];
        for (int i = 1; i < graph.length; i++) {
            if (edges[i]) {
                if (cur.contains(i) || (!dfsBipartition(graph, i, next, cur, visited))) {
                    return false;
                }
            }
        }
        return true;
    }

    // 887 https://leetcode.com/problems/super-egg-drop/description/
    public int superEggDrop(int K, int N) {
        int[][] dp = new int[N + 1][K + 1];
        int res = 0;
        while (dp[res][K] < N) {
            res++;
            for (int k = 1; k <= K; ++k)
                dp[res][k] = dp[res - 1][k - 1] + dp[res - 1][k] + 1;
        }
        return res;
    }

    // 888 https://leetcode.com/problems/fair-candy-swap/description/
    public int[] fairCandySwap(int[] A, int[] B) {
        int sumA = 0, sumB = 0;
        Set<Integer> setA = new HashSet<>();
        for (int i : A) {
            sumA += i;
            setA.add(i);
        }
        for (int i : B) {
            sumB += i;
        }
        int target = (sumA + sumB) / 2 - sumA;
        for (int i : B) {
            if (setA.contains(i - target)) {
                return new int[] { i - target, i };
            }
        }
        return null;
    }

    // 889
    // https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/description/
    public TreeNode constructFromPrePost(int[] pre, int[] post) {
        return buildTreeHelper(pre, 0, pre.length, post, 0, post.length);
    }

    private TreeNode buildTreeHelper(int[] pre, int i, int j, int[] post, int m, int n) {
        TreeNode root = new TreeNode(pre[i]);
        if (i == j - 1)
            return root;
        if (pre[i + 1] == post[n - 2]) {
            root.left = buildTreeHelper(pre, i + 1, j, post, m, n - 1);
        } else {
            int v1 = pre[i + 1];
            int v2 = post[n - 2];
            for (int k = i; k < j; k++) {
                if (pre[k] == v2) {
                    v2 = k;
                    break;
                }
            }
            for (int k = m; k < n; k++) {
                if (post[k] == v1) {
                    v1 = k;
                    break;
                }
            }
            root.left = buildTreeHelper(pre, i + 1, v2, post, m, v1 + 1);
            root.right = buildTreeHelper(pre, v2, j, post, v1 + 1, n - 1);
        }
        return root;
    }

    // 890 https://leetcode.com/problems/find-and-replace-pattern/description/
    public List<String> findAndReplacePattern(String[] words, String pattern) {
        List<String> res = new LinkedList<>();
        for (String s : words) {
            if (isMatching(s, pattern)) {
                res.add(s);
            }
        }
        return res;
    }

    private boolean isMatching(String s, String pattern) {
        Map<Character, Character> map = new HashMap<>();
        Set<Character> used = new HashSet<>();

        for (int i = 0; i < pattern.length(); i++) {
            Character c1 = pattern.charAt(i);
            Character c2 = s.charAt(i);
            if (map.containsKey(c1) && map.get(c1) != c2)
                return false;
            if (!map.containsKey(c1) && used.contains(c2))
                return false;
            map.put(c1, c2);
            used.add(c2);
        }
        return true;
    }

    // 891 https://leetcode.com/problems/sum-of-subsequence-widths/description/
    public int sumSubseqWidths(int[] A) {
        int Mod = (int) 1e9 + 7;
        long res = 0;

        Arrays.sort(A);
        long combination = 1;
        for (int i = 0; i < A.length; ++i, combination = (combination << 1) % Mod) {
            res = (res + A[i] * combination - A[A.length - i - 1] * combination) % Mod;
        }
        return (int) res;
    }

    // 892 https://leetcode.com/problems/surface-area-of-3d-shapes/description/
    public int surfaceArea(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int v = grid[i][j];

                if (v > 0)
                    res += 2;

                if (i == 0) {
                    res += v;
                } else {
                    res += Math.abs(v - grid[i - 1][j]);
                }

                if (j == 0) {
                    res += v;
                } else {
                    res += Math.abs(v - grid[i][j - 1]);
                }
            }
        }
        for (int i = 0; i < m; i++) {
            res += grid[i][n - 1];
        }
        for (int i = 0; i < n; i++) {
            res += grid[m - 1][i];
        }
        return res;
    }

    // 893
    // https://leetcode.com/problems/groups-of-special-equivalent-strings/description/
    public int numSpecialEquivGroups(String[] A) {
        boolean[] choosed = new boolean[A.length];

        int res = 0;

        for (int i = 0; i < A.length; i++) {
            if (choosed[i])
                continue;
            res += 1;
            for (int j = i + 1; j < A.length; j++) {
                if (choosed[j])
                    continue;
                if (isEquivGroups(A[i], A[j]))
                    choosed[j] = true;
            }
        }
        return res;
    }

    private boolean isEquivGroups(String a, String b) {
        int[] even = new int[26];
        int[] odd = new int[26];

        for (int i = 0; i < a.length(); i++) {
            if (i % 2 == 0) {
                even[a.charAt(i) - 'a'] += 1;
                even[b.charAt(i) - 'a'] -= 1;
            } else {
                odd[a.charAt(i) - 'a'] += 1;
                odd[b.charAt(i) - 'a'] -= 1;
            }
        }
        for (int i = 0; i < 26; i++) {
            if (even[i] != 0 || odd[i] != 0)
                return false;
        }
        return true;
    }

    // 894 https://leetcode.com/problems/all-possible-full-binary-trees/description/
    public List<TreeNode> allPossibleFBT(int N) {
        List<TreeNode> res = new LinkedList<>();
        if (N % 2 == 0)
            return res;
        if (N == 1) {
            res.add(new TreeNode(0));
            return res;
        }

        N -= 1;
        for (int i = 1; i < N; i += 2) {
            List<TreeNode> l1 = allPossibleFBT(i);
            List<TreeNode> l2 = allPossibleFBT(N - i);
            for (TreeNode n : l1) {
                for (TreeNode m : l2) {
                    TreeNode root = new TreeNode(0);
                    root.left = n;
                    root.right = m;
                    res.add(root);
                }
            }
        }
        return res;
    }

    // 895 https://leetcode.com/problems/maximum-frequency-stack/description/
    // javasolution.structdesign

    // 896 https://leetcode.com/problems/monotonic-array/description/
    public boolean isMonotonic(int[] A) {
        int type = 0;

        for (int i = 0; i < A.length - 1; i++) {
            if (A[i] == A[i + 1])
                continue;

            if (A[i] > A[i + 1]) {
                if (type == 1)
                    return false;
                type = -1;
            }
            if (A[i] < A[i + 1]) {
                if (type == -1)
                    return false;
                type = 1;
            }
        }
        return true;
    }

    // 897 https://leetcode.com/problems/increasing-order-search-tree/description/
    public TreeNode increasingBST(TreeNode root) {
        if (root == null)
            return root;

        TreeNode dummy = new TreeNode(0);
        TreeNode p = dummy, cur = root;
        Stack<TreeNode> st = new Stack<>();

        while (cur != null || st.size() > 0) {
            while (cur != null) {
                st.push(cur);
                cur = cur.left;
            }
            cur = st.pop();
            p.left = null;
            p.right = cur;
            p = p.right;
            cur = cur.right;
        }
        p.left = null;
        p.right = null;
        return dummy.right;
    }

    // 898 https://leetcode.com/problems/bitwise-ors-of-subarrays/description/
    // TODO: I don't know the how to come out this solution
    public int subarrayBitwiseORs(int[] A) {
        Set<Integer> res = new HashSet<>(), cur = new HashSet<>(), cur2;
        for (Integer i : A) {
            cur2 = new HashSet<>();
            cur2.add(i);
            for (Integer j : cur)
                cur2.add(i | j);
            res.addAll(cur = cur2);
        }
        return res.size();
    }

    // 899 https://leetcode.com/problems/orderly-queue/description/
    public String orderlyQueue(String S, int K) {
        if (K > 1) {
            char[] charArr = S.toCharArray();
            Arrays.sort(charArr);
            return new String(charArr);
        }
        String res = S;
        for (int i = 1; i < S.length(); i++) {
            String tmp = S.substring(i) + S.substring(0, i);
            if (res.compareTo(tmp) > 0)
                res = tmp;
        }
        return res;
    }

    // 900 https://leetcode.com/problems/rle-iterator/description/
    // javasolution.structdesign

    // 901 https://leetcode.com/problems/online-stock-span/description/
    // javasolution.structdesign

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
}
