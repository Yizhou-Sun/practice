package com.backup;

import java.util.*;
import java.lang.*;

import com.javasolution.util.*;
import com.javasolution.structdesign.*;

public class Solution {
    // 300
    // Given an unsorted array of integers, find the length of longest increasing
    // subsequence.
    // Example:
    // Input: [10,9,2,5,3,7,101,18]
    // Output: 4
    // Explanation: The longest increasing subsequence is [2,3,7,101], therefore the
    // length is 4.
    // Note:
    // There may be more than one LIS combination, it is only necessary for you to
    // return the length.
    // Your algorithm should run in O(n2) complexity.
    // Follow up: Could you improve it to O(n log n) time complexity?
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[nums.length];
        int res = 1;
        for (int i = 0; i < nums.length; i++) {
            dp[i] = 1;
        }
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    // 299
    // You are playing the following Bulls and Cows game with your friend: You write
    // down a number and ask your friend to guess what the number is. Each time your
    // friend makes a guess, you provide a hint that indicates how many digits in
    // said guess match your secret number exactly in both digit and position
    // (called "bulls") and how many digits match the secret number but locate in
    // the wrong position (called "cows"). Your friend will use successive guesses
    // and hints to eventually derive the secret number.
    // Write a function to return a hint according to the secret number and friend's
    // guess, use A to indicate the bulls and B to indicate the cows.
    // Please note that both secret number and friend's guess may contain duplicate
    // digits.
    // Example 1:
    // Input: secret = "1807", guess = "7810"
    // Output: "1A3B"
    // Explanation: 1 bull and 3 cows. The bull is 8, the cows are 0, 1 and 7.
    // Example 2:
    // Input: secret = "1123", guess = "0111"
    // Output: "1A1B"
    // Explanation: The 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a
    // cow.
    // Note: You may assume that the secret number and your friend's guess only
    // contain digits, and their lengths are always equal.
    public String getHint(String secret, String guess) {
        int[] dict = new int[10];
        int bulls = 0, cows = 0;
        for (int i = 0; i < secret.length(); i++) {
            dict[secret.charAt(i) - '0'] += 1;
        }
        for (int i = 0; i < guess.length(); i++) {
            if (guess.charAt(i) == secret.charAt(i)) {
                dict[guess.charAt(i) - '0'] -= 1;
                bulls++;
            }
        }
        for (int i = 0; i < guess.length(); i++) {
            if (guess.charAt(i) != secret.charAt(i) && dict[guess.charAt(i) - '0'] > 0) {
                dict[guess.charAt(i) - '0'] -= 1;
                cows++;
            }
        }
        return bulls + "A" + cows + "B";
    }

    // 282
    // Given a string that contains only digits 0-9 and a target value, return all
    // possibilities to add binary operators (not unary) +, -, or * between the
    // digits so they evaluate to the target value.
    // Example 1:
    // Input: num = "123", target = 6
    // Output: ["1+2+3", "1*2*3"]
    // Example 2:
    // Input: num = "232", target = 8
    // Output: ["2*3+2", "2+3*2"]
    // Example 3:
    // Input: num = "105", target = 5
    // Output: ["1*0+5","10-5"]
    // Example 4:
    // Input: num = "00", target = 0
    // Output: ["0+0", "0-0", "0*0"]
    // Example 5:
    // Input: num = "3456237490", target = 9191
    // Output: []
    public List<String> addOperators(String num, int target) {
        List<String> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        if (num == null || num.length() == 0)
            return res;
        addOperatorsHelper(res, sb, num, target, 0, 0, 0);
        return res;
    }

    private void addOperatorsHelper(List<String> res, StringBuilder sb, String num, int target, int p, long value,
            long multed) {
        if (p == num.length()) {
            if (value == target)
                res.add(sb.toString());
            return;
        }
        for (int i = p; i < num.length(); i++) {
            if (i != p && num.charAt(p) == '0')
                break;
            long cur = Long.parseLong(num.substring(p, i + 1));
            if (p == 0) {
                sb.append(cur);
                addOperatorsHelper(res, sb, num, target, i + 1, cur, cur);
                sb.delete(sb.length() - (i + 1 - p), sb.length());
            } else {
                sb.append('+');
                sb.append(cur);
                addOperatorsHelper(res, sb, num, target, i + 1, value + cur, -cur);
                sb.delete(sb.length() - i + p - 2, sb.length());

                sb.append('-');
                sb.append(cur);
                addOperatorsHelper(res, sb, num, target, i + 1, value - cur, multed);
                sb.delete(sb.length() - i + p - 2, sb.length());

                sb.append('*');
                sb.append(cur);
                addOperatorsHelper(res, sb, num, target, i + 1, value - multed + multed * cur, multed * cur);
                sb.delete(sb.length() - i + p - 2, sb.length());
            }
        }
        return;
    }

    // 264
    // Write a program to find the n-th ugly number.
    // Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.
    // Example:
    // Input: n = 10
    // Output: 12
    // Explanation: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 is the sequence of the first 10
    // ugly numbers.
    // Note:
    // 1 is typically treated as an ugly number.
    // n does not exceed 1690.
    public int nthUglyNumber(int n) {
        int[] uglyArr = new int[n];
        uglyArr[0] = 1;
        int seq2 = 2, seq3 = 3, seq5 = 5;
        int index2 = 0, index3 = 0, index5 = 0;
        for (int i = 1; i < n; i++) {
            int minVal = Math.min(seq2, Math.min(seq3, seq5));
            uglyArr[i] = minVal;
            if (seq2 == minVal)
                seq2 = 2 * uglyArr[++index2];
            if (seq3 == minVal)
                seq3 = 3 * uglyArr[++index3];
            if (seq5 == minVal)
                seq5 = 5 * uglyArr[++index5];
        }
        return uglyArr[n - 1];
    }

    // 260
    // Given an array of numbers nums, in which exactly two elements appear only
    // once and all the other elements appear exactly twice. Find the two elements
    // that appear only once.
    // Example:
    // Input: [1,2,1,3,2,5]
    // Output: [3,5]
    // Note:
    // The order of the result is not important. So in the above example, [5, 3] is
    // also correct.
    // Your algorithm should run in linear runtime complexity. Could you implement
    // it using only constant space complexity?
    public int[] singleNumber(int[] nums) {
        Set<Integer> mySet = new HashSet<>();
        for (int i : nums) {
            if (mySet.contains(i)) {
                mySet.remove(i);
            } else {
                mySet.add(i);
            }
        }

        int[] res = new int[2];
        int i = 0;
        for (int n : mySet) {
            res[i] = n;
            i++;
        }
        return res;
    }

    // 258
    // Given a non-negative integer num, repeatedly add all its digits until the
    // result has only one digit.
    // Example:
    // Input: 38
    // Output: 2
    // Explanation: The process is like: 3 + 8 = 11, 1 + 1 = 2.
    // Since 2 has only one digit, return it.
    // Follow up:
    // Could you do it without any loop/recursion in O(1) runtime?
    public int addDigits(int num) {
        // https://en.wikipedia.org/wiki/Digital_root
        return 1 + (num - 1) % 9;
    }

    // 257
    // Given a binary tree, return all root-to-leaf paths.
    // Note: A leaf is a node with no children.
    // Example:
    // Input:
    // 1
    // / \
    // 2 3
    // \
    // 5
    // Output: ["1->2->5", "1->3"]
    // Explanation: All root-to-leaf paths are: 1->2->5, 1->3
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root != null) {
            StringBuilder sb = new StringBuilder();
            sb.append(root.val);
            if (root.left == null && root.right == null) {
                res.add(sb.toString());
            }
            buildPathHelper(root.left, res, sb);
            buildPathHelper(root.right, res, sb);
        }
        return res;
    }

    private void buildPathHelper(TreeNode root, List<String> res, StringBuilder sb) {
        if (root == null) {
            return;
        }
        int n = sb.length();
        sb.append("->");
        sb.append(root.val);
        if (root.left == null && root.right == null) {
            res.add(sb.toString());
        }
        buildPathHelper(root.left, res, sb);
        buildPathHelper(root.right, res, sb);
        sb.delete(n, sb.length());
        return;
    }

    // 241
    // Given a string of numbers and operators, return all possible results from
    // computing all the different possible ways to group numbers and operators. The
    // valid operators are +, - and *.
    // Example 1:
    // Input: "2-1-1"
    // Output: [0, 2]
    // Explanation:
    // ((2-1)-1) = 0
    // (2-(1-1)) = 2
    // Example 2:
    // Input: "2*3-4*5"
    // Output: [-34, -14, -10, -10, 10]
    // Explanation:
    // (2*(3-(4*5))) = -34
    // ((2*3)-(4*5)) = -14
    // ((2*(3-4))*5) = -10
    // (2*((3-4)*5)) = -10
    // (((2*3)-4)*5) = 10
    public List<Integer> diffWaysToCompute(String input) {
        return diffWaysToComputeHelper(input, new HashMap<>());
    }

    private List<Integer> diffWaysToComputeHelper(String input, Map<String, List<Integer>> map) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);
            if (c == '+' || c == '-' || c == '*') {
                String leftString = input.substring(0, i);
                List<Integer> left;
                if (map.containsKey(leftString)) {
                    left = map.get(leftString);
                } else {
                    left = diffWaysToComputeHelper(leftString, map);
                }
                String rightString = input.substring(i + 1);

                List<Integer> right;
                if (map.containsKey(rightString)) {
                    right = map.get(rightString);
                } else {
                    right = diffWaysToComputeHelper(rightString, map);
                }

                for (int l : left) {
                    for (int r : right) {
                        switch (c) {
                        case '+':
                            res.add(l + r);
                            break;
                        case '-':
                            res.add(l - r);
                            break;
                        case '*':
                            res.add(l * r);
                            break;
                        }
                    }
                }
            }
        }
        if (res.size() == 0)
            res.add(Integer.valueOf(input));
        return res;
    }

    // 240
    // Write an efficient algorithm that searches for a value in an m x n matrix.
    // This matrix has the following properties:
    // Integers in each row are sorted in ascending from left to right.
    // Integers in each column are sorted in ascending from top to bottom.
    // Consider the following matrix:
    // [
    // [1, 4, 7, 11, 15],
    // [2, 5, 8, 12, 19],
    // [3, 6, 9, 16, 22],
    // [10, 13, 14, 17, 24],
    // [18, 21, 23, 26, 30]
    // ]
    // Example 1:
    // Input: matrix, target = 5
    // Output: true
    // Example 2:
    // Input: matrix, target = 20
    // Output: false
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        if (m == 0)
            return false;
        int n = matrix[0].length;
        if (n == 0)
            return false;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] <= target)
                for (int j = 0; j < n; j++) {
                    if (matrix[i][j] == target) {
                        return true;
                    }
                    if (matrix[i][j] > target) {
                        break;
                    }
                }
        }
        return false;
    }

    // 239
    // Given an array nums, there is a sliding window of size k which is moving from
    // the very left of the array to the very right. You can only see the k numbers
    // in the window. Each time the sliding window moves right by one position.
    // Return the max sliding window.
    // Example:
    // Input: nums = [1,3,-1,-3,5,3,6,7], and k = 3
    // Output: [3,3,5,5,6,7]
    // Explanation:
    // Window position Max
    // --------------- -----
    // [1 3 -1] -3 5 3 6 7 3
    // 1 [3 -1 -3] 5 3 6 7 3
    // 1 3 [-1 -3 5] 3 6 7 5
    // 1 3 -1 [-3 5 3] 6 7 5
    // 1 3 -1 -3 [5 3 6] 7 6
    // 1 3 -1 -3 5 [3 6 7] 7
    // Note:
    // You may assume k is always valid, 1 ≤ k ≤ input array's size for non-empty
    // array.
    // Follow up:
    // Could you solve it in linear time?
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> dq = new ArrayDeque<>();
        int len = nums.length;
        if (len == 0)
            return new int[0];
        int[] res = new int[len + 1 - k];
        for (int i = 0; i < len; i++) {
            while (!dq.isEmpty() && i - dq.peekFirst() >= k) {
                dq.pollFirst();
            }

            while (!dq.isEmpty() && nums[dq.peekLast()] < nums[i]) {
                dq.pollLast();
            }

            dq.offer(i);
            if (i + 1 - k >= 0) {
                res[i + 1 - k] = nums[dq.peekFirst()];
            }
        }
        return res;
    }

    // 238
    // Given an array nums of n integers where n > 1, return an array output such
    // that output[i] is equal to the product of all the elements of nums except
    // nums[i].
    // Example:
    // Input: [1,2,3,4]
    // Output: [24,12,8,6]
    // Note: Please solve it without division and in O(n).
    // Follow up:
    // Could you solve it with constant space complexity? (The output array does not
    // count as extra space for the purpose of space complexity analysis.)
    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        int temp = 1;

        for (int i = 0; i < nums.length; i++) {
            res[i] = temp;
            temp *= nums[i];
        }

        temp = nums[nums.length - 1];
        for (int i = nums.length - 2; i >= 0; i++) {
            res[i] *= temp;
            temp *= nums[i];
        }

        return res;
    }

    // 237
    // Write a function to delete a node (except the tail) in a singly linked list,
    // given only access to that node.
    // Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node
    // with value 3, the linked list should become 1 -> 2 -> 4 after calling your
    // function.
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    // 236 & 235
    // Given a binary tree, find the lowest common ancestor (LCA) of two given nodes
    // in the tree.
    // According to the definition of LCA on Wikipedia: “The lowest common ancestor
    // is defined between two nodes v and w as the lowest node in T that has both v
    // and w as descendants (where we allow a node to be a descendant of itself).”
    // Given the following binary search tree: root = [3,5,1,6,2,0,8,null,null,7,4]
    // _______3______
    // / \
    // ___5__ ___1__
    // / \ / \
    // 6 _2 0 8
    // / \
    // 7 4
    // Example 1:
    // Input: root, p = 5, q = 1
    // Output: 3
    // Explanation: The LCA of of nodes 5 and 1 is 3.
    // Example 2:
    // Input: root, p = 5, q = 4
    // Output: 5
    // Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant
    // of itself
    // according to the LCA definition.
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q)
            return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null)
            return root;
        return left != null ? left : right;
    }

    public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val > q.val) {
            TreeNode temp = p;
            p = q;
            q = temp;
        }
        if (root.val >= p.val && root.val <= q.val) {
            return root;
        }
        if (root.val < p.val) {
            return lowestCommonAncestorBST(root.right, p, q);
        } else {
            return lowestCommonAncestorBST(root.left, p, q);
        }
    }

    // 234
    // Given a singly linked list, determine if it is a palindrome.
    // Follow up:
    // Could you do it in O(n) time and O(1) space?
    public boolean isPalindrome(ListNode head) {
        Stack<ListNode> st = new Stack<>();
        ListNode cur = head;
        while (cur != null) {
            st.push(cur);
            cur = cur.next;
        }
        cur = head;
        while (cur != null && cur.val == st.pop().val) {
            cur = cur.next;
        }
        return st.isEmpty();
    }

    // 233
    // Given an integer n, count the total number of digit 1 appearing in all
    // non-negative integers less than or equal to n.
    // Example:
    // Input: 13
    // Output: 6
    // Explanation: Digit 1 occurred in the following numbers: 1, 10, 11, 12, 13.
    public int countDigitOne(int n) {
        int count = 0;
        for (long k = 1; k <= n; k *= 10) {
            long r = n / k, m = n % k;
            // sum up the count of ones on every place k
            count += (r + 8) / 10 * k + (r % 10 == 1 ? m + 1 : 0);
        }
        return count;
    }

    // 232
    // Implement the following operations of a queue using stacks.
    // push(x) -- Push element x to the back of queue.
    // pop() -- Removes the element from in front of queue.
    // peek() -- Get the front element.
    // empty() -- Return whether the queue is empty.
    // Notes:
    // You must use only standard operations of a stack -- which means only push to
    // top, peek/pop from top, size, and is empty operations are valid.
    // Depending on your language, stack may not be supported natively. You may
    // simulate a stack by using a list or deque (double-ended queue), as long as
    // you use only standard operations of a stack.
    // You may assume that all operations are valid (for example, no pop or peek
    // operations will be called on an empty queue).

    // 231
    // Given an integer, write a function to determine if it is a power of two.
    // Example 1:
    // Input: 1
    // Output: true
    // Example 2:
    // Input: 16
    // Output: true
    // Example 3:
    // Input: 218
    // Output: false
    public boolean isPowerOfTwo(int n) {
        double cur = 1;
        while (cur <= n) {
            if (cur == n)
                return true;
            cur *= 2;
        }
        return false;
    }

    // 230
    // Given a binary search tree, write a function kthSmallest to find the kth
    // smallest element in it.
    // Note:
    // You may assume k is always valid, 1 ≤ k ≤ BST's total elements.
    // Example 1:
    // Input: root = [3,1,4,null,2], k = 1
    // Output: 1
    // Example 2:
    // Input: root = [5,3,6,2,4,null,null,1], k = 3
    // Output: 3
    // Follow up:
    // What if the BST is modified (insert/delete operations) often and you need to
    // find the kth smallest frequently? How would you optimize the kthSmallest
    // routine?
    public int kthSmallest(TreeNode root, int k) {
        return kthSmallestHelper(root, k);
    }

    private int count = 0;

    private Integer kthSmallestHelper(TreeNode root, int k) {
        if (root.left != null) {
            Integer res = kthSmallestHelper(root.left, k);
            if (res != null) {
                return res;
            }
        }
        count += 1;

        if (k == count) {
            return root.val;
        }

        if (root.right != null) {
            Integer res = kthSmallestHelper(root.right, k);
            if (res != null) {
                return res;
            }
        }
        return null;
    }

    // 229
    // Given an integer array of size n, find all elements that appear more than ⌊
    // n/3 ⌋ times.
    // Note: The algorithm should run in linear time and in O(1) space.
    // Example 1:
    // Input: [3,2,3]
    // Output: [3]
    // Example 2:
    // Input: [1,1,1,3,3,2,2,2]
    // Output: [1,2]
    public List<Integer> majorityElement(int[] nums) {
        List<Integer> res = new ArrayList<>();
        int len = nums.length;
        if (len == 0)
            return res;

        int n1 = nums[0], n2 = nums[0], count1 = 0, count2 = 0;

        for (int i : nums) {
            if (i == n1) {
                count1 += 1;
            } else if (i == n2) {
                count2 += 1;
            } else if (count1 == 0) {
                n1 = i;
                count1 = 1;
            } else if (count2 == 0) {
                n2 = i;
                count2 = 1;
            } else {
                count1 -= 1;
                count2 -= 1;
            }
        }

        count1 = 0;
        count2 = 0;
        for (int i : nums) {
            if (i == n1)
                count1 += 1;
            else if (i == n2)
                count2 += 1;
        }
        if (count1 > len / 3)
            res.add(n1);
        if (count2 > len / 3)
            res.add(n2);
        return res;

        // Set<Integer> res = majorEleHelper(nums, 0, nums.length);
        // return (new LinkedList(res));
    }

    public Set<Integer> majorEleHelper(int[] nums, int start, int end) {
        Set<Integer> res = new HashSet<>();
        int len = end - start;

        if (len == 0) {
            return res;
        }
        if (len == 1) {
            res.add(nums[start]);
            return res;
        }

        Set<Integer> temp = majorEleHelper(nums, start, start + len / 3);
        temp.addAll(majorEleHelper(nums, start + len / 3, start + len * 2 / 3));
        temp.addAll(majorEleHelper(nums, start + len * 2 / 3, end));
        for (int i : temp) {
            if (isMajor(nums, i, start, end)) {
                res.add(i);
            }
        }
        return res;
    }

    private boolean isMajor(int[] nums, int k, int start, int end) {
        int count = 0, target = (end - start) / 3;
        for (int i = start; i < end; i++) {
            if (nums[i] == k) {
                count += 1;
            }
            if (count > target) {
                return true;
            }
        }
        return false;
    }

    // 228
    // Given a sorted integer array without duplicates, return the summary of its
    // ranges.
    // Example 1:
    // Input: [0,1,2,4,5,7]
    // Output: ["0->2","4->5","7"]
    // Explanation: 0,1,2 form a continuous range; 4,5 form a continuous range.
    // Example 2:
    // Input: [0,2,3,4,6,8,9]
    // Output: ["0","2->4","6","8->9"]
    // Explanation: 2,3,4 form a continuous range; 8,9 form a continuous range.
    public List<String> summaryRanges(int[] nums) {
        List<String> res = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        if (nums.length == 0)
            return res;
        int pre = 0;
        boolean single = true;
        for (int i : nums) {
            if (sb.length() == 0) {
                sb.append(i);
                single = true;
            } else if (pre + 1 == i) {
                single = false;
            } else {
                if (single) {
                    res.add(sb.toString());
                } else {
                    sb.append("->");
                    sb.append(pre);
                    res.add(sb.toString());
                }
                sb = new StringBuilder();
                sb.append(i);
                single = true;
            }
            pre = i;
        }
        if (single) {
            res.add(sb.toString());
        } else {
            sb.append("->");
            sb.append(pre);
            res.add(sb.toString());
        }
        return res;
    }

    // 226
    // Invert a binary tree.
    // 4
    // / \
    // 2 7
    // / \ / \
    // 1 3 6 9
    // TO
    // 4
    // / \
    // 7 2
    // / \ / \
    // 9 6 3 1
    // Trivia:
    // This problem was inspired by this original tweet by Max Howell:
    // Google: 90% of our engineers use the software you wrote (Homebrew), but you
    // can’t invert a binary tree on a whiteboard so f*** off.
    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return root;
        TreeNode left = invertTree(root.right);
        TreeNode right = invertTree(root.left);
        root.left = left;
        root.right = right;
        return root;
    }

    // 225
    // Implement the following operations of a stack using queues.
    // push(x) -- Push element x onto stack.
    // pop() -- Removes the element on top of the stack.
    // top() -- Get the top element.
    // empty() -- Return whether the stack is empty.
    // Notes:
    // You must use only standard operations of a queue -- which means only push to
    // back, peek/pop from front, size, and is empty operations are valid.
    // Depending on your language, queue may not be supported natively. You may
    // simulate a queue by using a list or deque (double-ended queue), as long as
    // you use only standard operations of a queue.
    // You may assume that all operations are valid (for example, no pop or top
    // operations will be called on an empty stack).

    // 224
    // Implement a basic calculator to evaluate a simple expression string.
    // The expression string may contain open ( and closing parentheses ), the plus
    // + or minus sign -, non-negative integers and empty spaces .
    // Example 1:
    // Input: "1 + 1"
    // Output: 2
    // Example 2:
    // Input: " 2-1 + 2 "
    // Output: 3
    // Example 3:
    // Input: "(1+(4+5+2)-3)+(6+8)"
    // Output: 23
    // Note:
    // You may assume that the given expression is always valid.
    // Do not use the eval built-in library function.
    public int calculate(String s) {
        int cur = 0;
        int sign = 1;
        int res = 0;
        Stack<Integer> st = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == ' ')
                continue;
            if (Character.isDigit(c)) {
                cur = cur * 10 + c - '0';
            } else {
                if (c == '(') {
                    st.push(res);
                    st.push(sign);
                    res = 0;
                    sign = 1;
                } else if (c == ')') {
                    res = res + sign * cur;
                    sign = st.pop();
                    cur = st.pop();
                    res = cur + sign * res;
                    cur = 0;
                    sign = 1;
                } else {
                    res = res + sign * cur;
                    sign = (c == '+' ? 1 : -1);
                    cur = 0;
                }
            }
        }
        return (res + sign * cur);
    }

    // 223
    // Find the total area covered by two rectilinear rectangles in a 2D plane.
    // Each rectangle is defined by its bottom left corner and top right corner as
    // shown in the figure.
    // Rectangle Area
    // Example:
    // Input: -3, 0, 3, 4, 0, -1, 9, 2
    // Output: 45
    // Note:
    // Assume that the total area is never beyond the maximum possible value of int.
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int area1 = (C - A) * (D - B);
        int area2 = (G - E) * (H - F);

        int left = Math.max(A, E);
        int right = Math.min(G, C);
        int bottom = Math.max(F, B);
        int top = Math.min(D, H);

        int overlap = 0;
        if (right > left && top > bottom)
            overlap = (right - left) * (top - bottom);

        return area1 + area2 - overlap;
    }

    // 222
    // Given a complete binary tree, count the number of nodes.
    // Note:
    // Definition of a complete binary tree from Wikipedia:
    // In a complete binary tree every level, except possibly the last, is
    // completely filled, and all nodes in the last level are as far left as
    // possible. It can have between 1 and 2h nodes inclusive at the last level h.
    // Example:
    // Input:
    // 1
    // / \
    // 2 3
    // / \ /
    // 4 5 6
    // Output: 6
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.add(root);
        int count = 1;

        while (!q.isEmpty()) {
            TreeNode temp = q.poll();
            System.out.println("val is " + temp.val);
            if (temp.val != -1000) {
                temp.val = -1000;

                if (temp.left != null) {
                    q.offer(temp.left);
                    count++;
                }

                if (temp.right != null) {
                    q.offer(temp.right);
                    count++;
                }
            }
        }
        return count;
    }

    // 221
    // Given a 2D binary matrix filled with 0's and 1's, find the largest square
    // containing only 1's and return its area.
    // Example:
    // Input:
    // 1 0 1 0 0
    // 1 0 1 1 1
    // 1 1 1 1 1
    // 1 0 0 1 0
    // Output: 4
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length;
        if (m == 0)
            return 0;
        int n = matrix[0].length;

        int res = Integer.MIN_VALUE;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                char c = matrix[i - 1][j - 1];
                if (c == '0') {
                    dp[i][j] = 0;
                } else {
                    int upleft = dp[i - 1][j - 1];
                    int up = dp[i - 1][j];
                    int left = dp[i][j - 1];
                    if (upleft == 0 || up == 0 || left == 0) {
                        dp[i][j] = 1;
                    } else {
                        int minVal = Math.min(upleft, Math.min(up, left));
                        int sqr = (int) Math.sqrt(minVal);
                        dp[i][j] = (int) Math.pow(sqr + 1, 2);
                    }
                }
                res = Math.max(res, dp[i][j]);
            }
        }
        return res;
    }

    // 220
    // Given an array of integers, find out whether there are two distinct indices i
    // and j in the array such that the absolute difference between nums[i] and
    // nums[j] is at most t and the absolute difference between i and j is at most
    // k.
    // Example 1:
    // Input: [1,2,3,1], k = 4, t = 0
    // Output: true
    // Example 2:
    // Input: [1,0,1,1], k = 1, t = 0
    // Output: true
    // Example 3:
    // Input: [4,2], k = 2, t = 1
    // Output: false
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeSet<Long> treeSet = new TreeSet<>();

        for (int i = 0; i < nums.length; i++) {
            Long ceiling = treeSet.ceiling((long) nums[i]);
            Long floor = treeSet.floor((long) nums[i]);
            if (ceiling != null && ceiling - nums[i] <= t || floor != null && nums[i] - floor <= t)
                return true;
            treeSet.add((long) nums[i]);
            if (i >= k) {
                treeSet.remove((long) nums[i - k]);
            }
        }
        return false;
    }

    // 219
    // Given an array of integers and an integer k, find out whether there are two
    // distinct indices i and j in the array such that nums[i] = nums[j] and the
    // absolute difference between i and j is at most k.
    // Example 1:
    // Input: [1,2,3,1], k = 3
    // Output: true
    // Example 2:
    // Input: [1,0,1,1], k = 1
    // Output: true
    // Example 3:
    // Input: [1,2,1], k = 0
    // Output: false
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> mySet = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (i > k)
                mySet.remove(nums[i - k - 1]);
            if (!mySet.add(nums[i]))
                return true;
        }
        return false;
    }

    // 218
    // A city's skyline is the outer contour of the silhouette formed by all the
    // buildings in that city when viewed from a distance. Now suppose you are given
    // the locations and height of all the buildings as shown on a cityscape photo
    // (Figure A), write a program to output the skyline formed by these buildings
    // collectively (Figure B).
    // The geometric information of each building is represented by a triplet of
    // integers [Li, Ri, Hi], where Li and Ri are the x coordinates of the left and
    // right edge of the ith building, respectively, and Hi is its height. It is
    // guaranteed that 0 ≤ Li, Ri ≤ INT_MAX, 0 < Hi ≤ INT_MAX, and Ri - Li > 0. Yo
    // 
    // may assume all buildings are perfect rectangles grounded on an absolutely
    // flat surface at height 0.
    // For instance, the dimensions of all buildings in Figure A are recorded as:
    // [[2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8]]
    // The output is a list of "key points" (red dots in Figure B) in the format of
    // [ [x1,y1], [x2, y2], [x3, y3], ... ] that uniquely defines a skyline. A key
    // point is the left endpoint of a horizontal line segment. Note that the last
    // key point, where the rightmost building ends, is merely used to mark the
    // termination of the skyline, and always has zero height. Also, the ground in
    // between any two adjacent buildings should be considered part of the skyline
    // contour.
    // For instance, the skyline in Figure B should be represented as:[ [2 10], [3
    // 15], [7 12], [12 0], [15 10], [20 8], [24, 0] ].
    // Notes:
    // The number of buildings in any input list is guaranteed to be in the range
    // [0, 10000].
    // The input list is already sorted in ascending order by the left x position
    // Li.
    // The output list must be sorted by the x position.
    // There must be no consecutive horizontal lines of equal height in the output
    // skyline. For instance, [...[2 3], [4 5], [7 5], [11 5], [12 7]...] is not
    // acceptable; the three lines of height 5 should be merged into one in the
    // final output as such: [...[2 3], [4 5], [12 7], ...]
    public List<int[]> getSkyline(int[][] buildings) {
        return divideAndconquer(buildings, 0, buildings.length - 1);
    }

    private List<int[]> divideAndconquer(int[][] buildings, int start, int end) {
        List<int[]> res = new LinkedList<>();

        if (start > end) {
            return res;
        } else if (start == end) {
            res.add(new int[] { buildings[start][0], buildings[start][2] });
            res.add(new int[] { buildings[start][1], 0 });
            return res;
        }

        int mid = start + (end - start) / 2;
        List<int[]> left = divideAndconquer(buildings, start, mid);
        List<int[]> right = divideAndconquer(buildings, mid + 1, end);

        int leftH = 0, rightH = 0;
        while (!left.isEmpty() || !right.isEmpty()) {
            long leftx = left.isEmpty() ? Long.MAX_VALUE : left.get(0)[0];
            long rightx = right.isEmpty() ? Long.MAX_VALUE : right.get(0)[0];
            int x = 0;
            if (leftx < rightx) {
                int[] temp = left.get(0);
                left.remove(0);
                x = temp[0];
                leftH = temp[1];
            } else if (leftx > rightx) {
                int[] temp = right.get(0);
                right.remove(0);
                x = temp[0];
                rightH = temp[1];
            } else {
                x = left.get(0)[0];
                leftH = left.get(0)[1];
                left.remove(0);
                rightH = right.get(0)[1];
                right.remove(0);
            }
            int h = Math.max(leftH, rightH);
            if (res.isEmpty() || h != res.get(res.size() - 1)[1]) {
                res.add(new int[] { x, h });
            }
        }
        return res;
    }

    // 217
    // Given an array of integers, find if the array contains any duplicates. Your
    // function should return true if any value appears at least twice in the array,
    // and it should return false if every element is distinct.
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> intSet = new HashSet<Integer>();
        for (int n : nums) {
            if (!intSet.add(n)) {
                return true;
            }
        }
        return false;
    }

    // 216
    // Find all possible combinations of k numbers that add up to a number n, given
    // that only numbers from 1 to 9 can be used and each combination should be a
    // unique set of numbers.
    // Example 1:
    // Input: k = 3, n = 7
    // Output: [[1,2,4]]
    // Example 2:
    // Input: k = 3, n = 9
    // Output: [[1,2,6], [1,3,5], [2,3,4]]
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new LinkedList<>();
        List<Integer> sumList = new ArrayList<>();
        for (int i = 1; i < 10; i++) {
            sumList.add(i);
            combiSum3Helper(res, sumList, k, n, i + 1, i);
            sumList.remove(sumList.size() - 1);
        }
        return res;
    }

    private void combiSum3Helper(List<List<Integer>> res, List<Integer> sumList, int k, int n, int cur, int sum) {
        if (sum > n || sumList.size() > k)
            return;
        if (sumList.size() == k) {
            if (sum == n)
                res.add(new LinkedList<>(sumList));
            return;
        }
        for (int i = cur; i < 10; i++) {
            sumList.add(i);
            combiSum3Helper(res, sumList, k, n, i + 1, sum + i);
            sumList.remove(sumList.size() - 1);
        }
    }

    // 215
    // Find the kth largest element in an unsorted array. Note that it is the kth
    // largest element in the sorted order, not the kth distinct element.
    // For example,
    // Given [3,2,1,5,6,4] and k = 2, return 5.
    // Note:
    // You may assume k is always valid, 1 ≤ k ≤ array's length.
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        int size = k;
        for (int n : nums) {
            pq.add(n);
            if (pq.size() > size) {
                pq.poll();
            }
        }
        return pq.peek();
    }

    // 214
    // Given a string S, you are allowed to convert it to a palindrome by adding
    // characters in front of it. Find and return the shortest palindrome you can
    // find by performing this transformation.
    // For example:
    // Given "aacecaaa", return "aaacecaaa".
    // Given "abcd", return "dcbabcd".
    public String shortestPalindrome(String s) {
        int mid = s.length() / 2;
        boolean isCenter = (s.length() % 2 != 0);
        StringBuilder sb = new StringBuilder(s);
        while (mid > 0) {
            if (isPalinm(sb, mid, isCenter)) {
                break;
            }
            if (isCenter) {
                isCenter = false;
            } else {
                mid -= 1;
                isCenter = true;
            }
        }
        return buildStr(sb, mid, isCenter);
    }

    private String buildStr(StringBuilder sb, int mid, boolean isCenter) {
        int j = mid;
        if (isCenter) {
            j = j + j + 1;
        } else {
            j = j + j;
        }
        while (j < sb.length()) {
            sb.insert(0, sb.charAt(j));
            j += 2;
        }
        return sb.toString();
    }

    private boolean isPalinm(StringBuilder sb, int mid, boolean isCenter) {
        int i, j;
        if (isCenter) {
            i = mid - 1;
            j = mid + 1;
        } else {
            i = mid - 1;
            j = mid;
        }
        while (i >= 0) {
            if (sb.charAt(i) == sb.charAt(j)) {
                i -= 1;
                j += 1;
            } else {
                return false;
            }
        }
        return true;
    }

    // 213
    // You are a professional robber planning to rob houses along a street. Each
    // house has a certain amount of money stashed, the only constraint stopping you
    // from robbing each of them is that adjacent houses have security system
    // connected and it will automatically contact the police if two adjacent houses
    // were broken into on the same night.
    // Given a list of non-negative integers representing the amount of money of
    // each house, determine the maximum amount of money you can rob tonight without
    // alerting the police.
    // After robbing those houses on that street, the thief has found himself a new
    // place for his thievery so that he will not get too much attention. This time,
    // all houses at this place are arranged in a circle. That means the first house
    // is the neighbor of the last one. Meanwhile, the security system for these
    // houses remain the same as for those in the previous street.
    // Given a list of non-negative integers representing the amount of money of
    // each house, determine the maximum amount of money you can rob tonight without
    // alerting the police.
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 0)
            return 0;
        int[] dp = new int[n + 1];
        dp[1] = nums[0];
        for (int i = 1; i < n; i++) {
            if (i == n - 1)
                dp[i + 1] = dp[i];
            else
                dp[i + 1] = Math.max(dp[i], dp[i - 1] + nums[i]);
        }
        int temp = dp[n];
        dp = new int[n + 1];
        for (int i = 1; i < n; i++) {
            dp[i + 1] = Math.max(dp[i], dp[i - 1] + nums[i]);
        }
        return Math.max(temp, dp[n]);
    }

    // 212
    // Given a 2D board and a list of words from the dictionary, find all words in
    // the board.
    // Each word must be constructed from letters of sequentially adjacent cell,
    // where "adjacent" cells are those horizontally or vertically neighboring. The
    // same letter cell may not be used more than once in a word.
    // For example,
    // Given words = ["oath","pea","eat","rain"] and board =
    // [
    // ['o','a','a','n'],
    // ['e','t','a','e'],
    // ['i','h','k','r'],
    // ['i','f','l','v']
    // ]
    // Return ["eat","oath"].
    // Note:
    // You may assume that all inputs are consist of lowercase letters a-z.
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new LinkedList<>();
        int m = board.length, n = board[0].length;
        Trie root = new Trie();
        for (String s : words) {
            root.insert(s);
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dfs(i, j, res, root, board, sb);
            }
        }
        return res;
    }

    private void dfs(int x, int y, List<String> res, Trie root, char[][] board, StringBuilder sb) {
        if (board[x][y] == '*')
            return;

        char c = board[x][y];
        if (root.nextLevel[c - 'a'] == null)
            return;
        Trie next = root.nextLevel[c - 'a'];
        board[x][y] = '*';
        sb.append(c);
        if (next.hasWord) {
            res.add(sb.toString());
            next.hasWord = false;
        }

        if (x > 0)
            dfs(x - 1, y, res, next, board, sb);
        if (y > 0)
            dfs(x, y - 1, res, next, board, sb);
        if (x < board.length - 1)
            dfs(x + 1, y, res, next, board, sb);
        if (y < board[0].length - 1)
            dfs(x, y + 1, res, next, board, sb);

        sb.deleteCharAt(sb.length() - 1);
        board[x][y] = c;
        return;
    }

    // 211
    // Design a data structure that supports the following two operations:
    // void addWord(word)
    // bool search(word)
    // search(word) can search a literal word or a regular expression string
    // containing only letters a-z or .. A . means it can represent any one letter.
    // For example:
    // addWord("bad")
    // addWord("dad")
    // addWord("mad")
    // search("pad") -> false
    // search("bad") -> true
    // search(".ad") -> true
    // search("b..") -> true
    // You may assume that all words are consist of lowercase letters a-z.

    // 210
    // There are a total of n courses you have to take, labeled from 0 to n - 1.
    // Some courses may have prerequisites, for example to take course 0 you have to
    // first take course 1, which is expressed as a pair: [0,1]
    // Given the total number of courses and a list of prerequisite pairs, return
    // the ordering of courses you should take to finish all courses.
    // There may be multiple correct orders, you just need to return one of them. If
    // it is impossible to finish all courses, return an empty array.
    // For example:
    // 2, [[1,0]]
    // There are a total of 2 courses to take. To take course 1 you should have
    // finished course 0. So the correct course order is [0,1]
    // 4, [[1,0],[2,0],[3,1],[3,2]]
    // There are a total of 4 courses to take. To take course 3 you should have
    // finished both courses 1 and 2. Both courses 1 and 2 should be taken after you
    // finished course 0. So one correct course order is [0,1,2,3]. Another correct
    // ordering is[0,2,1,3].
    // Note:
    // The input prerequisites is a graph represented by a list of edges, not
    // adjacency matrices. Read more about how a graph is represented.
    // You may assume that there are no duplicate edges in the input prerequisites.
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] res = new int[numCourses];

        Set<Integer> removed = new HashSet<>();
        boolean[] visited = new boolean[numCourses];
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++)
            graph.add(new ArrayList<>());
        for (int i = 0; i < prerequisites.length; i++)
            graph.get(prerequisites[i][0]).add(prerequisites[i][1]);

        for (int i = 0; i < numCourses; i++) {
            if (removed.contains(i))
                continue;
            if (!findOrderHelper(i, graph, removed, visited, res)) {
                return new int[0];
            }
        }
        return res;
    }

    private boolean findOrderHelper(int c, List<List<Integer>> graph, Set<Integer> removed, boolean[] visited,
            int[] res) {
        if (removed.contains(c))
            return true;
        if (visited[c])
            return false;

        visited[c] = true;
        for (int u : graph.get(c)) {
            if (removed.contains(u))
                continue;
            if (!findOrderHelper(u, graph, removed, visited, res)) {
                return false;
            }
        }
        res[removed.size()] = c;
        removed.add(c);
        return true;
    }

    // 209
    // Given an array of n positive integers and a positive integer s, find the
    // minimal length of a contiguous subarray of which the sum ≥ s. If there isn't
    // one, return 0 instead.
    // For example, given the array [2,3,1,2,4,3] and s = 7,
    // the subarray [4,3] has the minimal length under the problem constraint.
    // More practice: 1304
    // If you have figured out the O(n) solution, try coding another solution of
    // which the time complexity is O(n log n).
    public int minSubArrayLen(int s, int[] nums) {
        int n = nums.length;
        int start = 0, end = 1;
        int curSum = nums[0];
        int minLen = Integer.MAX_VALUE;
        while (end <= n) {
            if (curSum == s) {
                minLen = Math.min(minLen, end - start);
                end += 1;
            } else if (curSum < s) {
                curSum += nums[end];
                end += 1;
            } else {
                curSum -= nums[start];
                start += 1;
            }
        }
        if (minLen == Integer.MAX_VALUE)
            return 0;
        return minLen;
    }

    // 208
    // Implement a trie with insert, search, and startsWith methods.
    // Note:
    // You may assume that all inputs are consist of lowercase letters a-z in java
    // file

    // 207
    // There are a total of n courses you have to take, labeled from 0 to n - 1.
    // Some courses may have prerequisites, for example to take course 0 you have to
    // first take course 1, which is expressed as a pair: [0,1]
    // Given the total number of courses and a list of prerequisite pairs, is it
    // possible for you to finish all courses?
    // For example:
    // 2, [[1,0]]
    // There are a total of 2 courses to take. To take course 1 you should have
    // finished course 0. So it is possible.
    // 2, [[1,0],[0,1]]
    // There are a total of 2 courses to take. To take course 1 you should have
    // finished course 0, and to take course 0 you should also have finished course
    // 1. So it is impossible.
    // Note:
    // The input prerequisites is a graph represented by a list of edges, not
    // adjacency matrices. Read more about how a graph is represented.
    // You may assume that there are no duplicate edges in the input prerequisites.
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Set<Integer> removed = new HashSet<>();
        boolean[] visited = new boolean[numCourses];
        // convert to graph
        boolean[][] graph = new boolean[numCourses][numCourses];
        for (int[] arr : prerequisites) {
            graph[arr[0]][arr[1]] = true;
        }
        for (int i = 0; i < numCourses; i++) {
            if (!removed.contains(i)) {
                if (!canFinishHelper(i, graph, removed, visited)) {
                    return false;
                }
            }
        }
        return true;
    }

    private boolean canFinishHelper(int c, boolean[][] graph, Set<Integer> removed, boolean[] visited) {
        visited[c] = true;
        int n = graph.length;
        for (int i = 0; i < n; i++) {
            if (!graph[c][i]) {
                continue;
            }
            if (visited[i]) {
                return false;
            } else {
                if (!canFinishHelper(i, graph, removed, visited)) {
                    return false;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            graph[i][c] = false;
        }
        removed.add(c);
        return true;
    }

    // 206
    // Reverse a singly linked list.
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode tail = head.next;
        ListNode newhead = reverseList(head.next);
        tail.next = head;
        head.next = null;
        return newhead;
    }

    public ListNode reverseListIte(ListNode head) {
        if (head == null)
            return head;
        Stack<ListNode> st = new Stack<>();

        while (head != null) {
            st.push(head);
            head = head.next;
        }

        head = st.pop();
        ListNode cur = head;
        while (!st.isEmpty()) {
            cur.next = st.pop();
            cur = cur.next;
            cur.next = null;
        }
        return head;
    }

    // 205
    // Given two strings s and t, determine if they are isomorphic.
    // Two strings are isomorphic if the characters in s can be replaced to get t.
    // All occurrences of a character must be replaced with another character while
    // preserving the order of characters. No two characters may map to the same
    // character but a character may map to itself.
    // For example,
    // Given "egg", "add", return true.
    // Given "foo", "bar", return false.
    // Given "paper", "title", return true.
    // Note:
    // You may assume both s and t have the same length.
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> myMap = new HashMap<>();
        Set<Character> mySet = new HashSet<>();
        int n = s.length();

        for (int i = 0; i < n; i++) {
            char sc = s.charAt(i);
            char tc = t.charAt(i);
            if (myMap.containsKey(sc)) {
                if (myMap.get(sc) != tc) {
                    return false;
                }
            } else {
                if (mySet.contains(tc)) {
                    return false;
                }
                mySet.add(tc);
                myMap.put(sc, tc);
            }
        }
        return true;
    }

    // 204
    // Description:
    // Count the number of prime numbers less than a non-negative number, n.
    public int countPrimes(int n) {
        boolean[] notPrime = new boolean[n];
        int curMax = n - 1;
        int res = 0;
        for (int i = 2; i <= curMax; i++) {
            if (notPrime[i]) {
                continue;
            }
            res++;
            int j = 2;
            while (j * i <= curMax) {
                notPrime[j * i] = true;
                curMax = Math.max(curMax, i * j);
                j++;
            }
        }
        return res;
    }

    // 203
    // Remove all elements from a linked list of integers that have value val.
    // Example
    // Given: 1 --> 2 --> 6 --> 3 --> 4 --> 5 --> 6, val = 6
    // Return: 1 --> 2 --> 3 --> 4 --> 5
    public ListNode removeElements(ListNode head, int val) {
        while (head != null && head.val == val) {
            head = head.next;
        }

        ListNode node = head;
        ListNode pre = head;
        while (node != null) {
            if (node.val != val) {
                pre = node;
                node = node.next;
            } else {
                pre.next = node.next;
                node = node.next;
            }
        }
        return head;
    }

    // 202
    // Write an algorithm to determine if a number is "happy".
    // A happy number is a number defined by the following process: Starting with
    // any positive integer, replace the number by the sum of the squares of its
    // digits, and repeat the process until the number equals 1 (where it will
    // stay), or it loops endlessly in a cycle which does not include 1. Those
    // numbers for which this process ends in 1 are happy numbers.
    // Example: 19 is a happy number
    // 12 + 92 = 82
    // 82 + 22 = 68
    // 62 + 82 = 100
    // 12 + 02 + 02 = 1
    public boolean isHappy(int n) {
        Set<Integer> mySet = new HashSet<>();
        while (!mySet.contains(n)) {
            if (n == 1) {
                return true;
            }
            mySet.add(n);
            int temp = 0;
            while (n != 0) {
                int remainder = n % 10;
                n = n / 10;
                temp += remainder * remainder;
            }
            n = temp;
        }
        return false;
    }

    // 201
    // Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND
    // of all numbers in this range, inclusive.
    // For example, given the range [5, 7], you should return 4.
    public int rangeBitwiseAnd(int m, int n) {
        int i = 0;
        while (m != n) {
            m >>= 1;
            n >>= 1;
            i++;
        }
        return m << i;
    }
}
