package com.javasolution;

import java.util.*;
import java.lang.*;

import com.javasolution.util.*;
import com.javasolution.structdesign.*;

public class Solution {
    // 1 https://leetcode.com/problems/two-sum/description/
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                return (new int[] { map.get(nums[i]), i });
            } else {
                map.put(target - nums[i], i);
            }
        }
        return null;
    }

    // 2 https://leetcode.com/problems/add-two-numbers/description/
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        ListNode head = new ListNode(0);
        ListNode cur = head;

        while (l1 != null || l2 != null || carry != 0) {
            if (l1 != null) {
                carry += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                carry += l2.val;
                l2 = l2.next;
            }
            cur.next = new ListNode(carry % 10);
            cur = cur.next;
            carry /= 10;
        }
        return head.next;
    }

    // 3
    // https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        int head = 0;
        int tail = 0;
        int res = 0;

        while (head < s.length()) {
            if (map.containsKey(s.charAt(head)) && tail <= map.get(s.charAt(head))) {
                tail = map.get(s.charAt(head)) + 1;
            } else {
                map.put(s.charAt(head), head);
                head++;
                res = Math.max(res, head - tail);
            }
        }
        return res;
    }

    // 4 https://leetcode.com/problems/median-of-two-sorted-arrays/description/
    // TODO: Review this problem
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        return (getKthNum(nums1, 0, nums2, 0, (m + n + 1) / 2) + getKthNum(nums1, 0, nums2, 0, (m + n + 2) / 2)) / 2.0;
    }

    public double getKthNum(int[] nums1, int s1, int[] nums2, int s2, int k) {
        if (s1 >= nums1.length)
            return nums2[s2 + k - 1];
        if (s2 >= nums2.length)
            return nums1[s1 + k - 1];
        if (k == 1)
            return Math.min(nums1[s1], nums2[s2]);

        int mid1 = Integer.MAX_VALUE, mid2 = Integer.MAX_VALUE;
        if (s1 + k / 2 - 1 < nums1.length)
            mid1 = nums1[s1 + k / 2 - 1];
        if (s2 + k / 2 - 1 < nums2.length)
            mid2 = nums2[s2 + k / 2 - 1];
        if (mid1 < mid2) {
            return getKthNum(nums1, s1 + k / 2, nums2, s2, k - k / 2);
        } else {
            return getKthNum(nums1, s1, nums2, s2 + k / 2, k - k / 2);
        }
    }

    // 5 https://leetcode.com/problems/longest-palindromic-substring/description/
    public String longestPalindromeOn2(String s) {
        int len = s.length();
        if (len == 0)
            return "";
        boolean[][] dp = new boolean[len][len];
        int maxLen = 1;
        int head = 1, tail = 0;

        for (int i = 0; i < len; i++) {
            for (int j = 0; j + i < len; j++) {
                if (s.charAt(j) == s.charAt(j + i) && (i < 2 || dp[j + 1][j + i - 1])) {
                    dp[j][j + i] = true;
                }
                if (dp[j][j + i] && maxLen < i + 1) {
                    maxLen = i + 1;
                    head = j + i + 1;
                    tail = j;
                }
            }
        }
        return s.substring(tail, head);
    }

    public String longestPalindrome(String s) {
        String str = preProcess(s);
        int[] P = new int[str.length()];
        int center = 0, rIndex = 0;

        for (int i = 1; i < str.length() - 1; i++) {
            int i_ = 2 * center - i;
            P[i] = (rIndex > i) ? Math.min(rIndex - i, P[i_]) : 0;

            while (str.charAt(i + 1 + P[i]) == str.charAt(i - 1 - P[i])) {
                P[i]++;
            }
            if (i + P[i] > rIndex) {
                center = i;
                rIndex = i + P[i];
            }
        }
        int maxLen = 0;
        for (int i = 1; i < str.length() - 1; i++) {
            if (P[i] > maxLen) {
                maxLen = P[i];
                center = i;
            }
        }
        int start = (center - 1 - maxLen) / 2;
        return s.substring(start, start + maxLen);
    }

    private String preProcess(String s) {
        StringBuilder sb = new StringBuilder();
        sb.append("^#");
        for (int i = 0; i < s.length(); i++) {
            sb.append(s.charAt(i));
            sb.append('#');
        }
        sb.append('&');
        return sb.toString();
    }

    // 6 https://leetcode.com/problems/zigzag-conversion/description/
    public String convert(String s, int numRows) {
        if (numRows == 1)
            return s;
        int delta = 2 * numRows - 2;
        StringBuilder sb = new StringBuilder();

        for (int row = 0; row < numRows; row++) {
            int index = row;
            while (index < s.length()) {
                sb.append(s.charAt(index));
                index += delta;
                if (index - 2 * row < s.length() && row != 0 && row != numRows - 1) {
                    sb.append(s.charAt(index - 2 * row));
                }
            }
        }
        return sb.toString();
    }

    // 7 https://leetcode.com/problems/reverse-integer/description/
    public int reverse(int x) {
        int sign = x < 0 ? -1 : 1;
        long x_ = Math.abs((long) x);
        long res = 0;

        while (x_ != 0) {
            res = res * 10 + x_ % 10;
            x_ = x_ / 10;
        }
        res = res * sign;
        if (res < Integer.MAX_VALUE && res > Integer.MIN_VALUE) {
            return (int) res;
        } else {
            return 0;
        }
    }

    // 8 https://leetcode.com/problems/string-to-integer-atoi/description/
    public int myAtoi(String str) {
        int index = 0, sign = 1, total = 0;

        if (str.length() == 0)
            return 0; // empty
        while (index < str.length() && str.charAt(index) == ' ') // remove white spaces
            index++;
        if (index < str.length() && (str.charAt(index) == '+' || str.charAt(index) == '-')) { // handle sign
            sign = str.charAt(index) == '+' ? 1 : -1;
            index++;
        }

        while (index < str.length()) {
            int digit = str.charAt(index) - '0';
            if (digit < 0 || digit > 9)
                break;
            if (Integer.MAX_VALUE / 10 < total || Integer.MAX_VALUE / 10 == total && Integer.MAX_VALUE % 10 < digit)
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            total = total * 10 + digit;
            index++;
        }
        return total * sign;
    }

    // 9 https://leetcode.com/problems/palindrome-number/description/
    public boolean isPalindrome(int x) {
        if (x < 0)
            return false;
        int resx = 0;
        int orix = x;
        while (x != 0) {
            resx = resx * 10 + x % 10;
            x = x / 10;
        }
        return orix == resx;
    }

    // 10 https://leetcode.com/problems/regular-expression-matching/description/
    // TODO: String matching! Dynamic programing
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= p.length(); i++) {
            if (p.charAt(i - 1) == '*' && dp[0][i - 2]) {
                dp[0][i] = true;
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                char schar = s.charAt(i - 1);
                char pchar = p.charAt(j - 1);

                if (pchar == '*' && j > 1) {
                    dp[i][j] = dp[i][j - 2] || dp[i][j - 1];
                    if (p.charAt(j - 2) == s.charAt(i - 1) || p.charAt(j - 2) == '.') {
                        dp[i][j] |= dp[i - 1][j];
                    }
                } else {
                    dp[i][j] = (pchar == '.' || schar == pchar) && dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    // 11 https://leetcode.com/problems/container-with-most-water/description/
    public int maxArea(int[] height) {
        int res = 0;
        int i = 0, j = height.length - 1;
        while (i < j) {
            if (height[i] < height[j]) {
                res = Math.max(res, height[i] * (j - i));
                i++;
            } else {
                res = Math.max(res, height[j] * (j - i));
                j--;
            }
        }
        return res;
    }

    // 12 https://leetcode.com/problems/integer-to-roman/description/
    public String intToRoman(int num) {
        String M[] = { "", "M", "MM", "MMM" };
        String C[] = { "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" };
        String X[] = { "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" };
        String I[] = { "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" };
        return M[num / 1000] + C[(num % 1000) / 100] + X[(num % 100) / 10] + I[num % 10];
    }

    // 13 https://leetcode.com/problems/roman-to-integer/description/
    public int romanToInt(String s) {
        int res = 0;
        if (s.indexOf("IV") != -1)
            res -= 2;
        if (s.indexOf("IX") != -1)
            res -= 2;
        if (s.indexOf("XL") != -1)
            res -= 20;
        if (s.indexOf("XC") != -1)
            res -= 20;
        if (s.indexOf("CD") != -1)
            res -= 200;
        if (s.indexOf("CM") != -1)
            res -= 200;

        for (int count = 0; count < s.length(); count++) {
            char c = s.charAt(count);
            if (c == 'M')
                res += 1000;
            if (c == 'D')
                res += 500;
            if (c == 'C')
                res += 100;
            if (c == 'L')
                res += 50;
            if (c == 'X')
                res += 10;
            if (c == 'V')
                res += 5;
            if (c == 'I')
                res += 1;
        }
        return res;
    }

    // 14 https://leetcode.com/problems/longest-common-prefix/description/
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0)
            return "";
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < strs[0].length(); i++) {
            char c = strs[0].charAt(i);
            for (String str : strs) {
                if (i >= str.length() || c != str.charAt(i)) {
                    return sb.toString();
                }
            }
            sb.append(c);
        }
        return sb.toString();
    }

    // 15 https://leetcode.com/problems/3sum/description/
    // TODO: how to eliminate duplication!
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new LinkedList<>();
        Arrays.sort(nums);
        for (int i = 0; i + 2 < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) { // skip same result
                continue;
            }
            int j = i + 1, k = nums.length - 1;
            int target = -nums[i];
            while (j < k) {
                if (nums[j] + nums[k] == target) {
                    res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    j++;
                    k--;
                    while (j < k && nums[j] == nums[j - 1])
                        j++; // skip same result
                    while (j < k && nums[k] == nums[k + 1])
                        k--; // skip same result
                } else if (nums[j] + nums[k] > target) {
                    k--;
                } else {
                    j++;
                }
            }
        }
        return res;
    }

    // 16 https://leetcode.com/problems/3sum-closest/description/
    public int threeSumClosest(int[] nums, int target) {
        int result = nums[0] + nums[1] + nums[nums.length - 1];
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            int start = i + 1, end = nums.length - 1;
            while (start < end) {
                int sum = nums[i] + nums[start] + nums[end];
                if (sum > target) {
                    end--;
                } else {
                    start++;
                }
                if (Math.abs(sum - target) < Math.abs(result - target)) {
                    result = sum;
                }
            }
        }
        return result;
    }

    // 17
    // https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
    public List<String> letterCombinations(String digits) {
        List<String> res = new LinkedList<>();
        Map<Character, List<Character>> map = new HashMap<>();
        StringBuilder sb = new StringBuilder();

        map.put('2', Arrays.asList('a', 'b', 'c'));
        map.put('3', Arrays.asList('d', 'e', 'f'));
        map.put('4', Arrays.asList('g', 'h', 'i'));
        map.put('5', Arrays.asList('j', 'k', 'l'));
        map.put('6', Arrays.asList('m', 'n', 'o'));
        map.put('7', Arrays.asList('p', 'q', 'r', 's'));
        map.put('8', Arrays.asList('t', 'u', 'v'));
        map.put('9', Arrays.asList('w', 'x', 'y', 'z'));
        letterCombinations(digits, 0, map, sb, res);
        return res;
    }

    private void letterCombinations(String digits, int index, Map<Character, List<Character>> map, StringBuilder sb,
            List<String> res) {
        if (index == digits.length()) {
            res.add(sb.toString());
            return;
        }

        for (char c : map.get(digits.charAt(index))) {
            sb.append(c);
            letterCombinations(digits, index + 1, map, sb, res);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    // 18 https://leetcode.com/problems/4sum/description/
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> res = new LinkedList<>();
        for (int i = 0; i < nums.length - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            fourSumHelper(nums, target - nums[i], i + 1, res, nums[i]);
        }
        return res;
    }

    private void fourSumHelper(int[] nums, int t, int index, List<List<Integer>> res, int cur) {
        for (int i = index; i < nums.length - 2; i++) {
            if (i > index && nums[i] == nums[i - 1]) {
                continue;
            }
            int target = t - nums[i];
            int start = i + 1;
            int end = nums.length - 1;

            while (start < end) {
                int sum = nums[start] + nums[end];
                if (sum == target) {
                    res.add(Arrays.asList(cur, nums[i], nums[start], nums[end]));
                    start++;
                    end--;
                    while (start < end && nums[start] == nums[start - 1]) {
                        start++;
                    }
                    while (start < end && nums[end] == nums[end + 1]) {
                        end--;
                    }
                } else if (sum < target) {
                    start++;
                } else {
                    end--;
                }
            }
        }
    }

    // 19
    // https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode h1 = head, h2 = head;
        while (n-- > 0)
            h2 = h2.next;
        if (h2 == null)
            return head.next; // The head need to be removed, do it.
        h2 = h2.next;

        while (h2 != null) {
            h1 = h1.next;
            h2 = h2.next;
        }
        h1.next = h1.next.next; // the one after the h1 need to be removed
        return head;
    }

    // 20 https://leetcode.com/problems/valid-parentheses/description/
    public boolean isValid(String s) {
        Stack<Character> st = new Stack<>();
        for (char c : s.toCharArray()) {
            if (st.isEmpty()) {
                st.push(c);
            } else if (c == ')' && st.peek() == '(') {
                st.pop();
            } else if (c == '}' && st.peek() == '{') {
                st.pop();
            } else if (c == ']' && st.peek() == '[') {
                st.pop();
            } else {
                st.push(c);
            }
        }
        return st.isEmpty();
    }

    // 21 https://leetcode.com/problems/merge-two-sorted-lists/description/
    // Node: Recursion solution is more elegant
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
        // Iteration
        // ListNode dummy = new ListNode(0);
        // ListNode cur = dummy;
        // while (l1 != null && l2 != null) {
        // if (l1.val < l2.val) {
        // cur.next = l1;
        // cur = l1;
        // l1 = l1.next;
        // } else {
        // cur.next = l2;
        // cur = l2;
        // l2 = l2.next;
        // }
        // }
        // if (l1 != null) {
        // cur.next = l1;
        // } else {
        // cur.next = l2;
        // }
        // return dummy.next;
    }

    // 22 https://leetcode.com/problems/generate-parentheses/description/
    public List<String> generateParenthesis(int n) {
        if (n == 0)
            return Arrays.asList("");
        if (n == 1)
            return Arrays.asList("()");

        StringBuilder sb = new StringBuilder();
        List<String> res = new LinkedList<>();
        backtracing22(res, sb, n, 0, n);
        return res;
    }

    private void backtracing22(List<String> res, StringBuilder sb, int right, int left, int n) {
        if (sb.length() == 2 * n) {
            res.add(sb.toString());
            return;
        }
        if (right > 0) {
            sb.append("(");
            right--;
            left++;
            backtracing22(res, sb, right, left, n);
            sb.deleteCharAt(sb.length() - 1);
            right++;
            left--;
        }
        if (left > 0) {
            sb.append(")");
            left--;
            backtracing22(res, sb, right, left, n);
            sb.deleteCharAt(sb.length() - 1);
            left++;
        }
    }

    // 23 https://leetcode.com/problems/merge-k-sorted-lists/description/
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });
        for (ListNode l : lists) {
            while (l != null) {
                pq.add(l);
                l = l.next;
            }
        }
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (!pq.isEmpty()) {
            cur.next = pq.poll();
            cur = cur.next;
        }
        cur.next = null;
        return dummy.next;
    }
    // private ListNode mergeKListsHelper(ListNode[] lists) {
    // int minV = Integer.MAX_VALUE;
    // int index = 0;
    // for (int i = 0; i < lists.length; i++) {
    // if (lists[i] != null && lists[i].val < minV) {
    // index = i;
    // minV = lists[i].val;
    // }
    // }
    // if (minV == Integer.MAX_VALUE) {
    // return null;
    // } else {
    // ListNode l = lists[index];
    // lists[index] = lists[index].next;
    // l.next = mergeKListsHelper(lists);
    // return l;
    // }
    // }

    // 24 https://leetcode.com/problems/swap-nodes-in-pairs/description/
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode pre = dummy;
        ListNode cur = head;
        ListNode next = cur.next;

        while (cur != null && next != null) {
            pre.next = next;
            cur.next = next.next;
            next.next = cur;

            pre = cur;
            cur = cur.next;
            if (cur == null)
                break;
            next = cur.next;
        }
        return dummy.next;
    }

    // 25 https://leetcode.com/problems/reverse-nodes-in-k-group/description/
    // Note: Don't use stack !!!
    public ListNode reverseKGroup(ListNode head, int k) {
        Stack<ListNode> st = new Stack<>();
        int i = 0;
        ListNode cur = head;

        while (i < k && cur != null) {
            st.push(cur);
            cur = cur.next;
            i++;
        }
        if (i == k) {
            ListNode next = reverseKGroup(cur, k);
            head = st.pop();
            cur = head;
            while (!st.isEmpty()) {
                cur.next = st.pop();
                cur = cur.next;
            }
            cur.next = next;
        }
        return head;
    }

    // 26
    // https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0)
            return 0;

        int cur = 0;
        int index = cur + 1;
        while (index < nums.length) {
            if (nums[cur] != nums[index]) {
                cur++;
                nums[cur] = nums[index];
            }
            index++;
        }
        return cur + 1;
    }

    // 27 https://leetcode.com/problems/remove-element/description/
    public int removeElement(int[] nums, int val) {
        int i = -1, j = nums.length - 1;
        while (i < j) {
            if (nums[i + 1] != val) {
                i++;
            } else if (nums[j] == val) {
                j--;
            } else {
                nums[i + 1] = nums[j];
                i++;
                j--;
            }
        }
        return i + 1;
    }

    // 28 https://leetcode.com/problems/implement-strstr/description/
    // Note: Next time, use KMP!
    public int strStr(String haystack, String needle) {
        int lenN = needle.length();
        int lenH = haystack.length();

        int i = 0;
        while (i + lenN <= lenH) {
            if (needle.equals(haystack.substring(i, i + lenN)))
                return i;
            i++;
        }
        return -1;
    }

    // 29 https://leetcode.com/problems/divide-two-integers/description/
    public int divide(int dividend, int divisor) {
        int sign = 1;
        if (dividend < 0 && divisor < 0)
            sign = 1;
        else if (dividend < 0 || divisor < 0)
            sign = -1;

        long ldividend = Math.abs((long) dividend);
        long ldivisor = Math.abs((long) divisor);

        long res = divideHelper(ldividend, ldivisor);

        if ((sign == 1 && res > Integer.MAX_VALUE) || (sign == -1 && -res < Integer.MIN_VALUE)) {
            return Integer.MAX_VALUE;
        }

        return (int) res * sign;
    }

    private long divideHelper(long dividend, long divisor) {
        long sum = divisor;
        long res = 1;

        if (dividend < divisor)
            return 0;
        while (sum << 1 < dividend) {
            sum = sum << 1;
            res = 2 * res;
        }
        return res + divideHelper(dividend - sum, divisor);
    }

    // 30
    // https://leetcode.com/problems/substring-with-concatenation-of-all-words/description/
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new LinkedList<>();
        int n = words.length;

        if (n == 0) // 0 words
            return res;

        int m = words[0].length();
        int end = s.length() - m * n;
        if (end < 0) // s too short
            return res;

        Map<String, Integer> map = new HashMap<>();
        for (String w : words) {
            map.put(w, map.getOrDefault(w, 0) + 1);
        }

        for (int i = 0; i < end; i++) {
            Map<String, Integer> seen = new HashMap<>();
            int j = 0;
            while (j < n) {
                String word = s.substring(i + j * m, i + (j + 1) * m);
                if (map.containsKey(word)) {
                    seen.put(word, seen.getOrDefault(word, 0) + 1);
                    if (seen.get(word) > map.getOrDefault(word, 0)) {
                        break;
                    }
                } else {
                    break;
                }
                j++;
            }

            if (j == n) {
                res.add(i);
            }
        }
        return res;
    }

    // 31 https://leetcode.com/problems/next-permutation/description/
    // TODO: ......
    public void nextPermutation(int[] nums) {
        int i = nums.length - 1;
        while (i > 0 && nums[i] <= nums[i - 1]) {
            i--;
        }
        if (i == 0) {
            reverseArr(nums, 0, nums.length);
            return;
        }
        i--;
        int j = nums.length - 1;
        while (j >= i && nums[j] <= nums[i]) {
            j--;
        }
        swap(nums, i, j);
        reverseArr(nums, i + 1, nums.length);
    }

    private void reverseArr(int[] nums, int start, int end) {
        int i = start, j = end - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    // 30 https://leetcode.com/problems/longest-valid-parentheses/description/
    public int longestValidParentheses(String s) {
        return 0;
    }
}
