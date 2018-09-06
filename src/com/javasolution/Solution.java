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
                return (new int[]{map.get(nums[i]), i});
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

    // 3 https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
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
    // TODO: Review
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        return (getKthNum(nums1, 0, nums2, 0, (m + n + 1) / 2) + getKthNum(nums1, 0, nums2, 0, (m + n + 2) / 2)) / 2.0;
    }

    public double getKthNum(int[] nums1, int s1, int[] nums2, int s2, int k) {
        if (s1 >= nums1.length) return nums2[s2 + k - 1];
        if (s2 >= nums2.length) return nums1[s1 + k - 1];
        if (k == 1) return Math.min(nums1[s1], nums2[s2]);

        int mid1 = Integer.MAX_VALUE, mid2 = Integer.MAX_VALUE;
        if (s1 + k / 2 - 1 < nums1.length) mid1 = nums1[s1 + k / 2 - 1];
        if (s2 + k / 2 - 1 < nums2.length) mid2 = nums2[s2 + k / 2 - 1];
        if (mid1 < mid2) {
            return getKthNum(nums1, s1 + k / 2, nums2, s2, k - k / 2);
        } else {
            return getKthNum(nums1, s1, nums2, s2 + k / 2, k - k / 2);
        }
    }

    // 5 https://leetcode.com/problems/longest-palindromic-substring/description/
    public String longestPalindromeOn2(String s) {
        int len = s.length();
        if (len == 0) return "";
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
        if (numRows == 1) return s;
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
        int index = 0, sign = 1, total = 0, prev = 0;

        if (str.length() == 0) return 0; // empty
        while (index < str.length() && str.charAt(index) == ' ') // remove white spaces
            index++;
        if (index < str.length() && (str.charAt(index) == '+' || str.charAt(index) == '-')) { // handle sign
            sign = str.charAt(index) == '+' ? 1 : -1;
            index++;
        }

        while (index < str.length()) {
            int digit = str.charAt(index) - '0';
            if (digit < 0 || digit > 9) break;
            if (Integer.MAX_VALUE / 10 < total || Integer.MAX_VALUE / 10 == total && Integer.MAX_VALUE % 10 < digit)
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            total = total * 10 + digit;
            index++;
        }
        return total * sign;
    }

    // 9 https://leetcode.com/problems/palindrome-number/description/
    public boolean isPalindrome(int x) {
        return false;
    }

    // 10 https://leetcode.com/problems/regular-expression-matching/description/
    public boolean isMatch(String s, String p) {
        return false;
    }

    // 11 https://leetcode.com/problems/container-with-most-water/description/
    public int maxArea(int[] height) {
        return 0;
    }

    // 12 https://leetcode.com/problems/integer-to-roman/description/
    public String intToRoman(int num) {
        return "";
    }
    // 13 https://leetcode.com/problems/roman-to-integer/description/
    public int romanToInt(String s) {
        return 0;
    }

    // 14 https://leetcode.com/problems/longest-common-prefix/description/
    public String longestCommonPrefix(String[] strs) {
        return false;
    }

    // 15 https://leetcode.com/problems/3sum/description/
    public List<List<Integer>> threeSum(int[] nums) {
        return false;
    }

}
