package com.backup;

import java.util.*;
import java.lang.*;

import com.javasolution.util.*;

public class LCPart1 {

    // 200
    // Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
    // Example 1:
    // 11110
    // 11010
    // 11000
    // 00000
    // Answer: 1
    // Example 2:
    // 11000
    // 11000
    // 00100
    // 00011
    // Answer: 3
    char defaul = (char) 0;
    public int numIslands(char[][] grid) {
        int count = 0;
        int m = grid.length;
        if (m == 0) return count;
        int n = grid[0].length;
        char[][] map = new char[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (map[i][j] == defaul && grid[i][j] == '1') {
                    count++;
                    numIslandsHelper(grid, map, i, j, m, n);
                }
            }
        }
        return count;
    }
    private void numIslandsHelper(char[][] grid, char[][] map, int i, int j, int m, int n) {
        map[i][j] = '1';
        int up = i - 1;
        int down = i + 1;
        int left = j - 1;
        int right  = j + 1;

        if (up >= 0 && grid[up][j] == '1' && map[up][j] == defaul) {
            numIslandsHelper(grid, map, up, j, m, n);
        }
        if (down < m && grid[down][j] == '1' && map[down][j] == defaul) {
            numIslandsHelper(grid, map, down, j, m, n);
        }
        if (left >= 0 && grid[i][left] == '1' && map[i][left] == defaul) {
            numIslandsHelper(grid, map, i, left, m, n);
        }
        if (right < n && grid[i][right] == '1' && map[i][right] == defaul) {
            numIslandsHelper(grid, map, i, right, m, n);
        }
    }

    // 199
    // Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
    // For example:
    // Given the following binary tree,
    //    1            <---
    //  /   \
    // 2     3         <---
    //  \     \
    //   5     4       <---
    // You should return [1, 3, 4].
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        List<TreeNode> list = new LinkedList<>();
        list.add(root);
        while (!list.isEmpty()){
            res.add(list.get(list.size() - 1).val);
            List<TreeNode> temp = new LinkedList<>();
            for (TreeNode n : list) {
                if (n.left != null) temp.add(n.left);
                if (n.right != null) temp.add(n.right);
            }
            list = temp;
        }
        return res;
    }

    // 191
    // Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).
    // For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int res = 0;
        while (n != 0) {
            if ((n & 1) == 1) {
                res += 1;
            }
            n = n >>> 1;
        }
        return res;
    }

    // 190
    // Reverse bits of a given 32 bits unsigned integer.
    // For example, given input 43261596 (represented in binary as 00000010100101000001111010011100), return 964176192 (represented in binary as 00111001011110000010100101000000).
    // Follow up:
    // If this function is called many times, how would you optimize it?
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int res = 0;
        int i = 0;
        while (i < 32) {
            i++;
            res = res << 1;
            if ((n & 1) == 1) {
                res = res | 1;
            }
            n = n >> 1;

        }
        return res;
    }

    // 189
    // Rotate an array of n elements to the right by k steps.
    // For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].
    // Note:
    // Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.
    // [show hint]
    // Hint:
    // Could you do it in-place with O(1) extra space?
    public void rotate(int[] nums, int k) {
        int i = 0, j = nums.length - 1;
        k = k % nums.length;

        while (i < j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            i++;
            j--;
        }

        i = 0; j = k - 1;
        while (i < j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            i++;
            j--;
        }
        i = k; j = nums.length - 1;
        while (i < j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            i++;
            j--;
        }
    }

    // 188
    // Say you have an array for which the ith element is the price of a given stock on day i.
    // Design an algorithm to find the maximum profit. You may complete at most k transactions.
    // Note:
    // You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
    public int maxProfit(int K, int[] prices) {
        int n = prices.length;
        if (K >=  n / 2) {
            int res = 0;
            for (int i = 1; i < n; i++) {
                if (prices[i] > prices[i-1])
                    res += prices[i] - prices[i-1];
            }
            return res;
        }

        int[][] dp = new int[K + 1][n + 1];

        for (int i = 1; i <= K; i++) {
            int curMax = dp[i - 1][1] - prices[0];
            for (int j = 1; j <= n; j++) {
                curMax = Math.max(curMax, dp[i - 1][j] - prices[j - 1]);
                dp[i][j] = Math.max(dp[i][j - 1], prices[j - 1] + curMax);
            }
        }
        return dp[K][n];
    }

    // 187
    // All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T, for example: "ACGAATTCCG". When studying DNA, it is sometimes useful to identify repeated sequences within the DNA.
    // Write a function to find all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule.
    // For example,
    // Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",
    // Return:
    // ["AAAAACCCCC", "CCCCCAAAAA"].
    public List<String> findRepeatedDnaSequences(String s) {
        Set<String> dict = new HashSet<>();
        Set<String> repeated = new HashSet<>();
        int n = s.length();
        for (int i = 0; i <= n - 10; i++) {
            String curS = s.substring(i, i + 10);
            if (!dict.add(curS))
                repeated.add(curS);
        }
        return new LinkedList(repeated);
    }
    // discussion ans
    // public List<String> findRepeatedDnaSequences(String s) {
    //     Set<Integer> words = new HashSet<>();
    //     Set<Integer> doubleWords = new HashSet<>();
    //     List<String> rv = new ArrayList<>();
    //     char[] map = new char[26];
    //     //map['A' - 'A'] = 0;
    //     map['C' - 'A'] = 1;
    //     map['G' - 'A'] = 2;
    //     map['T' - 'A'] = 3;

    //     for(int i = 0; i < s.length() - 9; i++) {
    //         int v = 0;
    //         for(int j = i; j < i + 10; j++) {
    //             v <<= 2;
    //             v |= map[s.charAt(j) - 'A'];
    //         }
    //         if(!words.add(v) && doubleWords.add(v)) {
    //             rv.add(s.substring(i, i + 10));
    //         }
    //     }
    //     return rv;
    // }

    // 174
    // The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess.
    // The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.
    // Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms; other rooms are either empty (0's) or contain magic orbs that increase the knight's health (positive integers).
    // In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.
    // Write a function to determine the knight's minimum initial health so that he is able to rescue the princess.
    // For example, given the dungeon below, the initial health of the knight must be at least 7 if he follows the optimal path RIGHT-> RIGHT -> DOWN -> DOWN.
    // -2 (K)	-3	3
    // -5	-10	1
    // 10	30	-5 (P)
    // Notes:
    // The knight's health has no upper bound.
    // Any room can contain threats or power-ups, even the first room the knight enters and the bottom-right room where the princess is imprisoned.
    public int calculateMinimumHP(int[][] dungeon) {
        int M = dungeon.length;
        int N = dungeon[0].length;
        int[][] dp = new int[M][N];
        int i = M - 1; int j = N - 1;

        dp[i][j] = 1 + (dungeon[i][j] >= 0 ? 0 : Math.abs(dungeon[i][j]));
        for (i = M - 2, j = N - 1; i >= 0; i--) {
            int cur = dp[i + 1][j] - dungeon[i][j];
            dp[i][j] = cur > 0 ? cur : 1;
        }
        for (i = M - 1, j = N - 2; j >= 0; j--) {
            int cur = dp[i][j + 1] - dungeon[i][j];
            dp[i][j] = cur > 0 ? cur : 1;
        }
        for (i = M - 2; i >= 0; i--) {
            for (j = N - 2; j >=0; j--) {
                int minHeal = dp[i + 1][j] - dungeon[i][j];
                minHeal = Math.min(dp[i][j + 1] - dungeon[i][j], minHeal);
                dp[i][j] = minHeal > 0 ? minHeal : 1;
            }
        }
        return dp[0][0];
    }

    // 173
    // Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.
    // Calling next() will return the next smallest number in the BST.
    // Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
    // ********* In BSTIterator.java ********* //
    /**
     * Your BSTIterator will be called like this:
     * BSTIterator i = new BSTIterator(root);
     * while (i.hasNext()) v[f()] = i.next();
     */

    // 172
    // Given an integer n, return the number of trailing zeroes in n!.
    // Note: Your solution should be in logarithmic time complexity.
    public int trailingZeroes(int n) {
        int res = 0;
        int dividor = 5;
        int temp = 2;
        while (temp > 1) {
            temp = n / dividor;
            res += temp;
            dividor *= 5;
        }
        return res;
    }

    // 171
    // Related to question Excel Sheet Column Title
    // Given a column title as appear in an Excel sheet, return its corresponding column number.
    // For example:
    //     A -> 1
    //     B -> 2
    //     C -> 3
    //     ...
    //     Z -> 26
    //     AA -> 27
    //     AB -> 28
    public int titleToNumber(String s) {
        int res = 0;
        int n = s.length();
        for (char c : s.toCharArray()) {
            res += Math.pow(26, --n) * (c - 'A' + 1);
        }
        return res;
    }

    // 169
    // Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
    // You may assume that the array is non-empty and the majority element always exist in the array.
    public int majorityElement(int[] nums) {
        return majorityHelper(nums, 0, nums.length);
    }
    private int majorityHelper(int[] nums, int start, int end) {
        int size = end - start;
        int mid = start + size / 2;
        if (size <= 2) {
            return nums[start];
        }
        int candidate1 = majorityHelper(nums, start, mid);
        int i = start, n = end - start, count = 0;;
        while (i < end) {
            if (nums[i] == candidate1) {
                count++;
            }
            if (count > n / 2) {
                return candidate1;
            }
            i++;
        }
        int candidate2 = majorityHelper(nums, mid, end);
        i = start; count = 0;
        while (i < end) {
            if (nums[i] == candidate2) {
                count++;
            }
            if (count > n / 2) {
                return candidate2;
            }
            i++;
        }
        return Integer.MIN_VALUE;
    }

    // 168
    // Given a positive integer, return its corresponding column title as appear in an Excel sheet.
    // For example:
    //     1 -> A
    //     2 -> B
    //     3 -> C
    //     ...
    //     26 -> Z
    //     27 -> AA
    //     28 -> AB
    public String convertToTitle(int n) {
        if (n <= 0) return "";
        StringBuilder sb = new StringBuilder();
        while (n > 0) {
            n--;
            int remainder = n % 26;
            n = n / 26;
            sb.insert(0, (char) ('A' + remainder));
        }
        return sb.toString();
    }

    // 167
    // Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
    // The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

    // You may assume that each input would have exactly one solution and you may not use the same element twice.

    // Input: numbers={2, 7, 11, 15}, target=9
    // Output: index1=1, index2=2
    public int[] twoSum(int[] numbers, int target) {
        int[] res = new int[2];
        int start = 0, end = numbers.length - 1;

        while (start < end) {
            int sum = numbers[start] + numbers[end];
            if (sum > target) {
                end--;
            } else if (sum < target) {
                start++;
            } else {
                break;
            }
        }
        res[0] = start + 1; res[1] = end + 1;
        return res;
    }

    // 166
    // Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
    // If the fractional part is repeating, enclose the repeating part in parentheses.
    // For example,
    // Given numerator = 1, denominator = 2, return "0.5".
    // Given numerator = 2, denominator = 1, return "2".
    // Given numerator = 2, denominator = 3, return "0.(6)".
    public String fractionToDecimal(int numerator, int denominator) {
        StringBuilder sb = new StringBuilder();
        if (numerator < 0 || denominator < 0) {
            if (numerator > 0 || denominator > 0) {
                sb.append('-');
            }
        }
        long num = Math.abs((long) numerator);
        long den = Math.abs((long) denominator);
        long res = num / den;
        long remainder = num % den;

        sb.append(res);
        if (remainder == 0) {
            return sb.toString();
        }
        sb.append('.');
        HashMap<Long, Integer> map = new HashMap<>();
        StringBuilder digits = new StringBuilder();
        while (remainder != 0) {
            map.put(remainder, digits.length());
            remainder *= 10;
            res = remainder / den;
            remainder = remainder % den;
            digits.append(res);
            if (map.containsKey(remainder)) break;
        }
        if (remainder == 0) {
            sb.append(digits);
            return sb.toString();
        }
        int i = map.get(remainder);
        digits.insert(i, '(');
        sb.append(digits); sb.append(')');
        return sb.toString();
    }

    // 165
    // Compare two version numbers version1 and version2.
    // If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.
    // You may assume that the version strings are non-empty and contain only digits and the . character.
    // The . character does not represent a decimal point and is used to separate number sequences.
    // For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.
    // Here is an example of version numbers ordering:
    // 0.1 < 1.1 < 1.2 < 13.37
    public int compareVersion(String version1, String version2) {
        String[] ver1 = version1.split("\\.");
        String[] ver2 = version2.split("\\.");
        int n = Math.max(ver1.length, ver2.length);
        for (int i=0; i < n; i++) {
            Integer v1 = i < ver1.length ? Integer.parseInt(ver1[i]) : 0;
            Integer v2 = i < ver2.length ? Integer.parseInt(ver2[i]) : 0;
            int compare = v1.compareTo(v2);
            if (compare != 0) {
                return compare;
            }
        }
        return 0;
    }
    // 164
    // Given an unsorted array, find the maximum difference between the successive elements in its sorted form.
    // Try to solve it in linear time/space.
    // Return 0 if the array contains less than 2 elements.
    // You may assume all elements in the array are non-negative integers and fit in the 32-bit signed integer range.
    //*********************************************************/
    public int maximumGap(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int res = 0;
        if (n < 2) return res;
        for (int i = 1; i < n; i++) {
            res = Math.max(res, nums[i] - nums[i - 1]);
        }
        return res;
    }
    //*********************************************************/

    // 162
    // A peak element is an element that is greater than its neighbors.
    // Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.
    // The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
    // You may imagine that num[-1] = num[n] = -∞.
    // For example, in array [1, 2, 3, 1], 3 is a peak element and your function should return the index number 2.
    public int findPeakElement(int[] nums) {
        int i = 0;
        while (i + 1 < nums.length && nums[i] < nums[i + 1]) {
            i++;
        }
        return i;
    }

    // 160
    // Write a program to find the node at which the intersection of two singly linked lists begins.
    // For example, the following two linked lists:
    // A:          a1 → a2
    //                    ↘
    //                      c1 → c2 → c3
    //                    ↗
    // B:     b1 → b2 → b3
    // begin to intersect at node c1.
    // Notes:
    // If the two linked lists have no intersection at all, return null.
    // The linked lists must retain their original structure after the function returns.
    // You may assume there are no cycles anywhere in the entire linked structure.
    // Your code should preferably run in O(n) time and use only O(1) memory.
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int lenA = length(headA), lenB = length(headB);
        // move headA and headB to the same start point
        while (lenA > lenB) {
            headA = headA.next;
            lenA--;
        }
        while (lenA < lenB) {
            headB = headB.next;
            lenB--;
        }
        // find the intersection until end
        while (headA != headB) {
            headA = headA.next;
            headB = headB.next;
        }
        return headA;
    }
    private int length(ListNode node) {
        int length = 0;
        while (node != null) {
            node = node.next;
            length++;
        }
        return length;
    }

    // 151
    // Given an input string, reverse the string word by word.
    // For example,
    // Given s = "the sky is blue",
    // return "blue is sky the".
    public String reverseWords(String s) {
        String[] strs = s.split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i = strs.length - 1; i >= 0; i--) {
            if (strs[i].length() == 0) continue;
            if (sb.length() != 0) {
                sb.append(" ");
            }
            sb.append(strs[i]);
        }
        return sb.toString();
    }
    // Word ladder II
    // Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s) from beginWord to endWord, such that:
    // Only one letter can be changed at a time
    // Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
    // For example,
    // Given:
    // beginWord = "hit"
    // endWord = "cog"
    // wordList = ["hot","dot","dog","lot","log","cog"]
    // Return
    //   [
    //     ["hit","hot","dot","dog","cog"],
    //     ["hit","hot","lot","log","cog"]
    //   ]
    // Note:
    // Return an empty list if there is no such transformation sequence.
    // All words have the same length.
    // All words contain only lowercase alphabetic characters.
    // You may assume no duplicates in the word list. You may assume beginWord and endWord are non-empty and are not the same.
    // public List<List<String>> findLadders(String start, String end, List<String> wordList) {
    //     List<List<String>> results;
    //     List<String> list;
    //     Map<String,List<String>> map;
    //     Set<String> dict = new HashSet(wordList);
    //     results = new ArrayList<List<String>>();
    //     if (dict.size() == 0)
    //         return results;

    //     int curr=1,next=0;
    //     boolean found=false;
    //     list = new LinkedList<String>();
    //     map = new HashMap<String,List<String>>();

    //     Queue<String> queue= new ArrayDeque<String>();
    //     Set<String> unvisited = new HashSet<String>(dict);
    //     Set<String> visited = new HashSet<String>();

    //     queue.add(start);
    //     unvisited.add(end);
    //     unvisited.remove(start);
    //     //BFS
    //     while (!queue.isEmpty()) {

    //         String word = queue.poll();
    //         curr--;
    //         for (int i = 0; i < word.length(); i++){
    //            StringBuilder builder = new StringBuilder(word);
    //             for (char ch='a';  ch <= 'z'; ch++){
    //                 builder.setCharAt(i,ch);
    //                 String new_word=builder.toString();
    //                 if (unvisited.contains(new_word)){
    //                     //Handle queue
    //                     if (visited.add(new_word)){//Key statement,Avoid Duplicate queue insertion
    //                         next++;
    //                         queue.add(new_word);
    //                     }

    //                     if (map.containsKey(new_word))//Build Adjacent Graph
    //                         map.get(new_word).add(word);
    //                     else{
    //                         List<String> l= new LinkedList<String>();
    //                         l.add(word);
    //                         map.put(new_word, l);
    //                     }

    //                     if (new_word.equals(end)&&!found) found=true;

    //                 }

    //             }//End:Iteration from 'a' to 'z'
    //         }//End:Iteration from the first to the last
    //         if (curr==0){
    //             if (found) break;
    //             curr=next;
    //             next=0;
    //             unvisited.removeAll(visited);
    //             visited.clear();
    //         }
    //     }//End While
    //     System.out.println(map);
    //     // backTrace(end, start, results, list, map);

    //     return results;
    // }
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        Set<String> dict = new HashSet(wordList);
        Map<String, List<String>> map = new HashMap<>();
        Set<String> queue = new HashSet<>();
        Set<String> tmpQueue = new HashSet<>();
        Set<String> visited = new HashSet<>();
        Set<String> unvisited = new HashSet(dict);
        boolean found = false;

        queue.add(beginWord);
        while (!queue.isEmpty()) {
            for (String s : queue) {
                visited.add(s);
                StringBuilder sb = new StringBuilder(s);
                int n = sb.length();
                for (int i = 0; i < n; i++) {
                    char origin = sb.charAt(i);
                    for (char c = 'a'; c <= 'z'; c++) {
                        sb.setCharAt(i, c);
                        String str = sb.toString();
                        if (unvisited.contains(str)) {
                            visited.add(str);
                            tmpQueue.add(str);
                            if (map.containsKey(str)) {
                                map.get(str).add(s);
                            } else {
                                List<String> slist = new LinkedList<>();
                                slist.add(s);
                                map.put(str, slist);
                            }
                        }
                        if (str.equals(endWord)) {
                            found = true;
                        }
                    }
                    sb.setCharAt(i, origin);
                }
            }
            if (found) break;
            queue = tmpQueue;
            tmpQueue = new HashSet<>();
            unvisited.removeAll(visited);
            visited.clear();
        }
        List<List<String>> res = new LinkedList<>();
        List<String> list = new LinkedList<>();
        //System.out.println(map);
        backTrace(endWord, beginWord, res, list, map);
        return res;
    }
    private void backTrace(String word, String start, List<List<String>> res,
        List<String> list, Map<String, List<String>> map){
        if (word.equals(start)){
            list.add(0,start);
            res.add(new ArrayList<String>(list));
            list.remove(0);
            return;
        }
        list.add(0,word);
        if (map.get(word)!=null)
            for (String s : map.get(word))
                backTrace(s, start, res, list, map);
        list.remove(0);
    }

    // 150
    // Evaluate the value of an arithmetic expression in Reverse Polish Notation.
    // Valid operators are +, -, *, /. Each operand may be an integer or another expression.
    // Some examples:
    //   ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
    //   ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
    public int evalRPN(String[] tokens) {
        Stack<Integer> st = new Stack<>();
        for (String s : tokens) {
            if (s.equals("/")) {
                int b = st.pop();
                int a = st.pop();
                st.push(a / b);
            } else if (s.equals("*")) {
                int b = st.pop();
                int a = st.pop();
                st.push(a * b);
            } else if (s.equals("+")) {
                int b = st.pop();
                int a = st.pop();
                st.push(a + b);
            } else if (s.equals("-")) {
                int b = st.pop();
                int a = st.pop();
                st.push(a - b);
            } else {
                st.push(Integer.parseInt(s));
            }
        }
        return st.pop();
    }

    // 149
    // Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.
    /**
     * Definition for a point.
     * class Point {
     *     int x;
     *     int y;
     *     Point() { x = 0; y = 0; }
     *     Point(int a, int b) { x = a; y = b; }
     * }
     */
    public int maxPoints(Point[] points) {
        if (points==null) return 0;
        if (points.length<=2) return points.length;

        Map<Integer,Map<Integer,Integer>> map = new HashMap<Integer,Map<Integer,Integer>>();
        int result=0;
        for (int i=0;i<points.length;i++){
            map.clear();
            int overlap=0,max=0;
            for (int j=i+1;j<points.length;j++){
                int x=points[j].x-points[i].x;
                int y=points[j].y-points[i].y;
                if (x==0&&y==0){
                    overlap++;
                    continue;
                }
                int gcd=generateGCD(x,y);
                if (gcd!=0){
                    x/=gcd;
                    y/=gcd;
                }

                if (map.containsKey(x)){
                    if (map.get(x).containsKey(y)){
                        map.get(x).put(y, map.get(x).get(y)+1);
                    }else{
                        map.get(x).put(y, 1);
                    }
                }else{
                    Map<Integer,Integer> m = new HashMap<Integer,Integer>();
                    m.put(y, 1);
                    map.put(x, m);
                }
                max=Math.max(max, map.get(x).get(y));
            }
            result=Math.max(result, max+overlap+1);
        }
        return result;


    }
    private int generateGCD(int a,int b){

        if (b==0) return a;
        else return generateGCD(b,a%b);

    }


    // 148
    // Sort a linked list in O(n log n) time using constant space complexity.
    // ListNode dummyRes = new ListNode(0);
    // public class MergeResult {
    //     ListNode head;
    //     ListNode tail;

    //     MergeResult(ListNode h, ListNode t) { head = h; tail = t;}
    // }

    // public ListNode sortList(ListNode head) {
    //     if(head == null || head.next == null) return head;

    //     int length = length(head);

    //     ListNode dummy = new ListNode(0);
    //     dummy.next = head;
    //     MergeResult mr = new MergeResult(null, null);
    //     for(int step = 1; step < length; step <<= 1) {
    //         ListNode left = dummy.next;
    //         ListNode prev = dummy;
    //         while(left != null) {
    //             ListNode right = split(left, step);
    //             if(right == null) {
    //                 prev.next = left;
    //                 break;
    //             }
    //             ListNode next = split(right, step);
    //             merge(left, right, mr);
    //             prev.next = mr.head;
    //             prev = mr.tail;
    //             left = next;
    //         }
    //     }
    //     return dummy.next;
    // }

    // public ListNode split(ListNode head, int step) {
    //     while(head != null && step != 1) {
    //         head = head.next;
    //         step--;
    //     }
    //     if(head == null) return null;
    //     ListNode res = head.next;
    //     head.next = null;
    //     return res;
    // }

    // public int length(ListNode head) {
    //     int len = 0;
    //     while(head != null) {
    //         head = head.next;
    //         len++;
    //     }
    //     return len;
    // }

    // public void merge(ListNode head1, ListNode head2, MergeResult mr) {
    //     if(head2 == null) {
    //         mr.head = head1;
    //         mr.tail = null;
    //     }
    //     ListNode res = dummyRes;
    //     ListNode tail = res;
    //     while(head1 != null && head2 != null) {
    //         if(head1.val < head2.val) {
    //             tail.next = head1;
    //             head1 = head1.next;
    //         }else{
    //             tail.next = head2;
    //             head2 = head2.next;
    //         }
    //         tail = tail.next;
    //     }

    //     while(head1 != null) {
    //         tail.next = head1;
    //         head1 = head1.next;
    //         tail = tail.next;
    //     }

    //     while(head2 != null) {
    //         tail.next = head2;
    //         head2 = head2.next;
    //         tail = tail.next;
    //     }

    //     mr.head = res.next;
    //     mr.tail = tail;
    // }

    // 147
    // Sort a linked list using insertion sort.
    public ListNode insertionSortList(ListNode head) {
        if (head == null) return null;
        ListNode node = head, prenode = null;

        while (node != null) {
            ListNode cur = head, pre = null;
            while (cur != node && cur.val <= node.val) {
                pre = cur; cur = cur.next;
            }
            if (cur == node) {
                prenode = node;
                node = node.next;
            } else if (cur == head) {
                prenode.next = node.next;
                node.next = cur;
                head = node;
                node = prenode.next;

            } else {
                prenode.next = node.next;
                pre.next = node;
                node.next = cur;
                node = prenode.next;
            }
        }
        return head;
    }

    // 145
    // Given a binary tree, return the postorder traversal of its nodes' values.
    // For example:
    // Given binary tree [1,null,2,3],
    //    1
    //     \
    //      2
    //     /
    //    3
    // return [3,2,1].
    // Note: Recursive solution is trivial, could you do it iteratively?
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode(int x) { val = x; }
     * }
     */
    // Important, when you pop a node, ensure its children are traversed.
    public List<Integer> postorderTraversal(TreeNode root) {
        Stack<TreeNode> s = new Stack<TreeNode>();
        List<Integer> res = new ArrayList<Integer>();
        TreeNode cur = root;

        while (cur != null || !s.empty()) {
            while (!isLeaf(cur)) {
                s.push(cur);
                cur = cur.left;
            }
            if (cur != null) res.add(cur.val);
            while (!s.empty() && cur == s.peek().right) {
                cur = s.pop();
                res.add(cur.val);
            }
            if (s.empty()) cur = null;
            else cur = s.peek().right;
        }
        return res;
    }
    private boolean isLeaf(TreeNode r) {
        if (r == null) return true;
        return r.left == null && r.right == null;
    }

    // 144
    // Given a binary tree, return the preorder traversal of its nodes' values.
    // For example:
    // Given binary tree [1,null,2,3],
    //    1
    //     \
    //      2
    //     /
    //    3
    // return [1,2,3].
    // Note: Recursive solution is trivial, could you do it iteratively?
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode(int x) { val = x; }
     * }
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Stack<TreeNode> st = new Stack<>();
        st.push(root);
        while (!st.isEmpty()) {
            TreeNode node = st.pop();
            res.add(node.val);
            if (node.right != null) {
                st.push(node.right);
            }
            if (node.left != null) {
                st.push(node.left);
            }
        }
        return res;
    }

    // 143
    // Given a singly linked list L: L0→L1→…→Ln-1→Ln,
    // reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
    // You must do this in-place without altering the nodes' values.
    // For example,
    // Given {1,2,3,4}, reorder it to {1,4,2,3}.
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode(int x) { val = x; }
     * }
     */
    public void reorderList(ListNode head) {
        if (head == null) return;
        LinkedList<ListNode> backList = new LinkedList<>();
        ListNode p = head;
        while (p != null) {
            backList.addFirst(p);
            p = p.next;
        }
        ListNode start = head;
        ListNode end = backList.poll();
        while (start != end && start.next != end) {
            ListNode nextNode = start.next;
            start.next = end;
            end.next = nextNode;
            start = nextNode;
            end = backList.poll();
        }
        end.next = null;
        return;
    }

    // 142
    // Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
    // Note: Do not modify the linked list.
    // Follow up:
    // Can you solve it without using extra space?
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        boolean found = false;
        while (slow != null && fast != null) {
            slow = slow.next;
            fast = fast.next;
            if (fast == null) break;
            fast = fast.next;
            if (slow == fast) {
                found = true;
                break;
            }
        }
        if (!found) return null;
        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    // 141
    // Given a linked list, determine if it has a cycle in it.
    // Follow up:
    // Can you solve it without using extra space?
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (slow != null && fast != null) {
            slow = slow.next;
            fast = fast.next;
            if (fast == null) break;
            fast = fast.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }

    // 140
    // Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. You may assume the dictionary does not contain duplicate words.
    // Return all such possible sentences.
    // For example, given
    // s = "catsanddog",
    // dict = ["cat", "cats", "and", "sand", "dog"].
    // A solution is ["cats and dog", "cat sand dog"].
    public List<String> wordBreak(String s, List<String> wordDict) {
        HashMap<String, List<String>> map = new HashMap<>();
        return wordBreakHelper(s, wordDict, map);
    }
    private List<String> wordBreakHelper(String s, List<String> wordDict,
                                         HashMap<String, List<String>> map) {
        if (map.containsKey(s)) {
            return map.get(s);
        }
        List<String> res = new LinkedList<>();
        if (s.length() == 0) {
            res.add("");
            return res;
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                List<String> subStrList = wordBreakHelper(s.substring(word.length()), wordDict, map);
                for(String subStr : subStrList) {
                    res.add(word + (subStr.equals("") ? "" : " ") + subStr);
                }
            }
        }
        map.put(s, res);
        return res;
    }

    // 139
    // Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words. You may assume the dictionary does not contain duplicate words.
    // For example, given
    // s = "leetcode",
    // dict = ["leet", "code"].
    // Return true because "leetcode" can be segmented as "leet code".
    public boolean wordBreakBool(String s, List<String> wordDict) {
        Set<String> mySet = new HashSet<>(wordDict);
        int n = s.length();
        boolean[] dp = new boolean[n + 1];

        dp[0] = true;
        for (int i = 1; i <= n; i++) {
            dp[i] = mySet.contains(s.substring(0, i));
            for (int j = 1; j < i; j++) {
                if (mySet.contains(s.substring(j, i)) && dp[j]) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[n];
    }

    // 138
    // A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.
    // Return a deep copy of the list.
    /**
     * Definition for singly-linked list with a random pointer.
     * class RandomListNode {
     *     int label;
     *     RandomListNode next, random;
     *     RandomListNode(int x) { this.label = x; }
     * };
     */
    // public RandomListNode copyRandomList(RandomListNode head) {
    //     if (head == null) return null;

    //     RandomListNode p = head;
    //     while (p != null) {
    //         RandomListNode node = new RandomListNode(p.label);
    //         node.next = p.next;
    //         p.next = node;
    //         p = node.next;
    //     }

    //     p = head;
    //     while (p != null) {
    //         RandomListNode pNext = p.next;
    //         if (p.random != null)
    //             pNext.random = p.random.next;
    //         p = pNext.next;
    //     }

    //     RandomListNode newHead = head.next;
    //     RandomListNode pNext = newHead;
    //     p = head;
    //     while (p != null) {
    //         p.next = pNext.next;
    //         if (pNext.next != null)
    //             pNext.next = pNext.next.next;
    //         p = p.next;
    //         pNext = pNext.next;
    //     }

    //     return newHead;
    // }


    // 137
    // Given an array of integers, every element appears three times except for one, which appears exactly once. Find that single one.
    // Note:
    // Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
    public static int singleNumber(int[] nums) {
        int ones = 0, twos = 0;
        for(int i = 0; i < nums.length; i++){
            ones = (ones ^ nums[i]) & ~twos;
            twos = (twos ^ nums[i]) & ~ones;
        }
        return ones;
    }
    // The code seems tricky and hard to understand at first glance.
    // However, if you consider the problem in Boolean algebra form, everything becomes clear.
    // What we need to do is to store the number of '1’s of every bit. Since each of the 32 bits follow the same rules
    // we just need to consider 1 bit. We know a number appears 3 times at most, so we need 2 bits to store that. Now we have 4 state, 00, 01, 10 and 11, but we only need 3 of them.
    // In this solution, 00, 01 and 10 are chosen. Let ‘ones’ represents the first bit, ‘twos’ represents the second bit. Then we need to set rules for ‘ones’ and ‘twos’ so that they act as we hopes. The complete loop is 00->10->01->00(0->1->2->3/0).
    // For ‘ones’, we can get ‘ones = ones ^ A[i]; if (twos == 1) then ones = 0’, that can be tansformed to ‘ones = (ones ^ A[i]) & ~twos’.

    // Similarly, for ‘twos’, we can get ‘twos = twos ^ A[i]; if (ones* == 1) then twos = 0’ and ‘twos = (twos ^ A[i]) & ~ones’. Notice that ‘ones*’ is the value of ‘ones’ after calculation, that is why twos is
    // calculated later.
    // Here is another example. If a number appears 5 times at most, we can write a program using the same method. Now we need 3 bits and the loop is 000->100->010->110->001. The code looks like this:
    // int singleNumber(int A[], int n) {
    //     int na = 0, nb = 0, nc = 0;
    //     for(int i = 0; i < n; i++){
    //         nb = nb ^ (A[i] & na);
    //         na = (na ^ A[i]) & ~nc;
    //         nc = nc ^ (A[i] & ~na & ~nb);
    //     }
    //     return na & ~nb & ~nc;
    // }


    // 136
    // Given an array of integers, every element appears twice except for one. Find that single one.
    // Note:
    // Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
    public static int singleNumber2(int[] nums) {
        int res = 0;
        for (int i : nums) {
            res ^= i;
        }
        return res;
    }

    // 135
    // There are N children standing in a line. Each child is assigned a rating value.
    // You are giving candies to these children subjected to the following requirements:
    // Each child must have at least one candy.
    // Children with a higher rating get more candies than their neighbors.
    // What is the minimum candies you must give?
    public static int candy(int[] ratings) {
        int res = 0;
        int n = ratings.length;
        if (n == 0) return 0;
        int[] candys = new int[n];
        for (int i = 0; i < n; i++) {
            candys[i] = 1;
        }
        for (int i = 0; i < n - 1; i++) {
            if (ratings[i] < ratings[i + 1]) {
                candys[i + 1] = candys[i] + 1;
            }
        }
        for (int i = n - 1; i > 0; i--) {
            if (ratings[i] < ratings[i - 1] && candys[i - 1] <= candys[i]) {
                candys[i - 1] = candys[i] + 1;
            }
        }
        for (int i = 0; i < n; i++) {
            res += candys[i];
        }
        return res;
    }

    // 134
    // There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
    // You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1).
    // You begin the journey with an empty tank at one of the gas stations.
    // Return the starting gas station's index if you can travel around the circuit once, otherwise return -1.
    // Note:
    // The solution is guaranteed to be unique.
    public static int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length;
        int i = 0;
        int totalGas = 0;
        int totalCost = 0;
        for (int j = 0; j < n; j++) {
            totalCost += cost[j];
            totalGas += gas[j];
        }
        if (totalCost > totalGas) return -1;
        while (i < n) {
            // start from 0
            int res = cannotReach(i, gas, cost);
            if (res == i) {
                return res;
            } else {
                i = res;
            }
        }
        return -1;
    }
    private static int cannotReach(int start, int[] gas, int[] cost) {
        int n = gas.length;
        int i = (start + 1) % n;
        int curGas = gas[start];
        int nextCost = cost[start];
        while(i != start) {
            curGas -= nextCost;
            if (curGas < 0) {
                return i;
            } else {
                nextCost = cost[i];
                i = (i + 1) % n;
                curGas += gas[i];
            }
        }
        return start;
    }

    // 133
    // Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.

    // OJ's undirected graph serialization:
    // Nodes are labeled uniquely.

    // We use # as a separator for each node, and , as a separator for node label and each neighbor of the node.
    // As an example, consider the serialized graph {0,1,2#1,2#2,2}.

    // The graph has a total of three nodes, and therefore contains three parts as separated by #.

    // First node is labeled as 0. Connect node 0 to both nodes 1 and 2.
    // Second node is labeled as 1. Connect node 1 to node 2.
    // Third node is labeled as 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.
    // Visually, the graph looks like the following:

    //        1
    //       / \
    //      /   \
    //     0 --- 2
    //          / \
    //          \_/

    // Definition for undirected graph.
    // class UndirectedGraphNode {
    //     int label;
    //     List<UndirectedGraphNode> neighbors;
    //     UndirectedGraphNode(int x) { label = x; neighbors = new ArrayList<UndirectedGraphNode>(); }
    // };

    // public static UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
    //     if (node == null) return null;
    //     HashMap<Integer, UndirectedGraphNode> map = new HashMap<>();
    //     HashSet<Integer> set = new HashSet<>();

    //     int label = node.label;
    //     UndirectedGraphNode root =  new UndirectedGraphNode(label);
    //     map.put(label, root);
    //     set.add(label);

    //     for (UndirectedGraphNode n : node.neighbors) {
    //         int l = n.label;
    //         if (!set.contains(l)) { // visit it
    //             root.neighbors.add(cloneGraphHelper(n, map, set));
    //         } else {
    //             root.neighbors.add(map.get(l));
    //         }
    //     }
    //     return root;
    // }
    // private static UndirectedGraphNode cloneGraphHelper(UndirectedGraphNode node,
    //  HashMap<Integer, UndirectedGraphNode> map, HashSet<Integer> set) {
    //     int label = node.label;
    //     UndirectedGraphNode root;

    //     if (map.containsKey(label)) {
    //         root = map.get(label);
    //     } else {
    //         root =  new UndirectedGraphNode(label);
    //         map.put(label, root);
    //     }
    //     set.add(label);

    //     for (UndirectedGraphNode n : node.neighbors) {
    //         int l =  n.label;
    //         if (!set.contains(l)) {
    //             root.neighbors.add(cloneGraphHelper(n, map, set));
    //         } else {
    //             root.neighbors.add(map.get(l));
    //         }
    //     }
    //     return root;
    // }


    // 132
    // Given a string s, partition s such that every substring of the partition is a palindrome.
    // Return the minimum cuts needed for a palindrome partitioning of s.

    // For example, given s = "aab",
    // Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut.
    public static int minCut(String s) {
        int n  = s.length();
        if (n <= 1) return 0;
        boolean[][] isPalin = new boolean[n][n];
        int[] minRes = new int[n];
        for (int i = 0; i < n; i++) {
            isPalin[i][i] = true;
            minRes[i] = i;
        }
        minRes[0] = 0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(i) == s.charAt(j) &&
                        (j + 1 >= i || isPalin[j + 1][i - 1])) {
                    isPalin[j][i] = true;
                    if (j == 0) minRes[i] = 0;
                    else minRes[i] = Math.min(minRes[j - 1] + 1, minRes[i]);
                }
            }
        }
        return minRes[n - 1];
    }
}
