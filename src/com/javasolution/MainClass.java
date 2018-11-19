package com.javasolution;

import java.util.*;

import com.javasolution.util.*;

public class MainClass {
    public static void main(String[] args) {
        // int num = 231;
        int[] nums = { 58, -27, -11, 63, 90, 83, 61, -44, -39, 30 };
        int[][] twodArr = { { 0, 1 }, { 1, 0 } };
        // String s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT";
        // List<String> strList = Arrays.asList("hot","dot","dog","lot","log","cog");
        // char[][] grid =
        // {{'0','0','0','1'},{'1','1','0','1'},{'1','1','1','1'},{'0','1','1','1'},{'0','1','1','1'}};
        // TreeNode root = stringToTreeNode("[5,3,6,2,4,null,null,1]");
        // var nums1 = new int[] {1,3};
        // var nums2 = new int[] {2};
        // String[] strArr = {
        // "test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"
        // };

        var solution = new Solution_18();
        var res = solution.shortestSubarray(nums, 61);
        System.out.println(res);
    }

    public static TreeNode stringToTreeNode(String input) {
        input = input.trim();
        input = input.substring(1, input.length() - 1);
        if (input.length() == 0) {
            return null;
        }

        String[] parts = input.split(",");
        String item = parts[0];
        TreeNode root = new TreeNode(Integer.parseInt(item));
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        nodeQueue.add(root);

        int index = 1;
        while (!nodeQueue.isEmpty()) {
            TreeNode node = nodeQueue.remove();

            if (index == parts.length) {
                break;
            }

            item = parts[index++];
            item = item.trim();
            if (!item.equals("null")) {
                int leftNumber = Integer.parseInt(item);
                node.left = new TreeNode(leftNumber);
                nodeQueue.add(node.left);
            }

            if (index == parts.length) {
                break;
            }

            item = parts[index++];
            item = item.trim();
            if (!item.equals("null")) {
                int rightNumber = Integer.parseInt(item);
                node.right = new TreeNode(rightNumber);
                nodeQueue.add(node.right);
            }
        }
        return root;
    }
}
