package com.javasolution;

import java.util.*;
import java.lang.*;

import com.javasolution.util.*;

public class MainClass {
    public static void main(String[] args) {
        int num = 231;
        // int[] nums = {10,9,2,5,3,7,101,18};
        // int[][] twodArr = {{2,9,10}, {3,7,15}, {5,12,12}, {15,20,10}, {19,24,8}};
        // String s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT";
        // List<String> wordList = Arrays.asList("hot","dot","dog","lot","log","cog");
        // char[][] grid = {{'0','0','0','1'},{'1','1','0','1'},{'1','1','1','1'},{'0','1','1','1'},{'0','1','1','1'}};
        // TreeNode root = stringToTreeNode("[5,3,6,2,4,null,null,1]");
        var nums1 = new int[] {1,3};
        var nums2 = new int[] {2};
        Solution solution = new Solution();
        var res = solution.myAtoi("2147483648");
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
        while(!nodeQueue.isEmpty()) {
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
