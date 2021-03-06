package com.javasolution.structdesign;

import java.util.Stack;

import com.javasolution.util.TreeNode;

// 173
// Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.
// Calling next() will return the next smallest number in the BST.
// Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.

public class BSTIterator {
    private Stack<TreeNode> st;

    public BSTIterator(TreeNode root) {
        st = new Stack<>();
        TreeNode node = root;
        while (node != null) {
            st.push(node);
            node = node.left;
        }
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return !st.isEmpty();
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        TreeNode res = st.pop();
        TreeNode node = res.right;
        while (node != null) {
            st.push(node);
            node = node.left;
        }
        return res.val;
    }
}

/**
 * Definition for binary tree public class TreeNode { int val; TreeNode left;
 * TreeNode right; TreeNode(int x) { val = x; } }
 * <p>
 * Your BSTIterator will be called like this: BSTIterator i = new
 * BSTIterator(root); while (i.hasNext()) v[f()] = i.next();
 */
/**
 * Your BSTIterator will be called like this: BSTIterator i = new
 * BSTIterator(root); while (i.hasNext()) v[f()] = i.next();
 */