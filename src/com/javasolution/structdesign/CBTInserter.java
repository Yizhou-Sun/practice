package com.javasolution.structdesign;

import java.util.*;

import com.javasolution.util.TreeNode;

/**
 * Definition for a binary tree node. public class TreeNode { int val; TreeNode
 * left; TreeNode right; TreeNode(int x) { val = x; } }
 */
class CBTInserter {
    private TreeNode root;
    private Queue<TreeNode> curLevel;
    private Queue<TreeNode> nextLevel;

    public CBTInserter(TreeNode root) {
        this.root = root;
        curLevel = new ArrayDeque<>();
        nextLevel = new ArrayDeque<>();

        curLevel.offer(root);
        while (!curLevel.isEmpty() || !nextLevel.isEmpty()) {
            if (curLevel.isEmpty()) {
                curLevel = nextLevel;
                nextLevel = new ArrayDeque<>();
            }
            TreeNode node = curLevel.peek();

            if (node.left == null)
                return;
            nextLevel.offer(node.left);
            if (node.right == null)
                return;
            nextLevel.offer(node.right);
            curLevel.poll();
        }
    }

    public int insert(int v) {
        TreeNode head = curLevel.peek();
        TreeNode vnode = new TreeNode(v);

        nextLevel.offer(vnode);
        if (head.left == null) {
            head.left = vnode;
        } else {
            head.right = vnode;
            curLevel.poll();
            if (curLevel.isEmpty()) {
                curLevel = nextLevel;
                nextLevel = new ArrayDeque<>();
            }
        }
        return head.val;
    }

    public TreeNode get_root() {
        return this.root;
    }
}

/**
 * Your CBTInserter object will be instantiated and called as such: CBTInserter
 * obj = new CBTInserter(root); int param_1 = obj.insert(v); TreeNode param_2 =
 * obj.get_root();
 */