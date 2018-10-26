package com.javasolution.structdesign;

// 232
// Implement the following operations of a queue using stacks.
// push(x) -- Push element x to the back of queue.
// pop() -- Removes the element from in front of queue.
// peek() -- Get the front element.
// empty() -- Return whether the queue is empty.
// Notes:
// You must use only standard operations of a stack -- which means only push to top, peek/pop from top, size, and is empty operations are valid.
// Depending on your language, stack may not be supported natively. You may simulate a stack by using a list or deque (double-ended queue), as long as you use only standard operations of a stack.
// You may assume that all operations are valid (for example, no pop or peek operations will be called on an empty queue).

import java.util.Stack;

class MyQueue {
    Stack<Integer> st;

    /**
     * Initialize your data structure here.
     */
    public MyQueue() {
        st = new Stack<>();
    }

    /**
     * Push element x to the back of queue.
     */
    public void push(int x) {
        Stack<Integer> temp = new Stack<>();
        while (!st.isEmpty()) {
            temp.push(st.pop());
        }
        st.push(x);
        while (!temp.isEmpty()) {
            st.push(temp.pop());
        }
    }

    /**
     * Removes the element from in front of queue and returns that element.
     */
    public int pop() {
        return st.pop();
    }

    /**
     * Get the front element.
     */
    public int peek() {
        return st.peek();
    }

    /**
     * Returns whether the queue is empty.
     */
    public boolean empty() {
        return st.isEmpty();
    }
}

/**
 * Your MyQueue object will be instantiated and called as such: MyQueue obj =
 * new MyQueue(); obj.push(x); int param_2 = obj.pop(); int param_3 =
 * obj.peek(); boolean param_4 = obj.empty();
 */