package com.javasolution.structdesign;

import java.util.*;
// Hash map freq will count the frequence of elements.
// Hash map m is a map of stack.
// If element x has n frequence, we will push x n times in m[1], m[2] .. m[n]
// maxfreq records the maximum frequence.

// push(x) will push x tom[++freq[x]]
// pop() will pop from the m[maxfreq]
class FreqStack {
    private HashMap<Integer, Integer> freqMap;
    private HashMap<Integer, Stack<Integer>> stackMap;
    int curMax = 0;

    public FreqStack() {
        freqMap = new HashMap<Integer, Integer>();
        stackMap = new HashMap<Integer, Stack<Integer>>();
    }

    public void push(int x) {
        int freq = freqMap.getOrDefault(x, 0) + 1;
        freqMap.put(x, freq);
        curMax = Math.max(curMax, freq);
        Stack<Integer> st = stackMap.getOrDefault(freq, new Stack<Integer>());
        st.push(x);
        stackMap.put(freq, st);
    }

    public int pop() {
        Stack<Integer> st = stackMap.get(curMax);
        int res = st.pop();
        freqMap.put(res, curMax - 1);
        if (st.isEmpty())
            curMax -= 1;
        return res;
    }
}

/**
 * Your FreqStack object will be instantiated and called as such: FreqStack obj
 * = new FreqStack(); obj.push(x); int param_2 = obj.pop();
 */