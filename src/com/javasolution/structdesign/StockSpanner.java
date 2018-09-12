package com.javasolution.structdesign;

import java.util.Stack;

/**
 * 901 https://leetcode.com/problems/online-stock-span/description/ Your
 * StockSpanner object will be instantiated and called as such: StockSpanner obj
 * = new StockSpanner(); int param_1 = obj.next(price);
 */
class StockSpanner {
    Stack<int[]> st;

    public StockSpanner() {
        st = new Stack<>();
    }

    public int next(int price) {
        int count = 1;
        while (!st.isEmpty() && st.peek()[0] <= price) {
            count += st.pop()[1];
        }
        st.push(new int[] { price, count });
        return count;
    }
}
