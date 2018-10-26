package com.javasolution.structdesign;

/**
 * 900 https://leetcode.com/problems/rle-iterator/description/ Your RLEIterator
 * object will be instantiated and called as such: RLEIterator obj = new
 * RLEIterator(A); int param_1 = obj.next(n);
 */
public class RLEIterator {
    private int[] arr;
    private int cur;

    public RLEIterator(int[] A) {
        arr = A;
        cur = 0;
    }

    public int next(int n) {
        while (cur < arr.length && n > arr[cur]) {
            n -= arr[cur];
            cur += 2;
        }
        if (cur < arr.length) {
            arr[cur] -= n;
            return arr[cur + 1];
        }
        return -1;
    }
}
