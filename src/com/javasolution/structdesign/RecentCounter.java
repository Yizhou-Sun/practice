package com.javasolution.structdesign;

import java.util.*;

class RecentCounter {
    private TreeMap<Integer, Integer> timestamp;

    public RecentCounter() {
        timestamp = new TreeMap<>();
    }

    public int ping(int t) {
        Map.Entry<Integer, Integer> maxEntry = timestamp.lowerEntry(t);
        if (maxEntry == null) {
            timestamp.put(t, 1);
            return 1;
        }
        timestamp.put(t, maxEntry.getValue() + 1);

        Map.Entry<Integer, Integer> minEntry = timestamp.lowerEntry(t - 3000);
        if (minEntry == null) {
            return maxEntry.getValue() + 1;
        }

        return maxEntry.getValue() - minEntry.getValue() + 1;

    }
}

/**
 * Your RecentCounter object will be instantiated and called as such:
 * RecentCounter obj = new RecentCounter();
 * int param_1 = obj.ping(t);
 */