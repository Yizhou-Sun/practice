package com.javasolution.structdesign;

import java.util.TreeMap;

class TopVotedCandidate {
    private HashMap<Integer, Integer> freqMap;
    private HashMap<Integer, Stack<Integer>> stackMap;
    private TreeMap<Integer, Integer> treeMap;

    public TopVotedCandidate(int[] persons, int[] times) {
        freqMap = new HashMap<>();
        stackMap = new HashMap<>();
        treeMap = new TreeMap<>();

        int curMax = 0;
        for (int i = 0; i < persons.length; i++) {
            int t = times[i], p = persons[i];
            int freq = freqMap.getOrDefault(p, 0) + 1;
            freqMap.put(p, freq);
            if (curMax <= freq) {
                curMax = Math.max(curMax, freq);
                if (!stackMap.containsKey(freq)) {
                    stackMap.put(freq, new Stack<Integer>());
                }
                stackMap.get(freq).push(p);
                treeMap.put(t, stackMap.get(curMax).peek());
            }
        }
    }

    public int q(int t) {
        return treeMap.get(treeMap.floorKey(t));
    }
}

/**
 * Your TopVotedCandidate object will be instantiated and called as such:
 * TopVotedCandidate obj = new TopVotedCandidate(persons, times);
 * int param_1 = obj.q(t);
 */