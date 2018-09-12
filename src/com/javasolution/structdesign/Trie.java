package com.javasolution.structdesign;

// 208
// Implement a trie with insert, search, and startsWith methods.
// Note:
// You may assume that all inputs are consist of lowercase letters a-z in java file
public class Trie {
    public Trie[] nextLevel;
    public boolean hasWord;

    /** Initialize your data structure here. */
    public Trie() {
        this.nextLevel = new Trie[26];
        hasWord = false;
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
        int wordLen = word.length();
        if (wordLen == 0)
            return;

        char c = word.charAt(0);
        if (nextLevel[c - 'a'] == null)
            nextLevel[c - 'a'] = new Trie();
        insertHelper(word, 1, wordLen, nextLevel[c - 'a']);
        ;
    }

    private void insertHelper(String word, int i, int wordLen, Trie root) {
        if (i == wordLen) {
            root.hasWord = true;
            return;
        }
        char c = word.charAt(i);
        if (root.nextLevel[c - 'a'] == null)
            root.nextLevel[c - 'a'] = new Trie();
        insertHelper(word, i + 1, wordLen, root.nextLevel[c - 'a']);
    }

    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        int wordLen = word.length();
        if (wordLen == 0)
            return false;

        char c = word.charAt(0);
        if (nextLevel[c - 'a'] == null) {
            return false;
        }
        return searchHelper(word, 1, wordLen, nextLevel[c - 'a']);
    }

    private boolean searchHelper(String word, int i, int wordLen, Trie root) {
        if (i == wordLen) {
            return root.hasWord;
        }

        char c = word.charAt(i);
        if (root.nextLevel[c - 'a'] == null) {
            return false;
        }
        return searchHelper(word, i + 1, wordLen, root.nextLevel[c - 'a']);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        int wordLen = prefix.length();
        if (wordLen == 0)
            return false;

        char c = prefix.charAt(0);
        if (nextLevel[c - 'a'] == null) {
            return false;
        }
        return startsWithHelper(prefix, 1, wordLen, nextLevel[c - 'a']);
    }

    private boolean startsWithHelper(String prefix, int i, int wordLen, Trie root) {
        if (i == wordLen) {
            if (root.hasWord)
                return true;
            for (Trie node : root.nextLevel) {
                if (node != null) {
                    return true;
                }
            }
        }

        char c = prefix.charAt(i);
        if (root.nextLevel[c - 'a'] == null) {
            return false;
        }
        return startsWithHelper(prefix, i + 1, wordLen, root.nextLevel[c - 'a']);
    }
}

/**
 * Your Trie object will be instantiated and called as such: Trie obj = new
 * Trie(); obj.insert(word); boolean param_2 = obj.search(word); boolean param_3
 * = obj.startsWith(prefix);
 */