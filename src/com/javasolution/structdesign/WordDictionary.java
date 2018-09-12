package com.javasolution.structdesign;

// 211
// Design a data structure that supports the following two operations:
// void addWord(word)
// bool search(word)
// search(word) can search a literal word or a regular expression string containing only letters a-z or .. A . means it can represent any one letter.
// For example:
// addWord("bad")
// addWord("dad")
// addWord("mad")
// search("pad") -> false
// search("bad") -> true
// search(".ad") -> true
// search("b..") -> true
// Note:
// You may assume that all words are consist of lowercase letters a-z.

public class WordDictionary {
    WordDictionary[] nextLevel;
    boolean hasWord;

    /** Initialize your data structure here. */
    public WordDictionary() {
        this.nextLevel = new WordDictionary[26];
        hasWord = false;
    }

    /** Adds a word into the data structure. */
    public void addWord(String word) {
        int wordLen = word.length();
        if (wordLen == 0)
            return;

        char c = word.charAt(0);
        if (nextLevel[c - 'a'] == null)
            nextLevel[c - 'a'] = new WordDictionary();

        addWordHelper(word, 1, wordLen, nextLevel[c - 'a']);
        ;
    }

    private void addWordHelper(String word, int i, int wordLen, WordDictionary root) {
        if (i == wordLen) {
            root.hasWord = true;
            return;
        }

        char c = word.charAt(i);
        if (root.nextLevel[c - 'a'] == null)
            root.nextLevel[c - 'a'] = new WordDictionary();

        addWordHelper(word, i + 1, wordLen, root.nextLevel[c - 'a']);
    }

    /**
     * Returns if the word is in the data structure. A word could contain the dot
     * character '.' to represent any one letter.
     */
    public boolean search(String word) {
        int wordLen = word.length();
        if (wordLen == 0)
            return false;

        char c = word.charAt(0);
        if (c != '.') {
            if (nextLevel[c - 'a'] == null) {
                return false;
            }
            return searchHelper(word, 1, wordLen, nextLevel[c - 'a']);
        }
        for (WordDictionary dict : nextLevel) {
            if (dict != null && searchHelper(word, 1, wordLen, dict)) {
                return true;
            }
        }
        return false;
    }

    private boolean searchHelper(String word, int i, int wordLen, WordDictionary root) {
        if (i == wordLen) {
            return root.hasWord;
        }

        char c = word.charAt(i);
        if (c != '.') {
            if (root.nextLevel[c - 'a'] == null) {
                return false;
            }
            return searchHelper(word, i + 1, wordLen, root.nextLevel[c - 'a']);
        }
        for (WordDictionary dict : root.nextLevel) {
            if (dict != null && searchHelper(word, i + 1, wordLen, dict)) {
                return true;
            }
        }
        return false;
    }
}

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary obj = new WordDictionary(); obj.addWord(word); boolean param_2
 * = obj.search(word);
 */