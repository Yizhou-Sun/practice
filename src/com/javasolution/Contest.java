package com.javasolution;

import java.util.*;
import java.lang.*;

import com.javasolution.util.*;
import com.javasolution.structdesign.*;

public class Contest {
    // 902
    // https://leetcode.com/problems/numbers-at-most-n-given-digit-set/description/
    // TODO: Review!!!
    public int atMostNGivenDigitSet(String[] D, int N) {
        int result = 0;
        String s = String.valueOf(N);
        for (int i = 1; i <= s.length(); i++) {
            result += helper(D, i, Integer.toString(N));
        }
        return result;
    }

    private int helper(String[] D, int K, String s) {
        if (s.equals("0")) {
            return 0;
        }

        if (s.length() > K) {
            return (int) Math.pow(D.length, K);
        }

        int count = 0;
        int char0 = s.charAt(0) - '0';

        for (int i = 0; i < D.length; i++) {

            if (Integer.parseInt(D[i]) < char0) {
                count += helper(D, K - 1, s);
            } else if (Integer.parseInt(D[i]) == char0) {
                if (s.length() > 1) {
                    int charRem = Integer.parseInt(s.substring(1, 2)) == 0 ? 0 : Integer.parseInt(s.substring(1));
                    count += helper(D, K - 1, Integer.toString(charRem));
                } else {
                    count++;
                }
            }
        }
        return count;
    }

    // 903
    // https://leetcode.com/problems/valid-permutations-for-di-sequence/description/
    // TODO: A very good question!
    public int numPermsDISequence(String S) {
        int MOD = 1000000007;
        int n = S.length();

        int[][] dp = new int[n + 1][n + 1];
        for (int i = 0; i <= n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= i; j++) {
                if (S.charAt(i - 1) == 'I') {
                    for (int k = 0; k < j; ++k) {
                        dp[i][j] += dp[i - 1][k];
                        dp[i][j] %= MOD;
                    }
                } else {
                    for (int k = j; k < i; ++k) {
                        dp[i][j] += dp[i - 1][k];
                        dp[i][j] %= MOD;
                    }
                }
            }
        }

        long ans = 0;
        for (int x : dp[n]) {
            ans += x;
            ans %= MOD;
        }
        return (int) ans;
    }
}