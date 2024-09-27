public class Knapsack {

    public static int knapsackRecursive(int[] weights, int[] values, int n, int W) {
        return helper(weights, values, n, W);
    }

    private static int helper(int[] weights, int[] values, int i, int w) {
        if (i == 0 || w == 0) {
            return 0;
        }
        if (weights[i - 1] > w) {
            return helper(weights, values, i - 1, w);
        } else {
            int exclude = helper(weights, values, i - 1, w);
            int include = values[i - 1] + helper(weights, values, i - 1, w - weights[i - 1]);
            return Math.max(exclude, include);
        }
    }

    public static void main(String[] args) {
        int[] weights1 = {1, 3, 3, 1};
        int[] values1 = {3, 8, 4, 7};
        int n1 = 4;
        int W1 = 6;
        System.out.println(knapsackRecursive(weights1, values1, n1, W1));  // Output: 18

        int[] weights2 = {3, 1, 6, 10, 1, 4, 9, 1, 7, 2, 6, 1, 6, 2, 2, 4, 8, 1, 7, 3, 6, 2, 9, 5, 3, 3, 4, 7, 3, 5, 30, 50};
        int[] values2 = {7, 4, 9, 18, 9, 15, 4, 2, 6, 13, 18, 12, 12, 16, 19, 19, 10, 16, 14, 3, 14, 4, 15, 7, 5, 10, 10, 13, 19, 9, 8, 5};
        int n2 = 32;
        int W2 = 75;
        System.out.println(knapsackRecursive(weights2, values2, n2, W2));  // Output: 262
    }
}
