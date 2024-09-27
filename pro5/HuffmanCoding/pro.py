import heapq
from collections import Counter


class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.char is not None and other.char is not None:
            if self.weight == other.weight:
                return self.char < other.char
            return self.weight < other.weight
        elif self.char is not None:
            return True
        else:
            return False


def huffman_coding(text):
    frequency = Counter(text)

    priority_queue = [Node(weight, char) for char, weight in frequency.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = Node(left.weight + right.weight)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)

    root = priority_queue[0]

    huffman_codes = {}

    def generate_codes(node, prefix=""):
        if node.char is not None:
            huffman_codes[node.char] = prefix
        else:
            generate_codes(node.left, prefix + "0")
            generate_codes(node.right, prefix + "1")

    generate_codes(root)

    return huffman_codes

def main():
    text = input("Enter the ASCII-encoded text: ")

    huffman_codes = huffman_coding(text)

    print("Generated Prefix Codes:")
    for char, code in huffman_codes.items():
        print(f"Character '{char}': Code '{code}'")


if __name__ == "__main__":
    main()
