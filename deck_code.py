"""
Convert between deck codes and deck strings.
"""
import base64
from typing import List
import random
import json


# load name map and forbid list
deck_code_data = json.load(
    open(__file__.replace("deck_code.py", "data/deck_code_data.json"))
)
forbid_list = deck_code_data["forbid_list"][:]
characters_idx = []


# create forbid word trie
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def match(self, word):
        """
        test if word start with any word in trie
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
            if node.is_end_of_word:
                return True
        return False

    def search(self, word):
        """
        test if word contains any word in trie
        """
        for i in range(len(word)):
            if self.match(word[i:]):
                return True
        return False


forbidden_trie = Trie()
for forbid in forbid_list:
    forbidden_trie.insert(forbid)


def _deck_idx_to_deck_code_one(id_list: List[int], checksum: int) -> str:
    """
    Convert deck str to deck code. Versions about deck_str will be ignored.
    """
    binary = ""
    for idx in id_list:
        binary += "{:012b} ".format(idx)
    binary = binary.replace(" ", "")
    b8 = []
    for i in range(0, len(binary), 8):
        b8.append(int(binary[i: i + 8], 2))
    b8[-1] = b8[-1] * 16  # 4 zeros
    uint = list(zip(b8[:25], b8[25:]))
    uint = [list(x) for x in uint]
    uint = sum(uint, start=[])
    uint.append(0)
    uint = [(x + checksum) % 256 for x in uint]
    res = base64.b64encode(bytes(uint)).decode()
    return res


def deck_idx_to_deck_code(
    character_idx: list[int], card_idx: list[int], max_retry_time: int = 10000
) -> str:
    character_idx = character_idx[:]
    card_idx = card_idx[:]
    assert len(character_idx) == 3
    assert len(card_idx) == 30
    rand_checksum = list(range(256))
    random.shuffle(rand_checksum)
    for i in range(max_retry_time):
        # generate random checksum and shuffle cards
        if i > 255:
            checksum = random.randint(0, 255)
            random.shuffle(card_idx)
        else:
            checksum = rand_checksum[i]
        deck_code = _deck_idx_to_deck_code_one(character_idx + card_idx, checksum)
        code_lower = deck_code.lower().replace("+", "")
        if not forbidden_trie.search(code_lower):
            return deck_code
    raise ValueError("in generating deck code: retry time exceeded")


if __name__ == "__main__":
    # descs = json.load(open(__file__.replace('deck_code.py',
    #                                         'default_desc.json'),
    #                        encoding = 'utf8'))
    # map_eng = [''] * len(map_chinese)
    # for key, value in descs.items():
    #     if 'names' not in value:
    #         continue
    #     name_c = value['names']['zh-CN']
    #     if name_c in map_chinese:
    #         idx = map_chinese.index(name_c)
    #         map_eng[idx] = value['names']["en-US"]
    # print(map_eng)

    # import json
    # res = json.dumps({
    #     'name_map': name_map,
    #     'forbid_list': forbid_list,
    # }, indent = 2)
    # open('deck_code_data.json', 'w', encoding = 'utf8').write(res)

    ...
