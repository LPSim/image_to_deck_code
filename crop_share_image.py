from typing import Any, List
import cv2
import numpy as np
import json
import os
import imagehash
from PIL import Image
from tqdm import tqdm
import base64
import random


IMAGE_FOLDER = (
    r'C:\Users\zyr17\Documents\Projects\LPSim\frontend\collector\sprite'
)
PATCH_JSON = (
    r'data/'
)
BACKEND = 'cv2'


_CHARACTER_FEATS: Any = None
_CARD_FEATS: Any = None


# load name map and forbid list
deck_code_data = json.load(
    open(__file__.replace("crop_share_image.py", "data/deck_code_data.json"))
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


def get_image_feature_cv2(img):
    sift = cv2.SIFT_create()  # type: ignore
    kp, des = sift.detectAndCompute(img, None)
    return des


def get_image_feature_imagehash(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imagehash.dhash(img, hash_size = 128)


def get_flann_index_params():
    # 使用 FLANN 构建索引
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=5)
    return index_params, search_params


def build_flann_index(features):

    flann = cv2.FlannBasedMatcher(*get_flann_index_params())  # type: ignore
    flann.add([features])
    flann.train()
    return flann


def load_flann(cache_path):
    flann = cv2.FlannBasedMatcher(*get_flann_index_params())  # type: ignore
    flann.read(cache_path)
    return flann


def save_flann(flann: cv2.FlannBasedMatcher, cache_path):
    # TODO load result wrong
    flann.write(cache_path)


def cache_and_build_flann_index(img, cache_key = None, cache_folder = './cache/flann'):
    # TODO: currently save & load will become empty, disable it.
    # cache_path = f'{cache_folder}/{cache_key}'
    # if cache_key is not None and os.path.exists(cache_path):
    #     return load_flann(cache_path)
    feature = get_image_feature(img)
    flann = build_flann_index(feature)
    # save_flann(flann, cache_path)
    return flann


def find_best_match(names, flann, query_feature):
    matches = flann.knnMatch(query_feature, k=2)

    # 获取最匹配的特征的索引和相似度
    best_match_names = [m.trainIdx for m, n in matches]
    match_similarities = [m.distance for m, n in matches]
    print(best_match_names, len(names))
    best_match_names = [names[i] for i in best_match_names]

    return zip(best_match_names, match_similarities)


def compare_images_cv2(feat1, flann):
    des1 = feat1
    # kp2, des2 = feat2

    # 使用 FLANN 匹配器来匹配描述符
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=4)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore

    # print(des1.shape)
    matches = flann.knnMatch(des1, k=2)

    # 仅保留好的匹配项
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # 计算相似度
    similarity = len(good_matches) / len(matches)

    return similarity


def compare_images_imagehash(feat1, feat2):
    return 1 - (feat1 - feat2) / len(feat1.hash) ** 2


def get_helper():
    if BACKEND == 'cv2':
        return get_image_feature_cv2, compare_images_cv2, 2, 0.2
    elif BACKEND == 'imagehash':
        return get_image_feature_imagehash, compare_images_imagehash, 1.1, 0.6
    else:
        raise ValueError('unknown backend')


get_image_feature, compare_images, DIFF_THRESHOLD, MATCH_THRESHOLD = get_helper()


def warn_not_confident(
    sim, diff_threshold = DIFF_THRESHOLD, match_threshold = MATCH_THRESHOLD
):
    """
    diff_threshold: first should be how many times better than second
    match_threshold: how similar should it be
    """
    if sim[0][1] < match_threshold:
        print(f'{sim[0][0]} match too low: {sim[0][1]:.6f}')
    if sim[0][1] < diff_threshold * sim[1][1]:
        print(
            f'{sim[0][0]} not too much better than {sim[1][0]}: '
            f'{sim[0][1]:.6f} {sim[1][1]:.6f}'
        )


def do_one_img(
    character_feats: dict[int, tuple[int, int, str, Any]], 
    card_feats: dict[int, tuple[int, int, str, Any]], 
    current_character_feats: list[Any],
    current_card_feats: list[Any],
    verbose: bool = False,
):
    """
    get parts of img, and find their names, return a list of names. 
    """
    if verbose:
        current_character_feats = tqdm(current_character_feats)  # type: ignore
        current_card_feats = tqdm(current_card_feats)  # type: ignore
    characters_sim = []
    for current_character_feat in current_character_feats:
        character_sim = []
        for idx, [cid, csid, cname, character_feat] in (character_feats.items()):
            similarity = compare_images(current_character_feat, character_feat)
            character_sim.append([cid, csid, cname, similarity])
        character_sim.sort(key=lambda x: x[-1], reverse=True)
        characters_sim.append(character_sim)

    # calc sim, and do dp
    max_card_id = max(list(card_feats.keys()))
    sim_arr = np.full((len(current_card_feats) + 1, max_card_id + 1), -1.0)
    for idx, current_card_feat in enumerate(current_card_feats):
        for card_idx, [cid, csid, cname, card_feat] in (card_feats.items()):
            similarity = compare_images(current_card_feat, card_feat)
            sim_arr[idx, card_idx] = similarity
    dp = np.zeros_like(sim_arr)
    prev = np.zeros_like(dp, dtype=int)
    for i in range(len(current_card_feats)):
        for j in range(max_card_id + 1):
            if i == 0:
                dp[i, j] = sim_arr[i, j]
                prev[i, j] = -1
            else:
                prev[i, j] = np.argmax(dp[i - 1, :j + 1])
                dp[i, j] = sim_arr[i, j] + dp[i - 1, prev[i, j]]
    cards_sim = []
    x = len(current_card_feats) - 1
    y: int = np.argmax(dp[x]).item()
    while y != -1:
        cards_sim.append([*card_feats[y][:3], sim_arr[x, y]])
        y = prev[x, y]
        x -= 1
    assert x == y == -1
    cards_sim = cards_sim[::-1]

    for character_sim in characters_sim:
        warn_not_confident(character_sim)
    # for card_sim in cards_sim:
    #     warn_not_confident(card_sim)

    return [x[0] for x in characters_sim], [x for x in cards_sim]


def get_all_img_feat(patch_json = PATCH_JSON, image_folder = IMAGE_FOLDER):
    character_data = json.load(
        open(patch_json + '/guyu_characters.json', encoding = 'utf8'))
    card_data = json.load(
        open(patch_json + '/guyu_action_cards.json', encoding = 'utf8'))
    character_res = []
    card_res = []
    for title, data, res in [('character', character_data, character_res), 
                             ('card', card_data, card_res)]:
        for one_data in tqdm(data, desc = f'{title} feat init'):
            if 'shareId' not in one_data:
                continue
            obj_id = one_data['id']
            share_id = one_data['shareId']
            name = one_data['name']
            img_path = one_data['cardFace'].replace('UI_Gcg_CardFace_', '') + ".png"
            if not os.path.exists(os.path.join(image_folder, 'cardface', img_path)):
                print(f'{img_path} not exists')
                continue
            img = cv2.imread(os.path.join(image_folder, 'cardface', img_path))
            res.append((
                obj_id,
                share_id, 
                name,
                cache_and_build_flann_index(img, str(share_id)),
            ))
    character_res.sort(key=lambda x: x[0])
    card_res.sort(key=lambda x: x[0])
    character_res = {x: character_res[x] for x in range(len(character_res))}
    card_res = {x: card_res[x] for x in range(len(card_res))}
    return character_res, card_res


def crop_on_official_share_image(image):
    # image is PIL, crop cards from it.
    # # 定义RGB颜色
    # rgb_color = np.uint8([[[225, 212, 203]]])  # 例如，RGB颜色为 (220, 213, 205)
    # # 将RGB颜色转换为HSV颜色
    # hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    # print("HSV颜色值:", hsv_color)

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义背景颜色范围
    lower_color = np.array([10, 10, 210])
    upper_color = np.array([30, 30, 230])

    # 创建掩码
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # print(mask, mask.mean(), mask.std(), mask.shape)

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)

    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 获取包围矩形
    x, y, w, h = cv2.boundingRect(largest_contour)

    # print(x, y, w, h)

    # 裁剪图片
    cropped_image = image[y: y + h, x: x + w]
    cropped_hsv = hsv[y: y + h, x: x + w]
    cropped_mask = mask[y: y + h, x: x + w]

    # lower_color_cropped = np.array([3, 3, 210])
    # upper_color_cropped = np.array([30, 40, 230])
    lower_color_cropped = np.array([3, 3, 210])
    upper_color_cropped = np.array([230, 240, 230])
    cropped_mask = cv2.inRange(cropped_hsv, lower_color_cropped, upper_color_cropped)

    # print(cropped_mask.min(), cropped_mask.max(), 
    #     cropped_mask.mean(), cropped_mask.std())

    thresh = 255 - cropped_mask

    # 找到小矩形物体的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = thresh.shape

    iheight, iwidth = height * 0.11111, width * 0.117
    scale_multiplier = [0.9, 1.6]

    # 过滤出符合要求的矩形并统计数量
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            iheight * scale_multiplier[0] < h < iheight * scale_multiplier[1]
            and iwidth * scale_multiplier[0] < w < iwidth * scale_multiplier[1]
        ):
            rectangles.append((x, y, w, h))

    # 统计符合要求的矩形数量
    num_rectangles = len(rectangles)
    if num_rectangles != 33:
        print(f"符合要求的矩形数量与预期不符: {num_rectangles}")

    # print(f"符合要求的矩形数量: {num_rectangles}")

    # 按位置排序
    rectangles = sorted(rectangles, key=lambda r: (r[1] * 20 + r[0]))

    # cut the image
    output_images = []
    for i, (x, y, w, h) in enumerate(rectangles):
        output_images.append(cropped_image[y: y + h, x: x + w])

    # 绘制矩形
    for (x, y, w, h) in rectangles:
        cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 保存并显示结果
    # print(cropped_mask.shape)
    # cropped_image[cropped_mask.astype(bool)] = 255
    # cv2.imwrite('cropped_image_with_rectangles.jpg', cropped_image)
    # cv2.imshow('Cropped Image with Rectangles', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return output_images


def init_img_feat(patch_json = PATCH_JSON, image_folder = IMAGE_FOLDER):
    global _CHARACTER_FEATS
    global _CARD_FEATS
    if _CHARACTER_FEATS is None:
        _CHARACTER_FEATS, _CARD_FEATS = get_all_img_feat(patch_json, image_folder)
    return _CHARACTER_FEATS, _CARD_FEATS


def process_as_share_id(img_path, verbose: bool = False):
    sub_images = crop_on_official_share_image(cv2.imread(img_path))
    characters = sub_images[:3]
    cards = sub_images[3:]

    current_character_feats = [get_image_feature(character) for character in characters]
    current_card_feats = [get_image_feature(card) for card in cards]
    character_feats, card_feats = init_img_feat()
    res = do_one_img(
        character_feats, card_feats, current_character_feats, current_card_feats,
        verbose)
    if verbose:
        print([x[2] for x in res[0]], [x[2] for x in res[1]])
    return (
        [x[1] for x in res[0]],
        [x[1] for x in res[1]],
    )


def process_as_deck_code(img_path, verbose: bool = False):
    character_ids, card_ids = process_as_share_id(img_path, verbose)
    deck_code = deck_idx_to_deck_code(character_ids, card_ids)
    if verbose:
        print(deck_code)
    return deck_code


if __name__ == '__main__':
    # init_img_feat()

    image_path = r'C:\Users\zyr17\Downloads\ring-wedgness-metric.jpg'
    process_as_deck_code(image_path, True)
    print('done!')
