import json
from typing import Any
import cv2
import os
import numpy as np
import imagehash
from PIL import Image

from tqdm import tqdm


IMAGE_FOLDER = (
    r'C:\Users\zyr17\Documents\Projects\LPSim\frontend\collector\splitter\4.5'
)
GUYU_PATCH_JSON = (
    r'C:\Users\zyr17\Documents\Projects\ban-pick\frontend\src\guyu_json'
)
BACKEND = 'cv2'


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


def get_all_img_feat(patch_json = GUYU_PATCH_JSON, image_folder = IMAGE_FOLDER):
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
