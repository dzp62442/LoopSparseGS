import json
import os
import os.path as osp
import pickle as pkl
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import imageio.v2 as imageio


CENTER150_FILENAME = "bins_center150_v1.json"
CENTER150_SAMPLE_COUNT = 150
BIN_TOKEN_PATTERN = re.compile(r"^scene([0-9a-f]+)_bin(\d+)$")


@dataclass
class OmniSceneFrame:
    """单帧视角信息（用于预处理与加载）。"""
    image_name: str
    image_path: str
    depth_path: str
    conf_path: str
    intrinsics: np.ndarray  # 3x3, 像素坐标系
    c2w: np.ndarray         # 4x4
    width: int
    height: int
    depth_scale: float


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _replace_prefix(path: str, dataset_prefix: str, data_root: str) -> str:
    if path.startswith(dataset_prefix):
        return path.replace(dataset_prefix, data_root)
    return path


def _load_info(info: Dict) -> Tuple[str, np.ndarray, np.ndarray]:
    """读取 nuScenes/OmniScene 单帧信息，返回图像路径、c2w、w2c。"""
    img_path = info["data_path"]
    c2w = np.array(info["sensor2lidar_transform"], dtype=np.float32)

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
    return img_path, c2w, w2c


def _load_intrinsics(img_path: str) -> np.ndarray:
    """读取相机内参（像素坐标系）。"""
    param_path = img_path.replace("samples", "samples_param_small")
    param_path = param_path.replace("sweeps", "sweeps_param_small")
    param_path = param_path.replace(".jpg", ".json")
    param = json.load(open(param_path))
    return np.array(param["camera_intrinsic"], dtype=np.float32)


def _resize_image_and_intrinsics(
    img: Image.Image, intrinsics: np.ndarray, reso: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """按目标分辨率缩放图像与内参。"""
    tgt_h, tgt_w = reso
    if img.height == tgt_h and img.width == tgt_w:
        return np.array(img), intrinsics

    scale_h = tgt_h / img.height
    scale_w = tgt_w / img.width
    intrinsics = intrinsics.copy()
    intrinsics[0, 0] *= scale_w
    intrinsics[0, 2] *= scale_w
    intrinsics[1, 1] *= scale_h
    intrinsics[1, 2] *= scale_h
    img = img.resize((tgt_w, tgt_h))
    return np.array(img), intrinsics


def _resize_array(arr: np.ndarray, reso: Tuple[int, int]) -> np.ndarray:
    """将深度/置信度 resize 到目标分辨率。"""
    tgt_h, tgt_w = reso
    if arr.shape[0] == tgt_h and arr.shape[1] == tgt_w:
        return arr
    img = Image.fromarray(arr.astype(np.float32))
    img = img.resize((tgt_w, tgt_h), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def _load_metric_depth_and_conf(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 Metric3D 深度与置信度。"""
    depthm_path = img_path.replace("sweeps_small", "sweeps_dptm_small")
    depthm_path = depthm_path.replace("samples_small", "samples_dptm_small")
    depthm_path = depthm_path.replace(".jpg", "_dpt.npy")
    conf_path = depthm_path.replace("_dpt.npy", "_conf.npy")
    depth_m = np.load(depthm_path).astype(np.float32)
    conf_m = np.load(conf_path).astype(np.float32)
    return depth_m, conf_m


def _save_depth_png(depth_m: np.ndarray, depth_scale: float, out_path: str) -> None:
    """将深度保存为 uint16 PNG，并保留 scale。"""
    depth_int = np.clip(depth_m * depth_scale, 0, 65535).astype(np.uint16)
    imageio.imwrite(out_path, depth_int)


def _make_point_cloud(
    frames: List[OmniSceneFrame],
    images_rgb: List[np.ndarray],
    depths_m: List[np.ndarray],
    confs_m: List[np.ndarray],
    conf_threshold: float,
    max_points: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """由多视角深度生成初始点云。"""
    all_xyz = []
    all_rgb = []

    for frame, img, depth, conf in zip(frames, images_rgb, depths_m, confs_m):
        H, W = depth.shape
        fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
        cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]

        mask = depth > 0
        if conf is not None:
            mask &= conf > conf_threshold
        if not np.any(mask):
            continue

        ys, xs = np.where(mask)
        d = depth[ys, xs]
        x = (xs - cx) / fx * d
        y = (ys - cy) / fy * d
        z = d
        ones = np.ones_like(z)
        pts_cam = np.stack([x, y, z, ones], axis=1)
        pts_world = (frame.c2w @ pts_cam.T).T[:, :3]

        rgb = img[ys, xs, :3]
        all_xyz.append(pts_world)
        all_rgb.append(rgb)

    if not all_xyz:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    if xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz.astype(np.float32), rgb.astype(np.uint8)


def _store_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """保存点云为 PLY（带颜色）。"""
    from plyfile import PlyData, PlyElement

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    PlyData([vertex_element]).write(path)


def _load_json_object(path: str, description: str) -> Dict:
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"{description}不存在: {path}。Center150 清单只能由 SVF-GS 项目生成。"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{description}必须是 JSON object: {path}")
    return data


def _parse_bin_token(bin_token: str) -> Tuple[str, int]:
    if not isinstance(bin_token, str):
        raise ValueError(f"bin token 必须是字符串，实际为 {type(bin_token).__name__}")
    match = BIN_TOKEN_PATTERN.fullmatch(bin_token)
    if match is None:
        raise ValueError(f"非法 bin token: {bin_token}")
    return match.group(1), int(match.group(2))


def _expected_center150_tokens(val_manifest: Dict) -> List[str]:
    """仅在内存中复算 SVF-GS lower-median 规则，用于校验现有清单。"""
    all_bins = val_manifest.get("bins")
    adjacent_bins = val_manifest.get("adjacent_bins")
    if not isinstance(all_bins, list) or not isinstance(adjacent_bins, list):
        raise ValueError("bins_val_3.2m.json 必须包含 list 字段 'bins' 和 'adjacent_bins'")
    if len(adjacent_bins) != CENTER150_SAMPLE_COUNT:
        raise ValueError(
            f"OmniScene val 应包含 {CENTER150_SAMPLE_COUNT} 个场景，实际为 {len(adjacent_bins)}"
        )

    flattened_bins = [token for scene_bins in adjacent_bins for token in scene_bins]
    if flattened_bins != all_bins:
        raise ValueError("bins_val_3.2m.json 中 adjacent_bins 展平后与 bins 不完全一致")

    selected_bins = []
    selected_scenes = set()
    for scene_index, scene_bins in enumerate(adjacent_bins):
        if not isinstance(scene_bins, list) or not scene_bins:
            raise ValueError(f"val 场景分组 {scene_index} 为空或格式错误")
        parsed = [_parse_bin_token(token) for token in scene_bins]
        scene_tokens = {scene_token for scene_token, _ in parsed}
        bin_indices = [bin_index for _, bin_index in parsed]
        if len(scene_tokens) != 1:
            raise ValueError(f"val 场景分组 {scene_index} 混入了多个 scene token")
        if bin_indices != list(range(len(scene_bins))):
            raise ValueError(f"val 场景分组 {scene_index} 的 bin 序号不连续或未按序排列")

        scene_token = next(iter(scene_tokens))
        if scene_token in selected_scenes:
            raise ValueError(f"val 清单含重复 scene token: {scene_token}")
        selected_scenes.add(scene_token)
        selected_bins.append(scene_bins[(len(scene_bins) - 1) // 2])

    if len(selected_bins) != CENTER150_SAMPLE_COUNT or len(set(selected_bins)) != CENTER150_SAMPLE_COUNT:
        raise ValueError("从 val 清单复算后未得到 150 个唯一中央 bin")
    return selected_bins


def load_center150_tokens(version_dir: str) -> List[str]:
    """严格校验并加载 SVF-GS 生成的 Center150 清单；本项目不生成清单。"""
    val_manifest = _load_json_object(
        osp.join(version_dir, "bins_val_3.2m.json"), "OmniScene val 清单"
    )
    center150_manifest = _load_json_object(
        osp.join(version_dir, CENTER150_FILENAME), "SVF-GS Center150 清单"
    )
    actual_tokens = center150_manifest.get("bins")
    if not isinstance(actual_tokens, list):
        raise ValueError(f"{CENTER150_FILENAME} 必须包含 list 字段 'bins'")

    expected_tokens = _expected_center150_tokens(val_manifest)
    if actual_tokens != expected_tokens:
        mismatch_index = next(
            (idx for idx, pair in enumerate(zip(actual_tokens, expected_tokens)) if pair[0] != pair[1]),
            min(len(actual_tokens), len(expected_tokens)),
        )
        raise ValueError(
            f"{CENTER150_FILENAME} 与 SVF-GS lower-median 规则不一致，"
            f"首个差异索引为 {mismatch_index}"
        )

    bin_info_dir = osp.join(version_dir, "bin_infos_3.2m")
    missing_infos = [
        token for token in actual_tokens
        if not osp.isfile(osp.join(bin_info_dir, token + ".pkl"))
    ]
    if missing_infos:
        raise FileNotFoundError(
            f"Center150 有 {len(missing_infos)} 个 bin 缺少 bin info；"
            f"前几个为: {', '.join(missing_infos[:3])}"
        )
    return list(actual_tokens)


def get_bin_tokens(data_root: str, stage: str) -> List[str]:
    """获取 bin 列表（遵循 depthsplat 的 OmniScene 采样规则）。"""
    data_version = "interp_12Hz_trainval"
    version_dir = osp.join(data_root, data_version)
    if stage == "center150":
        return load_center150_tokens(version_dir)

    bins_path = osp.join(version_dir, "bins_val_3.2m.json")
    if stage == "train":
        bins_path = osp.join(data_root, data_version, "bins_train_3.2m.json")
    bins = json.load(open(bins_path))["bins"]
    if stage == "val":
        return bins[:30000:3000][:10]
    if stage == "test":
        return bins[0::14][:2048]
    if stage == "demo":
        # 与 depthsplat 一致的 demo 列表
        return [
            "scenee7ef871f77f44331aefdebc24ec034b7_bin010",
            "scenee7ef871f77f44331aefdebc24ec034b7_bin200",
            "scene30ae9c1092f6404a9e6aa0589e809780_bin100",
            "scene84e056bd8e994362a37cba45c0f75558_bin100",
            "scene717053dec2ef4baa913ba1e24c09edff_bin000",
            "scene82240fd6d5ba4375815f8a7fa1561361_bin050",
            "scene724957e51f464a9aa64a16458443786d_bin000",
            "scened3c39710e9da42f48b605824ce2a1927_bin050",
            "scene034256c9639044f98da7562ef3de3646_bin000",
            "scenee0b14a8e11994763acba690bbcc3f56a_bin080",
            "scene7e2d9f38f8eb409ea57b3864bb4ed098_bin150",
            "scene50ff554b3ecb4d208849d042b7643715_bin000",
        ]
    return bins


def _ensure_canonical_eval_names(test_json: str) -> None:
    """Keep the 18 evaluation views aligned with SVF-GS names 00--17."""
    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if len(frames) != 18:
        raise ValueError(f"OmniScene evaluation split must contain 18 views, got {len(frames)}: {test_json}")

    expected_names = [f"{idx:02d}" for idx in range(len(frames))]
    if [frame.get("image_name") for frame in frames] == expected_names:
        return

    for image_name, frame in zip(expected_names, frames):
        frame["image_name"] = image_name
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def preprocess_scene(
    data_root: str,
    output_root: str,
    bin_token: str,
    scene_prefix: str,
    reso: Tuple[int, int] = (112, 200),
    dataset_prefix: str = "/datasets/nuScenes",
    conf_threshold: float = 0.3,
    depth_scale: float = 1000.0,
) -> str:
    """预处理单个 bin，返回场景目录路径。"""
    data_version = "interp_12Hz_trainval"
    scene_dir = osp.join(output_root, f"{scene_prefix}_{bin_token}")
    images_train_dir = osp.join(scene_dir, "images", "train")
    images_test_dir = osp.join(scene_dir, "images", "test")
    depth_train_dir = osp.join(scene_dir, "depth", "train")
    depth_test_dir = osp.join(scene_dir, "depth", "test")
    conf_train_dir = osp.join(scene_dir, "depth", "conf")
    points_dir = osp.join(scene_dir, "points")
    cams_dir = osp.join(scene_dir, "cams")

    train_json = osp.join(cams_dir, "train.json")
    test_json = osp.join(cams_dir, "test.json")
    init_ply = osp.join(points_dir, "init_points.ply")
    if osp.exists(train_json) and osp.exists(test_json) and osp.exists(init_ply):
        _ensure_canonical_eval_names(test_json)
        return scene_dir

    _ensure_dir(images_train_dir)
    _ensure_dir(images_test_dir)
    _ensure_dir(depth_train_dir)
    _ensure_dir(depth_test_dir)
    _ensure_dir(conf_train_dir)
    _ensure_dir(points_dir)
    _ensure_dir(cams_dir)

    bin_info_path = osp.join(data_root, data_version, "bin_infos_3.2m", bin_token + ".pkl")
    with open(bin_info_path, "rb") as f:
        bin_info = pkl.load(f)

    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    sensor_info_center = {sensor: bin_info["sensor_info"][sensor][0] for sensor in camera_types + ["LIDAR_TOP"]}

    # 训练视角：key-frame（index=0）
    train_frames: List[OmniSceneFrame] = []
    train_imgs, train_depths, train_confs = [], [], []

    for cam in camera_types:
        info = sensor_info_center[cam]
        img_path, c2w, _ = _load_info(info)
        img_path = _replace_prefix(img_path, dataset_prefix, data_root)
        raw_img_path = img_path
        img_path = img_path.replace("samples", "samples_small").replace("sweeps", "sweeps_small")

        intr = _load_intrinsics(raw_img_path)
        img = Image.open(img_path).convert("RGB")
        img_np, intr = _resize_image_and_intrinsics(img, intr, reso)
        depth_m, conf_m = _load_metric_depth_and_conf(img_path)
        depth_m = _resize_array(depth_m, reso)
        conf_m = _resize_array(conf_m, reso)
        depth_m = np.where(conf_m > conf_threshold, depth_m, 0.0)

        image_name = f"{cam.lower()}_f0"
        image_out = osp.join(images_train_dir, image_name + ".png")
        depth_out = osp.join(depth_train_dir, image_name + ".png")
        conf_out = osp.join(conf_train_dir, image_name + ".npy")

        imageio.imwrite(image_out, img_np)
        _save_depth_png(depth_m, depth_scale, depth_out)
        np.save(conf_out, conf_m)

        train_frames.append(
            OmniSceneFrame(
                image_name=image_name,
                image_path=osp.relpath(image_out, scene_dir),
                depth_path=osp.relpath(depth_out, scene_dir),
                conf_path=osp.relpath(conf_out, scene_dir),
                intrinsics=intr,
                c2w=c2w,
                width=img_np.shape[1],
                height=img_np.shape[0],
                depth_scale=depth_scale,
            )
        )
        train_imgs.append(img_np)
        train_depths.append(depth_m)
        train_confs.append(conf_m)

    # 测试视角：index=1,2 + 训练视角拼回
    test_frames: List[OmniSceneFrame] = []
    for cam in camera_types:
        for frame_idx in [1, 2]:
            info = bin_info["sensor_info"][cam][frame_idx]
            img_path, c2w, _ = _load_info(info)
            img_path = _replace_prefix(img_path, dataset_prefix, data_root)
            raw_img_path = img_path
            img_path = img_path.replace("samples", "samples_small").replace("sweeps", "sweeps_small")

            intr = _load_intrinsics(raw_img_path)
            img = Image.open(img_path).convert("RGB")
            img_np, intr = _resize_image_and_intrinsics(img, intr, reso)
            depth_m, conf_m = _load_metric_depth_and_conf(img_path)
            depth_m = _resize_array(depth_m, reso)
            conf_m = _resize_array(conf_m, reso)
            depth_m = np.where(conf_m > conf_threshold, depth_m, 0.0)

            image_name = f"{cam.lower()}_f{frame_idx}"
            image_out = osp.join(images_test_dir, image_name + ".png")
            depth_out = osp.join(depth_test_dir, image_name + ".png")
            conf_out = osp.join(conf_train_dir, image_name + ".npy")

            imageio.imwrite(image_out, img_np)
            _save_depth_png(depth_m, depth_scale, depth_out)
            np.save(conf_out, conf_m)

            test_frames.append(
                OmniSceneFrame(
                    image_name=image_name,
                    image_path=osp.relpath(image_out, scene_dir),
                    depth_path=osp.relpath(depth_out, scene_dir),
                    conf_path=osp.relpath(conf_out, scene_dir),
                    intrinsics=intr,
                    c2w=c2w,
                    width=img_np.shape[1],
                    height=img_np.shape[0],
                    depth_scale=depth_scale,
                )
            )

    # 将训练视角追加到测试集（共 18 张）
    test_frames.extend(train_frames)

    # 保存相机信息
    def _dump_frames(frames: List[OmniSceneFrame], out_json: str) -> None:
        data = {
            "frames": [
                {
                    "image_name": f.image_name,
                    "file_path": f.image_path,
                    "depth_path": f.depth_path,
                    "conf_path": f.conf_path,
                    "intrinsics": f.intrinsics.tolist(),
                    "c2w": f.c2w.tolist(),
                    "width": f.width,
                    "height": f.height,
                    "depth_scale": f.depth_scale,
                }
                for f in frames
            ]
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    _dump_frames(train_frames, train_json)
    _dump_frames(test_frames, test_json)
    _ensure_canonical_eval_names(test_json)

    # 构建初始化点云
    xyz, rgb = _make_point_cloud(
        frames=train_frames,
        images_rgb=train_imgs,
        depths_m=train_depths,
        confs_m=train_confs,
        conf_threshold=conf_threshold,
    )
    _store_ply(init_ply, xyz, rgb)
    return scene_dir


def load_context_target(scene_dir: str) -> Dict[str, Dict]:
    """按 depthsplat 风格加载 context/target（用于调试或扩展）。"""
    def _load_split(split: str) -> Dict:
        json_path = osp.join(scene_dir, "cams", f"{split}.json")
        data = json.load(open(json_path))
        images, intrinsics, extrinsics = [], [], []
        for frame in data["frames"]:
            img_path = osp.join(scene_dir, frame["file_path"])
            img = imageio.imread(img_path)
            images.append(img)
            intrinsics.append(np.array(frame["intrinsics"], dtype=np.float32))
            extrinsics.append(np.array(frame["c2w"], dtype=np.float32))
        return {
            "image": np.stack(images, axis=0),
            "intrinsics": np.stack(intrinsics, axis=0),
            "extrinsics": np.stack(extrinsics, axis=0),
        }

    return {
        "context": _load_split("train"),
        "target": _load_split("test"),
    }
