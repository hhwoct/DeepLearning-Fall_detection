#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集修复脚本：合并 train/val 中拆散的 jpg+xml 配对，按标签分层重新 8:2 划分
"""
import os, shutil, random, re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path('dataset')
OLD_DIR = ROOT / '_old_split_backup'   # 旧划分备份
MERGE = ROOT / '_merge'                # 合并暂存
TRAIN_IMG = ROOT / 'train' / 'images'
VAL_IMG = ROOT / 'val' / 'images'
SEED = 42
VAL_RATIO = 0.2

random.seed(SEED)

def parse_label(xml_path):
    """有 down/fall/fallen 对象 -> 1(摔倒)，否则 0(正常)"""
    try:
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall('object'):
            name = obj.find('name')
            if name is not None and name.text:
                if name.text.strip().lower() in ('down', 'fall', 'fallen'):
                    return 1
    except Exception as e:
        print(f"  ⚠️ XML解析失败 {xml_path.name}: {e}")
    return 0

# ---------- 第一步：备份旧划分 ----------
print("=== 第一步：备份旧划分 ===")
for d in (TRAIN_IMG, VAL_IMG):
    for f in sorted(d.glob('*')):
        if f.name == '.DS_Store':
            continue
        target = OLD_DIR / d.parent.name / f.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.move(str(f), str(target))
        else:
            f.unlink()  # 理论上不会发生

# ---------- 第二步：从备份中配对收集 ----------
print("=== 第二步：配对收集 ===")
items = {}   # stem -> {'jpg': Path|None, 'xml': Path|None}
for f in sorted(OLD_DIR.rglob('*')):
    if not f.is_file() or f.name == '.DS_Store':
        continue
    if f.suffix.lower() == '.xml':
        items.setdefault(f.stem, {})['xml'] = f
    elif f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        items.setdefault(f.stem, {})['jpg'] = f

pairs, orphan_jpg, orphan_xml = [], [], []
for stem, d in sorted(items.items()):
    if d.get('jpg') and d.get('xml'):
        pairs.append((d['jpg'], d['xml']))
    elif d.get('jpg'):
        orphan_jpg.append(d['jpg'])
    elif d.get('xml'):
        orphan_xml.append(d['xml'])

print(f"配对成功: {len(pairs)} | 孤儿jpg: {len(orphan_jpg)} | 孤儿xml: {len(orphan_xml)}")
for f in orphan_jpg[:10]: print(f"  孤儿jpg: {f.name}")
for f in orphan_xml[:10]: print(f"  孤儿xml: {f.name}")

# ---------- 第三步：解析标签，分层划分 ----------
print("=== 第三步：分层划分 (8:2) ===")
labeled = []
for i, (jpg, xml) in enumerate(pairs):
    label = parse_label(xml)
    labeled.append((label, jpg, xml))
    if (i + 1) % 2000 == 0:
        print(f"  已解析 {i+1}/{len(pairs)}")

by_label = {0: [], 1: []}
for label, jpg, xml in labeled:
    by_label[label].append((jpg, xml))

train_set, val_set = [], []
for label, lst in by_label.items():
    random.shuffle(lst)
    n_val = max(1, int(len(lst) * VAL_RATIO)) if len(lst) > 5 else (1 if lst else 0)
    val_set.extend(lst[:n_val])
    train_set.extend(lst[n_val:])
random.shuffle(train_set)
random.shuffle(val_set)

cnt_tr = Counter(parse_label(xml) for _, xml in train_set)
cnt_va = Counter(parse_label(xml) for _, xml in val_set)
print(f"训练集: {len(train_set)} 张 | 正常={cnt_tr.get(0,0)} 摔倒={cnt_tr.get(1,0)}")
print(f"验证集: {len(val_set)} 张 | 正常={cnt_va.get(0,0)} 摔倒={cnt_va.get(1,0)}")

# ---------- 第四步：写入新划分 ----------
print("=== 第四步：写入新划分 ===")
def write_set(lst, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for jpg, xml in lst:
        shutil.copy2(str(jpg), str(dest / jpg.name))
        shutil.copy2(str(xml), str(dest / xml.name))

write_set(train_set, TRAIN_IMG)
write_set(val_set, VAL_IMG)
print(f"写入完成: train={len(train_set)}, val={len(val_set)}")

# ---------- 第五步：清理与验证 ----------
print("=== 第五步：验证 ===")
for name, d in (('train', TRAIN_IMG), ('val', VAL_IMG)):
    jpgs = sorted(f.name for f in d.glob('*.jpg'))
    xmls = sorted(f.name for f in d.glob('*.xml'))
    missing_xml = [j[:-4] for j in jpgs if f"{j[:-4]}.xml" not in xmls]
    print(f"{name}: jpg={len(jpgs)} xml={len(xmls)} 配对缺失={len(missing_xml)}")
    if missing_xml:
        print(f"  缺失示例: {missing_xml[:5]}")

shutil.rmtree(OLD_DIR, ignore_errors=True)
print("\n✅ 数据集修复完成！")
