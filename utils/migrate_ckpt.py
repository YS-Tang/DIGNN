#!/usr/bin/env python
"""迁移 DIGNN 旧 checkpoint 到 2026-09 架构清理后的 state_dict 布局。

背景: processor 曾把同一批 pml/iml 层同时注册在外层(processor.atm_bnd_pmls.*)
与 HGC/LCP 内层(processor.hgc.atm_bnd_pmls.*), state_dict 存双份键;
旧版架构(重构出 HGC/LCP 之前)则只有外层键。清理后代码只保留内层路径。

迁移规则(逐键):
1. 剥离 torch.compile 包装前缀 `._orig_mod.` (OptimizedModule.load_state_dict
   会转发到原模块, 剥离后新旧两种包装方式均可加载);
2. 新式 ckpt(外层+内层键并存): 删除外层冗余键, 保留内层;
3. 旧式 ckpt(仅外层键): 重命名为内层路径
   processor.atm_bnd_pmls. -> processor.hgc.atm_bnd_pmls.
   processor.bnd_ang_pmls. -> processor.hgc.bnd_ang_pmls.
   processor.ang_dih_pmls. -> processor.hgc.ang_dih_pmls.
   processor.atm_bnd_imls. -> processor.lcp.atm_bnd_imls.
其余键(encoder/decoder/global_layers/optimizer/epoch 等)原样保留。

用法:
    python migrate_ckpt.py <ckpt目录或单个ckpt> [--backup 备份目录] [--no-backup]
默认先把原文件拷贝到备份目录(默认 <目录>_backup_raw)再原地重写。
"""

import argparse
import glob
import os
import shutil
import sys

import torch

# 外层 -> 内层 的路径映射(作用于 model.processor. 之下)
OUTER_TO_INNER = {
    'processor.atm_bnd_pmls.': 'processor.hgc.atm_bnd_pmls.',
    'processor.bnd_ang_pmls.': 'processor.hgc.bnd_ang_pmls.',
    'processor.ang_dih_pmls.': 'processor.hgc.ang_dih_pmls.',
    'processor.atm_bnd_imls.': 'processor.lcp.atm_bnd_imls.',
}


def migrate_state_dict(sd: dict) -> tuple:
    """返回 (新state_dict, stats dict)。不修改传入的 sd。"""
    # Step 1: 剥离 _orig_mod(兼容 compile_model=True 保存的 ckpt)
    stripped = {}
    for k, v in sd.items():
        nk = k.replace('._orig_mod.', '.')
        stripped[nk] = v

    # Step 2: 判定新式(内外并存)还是旧式(仅外层)
    def has_inner(pref_inner):
        return any(k.startswith('model.' + pref_inner) for k in stripped)

    new_style = all(has_inner(p) for p in
                    ('processor.hgc.atm_bnd_pmls.', 'processor.hgc.bnd_ang_pmls.',
                     'processor.hgc.ang_dih_pmls.', 'processor.lcp.atm_bnd_imls.'))

    n_drop, n_rename = 0, 0
    out = {}
    for k, v in stripped.items():
        matched = None
        for outer, inner in OUTER_TO_INNER.items():
            if f'.{outer}' in k or k.startswith(outer):
                matched = (outer, inner)
                break
        if matched is None:
            out[k] = v
            continue
        outer, inner = matched
        if new_style:
            # 内层等价键已存在, 外层是冗余注册 -> 删除
            n_drop += 1
        else:
            # 旧式架构: 外层是唯一来源 -> 重命名为内层路径
            out[k.replace(outer, inner)] = v
            n_rename += 1
    return out, {'style': 'new' if new_style else 'old',
                 'dropped': n_drop, 'renamed': n_rename}


def migrate_file(path: str) -> dict:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not (isinstance(ckpt, dict) and 'state_dict' in ckpt):
        return {'status': 'skip', 'reason': f"非Lightning ckpt: {type(ckpt).__name__}"}
    sd = ckpt['state_dict']
    new_sd, stats = migrate_state_dict(sd)
    if new_sd == sd:
        stats['status'] = 'clean'
        return stats
    ckpt['state_dict'] = new_sd
    torch.save(ckpt, path)
    stats['status'] = 'migrated'
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', help='ckpt 文件或包含 ckpt 的目录')
    parser.add_argument('--backup', default=None,
                        help='备份目录(默认: <target>_backup_raw)')
    parser.add_argument('--no-backup', action='store_true', help='不备份直接原地重写')
    args = parser.parse_args()

    if os.path.isdir(args.target):
        files = sorted(glob.glob(os.path.join(args.target, '**', '*.ckpt'), recursive=True))
    else:
        files = [args.target]

    backup_dir = args.backup or (args.target.rstrip('/') + '_backup_raw')
    if not args.no_backup and os.path.isdir(args.target) and files:
        os.makedirs(backup_dir, exist_ok=True)

    ok = fail = skip = 0
    for f in files:
        rel = os.path.relpath(f, args.target if os.path.isdir(args.target) else os.path.dirname(f) or '.')
        try:
            # 先做备份(单文件也备份)
            if not args.no_backup:
                dst = os.path.join(backup_dir, rel.replace(os.sep, '__'))
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst) or backup_dir, exist_ok=True)
                    shutil.copy2(f, dst)
            stats = migrate_file(f)
            if stats.get('status') in ('migrated', 'clean'):
                ok += 1
                print(f"[{stats['status']:>8}] {rel} | style={stats['style']} "
                      f"drop={stats['dropped']} rename={stats['renamed']}")
            else:
                skip += 1
                print(f"[  SKIP  ] {rel} | {stats.get('reason')}")
        except Exception as e:
            fail += 1
            print(f"[  FAIL  ] {rel} | {type(e).__name__}: {e}")

    print(f"\n完成: migrated/clean={ok}, skip={skip}, fail={fail}")
    if not args.no_backup and ok + fail > 0:
        print(f"原始文件已备份至: {backup_dir}")
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
