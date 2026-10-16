import os
from pathlib import Path
import json, math, colorsys, warnings
import numpy as np, rasterio
from affine import Affine
from rasterio.windows import Window
from PIL import Image, ImageDraw
from scipy import ndimage as nd
from . import science
P = Path(os.environ.get('CROWN_RUN_ROOT','.')).resolve()
O = P / 'release'

def save_npz(path, arrays):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)

def native_rgb(dom, tf, size):
    with rasterio.open(dom) as src:
        c0, r0 = ~src.transform * (tf.c, tf.f)
        c1, r1 = ~src.transform * (tf.c + size * 0.1, tf.f - size * 0.1)
        c0, r0 = (math.floor(c0), math.floor(r0))
        c1, r1 = (math.ceil(c1), math.ceil(r1))
        win = Window(c0, r0, c1 - c0, r1 - r0)
        rgb = src.read([1, 2, 3], window=win, boundless=True, fill_value=0).transpose(1, 2, 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', rasterio.errors.NodataShadowWarning)
            valid = np.any(src.read_masks([1, 2, 3], window=win, boundless=True) > 0, axis=0)
        alpha = [i + 1 for i, c in enumerate(src.colorinterp) if c.name == 'alpha']
        if alpha:
            valid &= src.read(alpha[0], window=win, boundless=True, fill_value=0) > 0
        return (rgb, valid, src.window_transform(win), [c0, r0, c1 - c0, r1 - r0])

def pack(arrays, w, r, c, size):
    sl = np.s_[r:r + size, c:c + size]
    tf = Affine(*w['transform']) * Affine.translation(c, r)
    ids = arrays['candidate_ids'][sl]
    source_ids = np.unique(ids[ids > 0])
    lut = np.zeros(int(ids.max()) + 1, np.int32)
    lut[source_ids] = np.arange(1, len(source_ids) + 1)
    ids = lut[ids]
    rgb, nv, ntf, win = native_rgb(w['dom'], tf, size)
    valid = arrays['segmentation_valid'][sl]
    result = dict(schema=np.array('sam3_native_dom_dual_grid_v1'), rgb=rgb, rgb_native_valid=nv, rgb_transform=np.asarray(list(ntf)[:6]), label_transform=np.asarray(list(tf)[:6]), native_dom_window=np.asarray(win), native_dom_path=np.array(w['dom']), crs=np.array(w['crs']), label_resolution_m=np.array(0.1), instance_ids=ids, instance_keys=np.arange(1, len(source_ids) + 1, dtype=np.int32), processing_instance_ids=source_ids, rgb_valid=valid.copy(), valid_pixels=valid.copy(), ignore_pixels=~valid, training_ignore_pixels=np.zeros_like(valid), reliable_background=valid & (ids == 0), visible_top_ids=ids.copy(), height=arrays['height'][sl], height_valid=arrays['height_valid'][sl], height_bin=arrays['height_bin'][sl], height_observed_raw=arrays['height_observed_raw'][sl], geographic_forbidden=arrays['geographic_forbidden'][sl])
    assert result['rgb'].dtype == np.uint8 and result['instance_ids'].shape == (size, size)
    assert not np.any(ids[~valid]) and (not np.any(ids[result['geographic_forbidden']]))
    assert np.array_equal(np.unique(ids[ids > 0]), result['instance_keys'])
    assert np.isfinite(result['height'][result['height_valid']]).all()
    assert np.isnan(result['height'][~result['height_valid']]).all()
    return (result, tf, ntf, win)

def preview(a, path, width=1800):
    rgb = a['rgb']
    h, w = rgb.shape[:2]
    factor = min(1, width / max(h, w))
    pw, ph = (round(w * factor), round(h * factor))
    im = Image.fromarray(rgb).resize((pw, ph), Image.Resampling.LANCZOS)
    tf = Affine(*a['rgb_transform']) * Affine.scale(w / pw, h / ph)
    lt = Affine(*a['label_transform'])
    xs = tf.c + (np.arange(pw) + 0.5) * tf.a
    ys = tf.f + (np.arange(ph) + 0.5) * tf.e
    cc = np.floor((xs - lt.c) / 0.1).astype(int)
    rr = np.floor((lt.f - ys) / 0.1).astype(int)
    good = (rr[:, None] >= 0) & (rr[:, None] < a['instance_ids'].shape[0]) & (cc[None, :] >= 0) & (cc[None, :] < a['instance_ids'].shape[1])
    ids = a['instance_ids'][np.ix_(np.clip(rr, 0, a['instance_ids'].shape[0] - 1), np.clip(cc, 0, a['instance_ids'].shape[1] - 1))].copy()
    ids[~good] = 0
    out = np.asarray(im).copy()
    color = np.array([colorsys.hsv_to_rgb(k * 0.61803398875 % 1, 0.65, 1) for k in range(int(ids.max()) + 1)], np.float32) * 255
    fg = ids > 0
    out[fg] = (out[fg] * 0.72 + color[ids[fg]] * 0.28).astype(np.uint8)
    edge = fg & (nd.minimum_filter(ids, 3) != nd.maximum_filter(ids, 3))
    out[edge] = 255
    overlay = Image.fromarray(out)
    draw = ImageDraw.Draw(overlay)
    for ident, sl in enumerate(nd.find_objects(ids), 1):
        if sl is None:
            continue
        yy, xx = np.where(ids[sl] == ident)
        i = len(yy) // 2
        pos = (int(xx[i] + sl[1].start), int(yy[i] + sl[0].start))
        draw.text(pos, str(ident), fill='white', stroke_width=1, stroke_fill='black')
    path.mkdir(parents=True, exist_ok=True)
    im.save(path / 'RGB.jpg', quality=95)
    overlay.save(path / 'IDs.jpg', quality=95)
    return overlay

def export(arrays, w, solver, audits, out):
    name = w['scene']
    b = w['core_offset']
    core, tf, ntf, win = pack(arrays, w, b, b, 1024)
    native = O / 'native' / (name + '.npz')
    save_npz(native, core)
    rv = O / 'review' / name
    im = preview(core, rv)
    im.thumbnail((360, 360))
    im.save(rv / 'thumb.jpg', quality=90)
    crops = []
    pairs = []
    for letter, (r, c) in zip('ABCD', [(0, 0), (0, 512), (512, 0), (512, 512)]):
        cn = name + '_' + letter
        a, ctf, rtf, cwin = pack(arrays, w, b + r, b + c, 512)
        dest = O / 'crops' / (cn + '.npz')
        save_npz(dest, a)
        assert np.array_equal(a['instance_ids'] > 0, core['instance_ids'][r:r + 512, c:c + 512] > 0)
        assert np.array_equal(a['height'], core['height'][r:r + 512, c:c + 512], equal_nan=True)
        mapped = a['processing_instance_ids']
        coremap = {int(k): i + 1 for i, k in enumerate(core['processing_instance_ids'])}
        coreids = [coremap[int(k)] for k in mapped]
        row = dict(name=cn, scene=cn, core=name, uav=w['uav'], split='train', origin=[r, c], size=512, label_shape=[512, 512], native_path=str(dest), native_sha256=science.sha(dest), transform=list(ctf)[:6], crs=w['crs'], bounds=[ctf.c, ctf.f - 51.2, ctf.c + 51.2, ctf.f], rgb_shape=list(a['rgb'].shape[:2]), rgb_transform=list(rtf)[:6], native_dom_window=cwin, core_instance_ids=coreids, targets=len(mapped), foreground_pixels=int((a['instance_ids'] > 0).sum()), valid_pixels=int(a['valid_pixels'].sum()), query_capacity_reference=200, query_capacity_ok=len(mapped) <= 200, IDs_truncated=False, loader=str(P / 'native_dom_loader.py'), model_input_size=1008)
        science.write(rv / (cn + '.json'), row)
        crops.append(row)
        preview(a, rv / cn, width=1300)
        pairs.append(f'<section><h2>{cn} · {len(mapped)} IDs</h2><div class="pair"><img loading="lazy" src="{cn}/RGB.jpg"><img loading="lazy" src="{cn}/IDs.jpg"></div></section>')
    rec = dict(scene=name, scene_id=name, name=name, uav=w['uav'], split='train', spatial_group=w['spatial_group'], native_path=str(native), native_sha256=science.sha(native), size=1024, crop_size=512, crop_origins=[[0, 0], [0, 512], [512, 0], [512, 512]], resolution=0.1, crs=w['crs'], transform=list(tf)[:6], bounds=w['core_bounds'], processing_bounds=w['bounds'], buffer_m=b * 0.1, core_instances=len(core['instance_keys']), crops=crops, converged=solver['converged_full_pass'], schema='sam3_native_dom_dual_grid_v1', DOM_native=True, rgb_shape=list(core['rgb'].shape[:2]), rgb_transform=list(ntf)[:6], initialization='fresh_leaf_only', method='V1.2_hierarchical_joint_Gaussian_cost512_gate07', candidate_admission_min_vertical_ratio=0.7, active_instance_cost=512, graph_strength=1.5, old_generator_called=False, audits=audits, review_required=True, SAM3_run=False)
    science.write(rv / 'summary.json', rec)
    doc = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>' + name + '</title><style>body{font:17px/1.6 system-ui;margin:24px;background:#f5f7fa;color:#18252c}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%}section{margin:24px 0}</style><a href="../../index.html">返回全部窗口</a><h1>' + name + f""" · {len(core['instance_keys'])} 个核心实例</h1><p>新方法：0.1m体素、直接2m树顶、层级联合高斯与三项能量；完整收敛后清碎屑、填小孔、修轮廓。农田/电线为背景，无分割unknown。DOM在NPZ中保留原生像素，页面缩略图仅用于显示。</p><div class="pair"><img src="RGB.jpg"><img src="IDs.jpg"></div>""" + ''.join(pairs) + '</html>'
    (rv / 'index.html').write_text(doc)
    science.write(out / 'result.json', rec)
    return rec
