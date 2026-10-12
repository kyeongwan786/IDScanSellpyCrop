#!/usr/bin/env python3
"""
신분증 사진 -> 8.1cm x 5.1cm (셀피 인쇄용) 자동 크롭/펴기

- 어떤 배경(흰색/검정/나무/카펫 등)이든 신분증 사각형을 자동 감지
- 기울어짐/회전/원근 왜곡을 펴서 신분증이 8.1x5.1cm 캔버스에 꽉 차게 저장
- 얼굴 사진 위치로 180도 뒤집힘 자동 보정 (OpenCV 얼굴 검출)
- DPI 정보를 파일에 넣어 인쇄 시 실제 크기가 8.1x5.1cm 가 되도록 함

설치:  pip install opencv-python pillow numpy

사용법:
  python id_card_crop.py 사진.jpg                  -> 사진_selphy.png
  python id_card_crop.py ./원본폴더 -o ./결과폴더    -> 폴더 일괄 처리
  python id_card_crop.py 사진.jpg --debug          -> 감지 영역 표시 *_debug.jpg 도 저장
  python id_card_crop.py 사진.jpg --rotate 180     -> 방향이 틀리면 수동 회전
  python id_card_crop.py 사진.jpg --shrink 0.05    -> 안쪽 여백 조절 (기본 0.03)
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

TARGET_W_CM = 8.1
TARGET_H_CM = 5.1
CARD_RATIO = 85.6 / 53.98  # 신분증(ID-1) 가로세로비 ≈ 1.586
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WORK = 1000.0  # 감지용 작업 해상도(긴 변)


# ------------------------------------------------------------------ 유틸
def order_points(pts):
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def quad_sizes(q):
    tl, tr, br, bl = q
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return w, h


def load_image(path):
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------ 후보 마스크
def _auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    return cv2.Canny(gray, int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v)))


def _candidate_masks(small):
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sb = cv2.GaussianBlur(small, (5, 5), 0)
    masks = [cv2.Canny(blur, 30, 100), _auto_canny(blur), cv2.Canny(blur, 15, 50)]

    ec = np.zeros_like(gray)
    for ch in cv2.split(sb):
        ec = cv2.bitwise_or(ec, cv2.Canny(ch, 25, 80))
    masks.append(ec)

    _, o = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks += [o, cv2.bitwise_not(o)]
    masks.append(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 7))
    s = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))[1]
    _, so = cv2.threshold(cv2.GaussianBlur(s, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks += [so, cv2.bitwise_not(so)]

    # 이미지 테두리 색과 얼마나 다른지 (흰 종이/단색 배경용)
    lab = cv2.cvtColor(sb, cv2.COLOR_BGR2LAB).astype(np.float32)
    bw = max(4, int(0.03 * min(gray.shape)))
    border = np.concatenate([lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3),
                             lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    dist = np.clip(dist * 4, 0, 255).astype(np.uint8)
    _, dm = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(dm)
    return masks


# ------------------------------------------------------------------ 후보 채점
def _edge_contrast(lab, quad, off=4, n=40):
    """사각형 각 변의 안쪽/바깥쪽 색 차이 (신분증 경계는 대비가 뚜렷함)"""
    h, w = lab.shape[:2]
    c = quad.mean(axis=0)
    vals = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        L = np.linalg.norm(p1 - p0)
        if L < 5:
            return 0.0
        d = (p1 - p0) / L
        nrm = np.array([-d[1], d[0]])
        if np.dot(nrm, c - (p0 + p1) / 2) < 0:
            nrm = -nrm
        pts = p0[None] + np.linspace(0.1, 0.9, n)[:, None] * (p1 - p0)[None]
        a = pts + nrm * off
        b = pts - nrm * off
        a = np.clip(a, [0, 0], [w - 1, h - 1]).astype(int)
        b = np.clip(b, [0, 0], [w - 1, h - 1]).astype(int)
        diff = np.linalg.norm(lab[a[:, 1], a[:, 0]] - lab[b[:, 1], b[:, 0]], axis=1)
        vals.append(min(np.percentile(diff, 40), 40) / 40)
    vals = sorted(vals)
    return 0.5 * np.mean(vals) + 0.5 * vals[1]  # 한 변 정도는 약해도 허용


def _score_quad(quad, lab, hull_area):
    q = order_points(quad)
    h, w = lab.shape[:2]
    img_area = h * w
    ww, hh = quad_sizes(q)
    if min(ww, hh) < 20:
        return -1
    ratio = max(ww, hh) / min(ww, hh)
    if not (1.25 <= ratio <= 2.0):
        return -1
    qa = cv2.contourArea(q)
    frac = qa / img_area
    if frac < 0.04 or frac > 0.92:
        return -1
    if hull_area / max(qa, 1) < 0.80:  # 윤곽이 사각형과 안 맞으면 제외
        return -1
    ratio_sc = 1 - min(abs(ratio - CARD_RATIO) / 0.5, 0.85)
    contrast = _edge_contrast(lab, q)
    # 이미지 테두리에 붙은 꼭짓점 감점
    m = 0.012 * max(h, w)
    touch = sum((p[0] < m or p[1] < m or p[0] > w - m or p[1] > h - m) for p in q)
    border_pen = 0.45 ** touch
    return (contrast ** 1.5) * ratio_sc * (frac ** 0.4) * border_pen


def find_card_quad(img, return_score=False):
    h, w = img.shape[:2]
    scale = min(1.0, WORK / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else img.copy()
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    img_area = small.shape[0] * small.shape[1]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    best, best_sc, best_cnt = None, 0.0, None
    for m in _candidate_masks(small):
        m = cv2.dilate(m, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
        for c in cnts:
            if cv2.contourArea(c) < 0.04 * img_area:
                continue
            hull = cv2.convexHull(c)
            ha = cv2.contourArea(hull)
            peri = cv2.arcLength(hull, True)
            quads = []
            for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    quads.append(ap.reshape(4, 2).astype("float32"))
                    break
            quads.append(cv2.boxPoints(cv2.minAreaRect(hull)))
            for qd in quads:
                sc = _score_quad(qd, lab, ha)
                if sc > best_sc:
                    best, best_sc, best_cnt = qd, sc, hull
    if best is None:
        return (None, None, 0.0) if return_score else (None, None)
    result = (order_points(best / scale),
              best_cnt.reshape(-1, 2).astype('float32') / scale)
    return (*result, float(best_sc)) if return_score else result


# ------------------------------------------------------------------ 꼭짓점 정밀 보정
def _fit_line(pts):
    """Huber 직선 피팅 + 이상치 제거 후 재피팅. 반환: (점, 방향)"""
    pts = pts.astype(np.float32)
    for _ in range(3):
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        keep = d < max(1.5, np.percentile(d, 70))
        if keep.sum() < 15:
            break
        pts = pts[keep]
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(l1, l2):
    p, d = l1
    q, e = l2
    A = np.array([d, -e]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def _ransac_line(P, ang0, tol, iters=500, max_ang_deg=6.0, min_span=0.35, L=1.0, rng=None):
    """점들 중 가장 많은 점이 일직선으로 놓인 직선을 RANSAC 으로 찾음.
    (가려지거나 휘어진 구간보다, 길고 곧은 구간을 우선하므로 기울기가 안 틀어짐)"""
    rng = rng or np.random.default_rng(0)
    n = len(P)
    best_cnt, best = 0, None
    d0 = np.array([np.cos(ang0), np.sin(ang0)])
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        v = P[j] - P[i]
        nv = np.linalg.norm(v)
        if nv < min_span * L:
            continue
        v = v / nv
        if abs(v @ d0) < np.cos(np.deg2rad(max_ang_deg)):
            continue
        dist = np.abs((P[:, 0] - P[i, 0]) * v[1] - (P[:, 1] - P[i, 1]) * v[0])
        cnt = int((dist < tol).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (P[i], v, dist < tol)
    if best is None:
        return None
    return best


def _densify(poly, step=1.0):
    pts = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        k = max(1, int(np.linalg.norm(b - a) / step))
        pts.append(a[None] + np.linspace(0, 1, k, endpoint=False)[:, None] * (b - a)[None])
    return np.concatenate(pts).astype(np.float32)


def _poly_iou(a, b, shape):
    sc = 600.0 / max(shape[:2])
    h, w = int(shape[0] * sc) + 2, int(shape[1] * sc) + 2
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [(a * sc).astype(np.int32)], 1)
    cv2.fillPoly(mb, [(b * sc).astype(np.int32)], 1)
    u = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum()) / max(u, 1)


def refine_quad_silhouette(img, quad, sil):
    """카드의 바깥 윤곽선(silhouette) 점만 사용해 변마다 직선을 맞춤.
    카드 안쪽의 인쇄 선/글자 줄은 윤곽선에 없으므로 영향을 주지 않음."""
    try:
        H, W = img.shape[:2]
        P = _densify(sil)
        band = max(6.0, 0.02 * max(H, W))
        tol = max(1.5, 0.002 * max(H, W))
        lines = []
        for i in range(4):
            p0, p1 = quad[i], quad[(i + 1) % 4]
            d = p1 - p0
            L = float(np.sqrt(d @ d))
            t = ((P - p0) @ d) / (L * L)
            sd = np.abs((P[:, 0] - p0[0]) * d[1] - (P[:, 1] - p0[1]) * d[0]) / L
            sel = (t > 0.12) & (t < 0.88) & (sd < band)
            if sel.sum() < 30:
                return quad
            Ps = P[sel]
            r = _ransac_line(Ps, np.arctan2(d[1], d[0]), tol, max_ang_deg=4.0,
                             min_span=0.3, L=L)
            if r is None:
                return quad
            _, _, inl = r
            if inl.sum() < 30:
                return quad
            lines.append(_fit_line(Ps[inl]))
        new = []
        for i in range(4):
            pt = _intersect(lines[(i - 1) % 4], lines[i])
            if pt is None or np.linalg.norm(pt - quad[i]) > 0.04 * max(H, W):
                return quad
            new.append(pt)
        new = np.array(new, dtype="float32")
        # 윤곽선과의 겹침(IoU)이 오히려 나빠지면 원래 사각형 유지
        if _poly_iou(new, sil, img.shape) + 0.003 < _poly_iou(quad, sil, img.shape):
            return quad
        return new
    except Exception:
        return quad


def refine_quad(img, quad):
    """실제 가장자리 픽셀에 직선을 RANSAC 으로 맞춰 꼭짓점을 정밀 보정"""
    try:
        H, W = img.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        edges = np.zeros_like(gray)
        for ch in cv2.split(cv2.GaussianBlur(img, (3, 3), 0)) + (gray,):
            edges = cv2.bitwise_or(edges, _auto_canny(ch))
        ys, xs = np.nonzero(edges)
        P = np.stack([xs, ys], axis=1).astype(np.float32)
        band = max(8.0, 0.02 * max(H, W))
        tol = max(2.0, 0.003 * max(H, W))
        lines = []
        for i in range(4):
            p0, p1 = quad[i], quad[(i + 1) % 4]
            d = p1 - p0
            L = float(np.sqrt(d @ d))
            t = ((P - p0) @ d) / (L * L)
            sd = np.abs((P[:, 0] - p0[0]) * d[1] - (P[:, 1] - p0[1]) * d[0]) / L
            sel = (t > 0.03) & (t < 0.97) & (sd < band)
            if sel.sum() < 40:
                return quad
            Ps = P[sel]
            r = _ransac_line(Ps, np.arctan2(d[1], d[0]), tol, L=L)
            if r is None:
                return quad
            _, _, inl = r
            if inl.sum() < 40:
                return quad
            lines.append(_fit_line(Ps[inl]))
        new = []
        for i in range(4):
            pt = _intersect(lines[(i - 1) % 4], lines[i])
            if pt is None or np.linalg.norm(pt - quad[i]) > 0.05 * max(H, W):
                return quad
            new.append(pt)
        return np.array(new, dtype="float32")
    except Exception:
        return quad


# ------------------------------------------------------------------ 변환
def inset_quad(quad, shrink):
    """사각형을 중심 기준으로 안쪽으로 줄임 (shrink=0.03 -> 각 변이 가로/세로의 1.5%씩 안쪽)"""
    c = quad.mean(axis=0)
    return (c + (quad - c) * (1 - shrink)).astype("float32")


def warp_to_target(img, quad, out_w, out_h, shrink=0.0):
    q = order_points(quad)
    w, h = quad_sizes(q)
    if h > w:  # 세로로 서 있으면 가로가 되도록 4점 순서를 돌림
        q = np.array([q[3], q[0], q[1], q[2]], dtype="float32")
    if shrink > 0:
        c = q.mean(axis=0)
        q = c + (q - c) * (1 - shrink)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------ 평면화(rectify) 후 재보정
FLAT_W = 1200                      # 평면화 작업 해상도(카드 가로 px)
FLAT_H = int(round(FLAT_W / CARD_RATIO))
FLAT_M = int(round(0.10 * FLAT_H))  # 카드 주변 여백(변을 찾기 위한 배경 포함)


def landscape_quad(q):
    """4점을 [좌상,우상,우하,좌하] 로 정렬하고, 세로로 서 있으면 가로가 되도록 돌림"""
    q = order_points(q)
    w, h = quad_sizes(q)
    if h > w:
        q = np.array([q[3], q[0], q[1], q[2]], dtype="float32")
    return q


def flat_dst():
    m, W, H = FLAT_M, FLAT_W, FLAT_H
    return np.array([[m, m], [m + W, m], [m + W, m + H], [m, m + H]], dtype="float32")


def rectify(img, q):
    """카드(4점)를 정면 사각형으로 평면화. 반환: (flat 이미지, 원본->flat 호모그래피)"""
    Hm = cv2.getPerspectiveTransform(q.astype("float32"), flat_dst())
    size = (FLAT_W + 2 * FLAT_M, FLAT_H + 2 * FLAT_M)
    flat = cv2.warpPerspective(img, Hm, size, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return flat, Hm


def _edge_samples(G, side, nsamp=48, win=6):
    """평면화 이미지에서 한 변 근처를 따라가며 경계(가장 바깥쪽의 뚜렷한 기울기) 위치를 샘플링"""
    m, W, H = FLAT_M, FLAT_W, FLAT_H
    band = int(0.06 * H)
    pts = []
    for t in np.linspace(0.12, 0.88, nsamp):
        if side in ("top", "bottom"):
            x = int(m + t * W)
            y0 = (m if side == "top" else m + H) - band
            prof = G[y0:y0 + 2 * band + 1, x - win:x + win + 1].mean(axis=1)
        else:
            y = int(m + t * H)
            x0 = (m if side == "left" else m + W) - band
            prof = G[y - win:y + win + 1, x0:x0 + 2 * band + 1].mean(axis=0)
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
        thr = max(0.45 * prof.max(), 2.0 * np.median(prof) + 1e-3)
        peaks = [i for i in range(1, len(prof) - 1)
                 if prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] >= thr]
        if not peaks:
            continue
        i = peaks[0] if side in ("top", "left") else peaks[-1]  # 바깥쪽 우선
        if side in ("top", "bottom"):
            pts.append((x, y0 + i))
        else:
            pts.append((x0 + i, y))
    return np.array(pts, dtype=np.float32)


def refine_in_flat(flat):
    """평면 상태에서는 카드 변이 수평/수직이어야 함 -> 각 변을 직선으로 맞춰 4 꼭짓점(평면 좌표) 반환"""
    m, W, H = FLAT_M, FLAT_W, FLAT_H
    lab = cv2.cvtColor(cv2.GaussianBlur(flat, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    gx = sum(np.abs(cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3)) for c in range(3))
    gy = sum(np.abs(cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3)) for c in range(3))
    default = {"top": (np.array([0.0, m]), np.array([1.0, 0.0])),
               "bottom": (np.array([0.0, m + H]), np.array([1.0, 0.0])),
               "left": (np.array([m, 0.0]), np.array([0.0, 1.0])),
               "right": (np.array([m + W, 0.0]), np.array([0.0, 1.0]))}
    lines = {}
    for side in ("top", "right", "bottom", "left"):
        G = gy if side in ("top", "bottom") else gx
        P = _edge_samples(G, side)
        ang0 = 0.0 if side in ("top", "bottom") else np.pi / 2
        L = W if side in ("top", "bottom") else H
        line = None
        if len(P) >= 14:
            r = _ransac_line(P, ang0, 2.0, iters=400, max_ang_deg=5.0, min_span=0.3, L=L)
            if r is not None and r[2].sum() >= max(10, 0.4 * len(P)):
                line = _fit_line(P[r[2]])
        lines[side] = line if line is not None else default[side]
    tl = _intersect(lines["top"], lines["left"])
    tr = _intersect(lines["top"], lines["right"])
    br = _intersect(lines["bottom"], lines["right"])
    bl = _intersect(lines["bottom"], lines["left"])
    if any(v is None for v in (tl, tr, br, bl)):
        return flat_dst()
    out = np.array([tl, tr, br, bl], dtype=np.float32)
    # 초기값(카드 사각형)에서 너무 멀어지면 신뢰하지 않음
    if np.linalg.norm(out - flat_dst(), axis=1).max() > 0.07 * H:
        return flat_dst()
    return out


def flatten_and_refine(img, quad, iters=2):
    """평면화 -> 평면에서 변 재측정 -> 원본 좌표로 되돌려 꼭짓점 갱신 (반복)"""
    q = landscape_quad(quad)
    for _ in range(iters):
        flat, Hm = rectify(img, q)
        c_flat = refine_in_flat(flat)
        Hinv = np.linalg.inv(Hm)
        q = cv2.perspectiveTransform(c_flat.reshape(-1, 1, 2), Hinv).reshape(4, 2).astype("float32")
        if np.linalg.norm(c_flat - flat_dst(), axis=1).max() < 0.8:  # 수렴
            break
    return q


def crop_from_flat(img, q, out_w, out_h, shrink):
    """평면 좌표계에서 안쪽 여백을 준 뒤, 원본에서 한 번에 8.1x5.1 로 변환(재샘플링 1회)"""
    flat_c = flat_dst()
    c = flat_c.mean(axis=0)
    inner = (c + (flat_c - c) * (1 - shrink)).astype("float32")
    Hm = cv2.getPerspectiveTransform(q.astype("float32"), flat_c)
    inner_orig = cv2.perspectiveTransform(inner.reshape(-1, 1, 2), np.linalg.inv(Hm)).reshape(4, 2)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(inner_orig.astype("float32"), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE), inner_orig.astype("float32")


# ------------------------------------------------------------------ 방향(180도) 판별
_FACE = None


def _face_score(card):
    """바로 선 얼굴이 검출되면 점수를 반환. 검출기를 쓸 수 없으면 0"""
    global _FACE
    try:
        if _FACE is None:
            if not (hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")):
                _FACE = []
            else:
                _FACE = [cv2.CascadeClassifier(cv2.data.haarcascades + n) for n in
                         ("haarcascade_frontalface_default.xml",
                          "haarcascade_frontalface_alt2.xml")]
                _FACE = [d for d in _FACE if not d.empty()]
        if not _FACE:
            return 0.0
        gray = cv2.equalizeHist(cv2.cvtColor(card, cv2.COLOR_BGR2GRAY))
        best = 0.0
        for det in _FACE:
            for sf, mn in ((1.05, 3), (1.1, 2)):
                res = det.detectMultiScale3(gray, scaleFactor=sf, minNeighbors=mn,
                                            minSize=(35, 35), outputRejectLevels=True)
                if len(res) == 3 and len(res[0]):
                    best = max(best, float(np.max(res[2])) + 0.002 * float(
                        np.max(res[0][:, 2])))
        return best
    except Exception:
        return 0.0


def _tess_rotation(card):
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(card, output_type=pytesseract.Output.DICT)
        return int(osd.get("rotate", 0))
    except Exception:
        return None


def auto_orient(card):
    """card 는 가로형. 0도 / 180도 중 올바른 쪽을 고름. (확신 없으면 그대로)"""
    flipped = cv2.rotate(card, cv2.ROTATE_180)
    a, b = _face_score(card), _face_score(flipped)
    if max(a, b) > 0 and abs(a - b) > 1e-6:
        return card if a > b else flipped
    r = _tess_rotation(card)
    if r == 180:
        return flipped
    if max(a, b) == 0 and r is None:
        print("   (방향 자동 판별 불가: 결과가 거꾸로면 --rotate 180 사용)")
    return card


# ------------------------------------------------------------------ 처리
def process(path, out_path, dpi=300, shrink=0.03, rotate=None, debug=False):
    img = load_image(path)
    quad, sil = find_card_quad(img)
    if quad is None:
        print(f"[실패] 신분증 영역을 못 찾음: {path}")
        return False
    quad = refine_quad_silhouette(img, quad, sil)      # 1) 대략적 4점
    quad = flatten_and_refine(img, quad)               # 2) 평면화 후 변 재측정(반복)

    out_w = int(round(TARGET_W_CM / 2.54 * dpi))
    out_h = int(round(TARGET_H_CM / 2.54 * dpi))
    card, crop_quad = crop_from_flat(img, quad, out_w, out_h, shrink)  # 3) 평면 기준 안쪽 여백 + 변환

    if rotate is None:
        card = auto_orient(card)
    elif rotate == 180:
        card = cv2.rotate(card, cv2.ROTATE_180)

    pil = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    if os.path.splitext(out_path)[1].lower() in (".jpg", ".jpeg"):
        pil.save(out_path, quality=100, subsampling=0, dpi=(dpi, dpi))
    else:
        pil.save(out_path, dpi=(dpi, dpi))
    print(f"[완료] {path} -> {out_path} ({out_w}x{out_h}px @ {dpi}dpi)")

    if debug:
        dbg = img.copy()
        t = max(3, img.shape[1] // 300)
        cv2.polylines(dbg, [quad.astype(int).reshape(-1, 1, 2)], True, (0, 255, 255), max(1, t // 2))
        cv2.polylines(dbg, [crop_quad.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), t)
        for p in crop_quad:
            cv2.circle(dbg, tuple(int(v) for v in p), t * 3, (0, 0, 255), -1)
        cv2.imwrite(os.path.splitext(out_path)[0] + "_debug.jpg", dbg)
        flat, _ = rectify(img, quad)
        cv2.imwrite(os.path.splitext(out_path)[0] + "_flat.jpg", flat)
    return True


def main():
    ap = argparse.ArgumentParser(description="신분증 -> 8.1x5.1cm 셀피 인쇄용 자동 크롭")
    ap.add_argument("input", help="이미지 파일 또는 폴더")
    ap.add_argument("-o", "--output", help="출력 파일/폴더")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--shrink", type=float, default=0.03,
                    help="카드 가장자리에서 안쪽으로 줄이는 비율 (기본 0.03: 각 변 약 1.5%%/1.3mm "
                         "안쪽). 배경이 아직 보이면 0.05, 더 넓게 쓰려면 0.02")
    ap.add_argument("--rotate", type=int, choices=[0, 180], default=None,
                    help="자동 방향판별 대신 수동 지정 (0 또는 180)")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    if os.path.isdir(a.input):
        out_dir = a.output or os.path.join(a.input, "selphy_out")
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in sorted(os.listdir(a.input))
                 if os.path.splitext(f)[1].lower() in EXTS]
        ok = sum(process(os.path.join(a.input, f),
                         os.path.join(out_dir, os.path.splitext(f)[0] + "_selphy.png"),
                         a.dpi, a.shrink, a.rotate, a.debug) for f in files)
        print(f"\n{ok}/{len(files)} 장 처리 완료")
    else:
        out = a.output or os.path.splitext(a.input)[0] + "_selphy.png"
        if not process(a.input, out, a.dpi, a.shrink, a.rotate, a.debug):
            sys.exit(1)


if __name__ == "__main__":
    main()
