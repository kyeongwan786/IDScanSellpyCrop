#!/usr/bin/env python3
"""
신분증 사진 -> 8.1cm x 5.2cm (셀피 인쇄용) 자동 크롭/펴기

- 어떤 배경(흰색/검정/나무/카펫 등)이든 신분증 사각형을 자동 감지
- 기울어짐/회전/원근 왜곡을 펴서 신분증이 8.1x5.2cm 캔버스에 꽉 차게 저장
- 얼굴 사진 위치로 180도 뒤집힘 자동 보정 (OpenCV 얼굴 검출)
- DPI 정보를 파일에 넣어 인쇄 시 실제 크기가 8.1x5.2cm 가 되도록 함

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
TARGET_H_CM = 5.2
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


# ------------------------------------------------------------------ 직선 기반 후보 + 변 지지도 채점
def _grad_maps(small):
    """Lab 각 채널의 Sobel 기울기 + 크기(mag), 임계값 T"""
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    gxs = [cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)]
    gys = [cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)]
    # 구조 텐서: 임의 방향 n 의 기울기 제곱합 = n0²Jxx + 2n0n1Jxy + n1²Jyy (채널 6개 대신 3개만 조회)
    Jxx = sum(gx * gx for gx in gxs)
    Jyy = sum(gy * gy for gy in gys)
    Jxy = sum(gx * gy for gx, gy in zip(gxs, gys))
    mag = np.sqrt(Jxx + Jyy)
    T = max(22.0, 0.30 * float(np.percentile(mag, 97)))
    return gxs, gys, mag, T, (Jxx, Jxy, Jyy)


def _side_support(maps, p0, p1, n=48, reach=3):
    """변(p0->p1)을 따라, 변에 수직인 방향의 기울기가 충분히 큰 지점의 비율 (0~1)"""
    _, _, mag, T, (Jxx, Jxy, Jyy) = maps
    h, w = mag.shape
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 8:
        return 0.0
    d = v / L
    nrm = np.array([-d[1], d[0]])
    pts = p0[None] + np.linspace(0.06, 0.94, n)[:, None] * v[None]
    best = np.zeros(n, np.float32)
    for k in range(-reach, reach + 1):
        q = pts + nrm * k
        xi = np.clip(np.rint(q[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(q[:, 1]).astype(int), 0, h - 1)
        gn2 = nrm[0] * nrm[0] * Jxx[yi, xi] + 2 * nrm[0] * nrm[1] * Jxy[yi, xi] + nrm[1] * nrm[1] * Jyy[yi, xi]
        best = np.maximum(best, gn2)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1)
    return float(((np.sqrt(best) >= T) & inside).mean())


def _side_support_batch(maps, P0, P1, n=48, reach=3):
    """_side_support 를 변 K개에 대해 한 번에 계산 (P0, P1: (K,2))"""
    _, _, mag, T, (Jxx, Jxy, Jyy) = maps
    h, w = mag.shape
    V = P1 - P0
    L = np.linalg.norm(V, axis=1)
    Ls = np.maximum(L, 1e-6)
    N = np.stack([-V[:, 1] / Ls, V[:, 0] / Ls], axis=1)
    pts = P0[:, None, :] + np.linspace(0.06, 0.94, n)[None, :, None] * V[:, None, :]
    a, b, c = (N[:, 0] ** 2)[:, None], (2 * N[:, 0] * N[:, 1])[:, None], (N[:, 1] ** 2)[:, None]
    best = np.zeros(pts.shape[:2], np.float32)
    for k in range(-reach, reach + 1):
        q = pts + N[:, None, :] * k
        xi = np.clip(np.rint(q[..., 0]).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(q[..., 1]).astype(np.int32), 0, h - 1)
        best = np.maximum(best, a * Jxx[yi, xi] + b * Jxy[yi, xi] + c * Jyy[yi, xi])
    inside = (pts[..., 0] >= 0) & (pts[..., 0] <= w - 1) & (pts[..., 1] >= 0) & (pts[..., 1] <= h - 1)
    s = ((best >= T * T) & inside).mean(axis=1)
    s[L < 8] = 0.0
    return s


def _score_quads2_batch(quads, maps):
    """_score_quad2 (hull 없음) 를 여러 사각형에 대해 한 번에 계산. 반환: 점수 배열 (탈락은 -1)"""
    if len(quads) == 0:
        return np.zeros(0)
    h, w = maps[2].shape
    Q = np.stack([order_points(q) for q in quads]).astype(np.float32)
    e = np.linalg.norm(Q - np.roll(Q, -1, axis=1), axis=2)
    ww, hh = (e[:, 0] + e[:, 2]) / 2, (e[:, 1] + e[:, 3]) / 2
    ratio = np.maximum(ww, hh) / np.maximum(np.minimum(ww, hh), 1e-6)
    x, y = Q[..., 0], Q[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1))
    frac = area / (h * w)
    ok = (np.minimum(ww, hh) >= 20) & (ratio >= 1.25) & (ratio <= 2.0) & (frac >= 0.02) & (frac <= 0.95)
    out = np.full(len(Q), -1.0)
    if not ok.any():
        return out
    Qo = Q[ok]
    sups = np.sort(_side_support_batch(maps, Qo.reshape(-1, 2), np.roll(Qo, -1, axis=1).reshape(-1, 2))
                   .reshape(-1, 4), axis=1)
    sc = 0.5 * sups.mean(axis=1) + 0.5 * sups[:, 1]
    ratio_sc = np.exp(-0.5 * ((ratio[ok] - CARD_RATIO) / 0.12) ** 2)
    out[ok] = (sc ** 2) * ratio_sc * (frac[ok] ** 0.15)
    return out


def _score_quad2(quad, maps, hull_area=None):
    q = order_points(quad)
    h, w = maps[2].shape
    ww, hh = quad_sizes(q)
    if min(ww, hh) < 20:
        return -1.0
    ratio = max(ww, hh) / min(ww, hh)
    if not (1.25 <= ratio <= 2.0):
        return -1.0
    frac = cv2.contourArea(q) / (h * w)
    if frac < 0.02 or frac > 0.95:
        return -1.0
    if hull_area is not None and hull_area / max(cv2.contourArea(q), 1) < 0.80:
        return -1.0
    sups = sorted(_side_support_batch(maps, q, np.roll(q, -1, axis=0)))
    sc = 0.5 * float(np.mean(sups)) + 0.5 * sups[1]      # 한 변 정도는 약해도 허용
    ratio_sc = float(np.exp(-0.5 * ((ratio - CARD_RATIO) / 0.12) ** 2))
    return (sc ** 2) * ratio_sc * (frac ** 0.15)


def _detect_segments(gray):
    segs = None
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        r = lsd.detect(gray)[0]
        if r is not None:
            segs = r.reshape(-1, 4).astype(np.float64)
    except Exception:
        segs = None
    if segs is None:  # LSD 를 쓸 수 없는 OpenCV 빌드면 Hough 로 대체
        e = cv2.Canny(gray, 40, 120)
        r = cv2.HoughLinesP(e, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=8)
        segs = r.reshape(-1, 4).astype(np.float64) if r is not None else np.zeros((0, 4))
    return segs


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _merge_segments(segs, min_len, max_lines=160):
    """거의 같은 직선 위의 선분들을 하나의 직선으로 합침. 반환: [(점, 방향, tmin, tmax, 총길이)]"""
    if len(segs) == 0:
        return []
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs, L = segs[L >= 0.35 * min_len], L[L >= 0.35 * min_len]
    order = np.argsort(-L)
    clusters = []
    for idx in order:
        x0, y0, x1, y1 = segs[idx]
        d = np.array([x1 - x0, y1 - y0]) / L[idx]
        if d[0] < 0 or (abs(d[0]) < 1e-9 and d[1] < 0):
            d = -d
        nrm = np.array([-d[1], d[0]])
        mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        rho = float(mid @ nrm)
        placed = False
        for c in clusters:
            if abs(_cross2(c["d"], d)) < np.sin(np.deg2rad(3)) and c["d"] @ d > 0 \
                    and abs(rho - float(mid @ np.array([-c["d"][1], c["d"][0]]))) < 4.0 \
                    and abs(float((mid - c["mid"]) @ c["d"])) < c["span"] / 2 + L[idx] / 2 + 6.0 * min_len:   # 끊어진 선분도 같은 직선이면 이어 붙임 (실제 지지도는 채점에서 다시 검사)
                w0 = c["len"]
                nd = c["d"] * w0 + d * L[idx]
                c["d"] = nd / np.linalg.norm(nd)
                c["len"] += L[idx]
                c["pts"].extend([(x0, y0), (x1, y1)])
                P = np.array(c["pts"])
                t = P @ c["d"]
                c["mid"] = P.mean(axis=0)
                c["span"] = float(t.max() - t.min())
                placed = True
                break
        if not placed:
            clusters.append({"d": d, "len": float(L[idx]), "pts": [(x0, y0), (x1, y1)],
                             "mid": mid, "span": float(L[idx])})
    lines = []
    for c in clusters:
        P = np.array(c["pts"])
        d = c["d"]
        nrm = np.array([-d[1], d[0]])
        rho = float(np.mean(P @ nrm))
        t = P @ d
        base = rho * nrm
        lines.append((base, d, float(t.min()), float(t.max()), c["len"]))
    lines = [l for l in lines if l[3] - l[2] >= min_len]
    lines.sort(key=lambda l: -l[4])
    return lines[:max_lines]


def _line_intersection(la, lb):
    A = np.array([la[1], -lb[1]]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, lb[0] - la[0])
    return la[0] + t[0] * la[1]


def _quads_from_lines(lines, shape, min_side):
    """직각으로 만나는 직선쌍(모서리)을 이어서 4-사이클(=사각형) 후보를 만듦"""
    h, w = shape[:2]
    n = len(lines)
    sin22, sin50 = np.sin(np.deg2rad(22)), np.sin(np.deg2rad(50))
    lo_x, hi_x, lo_y, hi_y = -0.1 * w, 1.1 * w, -0.1 * h, 1.1 * h

    def parallel(i, j):
        return abs(_cross2(lines[i][1], lines[j][1])) < sin22

    # 모서리 그래프: 두 직선이 (대략) 직각이고, 교점이 두 선분의 끝 근처에 있어야 함
    corner = {}
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(_cross2(lines[i][1], lines[j][1])) < sin50:
                continue
            P = _line_intersection(lines[i], lines[j])
            if P is None or not (lo_x <= P[0] <= hi_x and lo_y <= P[1] <= hi_y):
                continue
            ok = True
            for ln in (lines[i], lines[j]):
                t = float(P @ ln[1])
                tol = max(25.0, 0.25 * (ln[3] - ln[2]))
                if t < ln[2] - tol or t > ln[3] + tol:
                    ok = False
                    break
            if ok:
                corner[(i, j)] = corner[(j, i)] = P
                adj[i].append(j)
                adj[j].append(i)

    quads, seen = [], set()
    for a in range(n):
        for b in adj[a]:
            for c in adj[b]:
                if c <= a or c == b or not parallel(a, c):
                    continue
                for d in adj[c]:
                    if d <= a or d in (a, b, c) or not parallel(b, d) or d not in adj[a]:
                        continue
                    key = frozenset((a, b, c, d))
                    if key in seen:
                        continue
                    seen.add(key)
                    pts = np.array([corner[(a, b)], corner[(b, c)], corner[(c, d)], corner[(d, a)]],
                                   dtype=np.float32)
                    if cv2.contourArea(pts) < 0.02 * w * h:
                        continue
                    # 각 변이 실제 검출된 선분으로 최소 35% 덮여야 함
                    good = True
                    for ln, p, q in ((lines[b], pts[0], pts[1]), (lines[c], pts[1], pts[2]),
                                     (lines[d], pts[2], pts[3]), (lines[a], pts[3], pts[0])):
                        t0, t1 = sorted((float(p @ ln[1]), float(q @ ln[1])))
                        if t1 - t0 < min_side or min(t1, ln[3]) - max(t0, ln[2]) < 0.35 * (t1 - t0):
                            good = False
                            break
                    if good:
                        quads.append(pts)
    # --- 변이 3개만 검출된 경우(ㄷ자): 마주보는 평행선 a,c 와 이들을 잇는 선 b -> 카드 비율로 나머지 한 변 추정
    for b in range(n):
        for a in adj[b]:
            for c in adj[b]:
                if c <= a or not parallel(a, c):
                    continue
                P1, P2 = corner[(a, b)], corner[(b, c)]
                Lb = float(np.linalg.norm(P2 - P1))
                if Lb < min_side:
                    continue
                for Lo in (Lb * CARD_RATIO, Lb / CARD_RATIO):
                    for sgn in (1.0, -1.0):
                        ua = lines[a][1] * sgn
                        uc = lines[c][1] * (1.0 if float(ua @ lines[c][1]) > 0 else -1.0)
                        P4, P3 = P1 + ua * Lo, P2 + uc * Lo
                        pts = np.array([P1, P2, P3, P4], dtype=np.float32)
                        if any(not (lo_x <= q[0] <= hi_x and lo_y <= q[1] <= hi_y) for q in pts):
                            continue
                        if cv2.contourArea(pts) < 0.02 * w * h:
                            continue
                        # a, c 변이 실제 검출된 선분으로 덮여야 함
                        good = True
                        for ln, p, q in ((lines[a], pts[3], pts[0]), (lines[c], pts[1], pts[2])):
                            t0, t1 = sorted((float(p @ ln[1]), float(q @ ln[1])))
                            if min(t1, ln[3]) - max(t0, ln[2]) < 0.35 * (t1 - t0):
                                good = False
                                break
                        if good:
                            quads.append(pts)
    return quads


# ------------------------------------------------------------------ 사각형 템플릿 탐색 (각도 x 크기 x 위치)
def _side_support_soft(maps, p0, p1, n=40, reach=3):
    """_side_support 의 연속값 버전(미세 조정용): 변을 따라 min(기울기/T, 1) 평균"""
    _, _, mag, T, (Jxx, Jxy, Jyy) = maps
    h, w = mag.shape
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 8:
        return 0.0
    nrm = np.array([-v[1], v[0]]) / L
    pts = p0[None] + np.linspace(0.06, 0.94, n)[:, None] * v[None]
    best = np.zeros(n, np.float32)
    for k in range(-reach, reach + 1):
        q = pts + nrm * k
        xi = np.clip(np.rint(q[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(q[:, 1]).astype(int), 0, h - 1)
        gn2 = nrm[0] * nrm[0] * Jxx[yi, xi] + 2 * nrm[0] * nrm[1] * Jxy[yi, xi] + nrm[1] * nrm[1] * Jyy[yi, xi]
        best = np.maximum(best, gn2)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1)
    return float((np.minimum(np.sqrt(best) / T, 1.0) * inside).mean())


def _ratio_term(q, sigma=0.25):
    ww, hh = quad_sizes(order_points(q))
    r = max(ww, hh) / max(min(ww, hh), 1e-6)
    return float(np.exp(-0.5 * ((r - CARD_RATIO) / sigma) ** 2))


def _obj_from_vals(vals, q=None):
    v = sorted(vals)
    o = 0.5 * float(np.mean(v)) + 0.5 * v[1]
    return o * (_ratio_term(q) if q is not None else 1.0)


def _soft_obj(maps, q):
    return _obj_from_vals([_side_support_soft(maps, q[i], q[(i + 1) % 4]) for i in range(4)], q)


def _valid_card_quad(q):
    ww, hh = quad_sizes(order_points(q))
    if min(ww, hh) < 15:
        return False
    r = max(ww, hh) / min(ww, hh)
    return 1.3 <= r <= 2.0 and cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2))


def _side_support_soft_batch(maps, P0, P1, n=40, reach=3):
    """_side_support_soft 를 변 K개에 대해 한 번에 계산 (P0, P1: (K,2))"""
    _, _, mag, T, (Jxx, Jxy, Jyy) = maps
    h, w = mag.shape
    V = P1 - P0
    L = np.linalg.norm(V, axis=1)
    Ls = np.maximum(L, 1e-6)
    N = np.stack([-V[:, 1] / Ls, V[:, 0] / Ls], axis=1)                     # (K,2)
    ts = np.linspace(0.06, 0.94, n)
    pts = P0[:, None, :] + ts[None, :, None] * V[:, None, :]                 # (K,n,2)
    a, b, c = (N[:, 0] ** 2)[:, None], (2 * N[:, 0] * N[:, 1])[:, None], (N[:, 1] ** 2)[:, None]
    best = np.zeros(pts.shape[:2], np.float32)
    for k in range(-reach, reach + 1):
        q = pts + N[:, None, :] * k
        xi = np.clip(np.rint(q[..., 0]).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(q[..., 1]).astype(np.int32), 0, h - 1)
        best = np.maximum(best, a * Jxx[yi, xi] + b * Jxy[yi, xi] + c * Jyy[yi, xi])
    inside = (pts[..., 0] >= 0) & (pts[..., 0] <= w - 1) & (pts[..., 1] >= 0) & (pts[..., 1] <= h - 1)
    s = (np.minimum(np.sqrt(best) / T, 1.0) * inside).mean(axis=1)
    s[L < 8] = 0.0
    return s


_DIRS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)], np.float32)


def _batch_obj(Q, vals):
    """Q: (M,4,2) 사각형들, vals: (M,4) 변 지지도 -> 목적함수 (M,), 유효하지 않으면 -1"""
    v = np.sort(vals, axis=1)
    o = 0.5 * v.mean(axis=1) + 0.5 * v[:, 1]
    e = np.linalg.norm(Q - np.roll(Q, -1, axis=1), axis=2)                  # 변 길이 (M,4)
    ww, hh = (e[:, 0] + e[:, 2]) / 2, (e[:, 1] + e[:, 3]) / 2
    r = np.maximum(ww, hh) / np.maximum(np.minimum(ww, hh), 1e-6)
    o = o * np.exp(-0.5 * ((r - CARD_RATIO) / 0.25) ** 2)
    # 볼록 + 변 순서 유지 확인 (외적 부호가 모두 같아야 함)
    d1 = np.roll(Q, -1, axis=1) - Q
    d2 = np.roll(Q, -2, axis=1) - np.roll(Q, -1, axis=1)
    cr = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    convex = np.all(cr > 0, axis=1) | np.all(cr < 0, axis=1)
    ok = convex & (np.minimum(ww, hh) >= 15) & (r >= 1.3) & (r <= 2.0)
    return np.where(ok, o, -1.0)


def _optimize_quad(q, maps, steps=(6.0, 3.0), max_iter=4, n=24, reach=2):
    """꼭짓점을 움직여 4변의 경계 지지도를 최대화 (좌표 하강).
    매 반복마다 '4꼭짓점 x 8방향' 32가지 이동을 한 번에(벡터화) 평가해 가장 좋은 것으로 이동"""
    q = order_points(q).astype(np.float32)
    vals = _side_support_soft_batch(maps, q, np.roll(q, -1, axis=0), n=n, reach=reach)
    best = float(_batch_obj(q[None], vals[None])[0])
    for st in steps:
        for _ in range(max_iter * 4):
            Q = np.repeat(q[None], 32, axis=0)
            ci = np.repeat(np.arange(4), 8)
            Q[np.arange(32), ci] += np.tile(_DIRS, (4, 1)) * st
            sp, sc = (ci - 1) % 4, ci                                     # 바뀌는 두 변
            idx = np.arange(32)
            P0 = np.concatenate([Q[idx, sp], Q[idx, sc]])
            P1 = np.concatenate([Q[idx, (sp + 1) % 4], Q[idx, (sc + 1) % 4]])
            sv = _side_support_soft_batch(maps, P0, P1, n=n, reach=reach)
            V = np.repeat(vals[None], 32, axis=0)
            V[idx, sp] = sv[:32]
            V[idx, sc] = sv[32:]
            obj = _batch_obj(Q, V)
            k = int(np.argmax(obj))
            if obj[k] <= best + 1e-4:
                break
            best, q, vals = float(obj[k]), Q[k], V[k]
    return q


def _rect_search(small, work=256, ratios=(CARD_RATIO,), top_k=8, theta_step=4):
    """회전된 사각형 템플릿(고정 비율)을 훑어 4변 지지도가 높은 후보를 찾음. 반환: small 좌표계의 4점 목록"""
    s = work / max(small.shape[:2])
    sm = cv2.resize(small, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gxs, gys, mag, T, (Jxx, Jxy, Jyy) = _grad_maps(sm)
    h, w = mag.shape
    phi = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)           # 기울기(법선) 방향, 범위 (-pi/2, pi/2]
    edge = mag >= T
    D = int(np.ceil(np.hypot(h, w))) + 4
    cx, cy = w / 2.0, h / 2.0
    tol = np.deg2rad(14)

    def angdiff(a, b):
        d = np.abs(a - b) % np.pi
        return np.minimum(d, np.pi - d)

    cands = []
    ker_h = np.ones((7, 1), np.float32)   # 가로변(상/하) 판정용: 세로로 +-3px 팽창
    ker_v = np.ones((1, 7), np.float32)
    for th in np.arange(0, 180, theta_step):
        t = np.deg2rad(th)
        Ah = (edge & (angdiff(phi, t + np.pi / 2) < tol)).astype(np.float32)   # 가로변: 법선이 t+90
        Av = (edge & (angdiff(phi, t) < tol)).astype(np.float32)               # 세로변: 법선이 t
        M = cv2.getRotationMatrix2D((cx, cy), float(th), 1.0)
        M[0, 2] += D / 2 - cx
        M[1, 2] += D / 2 - cy
        Rh = cv2.dilate(cv2.warpAffine(Ah, M, (D, D), flags=cv2.INTER_LINEAR), ker_h)
        Rv = cv2.dilate(cv2.warpAffine(Av, M, (D, D), flags=cv2.INTER_LINEAR), ker_v)
        Ch = np.concatenate([np.zeros((D, 1), np.float32), np.cumsum(Rh, axis=1)], axis=1)  # (D, D+1)
        Cv = np.concatenate([np.zeros((1, D), np.float32), np.cumsum(Rv, axis=0)], axis=0)  # (D+1, D)
        Minv = cv2.invertAffineTransform(M)
        for ratio in ratios:
            wk = 0.12 * work
            while wk <= 1.0 * work:
                ww = int(round(wk))
                hh = int(round(ww / ratio))
                wk *= 1.10
                if hh < 8 or ww >= D - 2 or hh >= D - 2:
                    continue
                xs = np.arange(0, D - ww - 1, 3)
                ys = np.arange(0, D - hh - 1, 3)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                y0 = ys[:, None]
                x0 = xs[None, :]
                y1 = y0 + hh
                x1 = x0 + ww
                top = (Ch[y0, x1] - Ch[y0, x0]) / ww
                bot = (Ch[y1, x1] - Ch[y1, x0]) / ww
                lef = (Cv[y1, x0] - Cv[y0, x0]) / hh
                rig = (Cv[y1, x1] - Cv[y0, x1]) / hh
                top, bot = np.clip(top, 0, 1), np.clip(bot, 0, 1)
                lef, rig = np.clip(lef, 0, 1), np.clip(rig, 0, 1)
                # 4개 중 두 번째로 작은 값 (partition 보다 훨씬 빠름)
                lo1, hi1 = np.minimum(top, bot), np.maximum(top, bot)
                lo2, hi2 = np.minimum(lef, rig), np.maximum(lef, rig)
                second = np.minimum(np.maximum(lo1, lo2), np.minimum(hi1, hi2))
                sc = (0.125 * (top + bot + lef + rig) + 0.5 * second) ** 2
                for _ in range(2):  # 같은 (각도, 크기)에서 서로 떨어진 최고점 2개
                    iy, ix = np.unravel_index(np.argmax(sc), sc.shape)
                    v = float(sc[iy, ix])
                    if v < 0.08:
                        break
                    X0, Y0 = int(xs[ix]), int(ys[iy])
                    cands.append((v, np.array([[X0, Y0], [X0 + ww, Y0], [X0 + ww, Y0 + hh], [X0, Y0 + hh]],
                                              dtype=np.float32), Minv))
                    sc[max(0, iy - hh // 6):iy + hh // 6 + 1, max(0, ix - ww // 6):ix + ww // 6 + 1] = 0
    cands.sort(key=lambda c: -c[0])
    out = []
    for v, qc, Minv in cands:
        q = cv2.transform(qc.reshape(-1, 1, 2), Minv).reshape(4, 2) / s   # 원래 small 좌표
        if any(np.linalg.norm(order_points(q) - order_points(o), axis=1).max() < 0.07 * max(small.shape[:2])
               for o in out):
            continue
        out.append(q.astype(np.float32))
        if len(out) >= top_k:
            break
    return out


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


def _sort_and_prefer_outer(cands, img_area, maps=None):
    """점수순 정렬 후, 1순위 후보를 완전히 감싸면서 점수가 충분히 높고 카드 비율에 맞는
    더 바깥쪽 후보가 있으면 그것을 1순위로 올림 (카드 안쪽 일부만 잡히는 오류 방지).
    단, 바깥 후보의 4변이 모두 실제 경계 위에 있어야 함: 카드를 정확히 잡은 후보가
    배경까지 걸친 '부푼' 사각형(변 일부가 허공에 떠 있음)으로 바뀌는 것을 막음"""
    def rank(c):
        return c[0] * (cv2.contourArea(order_points(c[1])) / img_area) ** 0.15

    def sides_ok(q):
        if maps is None:
            return True
        q = order_points(q).astype(np.float32)
        sup = np.sort(_side_support_batch(maps, q, np.roll(q, -1, axis=0)))
        return sup[0] >= 0.6 and sup.mean() >= 0.85

    cands = sorted(cands, key=lambda c: -rank(c))
    best = cands[0]
    for _ in range(2):
        bq = order_points(best[1]).astype(np.float32)
        ba = cv2.contourArea(bq)
        containers = []
        for c in cands:
            cq = order_points(c[1]).astype(np.float32)
            ca = cv2.contourArea(cq)
            if ca <= ba * 1.15 or ca > ba * 3.5 or c[0] < 0.45 * best[0]:
                continue
            if not sides_ok(cq):
                continue
            inter, _ = cv2.intersectConvexConvex(bq, cq)
            if inter < 0.9 * ba:
                continue
            ww, hh = quad_sizes(cq)
            if abs(max(ww, hh) / max(min(ww, hh), 1e-6) - CARD_RATIO) > 0.3:
                continue
            containers.append(c)
        if not containers:
            break
        best = max(containers, key=lambda c: c[0])
    return [best] + [c for c in cands if c is not best]


def find_card_quad(img, verbose=False, return_score=False):
    h, w = img.shape[:2]
    scale = min(1.0, WORK / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else img.copy()
    sh, sw = small.shape[:2]
    img_area = sh * sw
    maps = _grad_maps(small)
    cands = []  # (score, quad, silhouette_hull or None, source)

    # (A) 직선 기반 후보
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lines = _merge_segments(_detect_segments(gray), min_len=0.06 * max(sh, sw))
    lq = _quads_from_lines(lines, small.shape, min_side=0.05 * max(sh, sw))
    for qd, sc in zip(lq, _score_quads2_batch(lq, maps)):
        if sc > 0:
            cands.append((float(sc), qd, None, "lines"))

    # (C) 사각형 템플릿 탐색 후보 + 꼭짓점 미세 조정
    rect_c = []
    for q0 in _rect_search(small):
        q1 = _optimize_quad(q0, maps)
        sc = _score_quad2(q1, maps)
        if sc > 0:
            rect_c.append((sc, q1))
        # 템플릿은 원근 없는 직사각형이라 사다리꼴로 찍힌 카드보다 10~20px 작게 잡히는 경우가 많음.
        # 좁은 탐색폭(±2px)만으로는 그 거리의 진짜 경계를 못 봐서 엉뚱한 곳으로 흘러감
        # -> 넓은 폭(±8px)으로 먼저 끌어당긴 뒤 좁은 폭으로 다듬은 버전도 후보로 추가
        q2 = _optimize_quad(_optimize_quad(q0, maps, steps=(6.0, 3.0), reach=8),
                            maps, steps=(3.0, 1.5), reach=2)
        sc2 = _score_quad2(q2, maps)
        if sc2 > 0:
            rect_c.append((sc2, q2))
    rect_c.sort(key=lambda t: -t[0])
    for sc, q1 in rect_c:
        cands.append((sc, q1, None, "rect"))
    # 상위 후보를 키운 변형도 시도: 안쪽 인쇄선/사진 프레임에 걸린 작은 사각형 대신 바깥 경계를 찾기 위함
    for sc, q1 in rect_c[:3]:
        c0 = q1.mean(axis=0)
        for f in (1.2, 1.45, 1.75):
            q2 = (c0 + (q1 - c0) * f).astype(np.float32)
            if not _valid_card_quad(q2):
                continue
            q2 = _optimize_quad(q2, maps, steps=(12.0, 6.0, 3.0))
            sc2 = _score_quad2(q2, maps)
            if sc2 > 0:
                cands.append((sc2, q2, None, "rect-grow"))

    # (B) 마스크 윤곽 기반 후보 (배경이 단순할 때 강함)
    opt_starts, dedup_px = [], 0.015 * max(sh, sw)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for m in _candidate_masks(small):
        m = cv2.dilate(m, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
        for c in cnts:
            if cv2.contourArea(c) < 0.02 * img_area:
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
                sil = hull.reshape(-1, 2).astype("float32")
                sc = _score_quad2(qd, maps, hull_area=ha)
                if sc > 0:
                    cands.append((sc, qd, sil, "mask"))
                # 모서리가 크게 둥근 카드는 윤곽 사각형이 실제 변에서 살짝 벗어나 점수가 낮게 나옴
                # -> 변 지지도로 꼭짓점을 미세 조정한 버전도 후보로 추가
                qo = order_points(qd)
                # 마스크 10종에서 거의 같은 사각형이 반복해서 나오므로, 이미 조정한 것과 비슷하면 건너뜀
                if any(np.abs(qo - s0).max() < dedup_px for s0 in opt_starts):
                    continue
                opt_starts.append(qo)
                if _valid_card_quad(qo):
                    qo = _optimize_quad(qo, maps)
                    sc2 = _score_quad2(qo, maps, hull_area=ha)
                    if sc2 > 0:
                        cands.append((sc2, qo, sil, "mask-opt"))

    if not cands:
        return (None, None, 0.0) if return_score else (None, None)
    cands = _sort_and_prefer_outer(cands, img_area, maps)
    if verbose:
        for sc, qd, _, src in cands[:3]:
            print(f"   후보 {src} score={sc:.3f}")
    sc, qd, hull, src = cands[0]
    result = (order_points(qd / scale), (None if hull is None else hull / scale))
    return (*result, float(sc)) if return_score else result


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
FLAT_M = int(round(0.15 * FLAT_W))  # 카드 주변 여백(변을 찾기 위한 배경 포함). 변 재탐색 범위(가로의 10%)보다 넉넉해야 함


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
    band = int(0.05 * H)
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
        i = min(peaks, key=lambda k: abs(k - band))  # 현재 추정 위치에 가장 가까운 경계 (바깥으로 밀려나감 방지)
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


def flatten_and_refine(img, quad, iters=3, max_move=0.06):
    """평면화 -> 평면에서 변 재측정 -> 원본 좌표로 되돌려 꼭짓점 갱신 (반복).
    1차 검출에서 너무 멀어지면(다른 물체의 경계로 이탈) 1차 결과를 그대로 사용"""
    q0 = landscape_quad(quad)
    q = q0.copy()
    for _ in range(iters):
        flat, Hm = rectify(img, q)
        c_flat = refine_in_flat(flat)
        Hinv = np.linalg.inv(Hm)
        q = cv2.perspectiveTransform(c_flat.reshape(-1, 1, 2), Hinv).reshape(4, 2).astype("float32")
        if np.linalg.norm(c_flat - flat_dst(), axis=1).max() < 0.8:  # 수렴
            break
    # 총 이동량 점검: 1차 사각형 기준 평면 좌표에서 측정
    H0 = cv2.getPerspectiveTransform(q0, flat_dst())
    moved = cv2.perspectiveTransform(q.reshape(-1, 1, 2), H0).reshape(4, 2)
    if np.linalg.norm(moved - flat_dst(), axis=1).max() > max_move * FLAT_H:
        return q0
    return q


def _quad_quality(maps, s, q):
    """사각형 4변이 실제 경계(기울기)와 얼마나 잘 맞는지. 가장 나쁜 변에 비중을 둠 -> 한 변만 틀어져도 낮게 나옴"""
    q = (order_points(q) * s).astype(np.float32)
    v = np.sort(_side_support_soft_batch(maps, q, np.roll(q, -1, axis=0), n=80, reach=2))
    return 0.5 * float(np.mean(v)) + 0.5 * float(v[0])


def _flat_side_lines(flat, out_frac=0.10, in_frac=0.25, step=2, top_n=3):
    """평면화 이미지에서 변마다 '양 끝 위치(oa, ob)'를 독립적으로 바꿔가며 직선을 전수 탐색.
    현재 추정 변이 카드 안쪽 인쇄선/접힌 자국 등에 걸렸어도 근처의 진짜 경계를 다시 찾을 수 있음.
    반환: {side: [(score, oa, ob), ...]}  (oa/ob 는 현재 변 위치 기준 오프셋 px, 바깥쪽이 +)"""
    m, W, H = FLAT_M, FLAT_W, FLAT_H
    lab = cv2.cvtColor(cv2.GaussianBlur(flat, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    gx = sum(np.abs(cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3)) for c in range(3))
    gy = sum(np.abs(cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3)) for c in range(3))
    T = max(30.0, 0.35 * float(np.percentile(np.maximum(gx, gy), 97)))
    gyd = np.minimum(cv2.dilate(gy, np.ones((5, 1), np.float32)) / T, 1.0)
    gxd = np.minimum(cv2.dilate(gx, np.ones((1, 5), np.float32)) / T, 1.0)
    fh, fw = flat.shape[:2]
    ts = np.linspace(0.06, 0.94, 64)
    out = {}
    for side in ("top", "right", "bottom", "left"):
        L = W if side in ("top", "bottom") else H
        # 바깥(+)으로는 10%, 안쪽(-)으로는 25% 까지: 1차 감지가 바깥 배경 직선(매트 테두리 등)에 걸려 크게 부푼 경우도 복구
        # 범위는 변이 움직이는 방향의 카드 길이 기준 (위/아래 변은 세로 H, 좌/우 변은 가로 W).
        # 예전처럼 모두 H 기준이면 좌우 변은 가로의 6%밖에 못 움직여, 1차 감지가 카드 끝을 잘라먹었을 때 복구 못 함
        D = H if side in ("top", "bottom") else W
        offs = np.arange(-int(in_frac * D), int(out_frac * D) + 1, step)
        oa, ob = np.meshgrid(offs, offs, indexing="ij")
        sgn = -1 if side in ("top", "left") else 1          # 바깥쪽 방향
        base = {"top": m, "bottom": m + H, "left": m, "right": m + W}[side]
        pa = base + sgn * oa[..., None]
        pb = base + sgn * ob[..., None]
        perp = pa + (pb - pa) * ts                          # 변에 수직인 좌표
        along = (m + ts * L) * np.ones_like(perp)            # 변 방향 좌표
        if side in ("top", "bottom"):
            xs, ys, G = along, perp, gyd
        else:
            xs, ys, G = perp, along, gxd
        xi = np.clip(np.rint(xs).astype(int), 0, fw - 1)
        yi = np.clip(np.rint(ys).astype(int), 0, fh - 1)
        S = G[yi, xi].mean(axis=-1)
        picks = []
        S2 = S.copy()
        for _ in range(top_n):
            i, j = np.unravel_index(np.argmax(S2), S2.shape)
            if S2[i, j] <= 0:
                break
            picks.append((float(S2[i, j]), int(offs[i]), int(offs[j])))
            r = max(1, 8 // step)
            S2[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = -1
        if not any(a == 0 and b == 0 for _, a, b in picks):
            k0 = int(np.argmin(np.abs(offs)))
            picks.append((float(S[k0, k0]), 0, 0))
        out[side] = picks
    return out


def _flat_lines_to_quad(oa_ob):
    """변별 (oa, ob) 오프셋 -> 평면 좌표 4 꼭짓점"""
    m, W, H = FLAT_M, FLAT_W, FLAT_H
    (ta, tb), (ra, rb), (ba, bb), (la, lb) = oa_ob
    top = (np.array([m, m - ta], float), np.array([W, -(tb - ta)], float))
    bot = (np.array([m, m + H + ba], float), np.array([W, bb - ba], float))
    lef = (np.array([m - la, m], float), np.array([-(lb - la), H], float))
    rig = (np.array([m + W + ra, m], float), np.array([rb - ra, H], float))
    pts = [_intersect(top, lef), _intersect(top, rig), _intersect(bot, rig), _intersect(bot, lef)]
    if any(p is None for p in pts):
        return None
    return np.array(pts, dtype=np.float32)


def _quad_quality_batch(maps, s, quads):
    """_quad_quality 를 사각형 여러 개에 대해 한 번에 계산"""
    Q = np.stack([order_points(q) * s for q in quads]).astype(np.float32)        # (K,4,2)
    v = _side_support_soft_batch(maps, Q.reshape(-1, 2), np.roll(Q, -1, axis=1).reshape(-1, 2),
                                 n=80, reach=2).reshape(-1, 4)
    v.sort(axis=1)
    return 0.5 * v.mean(axis=1) + 0.5 * v[:, 0]


def search_sides(img, q, qual_batch_fn, ratio_fn):
    """평면화 -> 변별 직선 후보(각 3개) 조합 81개를 원본 좌표로 되돌려 전체 품질로 채점, 최고를 반환"""
    flat, Hm = rectify(img, q)
    lines = _flat_side_lines(flat)
    Hinv = np.linalg.inv(Hm)
    import itertools
    cand = []
    for combo in itertools.product(*(lines[s] for s in ("top", "right", "bottom", "left"))):
        fq = _flat_lines_to_quad([(a, b) for _, a, b in combo])
        if fq is None:
            continue
        oq = cv2.perspectiveTransform(fq.reshape(-1, 1, 2), Hinv).reshape(4, 2).astype("float32")
        if _valid_card_quad(oq):
            cand.append(oq)
    if not cand:
        return None
    sc = qual_batch_fn(cand) * np.array([ratio_fn(c) for c in cand])
    return landscape_quad(cand[int(np.argmax(sc))])


def true_aspect(q, shape, f_px=None):
    """원근으로 찌그러진 사각형(4점)의 '실제' 가로/세로 비를 추정 (Zhang & He 방식).
    초점거리를 모르면 일반 폰 메인 카메라(35mm 환산 약 26mm) 기준 0.75 x 긴 변으로 가정"""
    H, W = shape[:2]
    u0, v0 = W / 2.0, H / 2.0
    f = f_px or 0.75 * max(H, W)
    tl, tr, br, bl = [np.array([p[0], p[1], 1.0]) for p in landscape_quad(q)]
    try:
        k2 = np.dot(np.cross(tl, br), bl) / np.dot(np.cross(tr, br), bl)
        k3 = np.dot(np.cross(tl, br), tr) / np.dot(np.cross(bl, br), tr)
        Ki = np.array([[1 / f, 0, -u0 / f], [0, 1 / f, -v0 / f], [0, 0, 1]])
        a, b = Ki @ (k2 * tr - tl), Ki @ (k3 * bl - tl)
        r = float(np.linalg.norm(a) / np.linalg.norm(b))
        if np.isfinite(r) and r > 0:
            return r
    except Exception:
        pass
    w, h = quad_sizes(landscape_quad(q))
    return w / max(h, 1e-6)


def exif_focal_px(path, shape):
    """EXIF 35mm 환산 초점거리가 있으면 픽셀 단위 초점거리로 변환 (없으면 None)"""
    try:
        f35 = Image.open(path).getexif().get_ifd(0x8769).get(41989)
        if f35:
            return float(f35) * float(np.hypot(*shape[:2])) / 43.27
    except Exception:
        pass
    return None


def finalize_quad(img, quad, sil=None, verbose=False, f_px=None):
    """보정 경로 여러 개(그대로 / 윤곽선 보정 / 평면화 후 재측정 / 평면화 후 변별 직선 재탐색)를 모두 만들어 보고,
    4변이 실제 카드 경계에 가장 잘 맞는 것을 고름. 어느 한 단계가 틀어져도 결과가 삐뚤어지지 않게 함"""
    s = min(1.0, 1600.0 / max(img.shape[:2]))
    sm = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    maps = _grad_maps(sm)
    qual = lambda q: _quad_quality(maps, s, q)
    def ratio_term(q):
        # 원근을 걷어낸 실제 가로세로비가 카드 비율(1.586)에서 벗어나면 감점
        # (종이 접힌 선 / 매트 테두리 / 카드 안쪽 인쇄선 같은 엉뚱한 직선을 고르는 것 방지)
        r = true_aspect(q, img.shape, f_px)
        return float(np.exp(-0.5 * ((r - CARD_RATIO) / 0.07) ** 2))

    base = [("find", landscape_quad(quad))]
    if sil is not None:
        base.append(("sil", landscape_quad(refine_quad_silhouette(img, quad, sil))))
    opts = []
    for name, q in base:
        opts.append((name, q))
        opts.append((name + "+flat", landscape_quad(flatten_and_refine(img, q))))
    score = lambda q: qual(q) * ratio_term(q)
    cur = max(opts, key=lambda o: score(o[1]))[1]
    cur_sc = score(cur)
    for it in range(4):   # 한 번에 변 위치를 ±10% 까지만 찾으므로, 좋아지는 동안 반복해서 멀리 틀어진 경우도 따라감
        sq = search_sides(img, cur, lambda qs: _quad_quality_batch(maps, s, qs), ratio_term)
        if sq is None:
            break
        opts.append((f"search{it + 1}", sq))
        sc = score(sq)
        if sc <= cur_sc + 1e-3:
            break
        cur, cur_sc = sq, sc
    opts.append(("search+flat", landscape_quad(flatten_and_refine(img, cur))))
    scored = [(qual(q) * ratio_term(q), name, q) for name, q in opts]
    scored.sort(key=lambda r: -r[0])
    if verbose:
        for sc, name, _ in scored:
            print(f"   보정 {name:12s} {sc:.3f}")
    return scored[0][2]


def crop_from_flat(img, q, out_w, out_h, shrink):
    """평면 좌표계에서 안쪽 여백을 준 뒤 원본을 8.1x5.2로 변환(재샘플링 1회)"""
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
    """바로 선 얼굴이 검출되면 점수를 반환. 검출기를 쓸 수 없으면 0 (축소 이미지로 빠르게)"""
    global _FACE
    try:
        if _FACE is None:
            if not (hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")):
                _FACE = []
            else:
                _FACE = [cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
                         cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")]
                _FACE = [d for d in _FACE if not d.empty()]
        if not _FACE:
            return 0.0
        sc = 480.0 / card.shape[1]
        gray = cv2.equalizeHist(cv2.cvtColor(cv2.resize(card, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA),
                                             cv2.COLOR_BGR2GRAY))
        best = 0.0
        for det in _FACE:
            res = det.detectMultiScale3(gray, scaleFactor=1.1, minNeighbors=3,
                                        minSize=(30, 30), outputRejectLevels=True)
            if len(res) == 3 and len(res[0]):
                best = max(best, float(np.max(res[2])) + 0.002 * float(np.max(res[0][:, 2])))
            if best > 0:
                break
        return best
    except Exception:
        return 0.0


def _tess_rotation(card):
    try:
        import pytesseract
        # PyInstaller's one-file build extracts bundled tools under _MEIPASS.
        # Point pytesseract at that executable and its staged traineddata so
        # OSD does not silently fall back to an unrelated system install.
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            tesseract_dir = os.path.join(bundle_root, "tesseract")
            executable = os.path.join(tesseract_dir, "tesseract.exe")
            tessdata_dir = os.path.join(tesseract_dir, "tessdata")
            if os.path.isfile(executable):
                pytesseract.pytesseract.tesseract_cmd = executable
                if os.path.isdir(tessdata_dir):
                    os.environ["TESSDATA_PREFIX"] = tessdata_dir
        osd = pytesseract.image_to_osd(card, output_type=pytesseract.Output.DICT)
        return int(osd.get("rotate", 0))
    except Exception:
        return None


def auto_orient(card, return_rotation=False):
    """card 는 가로형. 0도 / 180도 중 올바른 쪽을 고름. (확신 없으면 그대로)"""
    flipped = cv2.rotate(card, cv2.ROTATE_180)
    a, b = _face_score(card), _face_score(flipped)
    rotation = 0
    if max(a, b) > 0 and abs(a - b) > 1e-6:
        rotation = 0 if a > b else 180
    else:
        r = _tess_rotation(card)
        if r == 180:
            rotation = 180
        elif max(a, b) == 0 and r is None:
            print("   (방향 자동 판별 불가: 결과가 거꾸로면 --rotate 180 사용)")
    if rotation == 180:
        card = flipped
    if return_rotation:
        return card, rotation
    return card


# ------------------------------------------------------------------ 처리
def process_image_data(img, source_path=None, dpi=300, shrink=0.03, rotate=None,
                       verbose=False):
    """Run the single card-detection/flattening pipeline on an OpenCV BGR image."""
    quad, sil, score = find_card_quad(img, verbose=verbose, return_score=True)
    if quad is None:
        return None
    focal_px = exif_focal_px(source_path, img.shape) if source_path else None
    quad = finalize_quad(img, quad, sil, verbose=verbose, f_px=focal_px)
    out_w = int(round(TARGET_W_CM / 2.54 * dpi))
    out_h = int(round(TARGET_H_CM / 2.54 * dpi))
    card, crop_quad = crop_from_flat(img, quad, out_w, out_h, shrink)
    if rotate is None:
        card, rotation = auto_orient(card, return_rotation=True)
    elif rotate == 180:
        card = cv2.rotate(card, cv2.ROTATE_180)
        rotation = 180
    else:
        rotation = 0
    return {
        "image": card,
        "quad": quad,
        "crop_quad": crop_quad,
        "confidence": float(score),
        "rotation": rotation,
        "dpi": dpi,
        "size": [out_w, out_h],
    }


def process(path, out_path, dpi=300, shrink=0.03, rotate=None, debug=False):
    img = load_image(path)
    result = process_image_data(img, source_path=path, dpi=dpi, shrink=shrink,
                                rotate=rotate, verbose=debug)
    if result is None:
        print(f"[실패] 신분증 영역을 못 찾음: {path}")
        return False
    card = result["image"]
    quad = result["quad"]
    crop_quad = result["crop_quad"]
    out_w, out_h = result["size"]

    pil = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    if os.path.splitext(out_path)[1].lower() in (".jpg", ".jpeg"):
        pil.save(out_path, quality=100, subsampling=0, dpi=(dpi, dpi))
    else:
        pil.save(out_path, dpi=(dpi, dpi), compress_level=1)   # 무손실 그대로, 압축만 가볍게(저장 빠름)
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


def _worker_init():
    cv2.setNumThreads(1)   # 프로세스 여러 개가 각자 스레드를 또 만들면 오히려 느려짐


def _process_task(tk):
    try:
        return process(*tk)
    except Exception as e:
        print(f"[오류] {tk[0]}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="신분증 -> 8.1x5.2cm 셀피 인쇄용 자동 크롭")
    ap.add_argument("input", help="이미지 파일 또는 폴더")
    ap.add_argument("-o", "--output", help="출력 파일/폴더")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--shrink", type=float, default=0.03,
                    help="카드 가장자리에서 안쪽으로 줄이는 비율 (기본 0.03: 각 변 약 1.5%%/1.3mm "
                         "안쪽). 배경이 아직 보이면 0.05, 더 넓게 쓰려면 0.02")
    ap.add_argument("--rotate", type=int, choices=[0, 180], default=None,
                    help="자동 방향판별 대신 수동 지정 (0 또는 180)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="폴더 처리 시 동시에 처리할 장 수 (기본: CPU 코어 수)")
    a = ap.parse_args()

    if os.path.isdir(a.input):
        out_dir = a.output or os.path.join(a.input, "selphy_out")
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in sorted(os.listdir(a.input))
                 if os.path.splitext(f)[1].lower() in EXTS]
        tasks = [(os.path.join(a.input, f), os.path.join(out_dir, os.path.splitext(f)[0] + "_selphy.png"),
                  a.dpi, a.shrink, a.rotate, a.debug) for f in files]
        jobs = max(1, min(a.jobs or (os.cpu_count() or 1), len(tasks)))
        if jobs == 1:
            ok = sum(_process_task(tk) for tk in tasks)
        else:   # 여러 장은 CPU 코어 수만큼 동시에 처리
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init) as ex:
                ok = sum(ex.map(_process_task, tasks))
        print(f"\n{ok}/{len(files)} 장 처리 완료")
    else:
        out = a.output or os.path.splitext(a.input)[0] + "_selphy.png"
        if not process(a.input, out, a.dpi, a.shrink, a.rotate, a.debug):
            sys.exit(1)


if __name__ == "__main__":
    main()
