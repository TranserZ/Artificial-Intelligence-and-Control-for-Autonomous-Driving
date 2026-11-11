"""어린이 보호구역 그림 파일(PNG) 생성 — 노면 (30) 표시, "어린이/보호구역" 글자, 표지판, 학교 이름.

SUMO 다각형으로는 한글 글자나 원형 표시를 그리기 어려워서 PNG를 만들어 POI 이미지로 붙인다.
road_builder.build()가 매번 호출해 env/assets/에 (다시) 만든다. Pillow나 한글 폰트가 없는
PC에서는 아무것도 하지 않고, 저장소에 커밋된 env/assets/*.png를 그대로 쓴다.

모든 그림은 "위쪽 = 차량 진행 방향"으로 그린다 (노면표시는 운전자 쪽에서 읽는 방향).

직접 실행:  python -m env.school_assets
"""
import os

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
PX = 200   # 1 m = 200 px
ROAD_RGB = (0, 0, 0)          # sumo-gui "real world" 차로 색
GRASS_RGB = (51, 128, 51)     # 배경(풀) 색

FILES = {
    "speed": "school_speed30.png",      # (30) 원형 표시 2.0 × 2.8 m
    "text": "school_text.png",          # "어린이/보호구역" 2.3 × 5.0 m
    "sign_start": "school_sign_start.png",
    "sign_end": "school_sign_end.png",
    "name": "school_name.png",
}
SIZE = {   # (가로 m, 세로 m) — POI width/height로 그대로 쓴다
    "speed": (2.0, 2.8), "text": (2.3, 5.0),
    "sign_start": (4.0, 2.6), "sign_end": (4.0, 2.0), "name": (24.0, 4.0),
}

_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
]


def _font_path():
    return next((p for p in _FONTS if os.path.exists(p)), None)


def _text_block(text, font_path, color, box_w, box_h, stretch=True):
    """글자를 그려 (box_w, box_h) 크기로 맞춘 RGBA 이미지. stretch면 세로로 늘인다 (노면 글자)."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, 200)
    l, t, r, b = font.getbbox(text)
    img = Image.new("RGBA", (r - l, b - t), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-l, -t), text, font=font, fill=color)
    if stretch:
        return img.resize((box_w, box_h), Image.LANCZOS)
    scale = min(box_w / img.width, box_h / img.height)
    return img.resize((max(int(img.width * scale), 1), max(int(img.height * scale), 1)), Image.LANCZOS)


def _paste_center(base, img, cx, cy):
    base.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))


def _speed_mark(font_path, size_m, limit_kmh):
    """(30): 적색 테두리 타원, 흰 바탕, 검은 숫자 (노면표시 518)."""
    from PIL import Image, ImageDraw
    w, h = int(size_m[0] * PX), int(size_m[1] * PX)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ring = int(0.16 * PX)
    d.ellipse((0, 0, w - 1, h - 1), fill=(200, 30, 30, 255))
    d.ellipse((ring, ring, w - 1 - ring, h - 1 - ring), fill=(255, 255, 255, 255))
    digits = _text_block(str(limit_kmh), font_path, (0, 0, 0, 255), int(w * 0.62), int(h * 0.55))
    _paste_center(img, digits, w / 2, h / 2)
    return img


def _zone_text(font_path, size_m, color):
    """'어린이/보호구역' 노란 두 줄. 운전자 쪽(아래)부터 읽으므로 아래가 '어린이'."""
    from PIL import Image
    w, h = int(size_m[0] * PX), int(size_m[1] * PX)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    line_h = int(h * 0.44)
    _paste_center(img, _text_block("보호구역", font_path, color, w, line_h), w / 2, h * 0.25)
    _paste_center(img, _text_block("어린이", font_path, color, int(w * 0.8), line_h), w / 2, h * 0.75)
    return img


def _sign(font_path, size_m, lines, bg, fg, border, slash=False, limit_kmh=None):
    """길가 표지판 (위에서 본 아이콘). lines: [(글자, 높이 비율)]."""
    from PIL import Image, ImageDraw
    w, h = int(size_m[0] * PX), int(size_m[1] * PX)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(0.08 * PX)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=pad * 2, fill=border)
    d.rounded_rectangle((pad, pad, w - 1 - pad, h - 1 - pad), radius=pad, fill=bg)
    text_w = w - 4 * pad
    if limit_kmh is not None:                       # 왼쪽에 (30) 원형
        mark = _speed_mark(font_path, (0.9, 0.9), limit_kmh)
        _paste_center(img, mark, 2 * pad + mark.width / 2, h / 2)
        text_w -= mark.width + pad
        x0 = 3 * pad + mark.width
    else:
        x0 = 2 * pad
    y = 2 * pad
    total = sum(r for _, r in lines)
    for text, ratio in lines:
        lh = int((h - 4 * pad) * ratio / total)
        block = _text_block(text, font_path, fg, text_w, int(lh * 0.85), stretch=False)
        _paste_center(img, block, x0 + text_w / 2, y + lh / 2)
        y += lh
    if slash:
        d.line((pad, h - pad, w - pad, pad), fill=(200, 30, 30, 255), width=int(0.12 * PX))
    return img


def make_assets(name="정평초등학교", limit_kmh=30, line_color="255,200,0", force=True) -> bool:
    """PNG를 env/assets/에 만든다. Pillow·한글 폰트가 없으면 False (기존 파일 사용)."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    font = _font_path()
    if font is None:
        return False
    os.makedirs(ASSET_DIR, exist_ok=True)
    yellow = tuple(int(c) for c in line_color.split(",")) + (255,)
    images = {
        "speed": _speed_mark(font, SIZE["speed"], limit_kmh),
        "text": _zone_text(font, SIZE["text"], yellow),
        "sign_start": _sign(font, SIZE["sign_start"],
                            [("어린이 보호구역", 3), ("여기부터 속도를 줄이시오", 2)],
                            (255, 210, 0, 255), (0, 0, 0, 255), (30, 30, 30, 255),
                            limit_kmh=limit_kmh),
        "sign_end": _sign(font, SIZE["sign_end"], [("어린이 보호구역", 3), ("해 제", 2)],
                          (255, 255, 255, 255), (0, 0, 0, 255), (30, 30, 30, 255), slash=True),
        "name": _text_block(name, font, (40, 40, 40, 255),
                            int(SIZE["name"][0] * PX / 4), int(SIZE["name"][1] * PX / 4),
                            stretch=False),
    }
    # sumo-gui는 차로를 POI 그림보다 나중에 그려서, 도로 위 그림의 투명한 부분은 차로가 아니라
    # 배경(풀색)으로 보인다 → 도로 위 그림은 아스팔트색, 길가 표지판은 풀색 바탕을 깔아 둔다.
    from PIL import Image
    for key, rgb in (("speed", ROAD_RGB), ("text", ROAD_RGB),
                     ("sign_start", GRASS_RGB), ("sign_end", GRASS_RGB)):
        bg = Image.new("RGBA", images[key].size, rgb + (255,))
        bg.alpha_composite(images[key])
        images[key] = bg
    for key, img in images.items():
        path = os.path.join(ASSET_DIR, FILES[key])
        if force or not os.path.exists(path):
            img.save(path, optimize=True)
    return True


if __name__ == "__main__":
    print("생성" if make_assets() else "Pillow 또는 한글 폰트 없음 — 기존 파일 사용")
