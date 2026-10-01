"""Рисует схему работы Stock Configurator: docs/assets/social-preview.png и docs/assets/stock-configurator-flow.png.

Схема концептуальная (не скриншот и не материал клиента). Запуск: python tools/render_social_preview.py
Нужны Pillow и шрифты Georgia и Segoe UI (в Windows они есть; на других системах пути можно заменить).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
FONTS = Path("C:/Windows/Fonts")
SERIF = FONTS / "georgia.ttf"
SANS = FONTS / "segoeui.ttf"
SANS_BOLD = FONTS / "segoeuib.ttf"
SANS_SEMI = FONTS / "seguisb.ttf"

BG = (14, 27, 46)
GRID = (24, 40, 64)
BOX = (26, 38, 64)
BOX_LINE = (62, 78, 108)
BLUE = (51, 88, 216)
BLUE_TEXT = (91, 120, 232)
CORAL = (244, 83, 60)
WHITE = (245, 247, 252)
MUTED = (150, 163, 190)
STEPS = [("01", "ЗАПРОС"), ("02", "ПАРАМЕТРЫ"), ("03", "ОСТАТКИ"), ("04", "ЧЕРНОВИК BOM")]


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def grid(draw: ImageDraw.ImageDraw, size: tuple[int, int], step: int) -> None:
    for x in range(0, size[0], step):
        draw.line((x, 0, x, size[1]), fill=GRID, width=1)
    for y in range(0, size[1], step):
        draw.line((0, y, size[0], y), fill=GRID, width=1)


def centered(draw, box, text, face, fill):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=face)
    x = box[0] + (box[2] - box[0] - (right - left)) / 2 - left
    y = box[1] + (box[3] - box[1] - (bottom - top)) / 2 - top
    draw.text((x, y), text, font=face, fill=fill)


def flow(draw, left, top, box_w, box_h, gap, small, large):
    for index, (number, label) in enumerate(STEPS):
        x = left + index * (box_w + gap)
        active = index == len(STEPS) - 1
        draw.rounded_rectangle(
            (x, top, x + box_w, top + box_h),
            radius=12,
            fill=BLUE if active else BOX,
            outline=None if active else BOX_LINE,
            width=2,
        )
        draw.text((x + 20, top + 18), number, font=small, fill=(220, 228, 255) if active else MUTED)
        centered(draw, (x, top, x + box_w, top + box_h), label, large, WHITE)
        if index < len(STEPS) - 1:
            cy = top + box_h // 2
            draw.line((x + box_w, cy, x + box_w + gap, cy), fill=BLUE, width=4)
            draw.ellipse((x + box_w + gap // 2 - 7, cy - 7, x + box_w + gap // 2 + 7, cy + 7), fill=BLUE)


def render_social() -> None:
    size = (1280, 640)
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    grid(draw, size, 64)
    draw.text((52, 30), "ПАВЕЛ ЛОГАЧЁВ / B2B-ПРОДУКТ НА ИИ", font=font(SANS_SEMI, 18), fill=BLUE_TEXT)
    right = font(SANS_SEMI, 18)
    text = "LOGACHEV.NET"
    draw.text((size[0] - 52 - draw.textlength(text, font=right), 30), text, font=right, fill=MUTED)
    draw.text((52, 62), "Stock Configurator", font=font(SERIF, 72), fill=WHITE)
    draw.text((55, 152), "Свободный запрос на инфраструктуру → черновик спецификации по остаткам", font=font(SANS, 28), fill=WHITE)
    flow(draw, 52, 268, 250, 142, 48, font(SANS_SEMI, 16), font(SANS_SEMI, 21))
    draw.rounded_rectangle((176, 477, 1104, 552), radius=8, fill=CORAL)
    centered(draw, (176, 477, 1104, 552), "Инженер подтверждает совместимость и итоговую конфигурацию", font(SANS_SEMI, 25), WHITE)
    draw.text((52, 601), "РАБОЧИЙ MVP / TELEGRAM · API · EXCEL", font=font(SANS_SEMI, 16), fill=MUTED)
    foot = "ПУБЛИЧНАЯ ВЕРСИЯ / СИНТЕТИЧЕСКИЕ ДАННЫЕ"
    draw.text((size[0] - 52 - draw.textlength(foot, font=font(SANS_SEMI, 16)), 601), foot, font=font(SANS_SEMI, 16), fill=MUTED)
    image.save(OUT / "social-preview.png", optimize=True)


def render_flow() -> None:
    size = (1600, 1200)
    image = Image.new("RGB", size, (17, 28, 47))
    draw = ImageDraw.Draw(image)
    grid(draw, size, 80)
    draw.text((64, 44), "ПРОДУКТ ДЛЯ ПРЕСЕЙЛА", font=font(SANS_SEMI, 20), fill=MUTED)
    draw.text((size[0] - 64 - draw.textlength("02", font=font(SANS_SEMI, 20)), 44), "02", font=font(SANS_SEMI, 20), fill=MUTED)
    title = font(SERIF, 62)
    draw.text((64, 128), "Свободный запрос превращается", font=title, fill=WHITE)
    draw.text((64, 212), "в черновик спецификации по остаткам", font=title, fill=WHITE)
    flow(draw, 76, 480, 310, 190, 62, font(SANS_SEMI, 18), font(SANS_BOLD, 26))
    draw.rounded_rectangle((250, 760, 1352, 838), radius=8, fill=CORAL)
    centered(draw, (250, 760, 1352, 838), "Инженер подтверждает совместимость и итоговую конфигурацию", font(SANS_BOLD, 28), WHITE)
    draw.text((64, 912), "СТАТУС", font=font(SANS_BOLD, 17), fill=MUTED)
    draw.text((64, 944), "РАБОЧИЙ MVP · TELEGRAM · API · EXCEL", font=font(SANS_BOLD, 28), fill=BLUE_TEXT)
    draw.text((64, 1008), "Лично вёл от описания процесса до проверки и передачи", font=font(SANS, 26), fill=MUTED)
    draw.text((64, 1156), "ПАВЕЛ ЛОГАЧЁВ  ·  LOGACHEV.NET", font=font(SANS_BOLD, 17), fill=MUTED)
    draw.ellipse((1511, 1141, 1535, 1165), fill=CORAL)
    image.save(OUT / "stock-configurator-flow.png", optimize=True)


if __name__ == "__main__":
    render_social()
    render_flow()
    print("Схемы Stock Configurator нарисованы")
