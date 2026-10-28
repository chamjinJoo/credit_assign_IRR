#!/usr/bin/env python3
"""Render a saved world-step SMAX trajectory as an animated GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
TEAM_COLORS = ((60, 170, 255), (255, 105, 85))
TEAM_LABELS = ("Trained policy", "Enemy bot")
UNIT_SPRITE_ORDER = ("marine", "marauder", "stalker", "scout", "zealot", "zergling", "hydralisk")
UNIT_SPRITE_PATH = ROOT / "assets" / "SMAX_replay" / "smax_unit_sprites.png"
UNIT_ICON_SIZE = 36
GRID_DIVISIONS = 8
PROJECTILE_STYLES = {
    "marine": "bullet",
    "marauder": "shell",
    "stalker": "laser",
    "scout": "laser",
    "hydralisk": "acid",
}
MELEE_STYLES = {"zealot": "blade", "zergling": "claw"}
WIDTH = 800
HEIGHT = 840
FIELD = (50, 100, 750, 800)
PROJECTILE_FRAMES = 3
MELEE_FRAMES = 4
HIT_FLASH_FRAMES = 5


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def _load_unit_sprites():
    sheet = Image.open(UNIT_SPRITE_PATH).convert("RGBA")
    sprites = {}
    cell_width = sheet.width / 4
    cell_height = sheet.height / 2
    for index, unit_type in enumerate(UNIT_SPRITE_ORDER):
        column, row = index % 4, index // 4
        cell = sheet.crop((
            round(column * cell_width),
            round(row * cell_height),
            round((column + 1) * cell_width),
            round((row + 1) * cell_height),
        ))
        visible = cell.getchannel("A").point(lambda alpha: 255 if alpha > 16 else 0).getbbox()
        if visible is None:
            raise ValueError(f"No visible sprite found for {unit_type} in {UNIT_SPRITE_PATH}")
        sprite = cell.crop(visible)
        sprite.thumbnail((UNIT_ICON_SIZE - 4, UNIT_ICON_SIZE - 4), Image.Resampling.LANCZOS)
        icon = Image.new("RGBA", (UNIT_ICON_SIZE, UNIT_ICON_SIZE), (0, 0, 0, 0))
        icon.alpha_composite(sprite, ((UNIT_ICON_SIZE - sprite.width) // 2, (UNIT_ICON_SIZE - sprite.height) // 2))
        flash_rgb = Image.blend(icon.convert("RGB"), Image.new("RGB", icon.size, "white"), 0.78)
        flash = flash_rgb.convert("RGBA")
        flash.putalpha(icon.getchannel("A"))
        sprites[unit_type] = (icon, flash)
    return sprites


def _events(data: dict[str, np.ndarray]):
    health = data["health"]
    alive = data["alive"].astype(bool)
    targets = data["targets"]
    cooldowns = data["cooldowns"]
    hits = [[] for _ in range(health.shape[1])]
    shots = []

    for frame in range(1, len(health)):
        damaged = (health[frame] < health[frame - 1] - 1e-4) & alive[frame]
        for unit_id in np.flatnonzero(damaged):
            hits[int(unit_id)].append(frame)

        cooldown_reset = cooldowns[frame] > cooldowns[frame - 1] + 0.1
        for unit_id in np.flatnonzero(cooldown_reset):
            target_id = int(targets[frame, unit_id])
            if (
                target_id >= 0
                and alive[frame, unit_id]
                and alive[frame, target_id]
            ):
                shots.append((frame, int(unit_id), target_id))

    return hits, shots


def _draw_projectile(draw, style, previous, point, color):
    px, py = point
    qx, qy = previous
    if style == "laser":
        draw.line((qx, qy, px, py), fill=(42, 157, 255), width=6)
        draw.line((qx, qy, px, py), fill=(222, 251, 255), width=2)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 255, 255))
    elif style == "acid":
        draw.line((qx, qy, px, py), fill=(65, 191, 81), width=4)
        draw.ellipse((px - 4, py - 3, px + 4, py + 3), fill=(162, 245, 105), outline=(224, 255, 185))
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(245, 255, 219))
    elif style == "shell":
        draw.line((qx, qy, px, py), fill=(239, 118, 57), width=5)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(255, 178, 74), outline=(255, 235, 177), width=1)
    else:
        draw.line((qx, qy, px, py), fill=color, width=3)
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(255, 246, 196), outline=(255, 255, 255))


def _draw_melee_swing(draw, style, age, attacker, target, color):
    ax, ay = attacker
    tx, ty = target
    dx, dy = tx - ax, ty - ay
    length = max(float(np.hypot(dx, dy)), 1.0)
    nx, ny = dx / length, dy / length
    center = (int((ax + tx) / 2), int((ay + ty) / 2))
    cx, cy = center
    reach = 15 + (7 if age == 1 else 0)
    if style == "blade":
        blade_color = tuple((channel + 255) // 2 for channel in color)
        draw.arc((cx - reach, cy - reach, cx + reach, cy + reach), 190, 350, fill=blade_color, width=5)
        draw.arc((cx - reach + 4, cy - reach + 4, cx + reach - 4, cy + reach - 4), 200, 340, fill=(245, 253, 255), width=2)
        if age < 2:
            start = (int(cx - ny * reach), int(cy + nx * reach))
            end = (int(cx + ny * reach), int(cy - nx * reach))
            draw.line((*start, *end), fill=(225, 248, 255), width=3)
    else:
        claw_color = (255, 164, 133) if color[0] > color[2] else (255, 205, 151)
        claw_width = 7
        a = (int(cx - nx * reach - ny * claw_width), int(cy - ny * reach + nx * claw_width))
        b = (int(cx + nx * reach + ny * claw_width), int(cy + ny * reach - nx * claw_width))
        c = (int(cx - nx * reach + ny * claw_width), int(cy - ny * reach - nx * claw_width))
        d = (int(cx + nx * reach - ny * claw_width), int(cy + ny * reach + nx * claw_width))
        draw.line((*a, *b), fill=claw_color, width=4)
        if age < 2:
            draw.line((*c, *d), fill=(255, 235, 196), width=3)


def render_frame(data: dict[str, np.ndarray], frame: int, fonts) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (16, 21, 31))
    draw = ImageDraw.Draw(image)
    positions = data["positions"]
    health = data["health"]
    alive = data["alive"]
    teams = data["unit_teams"][0]
    types = data["unit_types"][0]
    type_names = data["unit_types_by_id"]
    targets = data["targets"]
    max_health = data["max_health"]
    map_size = data["map_size"]
    hit_frames = data["_hit_frames"]
    attack_events = data["_shot_events"]
    map_name = str(data["map_name"].item())
    total_frames = len(positions)

    draw.text((36, 24), f"SMAX  /  {map_name}", font=fonts["title"], fill=(240, 244, 250))
    draw.text((38, 64), f"step {frame + 1} / {total_frames}", font=fonts["small"], fill=(154, 168, 188))
    draw.text((610, 65), f"{frame / 16:.1f}s", font=fonts["small"], fill=(210, 220, 234))

    left, top, right, bottom = FIELD
    draw.rectangle(FIELD, fill=(28, 37, 49), outline=(90, 106, 126), width=2)
    for tick in range(1, GRID_DIVISIONS):
        x = left + (right - left) * tick // GRID_DIVISIONS
        y = top + (bottom - top) * tick // GRID_DIVISIONS
        draw.line((x, top, x, bottom), fill=(42, 53, 68), width=1)
        draw.line((left, y, right, y), fill=(42, 53, 68), width=1)

    def xy(point):
        x = left + float(point[0]) / float(map_size[0]) * (right - left)
        y = bottom - float(point[1]) / float(map_size[1]) * (bottom - top)
        return int(x), int(y)

    melee_effects = []
    for unit_id in range(positions.shape[1]):
        if not alive[frame, unit_id]:
            continue

        x, y = xy(positions[frame, unit_id])
        team_id = int(teams[unit_id])
        color = TEAM_COLORS[team_id]
        unit_type = str(type_names[int(types[unit_id])]).lower()
        if unit_type not in data["_unit_sprites"]:
            raise ValueError(f"No sprite is configured for SMAX unit type {unit_type!r}")
        if unit_type in MELEE_STYLES:
            latest_swing = next(
                (
                    (launch_frame, target_id)
                    for launch_frame, shooter_id, target_id in reversed(attack_events)
                    if shooter_id == unit_id and 0 <= frame - launch_frame < MELEE_FRAMES
                ),
                None,
            )
            if latest_swing is not None:
                launch_frame, target_id = latest_swing
                age = frame - launch_frame
                target = xy(positions[frame, target_id])
                dx, dy = target[0] - x, target[1] - y
                stride = 9 if unit_type == "zealot" else 12
                scale = (1.0, 0.55, -0.15, -0.45)[age]
                x += int(dx / max(np.hypot(dx, dy), 1.0) * stride * scale)
                y += int(dy / max(np.hypot(dx, dy), 1.0) * stride * scale)
                melee_effects.append((MELEE_STYLES[unit_type], age, (x, y), target, color))
        recently_hit = [
            hit_frame for hit_frame in hit_frames[unit_id]
            if 0 <= frame - hit_frame < HIT_FLASH_FRAMES
        ]
        if recently_hit and (frame - recently_hit[-1]) % 2 == 0:
            icon = data["_unit_sprites"][unit_type][1]
        else:
            icon = data["_unit_sprites"][unit_type][0]
        badge_radius = UNIT_ICON_SIZE // 2 + 2
        draw.ellipse(
            (x - badge_radius, y - badge_radius, x + badge_radius, y + badge_radius),
            fill=(19, 26, 37),
        )
        image.paste(icon, (x - UNIT_ICON_SIZE // 2, y - UNIT_ICON_SIZE // 2), icon)
        draw.ellipse(
            (x - badge_radius, y - badge_radius, x + badge_radius, y + badge_radius),
            outline=color,
            width=3,
        )

        bar_width = 30
        bar_y = y + badge_radius + 4
        draw.rectangle((x - bar_width // 2, bar_y, x + bar_width // 2, bar_y + 4), fill=(51, 58, 69))
        ratio = max(0.0, min(1.0, float(health[frame, unit_id] / max_health[unit_id])))
        draw.rectangle((x - bar_width // 2, bar_y, x - bar_width // 2 + int(bar_width * ratio), bar_y + 4), fill=color)

    for style, age, attacker, target, color in melee_effects:
        _draw_melee_swing(draw, style, age, attacker, target, color)

    for launch_frame, shooter_id, target_id in attack_events:
        shooter_type = str(type_names[int(types[shooter_id])]).lower()
        if shooter_type in MELEE_STYLES:
            continue
        age = frame - launch_frame
        if not 0 <= age < PROJECTILE_FRAMES:
            continue
        impact_frame = min(len(positions) - 1, launch_frame + PROJECTILE_FRAMES)
        start = positions[launch_frame, shooter_id]
        end = positions[impact_frame, target_id]
        progress = (age + 1) / PROJECTILE_FRAMES
        point = start + (end - start) * progress
        previous = start + (end - start) * (age / PROJECTILE_FRAMES)
        px, py = xy(point)
        qx, qy = xy(previous)
        projectile_color = TEAM_COLORS[int(teams[shooter_id])]
        style = PROJECTILE_STYLES.get(shooter_type, "bullet")
        _draw_projectile(draw, style, (qx, qy), (px, py), projectile_color)

    for team_id, label in enumerate(TEAM_LABELS):
        x = 42 + team_id * 200
        draw.ellipse((x, 812, x + 16, 828), fill=(19, 26, 37), outline=TEAM_COLORS[team_id], width=3)
        draw.text((x + 24, 811), label, font=fonts["small"], fill=(205, 216, 231))

    alive_count = [int(np.count_nonzero(alive[frame, teams == team])) for team in (0, 1)]
    draw.text((520, 811), f"alive  {alive_count[0]}  :  {alive_count[1]}", font=fonts["small"], fill=(205, 216, 231))
    return image


def render_trajectory(source: Path, output: Path, fps: int, map_name: str | None = None) -> None:
    with np.load(source, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}

    if map_name is None:
        if "map_name" in data:
            map_name = str(data["map_name"].item())
        elif len(source.parents) > 5:
            map_name = source.parents[5].name
        else:
            raise ValueError("map name is not stored in this trajectory; pass --map-name")
    data["map_name"] = np.asarray(map_name)
    required = {"positions", "health", "alive", "unit_types", "unit_teams", "targets", "unit_types_by_id", "max_health", "map_size", "seed", "map_name"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Trajectory is missing fields: {sorted(missing)}")
    data["_unit_sprites"] = _load_unit_sprites()
    fonts = {"title": _font(28), "small": _font(16)}
    data["_hit_frames"], data["_shot_events"] = _events(data)
    frames = [
        render_frame(data, frame, fonts).quantize(colors=128, method=Image.Quantize.FASTOCTREE)
        for frame in range(len(data["positions"]))
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Saved {len(frames)} frames ({len(frames) / fps:.1f}s) to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("fps must be positive")
    output = args.output or ROOT / "assets" / "SMAX_replay" / "smax_replays" / f"{args.trajectory.stem}.gif"
    render_trajectory(args.trajectory, output, args.fps, args.map_name)


if __name__ == "__main__":
    main()
