"""Suvi Achu mech bot (mechfight/1) — survival-first battle royale bot.

Placement counts, not kills. Practice matches are 1 v 15: the house bots
share vision, never fight each other, and hunt the weakest mech first. So
this bot plays to outlive, not to outgun:

* stay quiet (walk slow near threats; gunfire is heard 320 m away),
* stay inside the next zone circle, moving early during hold phases,
* loot a laser (burst) or mortar (lob over cover) when it is safe,
* farm crates / NPC drones for cubes only while the area is quiet,
* fight at range with lead + torso-slew awareness; flee up close and
  shoot backwards while running,
* dodge telegraphed laser charges and incoming mortar shells,
* navigate with A* over a rasterized map of the buildings, so it does
  not walk into walls on the way to the zone.

Standard library only. Reads hello config instead of hard-coding numbers.
"""
import heapq
import json
import math
import os
import random
import struct
import time
from array import array
from collections import deque

import mechfight as mf

# ---------------------------------------------------------------- helpers


def ang_norm(a):
    """Wrap an angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def dist2(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def unit(ax, ay, bx, by):
    """Unit vector from (ax, ay) toward (bx, by); (0, 0) if degenerate."""
    dx, dy = bx - ax, by - ay
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (0.0, 0.0)
    return (dx / d, dy / d)


def point_in_poly(x, y, ring):
    """Ray casting. ring is a flat [x0, y0, x1, y1, ...] list."""
    inside = False
    n = len(ring) // 2
    j = n - 1
    for i in range(n):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if (yi > y) != (yj > y):
            xint = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < xint:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------- map


class ArenaMap:
    """Ground-plane navigation map built from v2/world.json + terrain.i16.

    A 4 m grid; a cell is blocked when a building/stadium prism reaches the
    ground there, or the terrain slope is too steep to walk. Bridges are not
    blocked (you walk under them). A* over the grid with line-of-sight
    smoothing gives waypoints. If the map files cannot be read, everything
    degrades to straight-line steering.
    """

    CELL = 8.0

    def __init__(self, map_dir):
        self.ok = False
        self.blocked = None
        try:
            with open(os.path.join(map_dir, "v2", "world.json")) as f:
                w = json.load(f)
            b = w["bounds"]
            self.minx, self.maxx = b["minX"], b["maxX"]
            self.miny, self.maxy = b["minY"], b["maxY"]
            self.nx = int((self.maxx - self.minx) / self.CELL) + 1
            self.ny = int((self.maxy - self.miny) / self.CELL) + 1
            self.blocked = bytearray(self.nx * self.ny)
            terr = w["terrain"]
            self.tx0, self.ty0 = terr["minX"], terr["minY"]
            self.tnx, self.tny = terr["nx"], terr["ny"]
            with open(os.path.join(map_dir, "v2", "terrain.i16"), "rb") as f:
                raw = f.read()
            self.terr = array("h")
            self.terr.frombytes(raw[: 2 * self.tnx * self.tny])
            if struct.pack("=h", 1) == struct.pack("<h", 1):
                pass  # little-endian host: raw int16s are already correct
            else:
                self.terr.byteswap()
            self._rasterize_prisms(w.get("prisms", []))
            self._rasterize_slopes()
            self.ok = True
        except Exception as e:
            mf.log(f"map load failed ({e}); falling back to straight-line steering")

    # -- terrain ------------------------------------------------------
    def ground_z(self, x, y):
        """Bilinear sample of terrain height (m) at (x, y)."""
        gx = (x - self.tx0)
        gy = (y - self.ty0)
        i = int(math.floor(gx))
        j = int(math.floor(gy))
        i = clamp(i, 0, self.tnx - 2)
        j = clamp(j, 0, self.tny - 2)
        fx, fy = gx - i, gy - j
        idx = j * self.tnx + i
        z00 = self.terr[idx] * 0.01
        z10 = self.terr[idx + 1] * 0.01
        z01 = self.terr[idx + self.tnx] * 0.01
        z11 = self.terr[idx + self.tnx + 1] * 0.01
        return (z00 * (1 - fx) + z10 * fx) * (1 - fy) + (z01 * (1 - fx) + z11 * fx) * fy

    # -- rasterization -------------------------------------------------
    def _cell(self, x, y):
        cx = int((x - self.minx) / self.CELL)
        cy = int((y - self.miny) / self.CELL)
        return cx, cy

    def _set_blocked(self, cx, cy):
        if 0 <= cx < self.nx and 0 <= cy < self.ny:
            self.blocked[cy * self.nx + cx] = 1

    def is_blocked(self, x, y):
        cx, cy = self._cell(x, y)
        if not (0 <= cx < self.nx and 0 <= cy < self.ny):
            return True
        return self.blocked[cy * self.nx + cx] == 1

    def _rasterize_prisms(self, prisms):
        for p in prisms:
            if p.get("kind") not in ("building", "stadium"):
                continue  # bridges: walk under them
            ring = p.get("ring") or []
            if len(ring) < 6:
                continue
            xs = ring[0::2]
            ys = ring[1::2]
            # Only prisms that reach the ground block ground movement.
            cxm = sum(xs) / len(xs)
            cym = sum(ys) / len(ys)
            if p.get("base", 0) > self.ground_z(cxm, cym) + 1.5:
                continue
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            c0x, c0y = self._cell(x0, y0)
            c1x, c1y = self._cell(x1, y1)
            holes = p.get("holes") or []
            for cy in range(max(0, c0y), min(self.ny - 1, c1y) + 1):
                for cx in range(max(0, c0x), min(self.nx - 1, c1x) + 1):
                    px = self.minx + (cx + 0.5) * self.CELL
                    py = self.miny + (cy + 0.5) * self.CELL
                    if point_in_poly(px, py, ring):
                        if any(point_in_poly(px, py, h) for h in holes):
                            continue
                        self._set_blocked(cx, cy)

    def _rasterize_slopes(self):
        # A cell is unwalkable when the ground rises too steeply across it.
        max_dz = self.CELL * math.tan(math.radians(50.0))
        for cy in range(self.ny):
            y = self.miny + (cy + 0.5) * self.CELL
            for cx in range(self.nx):
                x = self.minx + (cx + 0.5) * self.CELL
                z = self.ground_z(x, y)
                z2 = self.ground_z(x + self.CELL, y)
                z3 = self.ground_z(x, y + self.CELL)
                if abs(z2 - z) > max_dz or abs(z3 - z) > max_dz:
                    self._set_blocked(cx, cy)
        # Keep a margin inside the map edge (the edge is a wall).
        for cy in range(self.ny):
            for cx in (0, self.nx - 1):
                self._set_blocked(cx, cy)
        for cx in range(self.nx):
            for cy in (0, self.ny - 1):
                self._set_blocked(cx, cy)

    # -- pathfinding ----------------------------------------------------
    def nearest_free(self, x, y):
        """Nearest free point to (x, y), searching outward in rings."""
        if not self.is_blocked(x, y):
            return (x, y)
        for r in (4, 8, 12, 20, 32, 48):
            for k in range(12):
                a = k * math.pi / 6.0
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                if not self.is_blocked(px, py):
                    return (px, py)
        return (x, y)

    def line_clear(self, x0, y0, x1, y1):
        d = dist2(x0, y0, x1, y1)
        steps = max(1, int(d / 2.0))
        for i in range(steps + 1):
            t = i / steps
            if self.is_blocked(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
                return False
        return True

    def find_path(self, sx, sy, gx, gy, max_expand=20000):
        """A* (8-connected, no corner cutting) over flat arrays.
        Returns smoothed waypoints [(x, y), ...] from start to goal, or None."""
        if not self.ok:
            return None
        sx, sy = self.nearest_free(sx, sy)
        gx, gy = self.nearest_free(gx, gy)
        scx, scy = self._cell(sx, sy)
        gcx, gcy = self._cell(gx, gy)
        start = scy * self.nx + scx
        goal = gcy * self.nx + gcx
        if start == goal:
            return [(gx, gy)]
        nx, ny, blocked = self.nx, self.ny, self.blocked
        SQRT2 = math.sqrt(2.0)
        INF = float("inf")
        gscore = [INF] * (nx * ny)
        came = [-1] * (nx * ny)
        closed = bytearray(nx * ny)
        gscore[start] = 0.0
        # Octile heuristic, inlined via a small closure over (gcx, gcy).
        def h(ci):
            cx, cy = ci % nx, ci // nx
            dx = abs(cx - gcx)
            dy = abs(cy - gcy)
            return (dx if dx > dy else dy) + (SQRT2 - 1.0) * (dy if dx > dy else dx)

        open_heap = [(h(start), 0.0, start)]
        expanded = 0
        found = False
        t_start = time.perf_counter()
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if closed[cur]:
                continue
            closed[cur] = 1
            if cur == goal:
                found = True
                break
            expanded += 1
            if expanded > max_expand:
                return None
            if (expanded & 511) == 0 and time.perf_counter() - t_start > 0.02:
                return None  # never blow the 50 ms tick budget on one search
            cy, cx = divmod(cur, nx)
            row = cy * nx
            for dx in (-1, 0, 1):
                nxx = cx + dx
                if nxx < 0 or nxx >= nx:
                    continue
                for dy in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nyy = cy + dy
                    if nyy < 0 or nyy >= ny:
                        continue
                    ni = nyy * nx + nxx
                    if blocked[ni] or closed[ni]:
                        continue
                    if dx and dy and (blocked[row + nxx] or blocked[nyy * nx + cx]):
                        continue  # no cutting corners
                    ng = g + (SQRT2 if (dx and dy) else 1.0)
                    if ng < gscore[ni]:
                        gscore[ni] = ng
                        came[ni] = cur
                        heapq.heappush(open_heap, (ng + h(ni), ng, ni))
        if not found:
            return None
        # Reconstruct cell centers, then smooth with line-of-sight.
        cells = []
        cur = goal
        while cur != -1:
            cells.append((self.minx + (cur % self.nx + 0.5) * self.CELL,
                          self.miny + (cur // self.nx + 0.5) * self.CELL))
            cur = came[cur]
        cells.reverse()
        cells[0] = (sx, sy)
        cells[-1] = (gx, gy)
        pts = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self.line_clear(cells[i][0], cells[i][1], cells[j][0], cells[j][1]):
                j -= 1
            pts.append(cells[j])
            i = j
        return pts


# ---------------------------------------------------------------- bot

# Upgrade priority for a survival bot: live longer first, then see more,
# then move better, then hit harder.
UPGRADE_PRIORITY = [
    "armor", "plating", "sensors", "servos", "reactor", "capacitor",
    "nanites", "gyros", "jets",
    "gun_rate", "gun_damage", "gun_velocity", "gun_mag",
    "laser_damage", "laser_charge", "laser_pierce",
    "mortar_reload", "mortar_splash", "mortar_velocity",
    "lightning_range", "lightning_chain", "lightning_efficiency",
]

# Loot priority: laser (315 dmg hitscan at 280 m) > mortar (indirect fire
# over cover) > lightning (close-range panic weapon).
LOOT_PRIORITY = {"laser": 0, "mortar": 1, "lightning": 2}

WEAPON_SOUNDS = {"gun", "mortar", "laser", "laser_charge", "lightning",
                 "explosion", "npc_gun"}


class Bot:
    name = "Suvi Achu"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.rng = random.Random(int(hello.get("seed", 12345)))
        mech = self.cfg["mech"]
        self.walk = mech["walk_speed"]
        self.boost_speed = mech["boost_speed"]
        self.max_energy = mech["energy"]
        self.gun_range = self.cfg["weapons"]["gun"]["range"]
        self.gun_speed = self.cfg["weapons"]["gun"]["speed"]
        self.mortar_speed = self.cfg["weapons"]["mortar"]["speed"]
        self.mortar_grav = self.cfg["weapons"]["mortar"]["gravity"]
        self.laser_range = self.cfg["weapons"]["laser"]["range"]
        self.laser_charge_t = self.cfg["weapons"]["laser"]["charge"]
        self.light_range = self.cfg["weapons"]["lightning"]["range"]
        self.arena = ArenaMap(hello["map"]["dir"])

        # --- persistent state ---
        self.enemies = {}       # id -> dict(pos, vel, hp, cubes, weapons, charging, t)
        self.last_heard_weapon = -1e9   # t of last weapon-ish sound
        self.last_heard_bearing = None  # (bearing, dist, t)
        self.last_damage_t = -1e9
        self.last_damage_bearing = None
        self.pos_hist = deque(maxlen=20)  # (t, x, y)
        self.path = []           # waypoints [(x, y), ...]
        self.path_goal = None    # (x, y) the path was built for
        self.path_i = 0
        self.stuck_since = None
        self.detour_until = 0.0
        self.detour_side = 1
        self.stuck_count = 0
        self.stuck_window_t = 0.0
        self.strafe_dir = 1
        self.strafe_until = 0.0
        self.scan_yaw = 0.0
        self.patrol_goal = None
        self.patrol_until = 0.0
        self.laser_charging_since = None
        self.failed_take = {}    # pickup id -> t (don't retry at once)
        self.mode = "patrol"
        self.t = 0.0

    # -- lookups -------------------------------------------------------
    def weapon_slot(self, weapons, kind):
        for i, w in enumerate(weapons or []):
            if w and w.get("kind") == kind:
                return i
        return None

    def held_kinds(self, weapons):
        return {w["kind"] for w in (weapons or []) if w}

    def enemy_scale(self, e):
        return 1.0 + 0.05 * (e.get("cubes") or 0)

    # -- state update ----------------------------------------------------
    def update(self, obs):
        self.t = obs["t"]
        me = obs["self"]
        mx, my = me["pos"][0], me["pos"][1]
        self.pos_hist.append((self.t, mx, my))
        now = self.t
        for m in obs.get("mechs", []):
            if m.get("kind") == "mech":
                self.enemies[m["id"]] = {
                    "pos": list(m["pos"]), "vel": list(m.get("vel") or [0, 0, 0]),
                    "hp": m.get("hp"), "cubes": m.get("cubes") or 0,
                    "weapons": list(m.get("weapons") or []),
                    "charging": bool(m.get("charging")), "aim": list(m.get("aim") or [0, 0]),
                    "t": now, "seen": True,
                }
            # NPCs are not tracked as enemies; they are farm targets.
        # Expire stale enemy memory (positions go stale fast).
        for eid in [e for e, v in self.enemies.items() if now - v["t"] > 20.0]:
            del self.enemies[eid]
        for s in obs.get("sounds", []):
            if s.get("kind") in WEAPON_SOUNDS:
                self.last_heard_weapon = now
                self.last_heard_bearing = (s.get("bearing"), s.get("dist"), now)
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                self.last_damage_t = now
                self.last_damage_bearing = ev.get("bearing")
            elif ev.get("type") == "take_failed":
                self.failed_take[ev.get("id")] = now

    # -- safety ----------------------------------------------------------
    def threat_near(self, me, radius=150.0):
        """Nearest known enemy (seen recently) and its distance."""
        best, bestd = None, 1e18
        for e in self.enemies.values():
            if self.t - e["t"] > 12.0:
                continue
            d = dist2(me["pos"][0], me["pos"][1], e["pos"][0], e["pos"][1])
            if d < bestd:
                best, bestd = e, d
        return best, bestd

    def is_safe(self, me):
        """No enemy seen recently and no weapon sounds recently."""
        _, d = self.threat_near(me, 200.0)
        if d < 200.0:
            return False
        if self.t - self.last_heard_weapon < 12.0:
            return False
        if self.t - self.last_damage_t < 10.0:
            return False
        return True

    # -- zone ------------------------------------------------------------
    def zone_now(self, obs):
        """Current circle (interpolated while shrinking) and next circle."""
        z = obs["zone"]
        t = self.t
        if t < z["shrink_start"]:
            return (z["center"][0], z["center"][1], z["radius"]), \
                   (z["next_center"][0], z["next_center"][1], z["next_radius"])
        if t >= z["shrink_end"]:
            return (z["next_center"][0], z["next_center"][1], z["next_radius"]), \
                   (z["next_center"][0], z["next_center"][1], z["next_radius"])
        f = (t - z["shrink_start"]) / max(1e-6, z["shrink_end"] - z["shrink_start"])
        cx = z["center"][0] + (z["next_center"][0] - z["center"][0]) * f
        cy = z["center"][1] + (z["next_center"][1] - z["center"][1]) * f
        r = z["radius"] + (z["next_radius"] - z["radius"]) * f
        return (cx, cy, r), (z["next_center"][0], z["next_center"][1], z["next_radius"])

    # -- upgrades ----------------------------------------------------------
    def choose_upgrade(self, offer):
        best_i, best_rank = 0, 1e9
        for i, ch in enumerate(offer.get("choices", [])):
            try:
                rank = UPGRADE_PRIORITY.index(ch["id"])
            except ValueError:
                rank = 999
            if rank < best_rank:
                best_rank, best_i = rank, i
        return best_i

    # -- aiming ------------------------------------------------------------
    def gun_aim(self, me, tgt):
        """Aim [yaw, pitch] to hit a moving target's center with the gun."""
        mx, my, mz = me["pos"]
        muzzle_z = mz + me["stats"].get("muzzle_height", 7.0)
        tx, ty, tz = tgt["pos"]
        vx, vy = tgt["vel"][0], tgt["vel"][1]
        d = dist2(mx, my, tx, ty)
        stats = None
        for w in me["weapons"]:
            if w and w.get("kind") == "gun":
                stats = w["stats"]
                break
        speed = (stats or {}).get("speed", self.gun_speed)
        tof = d / max(1.0, speed)
        px, py = tx + vx * tof, ty + vy * tof
        pz = tz + 4.5 * self.enemy_scale(tgt)
        dd = dist2(mx, my, px, py)
        yaw = math.atan2(py - my, px - mx)
        pitch = math.atan2(pz - muzzle_z, dd)
        return [yaw, pitch], d

    def laser_aim(self, me, tgt):
        mx, my, mz = me["pos"]
        muzzle_z = mz + me["stats"].get("muzzle_height", 7.0)
        tx, ty, tz = tgt["pos"]
        px, py = tx + tgt["vel"][0] * 0.05, ty + tgt["vel"][1] * 0.05
        pz = tz + 4.5 * self.enemy_scale(tgt)
        dd = dist2(mx, my, px, py)
        return [math.atan2(py - my, px - mx), math.atan2(pz - muzzle_z, dd)], dd

    def mortar_aim(self, me, tgt):
        """Ballistic aim at a (possibly moving) target. Returns ([yaw, pitch], dist)
        or (None, dist) when the target is out of mortar range."""
        mx, my, mz = me["pos"]
        muzzle_z = mz + me["stats"].get("muzzle_height", 7.0)
        v, g = self.mortar_speed, self.mortar_grav
        tx, ty, tz = tgt["pos"]
        vx, vy = tgt["vel"][0], tgt["vel"][1]
        tz = tz + 4.5 * self.enemy_scale(tgt)
        theta = None
        tof = 0.0
        d = dist2(mx, my, tx, ty)
        for _ in range(3):
            dx, dy = tx - mx, ty - my
            d = math.hypot(dx, dy)
            dz = tz - muzzle_z
            disc = v ** 4 - g * (g * d * d + 2.0 * dz * v * v)
            if disc < 0 or d < 1.0:
                return None, d
            theta = math.atan((v * v - math.sqrt(disc)) / (g * d))
            tof = d / max(1e-3, v * math.cos(theta))
            tx, ty = tgt["pos"][0] + vx * tof, tgt["pos"][1] + vy * tof
        yaw = math.atan2(ty - my, tx - mx)
        return [yaw, theta], d

    def aim_error(self, me, want):
        dyaw = abs(ang_norm(want[0] - me["aim"][0]))
        dpit = abs(want[1] - me["aim"][1])
        return dyaw, dpit

    # -- movement / pathing -------------------------------------------------
    def set_goal(self, gx, gy):
        if self.path_goal is None or dist2(gx, gy, self.path_goal[0], self.path_goal[1]) > 12.0:
            self.path_goal = (gx, gy)
            self.path = []
            self.path_i = 0

    def move_toward(self, me, gx, gy, speed_frac):
        """Return (move [x, y], stuck bool) steering toward (gx, gy)."""
        mx, my = me["pos"][0], me["pos"][1]
        if dist2(mx, my, gx, gy) < 3.0:
            return [0.0, 0.0], False
        if self.arena.ok:
            self.set_goal(gx, gy)
            if not self.path:
                self.path = self.arena.find_path(mx, my, gx, gy) or [(gx, gy)]
                self.path_i = 0
            # Advance past reached waypoints.
            while self.path_i < len(self.path) and \
                    dist2(mx, my, self.path[self.path_i][0], self.path[self.path_i][1]) < 3.5:
                self.path_i += 1
            if self.path_i >= len(self.path):
                self.path = []  # arrived; replan fresh next tick
                return [0.0, 0.0], False
            wx, wy = self.path[self.path_i]
        else:
            wx, wy = gx, gy
        dx, dy = unit(mx, my, wx, wy)
        return [dx * speed_frac, dy * speed_frac], self.is_stuck(me)

    def is_stuck(self, me):
        """Commanded to move but barely displaced over the last ~1 s."""
        if len(self.pos_hist) < 10:
            return False
        t0, x0, y0 = self.pos_hist[0]
        t1, x1, y1 = self.pos_hist[-1]
        if t1 - t0 < 0.8:
            return False
        return dist2(x0, y0, x1, y1) < 1.2

    def detour_goal(self, me, gx, gy):
        """Side-step goal when stuck: 25 m perpendicular to the goal direction."""
        mx, my = me["pos"][0], me["pos"][1]
        dx, dy = unit(mx, my, gx, gy)
        self.detour_side *= -1
        px, py = -dy * self.detour_side, dx * self.detour_side
        return (mx + px * 25.0 + dx * 10.0, my + py * 25.0 + dy * 10.0)

    # -- dodge ---------------------------------------------------------------
    def dodge_move(self, obs, me):
        """Sidestep incoming projectiles. Returns a move vector or None."""
        mx, my, mz = me["pos"]
        best = None
        for p in obs.get("projectiles", []):
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            rx, ry, rz = px - mx, py - my, pz - (mz + 4.5)
            vv = vx * vx + vy * vy + vz * vz
            if vv < 1.0:
                continue
            tstar = -(rx * vx + ry * vy + rz * vz) / vv
            if tstar < 0 or tstar > 1.2:
                continue
            cx, cy = rx + vx * tstar, ry + vy * tstar
            if math.hypot(cx, cy) < 6.0:
                best = (vx, vy)
        if best is None:
            # Mortar shell falling near us: run from the impact point.
            for p in obs.get("projectiles", []):
                if p.get("kind") != "mortar":
                    continue
                px, py, pz = p["pos"]
                vx, vy, vz = p["vel"]
                gz = self.arena.ground_z(px, py) if self.arena.ok else pz
                disc = vz * vz + 2 * 22.0 * max(0.0, pz - gz)
                tfall = (vz + math.sqrt(disc)) / 22.0 if disc > 0 else 0.0
                ix, iy = px + vx * tfall, py + vy * tfall
                if dist2(mx, my, ix, iy) < 12.0:
                    dx, dy = unit(ix, iy, mx, my)
                    return [dx, dy]
            return None
        # Strafe perpendicular to the bullet's horizontal path.
        vx, vy = best
        d = math.hypot(vx, vy) or 1.0
        px, py = -vy / d, vx / d
        return [px, py]


    # -- decisions ------------------------------------------------------------
    def act(self, obs):
        try:
            return self._act(obs)
        except Exception:
            import traceback
            mf.log(f"act() error at tick {obs.get('tick')}:\n{traceback.format_exc()}")
            return {"tick": obs["tick"]}

    def _act(self, obs):
        self.update(obs)
        me = obs["self"]
        mx, my, mz = me["pos"]
        t = self.t
        act = {"tick": obs["tick"]}
        weapons = me.get("weapons") or []
        held = self.held_kinds(weapons)
        hp_frac = me["hp"] / max(1.0, me["max_hp"])

        # --- upgrades: answer immediately ---
        if obs.get("offer"):
            act["pick"] = self.choose_upgrade(obs["offer"])

        # --- zone state ---
        (cx, cy, cr), (ncx, ncy, nr) = self.zone_now(obs)
        d_next = dist2(mx, my, ncx, ncy)
        outside_now = dist2(mx, my, cx, cy) > cr
        need_next = d_next > max(0.0, nr - 20.0)
        z = obs["zone"]
        # Time-aware trigger: start for the next circle when the time left
        # until it closes barely covers the walk there (plus a margin), so
        # long treks start early and short ones leave time to loot.
        d_edge = max(0.0, d_next - max(0.0, nr - 20.0))
        time_needed = d_edge / max(1.0, self.walk)
        shrink_soon = (z["shrink_end"] - t) < time_needed + 15.0

        # --- threats ---
        visible_mechs = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        threat, threat_d = self.threat_near(me)
        # Nearest visible mech (fresh info beats memory).
        vis_best, vis_d = None, 1e18
        for m in visible_mechs:
            d = dist2(mx, my, m["pos"][0], m["pos"][1])
            if d < vis_d:
                vis_best, vis_d = m, d
        if vis_best is not None:
            threat, threat_d = {
                "pos": list(vis_best["pos"]), "vel": list(vis_best.get("vel") or [0, 0, 0]),
                "hp": vis_best.get("hp"), "cubes": vis_best.get("cubes") or 0,
                "weapons": list(vis_best.get("weapons") or []),
                "charging": bool(vis_best.get("charging")), "t": t,
            }, vis_d
        recently_hurt = (t - self.last_damage_t) < 1.5
        # A charging laser aimed at us is a telegraphed shot: dodge it.
        laser_aimed_at_us = False
        for m in visible_mechs:
            if m.get("charging"):
                bx = math.atan2(my - m["pos"][1], mx - m["pos"][0])
                if abs(ang_norm(bx - m["aim"][0])) < 0.12:
                    laser_aimed_at_us = True

        # --- mode selection ---
        mode = "patrol"
        goal = None
        target = None       # combat/farm target dict with pos/vel/cubes
        speed = 1.0
        boost = False
        jump = False
        if threat is not None and (threat_d < 55.0 or recently_hurt):
            mode = "flee"
        elif recently_hurt and self.last_damage_bearing is not None:
            mode = "flee"  # hurt by someone unseen: run from the bearing
        elif outside_now:
            mode = "zone"
        elif threat is not None and threat_d <= 200.0:
            mode = "combat"
        elif need_next and shrink_soon:
            mode = "zone"
        else:
            mode = "opportunity"

        # --- opportunity mode: loot / drop / farm / collect, else patrol ---
        loot_goal = None
        loot_id = None
        loot_kind = None
        loot_z = None
        drop_goal = None
        farm_target = None
        if mode == "opportunity" and self.is_safe(me):
            # Weapon loot (best kind we do not already hold, nearest first).
            # Skip pickups far above us: a quarter of floor loot is on roofs
            # and this bot stays on the ground.
            slot1_kind = weapons[1]["kind"] if len(weapons) > 1 and weapons[1] else None
            best_pick = None
            for p in obs.get("pickups", []):
                k = p.get("kind")
                if k in held or k == "gun":
                    continue
                if k not in LOOT_PRIORITY:
                    continue
                if slot1_kind and LOOT_PRIORITY[k] >= LOOT_PRIORITY.get(slot1_kind, 99):
                    continue  # already holding something at least as good
                if p["pos"][2] > mz + 6.0:
                    continue  # rooftop pickup, out of reach on foot
                d = dist2(mx, my, p["pos"][0], p["pos"][1])
                if d > 150.0:
                    continue
                rank = (LOOT_PRIORITY[k], d)
                if best_pick is None or rank < best_pick[0]:
                    best_pick = (rank, p)
            if best_pick is not None:
                loot_goal = (best_pick[1]["pos"][0], best_pick[1]["pos"][1])
                loot_id = best_pick[1]["id"]
                loot_kind = best_pick[1]["kind"]
                loot_z = best_pick[1]["pos"][2]
            # Supply drop on the ground: 3 cubes + full HP, if nobody is near.
            if loot_goal is None:
                for d in obs.get("drops", []):
                    if d.get("lands_in", 99) > 0:
                        continue
                    dd = dist2(mx, my, d["pos"][0], d["pos"][1])
                    if dd < 250.0:
                        drop_goal = (d["pos"][0], d["pos"][1])
                        break
            # Farm: nearest crate or NPC drone in gun range.
            if loot_goal is None and drop_goal is None:
                best_farm, best_fd = None, 1e18
                for c in obs.get("crates", []):
                    d = dist2(mx, my, c["pos"][0], c["pos"][1])
                    if d < best_fd:
                        best_farm, best_fd = {"pos": list(c["pos"]), "vel": [0, 0, 0],
                                              "cubes": 0, "hp": c.get("hp"), "kind": "crate"}, d
                for n in obs.get("mechs", []):
                    if n.get("kind") != "npc":
                        continue
                    d = dist2(mx, my, n["pos"][0], n["pos"][1])
                    if d < best_fd:
                        best_farm, best_fd = {"pos": list(n["pos"]),
                                              "vel": list(n.get("vel") or [0, 0, 0]),
                                              "cubes": 0, "hp": n.get("hp"), "kind": "npc"}, d
                if best_farm is not None and best_fd < 130.0:
                    farm_target = best_farm
            # Loose cubes on the ground.
            if loot_goal is None and drop_goal is None and farm_target is None:
                best_cube, best_cd = None, 1e18
                for c in obs.get("cubes", []):
                    d = dist2(mx, my, c["pos"][0], c["pos"][1])
                    if d < best_cd:
                        best_cube, best_cd = (c["pos"][0], c["pos"][1]), d
                if best_cube is not None and best_cd < 60.0:
                    goal = best_cube
                    mode = "collect"
                    speed = 1.0 if best_cd > 30.0 else 0.45

        # --- resolve the mode into a movement goal ---
        aim = None
        fire = [False, False]
        if mode == "flee":
            if threat is not None:
                tx, ty = threat["pos"][0], threat["pos"][1]
                ax, ay = unit(tx, ty, mx, my)
            elif self.last_damage_bearing is not None:
                b = self.last_damage_bearing
                ax, ay = -math.cos(b), -math.sin(b)
            else:
                ax, ay = unit(mx, my, ncx, ncy)
            gx, gy = mx + ax * 120.0, my + ay * 120.0
            if need_next:
                bx, by = unit(mx, my, ncx, ncy)
                gx, gy = mx + (ax * 1.2 + bx * 0.8) * 100.0, my + (ay * 1.2 + by * 0.8) * 100.0
            goal = (gx, gy)
            speed = 1.0
            if me["energy"] > 30.0:
                boost = True
            target = threat  # shoot backwards while running
        elif mode == "zone":
            goal = (ncx, ncy)
            speed = 1.0
            if outside_now and dist2(mx, my, cx, cy) - cr > 100.0 and me["energy"] > 50.0:
                boost = True
        elif mode == "combat":
            target = threat
            # Keep distance: back off when close, strafe at range.
            if threat_d < 90.0:
                ax, ay = unit(threat["pos"][0], threat["pos"][1], mx, my)
                goal = (mx + ax * 50.0, my + ay * 50.0)
            else:
                if t > self.strafe_until:
                    self.strafe_dir *= -1
                    self.strafe_until = t + 2.5
                px, py = -unit(threat["pos"][0], threat["pos"][1], mx, my)[1], \
                    unit(threat["pos"][0], threat["pos"][1], mx, my)[0]
                goal = (mx + px * self.strafe_dir * 35.0, my + py * self.strafe_dir * 35.0)
            speed = 1.0
        elif loot_goal is not None:
            goal = loot_goal
            speed = 1.0 if dist2(mx, my, loot_goal[0], loot_goal[1]) > 40.0 else 0.45
            mode = "loot"
        elif drop_goal is not None:
            goal = drop_goal
            speed = 1.0 if dist2(mx, my, drop_goal[0], drop_goal[1]) > 40.0 else 0.45
            mode = "drop"
        elif farm_target is not None:
            target = farm_target
            fd = dist2(mx, my, farm_target["pos"][0], farm_target["pos"][1])
            if fd > 110.0:
                goal = (farm_target["pos"][0], farm_target["pos"][1])
                speed = 1.0
            else:
                # Hold range and strafe slowly so NPC bullets miss.
                if t > self.strafe_until:
                    self.strafe_dir *= -1
                    self.strafe_until = t + 2.0
                ax, ay = unit(farm_target["pos"][0], farm_target["pos"][1], mx, my)
                goal = (mx + (-ay) * self.strafe_dir * 20.0 + ax * 5.0,
                        my + (ax) * self.strafe_dir * 20.0 + ay * 5.0)
                speed = 0.45
            mode = "farm"
        elif mode == "collect":
            pass  # goal already set
        else:
            # Patrol near the next zone center: drift to a nearby point,
            # then scan (rotate the torso) to spot threats and loot early.
            if self.patrol_goal is None or t > self.patrol_until or \
                    dist2(mx, my, self.patrol_goal[0], self.patrol_goal[1]) < 5.0:
                px = clamp(ncx + self.rng.uniform(-40, 40), ncx - max(0.0, nr - 10), ncx + max(0.0, nr - 10))
                py = clamp(ncy + self.rng.uniform(-40, 40), ncy - max(0.0, nr - 10), ncy + max(0.0, nr - 10))
                self.patrol_goal = (px, py)
                self.patrol_until = t + 12.0
            goal = self.patrol_goal
            # Travel fast when the patrol point is far (it sits near the
            # next zone center, so this is the zone move), slow and quiet
            # once there.
            speed = 1.0 if dist2(mx, my, goal[0], goal[1]) > 60.0 else 0.45
            self.scan_yaw = ang_norm(self.scan_yaw + 0.05)

        # --- dodging overrides movement (but not a zone emergency) ---
        dodge = self.dodge_move(obs, me)
        if dodge is not None and mode != "zone":
            move = dodge
            if laser_aimed_at_us:
                boost = me["energy"] > 25.0
        elif goal is not None:
            move, stuck = self.move_toward(me, goal[0], goal[1], speed)
            if stuck and mode in ("flee", "zone", "loot", "drop", "farm", "collect"):
                if t > self.detour_until:
                    dg = self.detour_goal(me, goal[0], goal[1])
                    self.detour_until = t + 2.0
                    self.set_goal(dg[0], dg[1])
                    self.path = []
                    move, _ = self.move_toward(me, dg[0], dg[1], speed)
                    jump = True
        else:
            move = [0.0, 0.0]

        # --- take / drop a weapon when standing on it ---
        if loot_id is not None and loot_goal is not None:
            ld = dist2(mx, my, loot_goal[0], loot_goal[1])
            close_z = loot_z is None or abs(loot_z - mz) < 4.0
            if ld < 4.0 and close_z and self.t - self.failed_take.get(loot_id, -1e9) > 5.0:
                act["take"] = loot_id
                slot1 = weapons[1] if len(weapons) > 1 else None
                if slot1 is not None:
                    act["drop"] = 1
        act["move"] = move
        act["boost"] = bool(boost)
        act["jump"] = bool(jump)

        # --- aiming and firing ---
        # The torso can only track one solution per tick, so the weapons get
        # a strict priority for the aim: lightning (yaw only) > laser charge
        # > mortar > gun. A weapon only fires when the torso's actual aim is
        # already on its solution.
        if target is not None:
            is_mech = target is threat
            td = dist2(mx, my, target["pos"][0], target["pos"][1])
            gun_i = self.weapon_slot(weapons, "gun")
            laser_i = self.weapon_slot(weapons, "laser")
            mortar_i = self.weapon_slot(weapons, "mortar")
            light_i = self.weapon_slot(weapons, "lightning")
            aim_taken = False

            # Lightning up close beats the gun (240 dps + chains).
            if is_mech and light_i is not None and td < 40.0 and me["energy"] > 20.0:
                laim, _ = self.laser_aim(me, target)
                aim = [laim[0], 0.0]
                aim_taken = True
                dyaw, _ = self.aim_error(me, aim)
                if dyaw < 0.20:
                    fire[light_i] = True
            # Laser: charge in combat at range, release on a full charge when
            # the aim is on. While charging, the torso tracks the laser.
            if laser_i is not None and is_mech and 60.0 <= td <= 200.0 and mode == "combat":
                lw = weapons[laser_i]
                charge = lw.get("charge") or 0.0
                laim, _ = self.laser_aim(me, target)
                aim = laim
                aim_taken = True
                dyaw, dpit = self.aim_error(me, laim)
                if charge <= 0.0:
                    if dyaw < 0.5:
                        fire[laser_i] = True
                        self.laser_charging_since = t
                elif charge >= 0.7 and dyaw < 0.03 and dpit < 0.03:
                    fire[laser_i] = False  # release: beam fires along the aim
                    self.laser_charging_since = None
                elif self.laser_charging_since is not None and \
                        t - self.laser_charging_since > 2.2:
                    fire[laser_i] = False  # give up before the auto-fire
                    self.laser_charging_since = None
                else:
                    fire[laser_i] = True  # keep charging
            # Mortar: at visible targets, or at a last-known position (lob
            # over cover) when the target ducked out of sight < 5 s ago.
            if mortar_i is not None and is_mech and 60.0 <= td <= 220.0 and not aim_taken:
                mw = weapons[mortar_i]
                fresh = (vis_best is not None) or (self.t - threat.get("t", 0) < 5.0)
                if fresh and (mw.get("cooldown") or 0.0) <= 0.02:
                    maim, _ = self.mortar_aim(me, target)
                    if maim is not None:
                        aim = maim
                        aim_taken = True
                        dyaw, dpit = self.aim_error(me, maim)
                        if dyaw < 0.05 and dpit < 0.05:
                            fire[mortar_i] = True
            # Gun: fire when the (slewed) torso is actually on target.
            if gun_i is not None:
                gw = weapons[gun_i]
                gaim, _ = self.gun_aim(me, target)
                if not aim_taken:
                    aim = gaim
                dyaw, dpit = self.aim_error(me, gaim)
                ammo = gw.get("ammo")
                reloading = (gw.get("reload") or 0.0) > 0.0
                if ammo == 0 and not reloading:
                    act["reload"] = True
                elif td <= 200.0 and (gw.get("cooldown") or 0.0) <= 0.02 \
                        and not reloading and (ammo or 0) > 0 \
                        and dyaw < 0.035 and dpit < 0.035:
                    fire[gun_i] = True
        else:
            # No target: look where we are going, or scan while patrolling.
            if mode == "patrol":
                aim = [self.scan_yaw, 0.0]
            elif move and (abs(move[0]) > 0.05 or abs(move[1]) > 0.05):
                aim = [math.atan2(move[1], move[0]), 0.0]
            else:
                aim = [self.scan_yaw, 0.0]
            # Top up the gun magazine while it is quiet.
            gun_i = self.weapon_slot(weapons, "gun")
            if gun_i is not None:
                gw = weapons[gun_i]
                if (gw.get("ammo") or 0) < (gw.get("stats", {}).get("mag", 12)) \
                        and (gw.get("reload") or 0.0) <= 0.0 and self.is_safe(me):
                    act["reload"] = True

        if aim is not None:
            act["aim"] = [float(aim[0]), float(clamp(aim[1], -0.7, 1.3))]
        if any(fire):
            act["fire"] = fire
        self.mode = mode
        return act


mf.run(Bot)
