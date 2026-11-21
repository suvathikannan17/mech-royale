# Suvi Achu — Mech Royale bot

A survival-first bot for the Builders Mech Royale 16-mech battle royale
(`mechfight/1`).

Placement counts, not kills — and practice matches are 1 v 15, with the
house bots sharing vision and hunting the weakest mech first. So this bot
plays to outlive:

- **Stay quiet.** Walks slowly near threats; gunfire is heard 320 m away,
  so it only shoots when it has a target and only farms crates / NPC
  drones while the area is quiet.
- **Stay in the zone.** Moves to the next circle early, during hold phases,
  using A* over a rasterized map of the campus buildings so it does not
  walk into walls.
- **Loot smart.** Laser first (315 damage hitscan at 280 m), then mortar
  (lobs over cover at last-known positions), then lightning (close range).
- **Fight at range, flee up close.** Leads moving targets with the gun,
  charges the laser and releases on a full charge when the torso is on
  target, and shoots backwards while running away.
- **Dodge.** Sidesteps incoming bullets and mortar shells, and breaks away
  from a telegraphed laser charge aimed at it.
- **Upgrades:** armor and plating first, then sensors, speed, and energy.

## Files

- `bot.py` — the bot (standard library only)
- `mechfight.py` — the kit's stdin/stdout wrapper (unchanged)
- `Dockerfile` — `python:3.13-slim`, entrypoint `bot.py`

## Test

```sh
python3 ../mechfight-kit/check/check.py -- python3 -u bot.py
docker build -t my-bot .
python3 ../mechfight-kit/check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/../mechfight-kit/arena:/arena:ro" my-bot
```

Team: Suvi Achu (Suvathi Kannan, Achudan Thudukutchi Sadhasivam).
License: MIT.
