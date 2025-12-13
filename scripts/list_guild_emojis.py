#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tomllib
from pathlib import Path
from typing import Iterable, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tuner.discord_client import connect_tuner_client

DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/list_guild_emojis.py",
        description="Dump all custom emoji definitions for one or more Discord servers.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tuner config TOML (defaults to tuner/config.toml when present).",
    )
    parser.add_argument(
        "--guilds",
        "-g",
        nargs="+",
        type=int,
        default=None,
        help="Discord guild (server) IDs to inspect (overrides config).",
    )
    parser.add_argument(
        "--channels",
        "-c",
        nargs="+",
        type=int,
        default=None,
        help="Channel IDs whose parent guilds should be inspected (overrides config).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Discord bot token. Defaults to the value resolved from config/env.",
    )
    parser.add_argument(
        "--token-env",
        type=str,
        default=None,
        help="Environment variable that stores the Discord token (overrides config).",
    )
    return parser


def _load_file_config(path: Path | None) -> tuple[dict, Path | None]:
    target = path or DEFAULT_CONFIG
    resolved = target.resolve()
    if resolved.is_file():
        with resolved.open("rb") as handle:
            return tomllib.load(handle), resolved
    return {}, None


def _coerce_int_list(raw: Sequence[int | str] | None) -> list[int]:
    results: list[int] = []
    if not raw:
        return results
    for item in raw:
        if item is None:
            continue
        try:
            results.append(int(item))
        except (TypeError, ValueError):
            continue
    return results


async def _fetch_emojis(
    *,
    token: str,
    guild_ids: Iterable[int],
    channel_ids: Iterable[int],
) -> Set[str]:
    collected: Set[str] = set()

    async with connect_tuner_client(token) as client:
        resolved_guilds: Set[int] = {gid for gid in guild_ids if gid}

        derived_channels = [cid for cid in channel_ids if cid]
        if derived_channels:
            for channel_id in derived_channels:
                try:
                    channel = await client.fetch_channel(channel_id)
                except Exception as exc:  # pragma: no cover - network side effect
                    print(
                        f"[warn] Failed to inspect channel {channel_id}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                guild = getattr(channel, "guild", None)
                if guild is not None and getattr(guild, "id", None):
                    resolved_guilds.add(guild.id)
                elif getattr(channel, "guild_id", None):
                    resolved_guilds.add(int(channel.guild_id))

        if not resolved_guilds:
            print(
                "No guild IDs resolved; provide --guilds or ensure dataset.channels is populated.",
                file=sys.stderr,
            )
            return set()

        for guild_id in sorted(resolved_guilds):
            try:
                guild = await client.fetch_guild(guild_id)
                emojis = await guild.fetch_emojis()
            except Exception as exc:  # pragma: no cover - network side effect
                print(
                    f"[error] Failed to fetch emojis for guild {guild_id}: {exc}",
                    file=sys.stderr,
                )
                continue

            name = getattr(guild, "name", str(guild_id))
            print(f"\nGuild: {name} ({guild_id}) — {len(emojis)} custom emojis")
            if not emojis:
                print("  (no custom emojis)")
                continue

            for emoji in sorted(emojis, key=lambda item: item.name or ""):
                prefix = "a" if getattr(emoji, "animated", False) else ""
                discord_markup = f"<{prefix}:{emoji.name}:{emoji.id}>"
                print(f"  - {emoji.name}: {discord_markup}")
                collected.add(discord_markup)

    return collected


def _format_toml_list(entries: Sequence[str]) -> str:
    if not entries:
        return "[]"
    quoted = ", ".join(f'"{entry}"' for entry in entries)
    return f"[{quoted}]"


def _update_config_assistant_emojis(
    *,
    path: Path,
    tokens: Sequence[str],
) -> None:
    if not path.is_file():
        print(f"[warn] Config file {path} not found. Skipping update.")
        return

    existing = path.read_text(encoding="utf-8").splitlines()
    new_value = _format_toml_list(tokens)
    target_line = f"assistant_custom_emojis = {new_value}"

    for idx, line in enumerate(existing):
        stripped = line.lstrip()
        if not stripped.startswith("assistant_custom_emojis"):
            continue

        indent = line[: len(line) - len(stripped)]
        comment = ""
        if "#" in line:
            comment = line[line.index("#") :]
        existing[idx] = f"{indent}{target_line}"
        if comment:
            existing[idx] += f" {comment}"
        break
    else:
        # Append to dataset block (before reuse_raw if possible)
        inserted = False
        for idx, line in enumerate(existing):
            stripped = line.strip()
            if stripped.startswith("reuse_raw"):
                existing.insert(idx, f"{target_line}")
                inserted = True
                break
        if not inserted:
            existing.append(target_line)

    path.write_text("\n".join(existing) + "\n", encoding="utf-8")
    print(f"Updated assistant_custom_emojis in {path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    file_config, resolved_config_path = _load_file_config(args.config)
    finetune_cfg = file_config.get("finetune", {}) if isinstance(file_config, dict) else {}
    dataset_cfg = finetune_cfg.get("dataset", {}) or file_config.get("dataset", {})
    discord_cfg = finetune_cfg.get("discord", {}) or file_config.get("discord", {})

    token = args.token
    token_env = args.token_env or discord_cfg.get("token_env", "DISCORD_API_TOKEN")
    if not token:
        inline_token = discord_cfg.get("token")
        if inline_token:
            token = inline_token
    if not token:
        token = os.getenv(str(token_env))
    if not token:
        parser.error(
            f"Discord token missing. Provide --token, set discord.token, or export {token_env}."
        )

    guild_ids = args.guilds or _coerce_int_list(discord_cfg.get("guilds"))
    channel_ids = args.channels or _coerce_int_list(dataset_cfg.get("channels"))
    if resolved_config_path:
        config_path = resolved_config_path
    else:
        config_path = (args.config or DEFAULT_CONFIG).resolve()

    emoji_set = asyncio.run(
        _fetch_emojis(
            token=token,
            guild_ids=guild_ids,
            channel_ids=channel_ids,
        )
    )

    if not emoji_set:
        print("\nNo custom emojis discovered; config left unchanged.")
        return

    sorted_tokens = sorted(emoji_set)
    print(f"\nDiscovered {len(sorted_tokens)} unique custom emojis.")
    _update_config_assistant_emojis(path=config_path, tokens=sorted_tokens)
    print("assistant_custom_emojis now set to:")
    print(_format_toml_list(sorted_tokens))


if __name__ == "__main__":
    main()
