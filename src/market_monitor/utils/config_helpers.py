from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ruamel.yaml import YAML

reader = YAML(typ='safe')


@dataclass
class ConfigEntry:
    """Describes an available configuration option."""

    name: str
    path: Optional[Path]
    description: Optional[str] = None
    alias_of: Optional[str] = None


def _project_root() -> Path:
    # Dalla posizione del file (src/market_monitor/utils/)
    # risali a src/market_monitor/ -> src/ -> root progetto
    return Path(__file__).parent.parent.parent.parent


def _search_dirs() -> List[Path]:
    script_dir = _project_root()  # root del progetto
    return [
        script_dir / "etc" / "config",  # root/etc/config/
        script_dir / "config",  # root/config/
        script_dir / "src" / "config",  # root/src/config/
        script_dir,  # root/
    ]


def load_aliases() -> Dict[str, str]:
    """
    Load config aliases from aliases.yaml file.
    Searches in the same locations as config files.
    """
    for search_dir in _search_dirs():
        aliases_path = search_dir / "aliases.yaml"
        if aliases_path.exists():
            with open(aliases_path, 'r') as f:
                data = reader.load(f)
                return data.get('aliases', {})

    # Default fallback se non trova il file
    return {}


def _iter_config_files() -> Iterable[Path]:
    """Yield config files across all supported directories."""
    for search_dir in _search_dirs():
        if search_dir.exists():
            for pattern in ('*.yaml', '*.yml'):
                yield from search_dir.glob(pattern)


def _read_description(path: Path) -> Optional[str]:
    try:
        with open(path, 'r') as f:
            data = reader.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # Check simple top-level description or nested meta.description
    return data.get("description") or data.get("meta", {}).get("description")


def _unique_configs() -> Dict[str, Path]:
    """Return map of config name -> first-found path."""
    configs: Dict[str, Path] = {}
    for config_file in _iter_config_files():
        if config_file.stem == "aliases":
            continue
        configs.setdefault(config_file.stem, config_file)
    return configs


def find_all_configs() -> List[str]:
    """Find all available config files."""
    return sorted(_unique_configs())


def find_config(config_name: str) -> Path:
    """
    Search for config file in standard locations.
    Accepts config name with or without .yaml extension.
    """
    # Ensure we have the .yaml extension for searching
    if not config_name.endswith('.yaml') and not config_name.endswith('.yml'):
        config_variants = [f"{config_name}.yaml", f"{config_name}.yml"]
    else:
        config_variants = [config_name]

    # Search for config in all locations
    for search_dir in _search_dirs():
        for variant in config_variants:
            config_path = search_dir / variant
            if config_path.exists():
                return config_path

    # If not found, show helpful error message
    searched_paths = []
    for search_dir in _search_dirs():
        for variant in config_variants:
            searched_paths.append(str(search_dir / variant))

    raise FileNotFoundError(
        f"Config file '{config_name}' not found.\n"
        f"Searched in:\n" + "\n".join(f"  - {p}" for p in searched_paths)
    )


def resolve_config(name_or_path: str) -> Path:
    """
    Resolve alias or config name to actual config path.

    Args:
        name_or_path: Can be an alias (e.g. 'prod'),
                      a config name (e.g. 'production.yaml'),
                      or a direct path

    Returns:
        Path object to the config file
    """
    entry = resolve_config_entry(name_or_path)
    if not entry.path:
        raise FileNotFoundError(f"Config file '{name_or_path}' not found.")
    return entry.path


def resolve_config_entry(name_or_path: str) -> ConfigEntry:
    """
    Resolve user-provided name or alias into a ConfigEntry.
    The entry retains alias metadata for richer UX.
    """
    aliases = load_aliases()

    if name_or_path in aliases:
        config_name = aliases[name_or_path]
        alias_of = config_name
    else:
        config_name = name_or_path
        alias_of = None

    path = find_config(config_name)
    description = _read_description(path)
    return ConfigEntry(name=name_or_path, path=path, description=description, alias_of=alias_of)


def list_available():
    """List available aliases and config files."""
    aliases = load_aliases()
    configs = _unique_configs()

    print("Available aliases:")
    if aliases:
        for alias, config in sorted(aliases.items()):
            desc = _read_description(configs.get(config, find_config(config)))
            suffix = f" — {desc}" if desc else ""
            print(f"  {alias:12s} -> {config}{suffix}")
    else:
        print("  (no aliases configured)")

    print("\nAvailable config files:")
    if configs:
        for name, path in sorted(configs.items()):
            desc = _read_description(path)
            suffix = f" — {desc}" if desc else ""
            print(f"  - {name}{suffix}")
    else:
        print("  (no config files found)")


def available_names() -> Sequence[str]:
    """Return all known config and alias names for suggestion purposes."""
    names = set(find_all_configs())
    names.update(load_aliases())
    return sorted(names)


def suggest_names(name: str, max_suggestions: int = 3) -> List[str]:
    """Return close matches for a missing config/alias name."""
    return get_close_matches(name, available_names(), n=max_suggestions, cutoff=0.6)
