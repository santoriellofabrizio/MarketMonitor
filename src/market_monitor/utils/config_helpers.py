from pathlib import Path

from ruamel.yaml import YAML

reader = YAML(typ='safe')


def load_aliases():
    """
    Load config aliases from aliases.yaml file.
    Searches in the same locations as config files.
    """
    # Dalla posizione del file (src/market_monitor/utils/)
    # risali a src/market_monitor/ -> src/ -> root progetto
    script_dir = Path(__file__).parent.parent.parent.parent  # root del progetto

    search_dirs = [
        script_dir / "etc" / "config",  # root/etc/config/
        script_dir / "config",  # root/config/
        script_dir / "src" / "config",  # root/src/config/
        script_dir,  # root/
    ]

    for search_dir in search_dirs:
        aliases_path = search_dir / "aliases.yaml"
        if aliases_path.exists():
            with open(aliases_path, 'r') as f:
                data = reader.load(f)
                return data.get('aliases', {})  # Aggiungi .get('aliases', {})

    # Default fallback se non trova il file
    return {}


def find_all_configs():
    """Find all available config files."""
    # Root del progetto
    script_dir = Path(__file__).parent.parent.parent.parent

    search_dirs = [
        script_dir / "etc" / "config",
        script_dir / "config",
        script_dir / "src" / "config",
        script_dir,
    ]

    configs = set()
    for search_dir in search_dirs:
        if search_dir.exists():
            for pattern in ['*.yaml', '*.yml']:
                for config_file in search_dir.glob(pattern):
                    # Escludi aliases.yaml dalla lista
                    if config_file.stem != 'aliases':
                        configs.add(config_file.stem)

    return sorted(configs)


def find_config(config_name):
    """
    Search for config file in standard locations.
    Accepts config name with or without .yaml extension.
    """
    # Ensure we have the .yaml extension for searching
    if not config_name.endswith('.yaml') and not config_name.endswith('.yml'):
        config_variants = [f"{config_name}.yaml", f"{config_name}.yml"]
    else:
        config_variants = [config_name]

    # Root del progetto
    script_dir = Path(__file__).parent.parent.parent.parent

    # Define search paths - STESSO ORDINE di load_aliases
    search_dirs = [
        script_dir / "etc" / "config",
        script_dir / "config",
        script_dir / "src" / "config",
        script_dir,
    ]

    # Search for config in all locations
    for search_dir in search_dirs:
        for variant in config_variants:
            config_path = search_dir / variant
            if config_path.exists():
                return config_path

    # If not found, show helpful error message
    searched_paths = []
    for search_dir in search_dirs:
        for variant in config_variants:
            searched_paths.append(str(search_dir / variant))

    raise FileNotFoundError(
        f"Config file '{config_name}' not found.\n"
        f"Searched in:\n" + "\n".join(f"  - {p}" for p in searched_paths)
    )


def resolve_config(name_or_path):
    """
    Resolve alias or config name to actual config path.

    Args:
        name_or_path: Can be an alias (e.g. 'prod'),
                      a config name (e.g. 'production.yaml'),
                      or a direct path

    Returns:
        Path object to the config file
    """
    # Load aliases from file
    aliases = load_aliases()

    # Check if it's an alias
    if name_or_path in aliases:
        config_name = aliases[name_or_path]
        print(f"Using alias '{name_or_path}' -> {config_name}")
    else:
        config_name = name_or_path

    return find_config(config_name)


def list_available():
    """List available aliases and config files."""
    aliases = load_aliases()

    print("Available aliases:")
    if aliases:
        for alias, config in aliases.items():
            print(f"  {alias:12s} -> {config}")
    else:
        print("  (no aliases configured)")

    print("\nAvailable config files:")
    configs = find_all_configs()
    if configs:
        for cfg in configs:
            print(f"  - {cfg}")
    else:
        print("  (no config files found)")