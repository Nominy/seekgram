"""Prints the DLL imports (normal and delay-load) of PE files.
Usage: uv run --no-project --with pefile python scripts/pe_imports.py <dll>..."""
import sys
import pefile
for path in sys.argv[1:]:
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"], pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
    normal = sorted(e.dll.decode() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []))
    delay = sorted(e.dll.decode() for e in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []))
    print(path.replace("\\", "/").split("/")[-1], "imports:", ", ".join(normal), "| delay:", ", ".join(delay))
