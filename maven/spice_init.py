import os
import re
from collections import defaultdict

def spice_init(kernel_root, output_file="maven_kernel.txt"):
    """
    Generate SPICE kernel init file in current directory, keeping only latest CK versions.
    Filters out mvn_app_pred_, mvn_sc_pred_, trj_orb_, .htm/.html, and retains only highest vNN per date.

    Parameters
    ----------
    kernel_root : str
        The root path to SPICE kernel files (e.g., 'F:\\kernels')
    output_file : str
        The name of the kernel list text file to write (default: 'maven_kernel.txt')
    """

    valid_extensions = ('.ti', '.tf', '.tls', '.tsc', '.tpc', '.bsp', '.bc')
    excluded_keywords = []
    excluded_extensions = ('.htm', '.html')

    # Normalize path
    kernel_root = kernel_root.replace("\\", "/")
    if not kernel_root.endswith("/"):
        kernel_root += "/"

    # Step 1: find all valid kernel files
    raw_files = []
    for root, _, files in os.walk(kernel_root):
        for file in files:
            file_lc = file.lower()
            full_path = os.path.join(root, file).replace("\\", "/")

            if (
                file_lc.endswith(valid_extensions)
                and not any(kw in file_lc for kw in excluded_keywords)
                and not file_lc.endswith(excluded_extensions)
            ):
                raw_files.append(full_path)

    # Step 2: filter duplicate CK versions
    latest_ck_files = {}
    versioned_ck_pattern = re.compile(r"(.+?)_v(\d{2})\.bc$", re.IGNORECASE)

    for f in raw_files:
        if not f.lower().endswith(".bc"):
            continue
        match = versioned_ck_pattern.search(os.path.basename(f))
        if match:
            prefix = match.group(1)
            vnum = int(match.group(2))
            if prefix not in latest_ck_files or vnum > latest_ck_files[prefix][1]:
                latest_ck_files[prefix] = (f, vnum)
        else:
            # Keep non-versioned .bc files
            latest_ck_files[f] = (f, 0)

    # Final list: merge with non-bc files
    final_files = [
        v[0] for k, v in latest_ck_files.items()
    ] + [f for f in raw_files if not f.endswith(".bc")]

    final_files = sorted(set(final_files))

    # Step 3: write to init file
    with open(output_file, "w") as f:
        f.write("\\begindata\n")
        f.write("KERNELS_TO_LOAD = (\n")
        for k in final_files:
            f.write(f"                  '{k}'\n")
        f.write("                  )\n")
        f.write("\\begintext\n")

    print(f"[INFO] Kernel init written to: {os.path.abspath(output_file)}")
    print(f"[INFO] Total {len(final_files)} kernel files included (latest versions only).")

if __name__ == "__main__":
    spice_init("F:\\kernels")
