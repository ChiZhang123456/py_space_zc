import spiceypy as spice
import os

def load_maven_spice():
    """
    Load MAVEN SPICE kernels from 'maven_kernel.txt' in the same directory as this module.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_path, "maven_kernel.txt")
    spice.furnsh(filename)
    print(f"[INFO] SPICE kernels loaded from: {filename}")
