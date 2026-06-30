from __future__ import annotations

import taichi as ti

from fsi.reference_cases import default_reference_data_dir, generate_reference_datasets


def main() -> None:
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False, offline_cache=False)

    output_dir = default_reference_data_dir()
    print("This overwrites committed reference JSON files.")
    print("Review the generated git diff before committing refreshed references.")
    paths = generate_reference_datasets(output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
