from __future__ import annotations

import json

from crackfann.materialization.local_ann_store import available_backends


def main() -> None:
    print(json.dumps(available_backends(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
