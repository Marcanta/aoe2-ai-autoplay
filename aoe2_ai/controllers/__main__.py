"""Controller CLIs.

    python -m aoe2_ai.controllers [options]                 # Town Center controller
    python -m aoe2_ai.controllers idle-villager [options]   # Idle villager controller
    python -m aoe2_ai.controllers assign-sheep [options]    # Villager assignment (sheep) controller
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] in ("idle-villager", "idle"):
    from .idle_villager import main

    main(sys.argv[2:])
elif len(sys.argv) > 1 and sys.argv[1] in ("assign-sheep", "sheep"):
    from .villager_assignment import main

    main(sys.argv[2:])
else:
    from .town_center import main

    main()
