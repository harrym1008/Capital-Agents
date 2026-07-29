import pandas as pd

from llm.tools.registry_builder import buildToolRegistry
from llm.boardroom_engine import generateBoardroom

from collectors.constants import UTC, NEW_YORK



if __name__ == "__main__":
    timestampStr = "2026-07-25"
    timestamp = pd.Timestamp(f"{timestampStr} 09:00").tz_localize(NEW_YORK).tz_convert(UTC)

    toolRegistry = buildToolRegistry()

    testFunc = "fetchCompanyValuationMetrics"
    testArgs = {"ticker": "NVDA"}

    result = toolRegistry.executeTool(
        toolName=testFunc,
        timestamp=timestamp,
        arguments=testArgs
    )

    import json
    print(json.dumps(result, indent=4))
    print(type(result))