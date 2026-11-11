import asyncio

from orchestrator.graphs.main_pipeline import build_pipeline_graph


async def main() -> None:
    graph = build_pipeline_graph().compile()
    while True:
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
