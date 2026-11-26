import asyncio
import time

from common.config import settings
from common.inference.schemas import RerankResult

from ml_inference.engine import MLEngine


class BatchingQueueWorker:
    """Dynamic micro-batching worker aggregating concurrent requests into batches."""

    def __init__(
        self,
        engine: MLEngine,
        max_batch_size: int | None = None,
        max_wait_ms: int | None = None,
    ) -> None:
        self.engine = engine
        self.max_batch_size = max_batch_size or settings.ml_batch_max_size
        self.max_wait_ms = max_wait_ms or settings.ml_batch_max_wait_ms

        self._embed_queue: (
            asyncio.Queue[tuple[list[str], asyncio.Future[list[list[float]]]]] | None
        ) = None
        self._rerank_queue: (
            asyncio.Queue[
                tuple[
                    str, list[str], str | None, bool, int | None, asyncio.Future[list[RerankResult]]
                ]
            ]
            | None
        ) = None

        self._running = False
        self._embed_task: asyncio.Task[None] | None = None
        self._rerank_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._running = True
        self._embed_queue = asyncio.Queue()
        self._rerank_queue = asyncio.Queue()
        self._embed_task = asyncio.create_task(self._process_embed_queue())
        self._rerank_task = asyncio.create_task(self._process_rerank_queue())

    async def stop(self) -> None:
        self._running = False
        if self._embed_task is not None:
            self._embed_task.cancel()
            try:
                await self._embed_task
            except asyncio.CancelledError:
                pass
            self._embed_task = None
        if self._rerank_task is not None:
            self._rerank_task.cancel()
            try:
                await self._rerank_task
            except asyncio.CancelledError:
                pass
            self._rerank_task = None

    async def enqueue_embed(self, texts: list[str]) -> list[list[float]]:
        if not self._running or self._embed_queue is None:
            self.start()
        assert self._embed_queue is not None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[list[float]]] = loop.create_future()
        await self._embed_queue.put((texts, future))
        return await future

    async def enqueue_rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
        return_documents: bool = False,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not self._running or self._rerank_queue is None:
            self.start()
        assert self._rerank_queue is not None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[RerankResult]] = loop.create_future()
        await self._rerank_queue.put(
            (query, documents, instruction, return_documents, top_n, future)
        )
        return await future

    async def _process_embed_queue(self) -> None:
        while self._running:
            if self._embed_queue is None:
                break
            try:
                first_item = await self._embed_queue.get()
            except asyncio.CancelledError:
                break

            batch = [first_item]
            deadline = time.perf_counter() + (self.max_wait_ms / 1000.0)

            while len(batch) < self.max_batch_size:
                remaining_time = deadline - time.perf_counter()
                if remaining_time <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._embed_queue.get(), timeout=remaining_time)
                    batch.append(item)
                except (TimeoutError, asyncio.QueueEmpty):
                    break
                except asyncio.CancelledError:
                    return

            all_texts: list[str] = []
            slices: list[tuple[int, int]] = []
            for texts, _ in batch:
                start_idx = len(all_texts)
                all_texts.extend(texts)
                slices.append((start_idx, len(all_texts)))

            try:
                if self.engine.cuda_lock is not None:
                    async with self.engine.cuda_lock:
                        all_embeddings = await asyncio.to_thread(
                            self.engine.compute_embeddings, all_texts
                        )
                else:
                    all_embeddings = await asyncio.to_thread(
                        self.engine.compute_embeddings, all_texts
                    )

                for (_, future), (st, end) in zip(batch, slices, strict=False):
                    if not future.done():
                        future.set_result(all_embeddings[st:end])
            except Exception as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)

    async def _process_rerank_queue(self) -> None:
        while self._running:
            if self._rerank_queue is None:
                break
            try:
                item = await self._rerank_queue.get()
            except asyncio.CancelledError:
                break

            query, documents, instruction, return_documents, top_n, future = item
            try:
                if self.engine.cuda_lock is not None:
                    async with self.engine.cuda_lock:
                        results = await asyncio.to_thread(
                            self.engine.compute_rerank,
                            query,
                            documents,
                            instruction,
                            return_documents,
                            top_n,
                        )
                else:
                    results = await asyncio.to_thread(
                        self.engine.compute_rerank,
                        query,
                        documents,
                        instruction,
                        return_documents,
                        top_n,
                    )

                if not future.done():
                    future.set_result(results)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
