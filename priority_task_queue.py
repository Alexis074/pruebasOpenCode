"""Sistema de cola de tareas con prioridad, seguro para hilos (thread-safe).

Ejecutar con:
    python priority_task_queue.py

Características:
  * Tareas con prioridad ALTA / MEDIA / BAJA y un payload arbitrario.
  * WorkerPool que consume la cola siempre por orden de prioridad (FIFO entre
    tareas de la misma prioridad).
  * Cancelación de tareas pendientes y captura de excepciones por tarea.
  * Cada tarea devuelve un concurrent.futures.Future con su resultado o error.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable
from uuid import uuid4

LOG = logging.getLogger("taskqueue")


# ---------------------------------------------------------------------------
# Modelo de tareas
# ---------------------------------------------------------------------------
class Priority(IntEnum):
    """Menor valor = mayor prioridad (así lo ordena heapq internamente)."""

    ALTA = 0
    MEDIA = 1
    BAJA = 2


class TaskStatus(str, Enum):
    PENDING = "PENDIENTE"
    RUNNING = "EN EJECUCIÓN"
    DONE = "COMPLETADA"
    FAILED = "FALLIDA"
    CANCELLED = "CANCELADA"


@dataclass
class Task:
    """Una unidad de trabajo con prioridad y datos de entrada."""

    name: str
    priority: Priority
    func: Callable[[Any], Any]
    payload: Any = None
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    status: TaskStatus = TaskStatus.PENDING

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise TypeError(f"La tarea '{self.name}' necesita una función ejecutable")


@dataclass(order=True)
class _QueueItem:
    """Elemento interno de la cola.

    La comparación usa solo (prioridad, secuencia). La secuencia garantiza un
    orden FIFO estable entre tareas con la misma prioridad y evita comparar
    objetos Task (que no son ordenables).
    """

    priority: int
    sequence: int
    task: Task = field(compare=False)
    future: Future = field(compare=False)


# ---------------------------------------------------------------------------
# WorkerPool
# ---------------------------------------------------------------------------
class WorkerPool:
    """Pool de hilos que consume una cola de prioridad thread-safe."""

    def __init__(self, num_workers: int = 4, name: str = "worker") -> None:
        if num_workers < 1:
            raise ValueError("num_workers debe ser >= 1")

        self._num_workers = num_workers
        self._pool_name = name
        self._queue: queue.PriorityQueue = queue.PriorityQueue()

        # Un único lock protege el estado compartido (mapas de tareas/futuros,
        # contador de secuencia, conjunto de canceladas...). RLock para poder
        # reutilizar helpers internos sin miedo a reentradas.
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._started = False

        self._threads = [
            threading.Thread(
                target=self._worker_loop,
                name=f"{name}-{i}",
                daemon=True,
            )
            for i in range(1, num_workers + 1)
        ]

        self._tasks: dict[str, Task] = {}
        self._futures: dict[str, Future] = {}
        self._cancelled: set[str] = set()
        self._sequence = 0

    # -- ciclo de vida ------------------------------------------------------
    def start(self) -> None:
        """Arranca los hilos trabajadores (idempotente)."""
        with self._lock:
            if self._started:
                return
            if self._stop.is_set():
                raise RuntimeError("El pool ya fue cerrado")
            self._started = True
            for thread in self._threads:
                thread.start()
        LOG.info("Pool '%s' iniciado con %d workers", self._pool_name, self._num_workers)

    def shutdown(self, wait: bool = True) -> None:
        """Señala a los workers que terminen.

        Los elementos que ya estén en la cola se terminan de drenar antes de
        que cada worker salga. No se aceptan nuevas tareas tras el cierre.
        """
        with self._lock:
            self._stop.set()
        if wait:
            for thread in self._threads:
                if thread.is_alive():
                    thread.join()
        LOG.info("Pool '%s' cerrado", self._pool_name)

    def join(self) -> None:
        """Bloquea hasta que todas las tareas encoladas hayan sido procesadas."""
        self._queue.join()

    # -- API de tareas ------------------------------------------------------
    def submit(self, task: Task) -> Future:
        """Encola una tarea y devuelve un Future para consultar su resultado."""
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("No se pueden enviar tareas a un pool cerrado")
            if task.id in self._tasks:
                raise ValueError(f"ID de tarea duplicado: {task.id}")

            future: Future = Future()
            self._sequence += 1
            self._tasks[task.id] = task
            self._futures[task.id] = future
            item = _QueueItem(int(task.priority), self._sequence, task, future)
            self._queue.put(item)

        LOG.info(
            "Encolada   %-18s prioridad=%-5s id=%s",
            task.name,
            task.priority.name,
            task.id,
        )
        return future

    def cancel(self, task_id: str) -> bool:
        """Cancela una tarea si aún está pendiente.

        Devuelve True si se canceló. Si la tarea ya se está ejecutando o
        terminó, no se puede interrumpir y devuelve False.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            future = self._futures.get(task_id)
            if task is None or future is None:
                return False
            # Future.cancel() solo tiene éxito si aún no ha empezado a correr.
            if not future.cancel():
                return False

            self._cancelled.add(task_id)
            task.status = TaskStatus.CANCELLED

        LOG.warning("Cancelada  %-18s id=%s", task.name, task.id)
        return True

    # -- internos -----------------------------------------------------------
    def _begin(self, task: Task, future: Future) -> bool:
        """Marca la tarea como en ejecución de forma atómica."""
        with self._lock:
            if task.id in self._cancelled:
                task.status = TaskStatus.CANCELLED
                return False
            if not future.set_running_or_notify_cancel():
                return False
            task.status = TaskStatus.RUNNING
            return True

    def _finish(
        self,
        task: Task,
        future: Future,
        result: Any = None,
        exc: BaseException | None = None,
    ) -> None:
        with self._lock:
            task.status = TaskStatus.FAILED if exc is not None else TaskStatus.DONE

        if exc is not None:
            future.set_exception(exc)
        else:
            future.set_result(result)

    def _worker_loop(self) -> None:
        thread_name = threading.current_thread().name
        LOG.debug("[%s] listo", thread_name)

        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue

            task = item.task
            try:
                # Puede haber sido cancelada mientras esperaba en la cola.
                if not self._begin(task, item.future):
                    continue

                LOG.info(
                    "[%s] Ejecutando %-18s prioridad=%s",
                    thread_name,
                    task.name,
                    task.priority.name,
                )
                try:
                    result = task.func(task.payload)
                except Exception as exc:  # una tarea rota no debe matar al worker
                    self._finish(task, item.future, exc=exc)
                    LOG.error(
                        "[%s] FALLO      %-18s -> %s: %s",
                        thread_name,
                        task.name,
                        type(exc).__name__,
                        exc,
                    )
                    LOG.debug(
                        "Traza de la tarea '%s':\n%s",
                        task.name,
                        traceback.format_exc(),
                    )
                else:
                    self._finish(task, item.future, result=result)
                    LOG.info(
                        "[%s] Completada %-18s resultado=%r",
                        thread_name,
                        task.name,
                        result,
                    )
            finally:
                # Limpieza de estado y contabilidad de la cola, pase lo que pase.
                with self._lock:
                    self._tasks.pop(task.id, None)
                    self._futures.pop(task.id, None)
                    self._cancelled.discard(task.id)
                self._queue.task_done()

        LOG.debug("[%s] detenido", thread_name)


# ---------------------------------------------------------------------------
# Funciones de ejemplo
# ---------------------------------------------------------------------------
def calcular(payload: dict) -> str:
    """Simula un cálculo costoso."""
    time.sleep(payload.get("delay", 0.1))
    n = payload["n"]
    total = sum(i * i for i in range(1, n + 1))
    return f"suma de cuadrados 1..{n} = {total}"


def procesar_texto(payload: dict) -> str:
    """Simula el procesamiento de un texto."""
    time.sleep(payload.get("delay", 0.1))
    return payload["texto"].upper()


def tarea_que_falla(payload: dict) -> None:
    """Fuerza un error para demostrar el manejo de excepciones."""
    time.sleep(payload.get("delay", 0.1))
    raise ValueError(f"fallo simulado al procesar el lote {payload['lote']!r}")


# ---------------------------------------------------------------------------
# Demo ejecutable
# ---------------------------------------------------------------------------
def configurar_logging() -> None:
    # logging escribe en stderr por defecto; forzamos UTF-8 en ambas salidas
    # para que los acentos se vean bien también cuando la salida es una tubería.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(threadName)-12s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    configurar_logging()

    # 2 workers: suficiente para ver concurrencia real y, a la vez, que la cola
    # acumule tareas y se aprecie el orden por prioridad.
    pool = WorkerPool(num_workers=2, name="worker")

    definiciones = [
        # (nombre, prioridad, función, payload)
        ("baja-informe",        Priority.BAJA,  procesar_texto,   {"texto": "informe mensual", "delay": 0.12}),
        ("alta-pago",           Priority.ALTA,  calcular,         {"n": 10, "delay": 0.10}),
        ("media-email",         Priority.MEDIA, procesar_texto,   {"texto": "correo de bienvenida", "delay": 0.10}),
        ("baja-limpieza",       Priority.BAJA,  calcular,         {"n": 5, "delay": 0.12}),
        ("alta-inventario",     Priority.ALTA,  tarea_que_falla,  {"lote": "A-17", "delay": 0.08}),
        ("alta-login",          Priority.ALTA,  procesar_texto,   {"texto": "sesion validada", "delay": 0.08}),
        ("media-reporte",       Priority.MEDIA, calcular,         {"n": 7, "delay": 0.10}),
        ("baja-archivo",        Priority.BAJA,  procesar_texto,   {"texto": "archivo temporal", "delay": 0.12}),
        ("media-notificacion",  Priority.MEDIA, procesar_texto,   {"texto": "notificacion push", "delay": 0.10}),
        ("media-backup",        Priority.MEDIA, calcular,         {"n": 3, "delay": 0.10}),
    ]

    tareas: dict[str, Task] = {}
    futuros: dict[str, Future] = {}

    LOG.info("=== Enviando 10 tareas (aún sin workers activos) ===")
    for nombre, prioridad, func, payload in definiciones:
        task = Task(name=nombre, priority=prioridad, func=func, payload=payload)
        tareas[nombre] = task
        futuros[nombre] = pool.submit(task)

    # Cancelación: 'baja-archivo' sigue en la cola, nadie la ha tomado todavía.
    LOG.info("=== Cancelando una tarea pendiente ===")
    cancelada = tareas["baja-archivo"]
    if not pool.cancel(cancelada.id):
        LOG.error("No se pudo cancelar '%s' (ya estaba en ejecución)", cancelada.name)

    LOG.info("=== Arrancando workers ===")
    pool.start()

    # Espera a que se procesen todas las tareas encoladas (incluidas las
    # canceladas, que se descartan al sacarlas de la cola).
    pool.join()
    pool.shutdown()

    LOG.info("=== Resumen de resultados ===")
    for nombre, future in futuros.items():
        if future.cancelled():
            LOG.info("%-20s -> CANCELADA", nombre)
        elif future.exception() is not None:
            LOG.info("%-20s -> FALLÓ (%s)", nombre, future.exception())
        else:
            LOG.info("%-20s -> OK: %s", nombre, future.result())


if __name__ == "__main__":
    main()
