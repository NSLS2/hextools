"""Edit a plan queue as the Bluesky Queue Server does.

Positions, UID rules and error messages follow ``bluesky_queueserver``'s
``PlanQueueOperations``, so a queue edited in process behaves as one edited
through the Queue Server.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, Literal, Protocol, TypeVar

#: An index into the queue (negative counts from the back), ``"front"`` or ``"back"``.
QueuePosition = int | Literal["front", "back"]


class QueueError(Exception):
    """A queue operation failed; ``str`` says why, as the Queue Server's ``msg``."""


class _Item(Protocol):
    @property
    def item_uid(self) -> str: ...


T = TypeVar("T", bound=_Item)


class QueueEditor(Generic[T]):
    """Edit a list of queue items in place.

    A failed operation may leave the list partly edited, so edit a copy and keep
    it only if every operation succeeds.
    """

    def __init__(self, items: Sequence[T]):
        self.items = list(items)

    def get(self, *, pos: QueuePosition | None = None, uid: str | None = None) -> T:
        """Return the item at ``pos`` or with ``uid``; the back item by default."""
        return self.items[self._index(pos, uid)]

    def add(
        self,
        item: T,
        *,
        pos: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> T:
        """Insert ``item``, at the back by default.

        An index past either end puts the item at that end, and ``-1`` is the back.
        """
        if pos is not None and (before_uid is not None or after_uid is not None):
            raise QueueError("Ambiguous parameters: plan position and UID is specified")
        if before_uid is not None and after_uid is not None:
            raise QueueError(
                "Ambiguous parameters: request to insert the plan before and after "
                "the reference plan"
            )
        self._check_new_uid(item.item_uid)
        size = len(self.items)
        if before_uid is not None:
            index = self._index_of(before_uid)
        elif after_uid is not None:
            index = self._index_of(after_uid) + 1
        elif pos is None or pos == "back":
            index = size
        elif pos == "front":
            index = 0
        elif isinstance(pos, int) and not isinstance(pos, bool):
            index = min(pos, size) if pos >= 0 else max(size + pos + 1, 0)
        else:
            raise QueueError(f"Parameter 'pos' has incorrect value: pos={pos!r}")
        self.items.insert(index, item)
        return item

    def add_batch(
        self,
        items: Sequence[T],
        *,
        pos: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> list[T]:
        """Insert ``items`` together, in order, where :meth:`add` puts one."""
        added: list[T] = []
        for item in items:
            try:
                if added:
                    self.add(item, after_uid=added[-1].item_uid)
                else:
                    self.add(item, pos=pos, before_uid=before_uid, after_uid=after_uid)
            except QueueError as error:
                raise QueueError(f"Failed to add item to queue: {error}") from error
            added.append(item)
        return added

    def replace(self, item: T, *, item_uid: str) -> T:
        """Put ``item`` in place of the item with ``item_uid``."""
        index = self._index_of(item_uid)
        if item.item_uid != item_uid:
            self._check_new_uid(item.item_uid)
        self.items[index] = item
        return item

    def remove(self, *, pos: QueuePosition | None = None, uid: str | None = None) -> T:
        """Remove and return the item at ``pos`` or with ``uid``; the back one."""
        return self.items.pop(self._index(pos, uid))

    def remove_batch(
        self, *, uids: Sequence[str], ignore_missing: bool = True
    ) -> list[T]:
        """Remove the items with ``uids``, skipping missing ones if allowed."""
        if not ignore_missing:
            self._check_batch(uids)
        removed = []
        for uid in uids:
            try:
                removed.append(self.remove(uid=uid))
            except QueueError:
                pass
        return removed

    def move(
        self,
        *,
        pos: QueuePosition | None = None,
        uid: str | None = None,
        pos_dest: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> T:
        """Move the item at ``pos`` or with ``uid`` to ``pos_dest``, or by another."""
        if pos is None and uid is None:
            raise QueueError("Source position or UID is not specified.")
        if pos_dest is None and before_uid is None and after_uid is None:
            raise QueueError("Destination position or UID is not specified.")
        if pos is not None and uid is not None:
            raise QueueError(
                "Ambiguous parameters: Both position and uid is specified for the "
                "source plan."
            )
        if pos_dest is not None and (before_uid is not None or after_uid is not None):
            raise QueueError(
                "Ambiguous parameters: Both position and uid is specified for the "
                "destination plan."
            )
        if before_uid is not None and after_uid is not None:
            raise QueueError(
                "Ambiguous parameters: source should be moved 'before' and 'after' "
                "the destination."
            )
        source = f"UID {uid!r}" if uid is not None else f"position {pos}"
        try:
            src = self._index(pos, uid)
        except QueueError as error:
            raise QueueError(
                f"Source plan ({source}) was not found: {error}."
            ) from error
        if pos_dest is None:
            dest_uid = before_uid if before_uid is not None else after_uid
            destination = f"UID {dest_uid!r}"
        else:
            destination = f"position {pos_dest}"
        try:
            if pos_dest is None:
                dest = self._index_of(dest_uid)  # type: ignore[arg-type]
                before = before_uid is not None
            else:
                dest = self._index_at(pos_dest)
                # A plan moved to an index takes that place, whichever way it moves.
                before = pos_dest == "front" or (pos_dest != "back" and src > dest)
        except QueueError as error:
            raise QueueError(
                f"Destination plan ({destination}) was not found: {error}."
            ) from error
        if src == dest:
            return self.items[src]
        dest_uid = self.items[dest].item_uid
        item = self.items.pop(src)
        self.items.insert(self._index_of(dest_uid) + (0 if before else 1), item)
        return item

    def move_batch(
        self,
        *,
        uids: Sequence[str],
        pos_dest: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
        reorder: bool = False,
    ) -> list[T]:
        """Move the items with ``uids`` together to where :meth:`move` puts one.

        They keep the order of ``uids``, or with ``reorder`` their queue order.
        """
        destinations = [pos_dest, before_uid, after_uid].count(None)
        if destinations == 3:
            raise QueueError(
                "Destination for the batch is not specified: use parameters "
                "'pos_dest', 'before_uid' or 'after_uid'"
            )
        if destinations < 2:
            raise QueueError(
                "The function was called with more than one mutually exclusive "
                "parameter ('pos_dest', 'before_uid', 'after_uid')"
            )
        self._check_batch(uids)
        for name, ref in (("before_uid", before_uid), ("after_uid", after_uid)):
            if ref is not None and ref in uids:
                raise QueueError(
                    f"Parameter {name!r}: item with UID {ref!r} is in the batch"
                )
        if reorder:
            uids = sorted(uids, key=self._index_of)
        moved: list[T] = []
        for uid in uids:
            if moved:
                moved.append(self.move(uid=uid, after_uid=moved[-1].item_uid))
            else:
                moved.append(
                    self.move(
                        uid=uid,
                        pos_dest=pos_dest,
                        before_uid=before_uid,
                        after_uid=after_uid,
                    )
                )
        return moved

    def _index(self, pos: QueuePosition | None, uid: str | None) -> int:
        if pos is not None and uid is not None:
            raise QueueError("Ambiguous parameters: plan position and UID is specified")
        if uid is not None:
            return self._index_of(uid)
        return self._index_at("back" if pos is None else pos)

    def _index_of(self, uid: str) -> int:
        for index, item in enumerate(self.items):
            if item.item_uid == uid:
                return index
        raise QueueError(f"Item with UID {uid!r} is not in the queue")

    def _index_at(self, pos: QueuePosition) -> int:
        size = len(self.items)
        if pos in ("front", "back"):
            if not size:
                raise QueueError("Queue is empty")
            return 0 if pos == "front" else size - 1
        if isinstance(pos, bool) or not isinstance(pos, int):
            raise QueueError(f"Parameter 'pos' has incorrect value: pos={pos!r}")
        if not -size <= pos < size:
            raise QueueError(f"Index {pos} is out of range")
        return pos % size

    def _check_new_uid(self, uid: str) -> None:
        if not uid:
            raise QueueError("Item does not have UID.")
        if any(item.item_uid == uid for item in self.items):
            raise QueueError(f"Item with UID {uid} is already in the queue")

    def _check_batch(self, uids: Sequence[str]) -> None:
        if repeated := len(uids) - len(set(uids)):
            raise QueueError(f"The list contains repeated UIDs ({repeated} UIDs)")
        present = {item.item_uid for item in self.items}
        if missing := [uid for uid in uids if uid not in present]:
            raise QueueError(
                f"The queue does not contain items with the following UIDs: {missing}"
            )
