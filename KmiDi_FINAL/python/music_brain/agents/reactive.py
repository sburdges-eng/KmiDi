#!/usr/bin/env python3
"""
Reactive State Management for UnifiedHub.

Provides observable state containers that automatically notify subscribers
when properties change. Enables seamless UI integration and event-driven
architectures.

Usage:
    from music_brain.agents.reactive import ReactiveState, observe

    # Create reactive state
    state = ReactiveState({"tempo": 120, "playing": False})

    # Subscribe to all changes
    state.subscribe(lambda key, old, new: print(f"{key}: {old} -> {new}"))

    # Subscribe to specific keys
    state.subscribe(lambda k, o, n: print(f"Tempo changed!"), keys=["tempo"])

    # Changes trigger callbacks
    state.tempo = 140  # prints: "tempo: 120 -> 140" and "Tempo changed!"

    # Observe decorator for methods
    class MyClass:
        @observe("voice_state")
        def set_vowel(self, vowel):
            self._vowel = vowel
            return {"vowel": vowel}  # Returned dict is broadcast
"""

from __future__ import annotations

import asyncio
import functools
import threading
import weakref
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

T = TypeVar("T")

# Type alias for state change callbacks
StateCallback = Callable[[str, Any, Any], None]  # (key, old_value, new_value)
AsyncStateCallback = Callable[[str, Any, Any], "asyncio.Future[None]"]


# =============================================================================
# Core Reactive Primitives
# =============================================================================


class Observable:
    """
    Base mixin for objects that can notify observers of changes.

    Supports both sync and async callbacks.
    """

    def __init__(self):
        self._observers: List[weakref.ref] = []
        self._async_observers: List[weakref.ref] = []
        self._lock = threading.RLock()

    def add_observer(
        self,
        callback: Union[StateCallback, AsyncStateCallback],
        is_async: bool = False,
    ) -> Callable[[], None]:
        """
        Add an observer callback.

        Args:
            callback: Function called on state change
            is_async: True if callback is async

        Returns:
            Unsubscribe function
        """
        ref = weakref.ref(callback) if hasattr(callback, "__self__") else lambda: callback

        with self._lock:
            if is_async:
                self._async_observers.append(ref)
            else:
                self._observers.append(ref)

        def unsubscribe():
            with self._lock:
                if is_async:
                    self._async_observers = [r for r in self._async_observers if r() is not callback]
                else:
                    self._observers = [r for r in self._observers if r() is not callback]

        return unsubscribe

    def remove_observer(self, callback: Union[StateCallback, AsyncStateCallback]):
        """Remove an observer by reference."""
        with self._lock:
            self._observers = [r for r in self._observers if r() is not callback]
            self._async_observers = [r for r in self._async_observers if r() is not callback]

    def notify(self, key: str, old_value: Any, new_value: Any):
        """Notify all observers of a change."""
        # Clean dead refs and call sync observers
        with self._lock:
            live_observers = []
            for ref in self._observers:
                cb = ref()
                if cb is not None:
                    live_observers.append(ref)
                    try:
                        cb(key, old_value, new_value)
                    except Exception as e:
                        print(f"[ReactiveState] Observer error: {e}")
            self._observers = live_observers

            # Schedule async observers
            live_async = []
            for ref in self._async_observers:
                cb = ref()
                if cb is not None:
                    live_async.append(ref)
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(cb(key, old_value, new_value))
                    except RuntimeError:
                        # No running loop - skip async callback
                        pass
            self._async_observers = live_async


class ReactiveState(Observable, Generic[T]):
    """
    Observable state container with automatic change detection.

    Wraps a dict or dataclass and fires callbacks when properties change.

    Example:
        state = ReactiveState({"tempo": 120})
        state.subscribe(lambda k, o, n: print(f"{k} changed"))
        state.tempo = 140  # Triggers callback
    """

    def __init__(
        self,
        initial: Optional[Union[Dict[str, Any], T]] = None,
        name: str = "state",
    ):
        super().__init__()
        self._name = name
        self._key_filters: Dict[int, Set[str]] = {}  # observer_id -> filtered keys

        # Initialize internal state
        if initial is None:
            self._data: Dict[str, Any] = {}
        elif is_dataclass(initial) and not isinstance(initial, type):
            self._data = asdict(initial)
        elif isinstance(initial, dict):
            self._data = dict(initial)
        else:
            self._data = {}

    def subscribe(
        self,
        callback: StateCallback,
        keys: Optional[List[str]] = None,
        is_async: bool = False,
    ) -> Callable[[], None]:
        """
        Subscribe to state changes.

        Args:
            callback: Called with (key, old_value, new_value)
            keys: If provided, only notify for these keys
            is_async: True if callback is async

        Returns:
            Unsubscribe function
        """
        if keys:
            self._key_filters[id(callback)] = set(keys)

        unsub = self.add_observer(callback, is_async)

        def full_unsub():
            self._key_filters.pop(id(callback), None)
            unsub()

        return full_unsub

    def notify(self, key: str, old_value: Any, new_value: Any):
        """Override to apply key filters."""
        with self._lock:
            live_observers = []
            for ref in self._observers:
                cb = ref()
                if cb is None:
                    continue
                live_observers.append(ref)

                # Check key filter
                allowed_keys = self._key_filters.get(id(cb))
                if allowed_keys and key not in allowed_keys:
                    continue

                try:
                    cb(key, old_value, new_value)
                except Exception as e:
                    print(f"[ReactiveState:{self._name}] Observer error: {e}")

            self._observers = live_observers

            # Async observers
            live_async = []
            for ref in self._async_observers:
                cb = ref()
                if cb is None:
                    continue
                live_async.append(ref)

                allowed_keys = self._key_filters.get(id(cb))
                if allowed_keys and key not in allowed_keys:
                    continue

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(cb(key, old_value, new_value))
                except RuntimeError:
                    pass

            self._async_observers = live_async

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        old_value = self._data.get(name)
        if old_value != value:
            self._data[name] = value
            self.notify(name, old_value, value)

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def __setitem__(self, key: str, value: Any):
        old_value = self._data.get(key)
        if old_value != value:
            self._data[key] = value
            self.notify(key, old_value, value)

    def update(self, changes: Dict[str, Any], silent: bool = False):
        """
        Update multiple values at once.

        Args:
            changes: Dict of key -> new_value
            silent: If True, don't trigger callbacks
        """
        for key, value in changes.items():
            old_value = self._data.get(key)
            if old_value != value:
                self._data[key] = value
                if not silent:
                    self.notify(key, old_value, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with optional default."""
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of the state as a dict."""
        return dict(self._data)

    def snapshot(self) -> Dict[str, Any]:
        """Alias for to_dict - returns immutable snapshot."""
        return self.to_dict()

    def __repr__(self) -> str:
        return f"ReactiveState({self._name}, {self._data})"


# =============================================================================
# Reactive Dataclass Wrapper
# =============================================================================


def reactive_dataclass(cls: type) -> type:
    """
    Decorator that makes a dataclass reactive.

    All attribute changes will trigger notifications.

    Example:
        @reactive_dataclass
        @dataclass
        class VoiceState:
            vowel: str = "A"
            pitch: int = 60
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} must be a dataclass")

    original_init = cls.__init__
    original_setattr = cls.__setattr__ if hasattr(cls, "__setattr__") else object.__setattr__

    def new_init(self, *args, **kwargs):
        object.__setattr__(self, "_reactive_observers", [])
        object.__setattr__(self, "_reactive_lock", threading.RLock())
        object.__setattr__(self, "_reactive_initializing", True)
        original_init(self, *args, **kwargs)
        object.__setattr__(self, "_reactive_initializing", False)

    def new_setattr(self, name: str, value: Any):
        if name.startswith("_reactive"):
            object.__setattr__(self, name, value)
            return

        old_value = getattr(self, name, None)
        original_setattr(self, name, value)

        # Only notify after init and if value changed
        if not getattr(self, "_reactive_initializing", True) and old_value != value:
            self._notify_change(name, old_value, value)

    def subscribe(self, callback: StateCallback) -> Callable[[], None]:
        """Subscribe to state changes."""
        with self._reactive_lock:
            self._reactive_observers.append(callback)

        def unsub():
            with self._reactive_lock:
                if callback in self._reactive_observers:
                    self._reactive_observers.remove(callback)

        return unsub

    def _notify_change(self, key: str, old_value: Any, new_value: Any):
        with self._reactive_lock:
            for cb in list(self._reactive_observers):
                try:
                    cb(key, old_value, new_value)
                except Exception as e:
                    print(f"[reactive_dataclass] Observer error: {e}")

    cls.__init__ = new_init
    cls.__setattr__ = new_setattr
    cls.subscribe = subscribe
    cls._notify_change = _notify_change

    return cls


# =============================================================================
# Method Decorator for Observing Changes
# =============================================================================


def observe(state_name: str):
    """
    Decorator that broadcasts method return value as a state change.

    The decorated method should return a dict of changed values,
    which will be broadcast to observers of the named state.

    Example:
        class Hub:
            def __init__(self):
                self.voice_state = ReactiveState(name="voice_state")

            @observe("voice_state")
            def set_vowel(self, vowel: str):
                self._voice.set_vowel(vowel)
                return {"vowel": vowel}  # This gets broadcast
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            # Get the reactive state
            state = getattr(self, state_name, None)
            if state is None:
                return result

            # If result is a dict, broadcast changes
            if isinstance(result, dict):
                for key, value in result.items():
                    old_value = state.get(key) if hasattr(state, "get") else getattr(state, key, None)
                    if hasattr(state, "update"):
                        state.update({key: value})
                    else:
                        setattr(state, key, value)

            return result

        return wrapper

    return decorator


# =============================================================================
# State Aggregator (combines multiple reactive states)
# =============================================================================


class StateAggregator(Observable):
    """
    Combines multiple ReactiveState objects into a single observable.

    Useful for watching the entire hub state from one subscription.

    Example:
        agg = StateAggregator()
        agg.add("voice", voice_state)
        agg.add("daw", daw_state)
        agg.subscribe(lambda k, o, n: print(f"{k} changed"))
    """

    def __init__(self):
        super().__init__()
        self._states: Dict[str, ReactiveState] = {}
        self._unsubs: Dict[str, Callable] = {}

    def add(self, name: str, state: ReactiveState):
        """Add a state to aggregate."""
        if name in self._states:
            self.remove(name)

        self._states[name] = state

        def on_change(key: str, old: Any, new: Any):
            self.notify(f"{name}.{key}", old, new)

        unsub = state.subscribe(on_change)
        self._unsubs[name] = unsub

    def remove(self, name: str):
        """Remove a state from aggregation."""
        if name in self._unsubs:
            self._unsubs[name]()
            del self._unsubs[name]
        self._states.pop(name, None)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Get snapshot of all states."""
        return {name: state.to_dict() for name, state in self._states.items()}

    def subscribe_all(
        self,
        callback: Callable[[str, Any, Any], None],
        is_async: bool = False,
    ) -> Callable[[], None]:
        """Subscribe to changes from all states."""
        return self.add_observer(callback, is_async)


# =============================================================================
# Computed State (derived from other reactive states)
# =============================================================================


class ComputedState(Observable, Generic[T]):
    """
    Reactive value derived from other reactive states.

    Re-computes when dependencies change.

    Example:
        voice = ReactiveState({"pitch": 60, "velocity": 100})
        energy = ComputedState(
            lambda: voice.pitch * voice.velocity / 12700,
            [voice],
            keys=["pitch", "velocity"]
        )
        energy.subscribe(lambda k, o, n: print(f"Energy: {n}"))
    """

    def __init__(
        self,
        compute: Callable[[], T],
        dependencies: List[ReactiveState],
        keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self._compute = compute
        self._value: Optional[T] = None
        self._unsubs: List[Callable] = []

        # Subscribe to dependencies
        for dep in dependencies:
            unsub = dep.subscribe(self._on_dep_change, keys=keys)
            self._unsubs.append(unsub)

        # Initial compute
        self._recompute()

    def _on_dep_change(self, key: str, old: Any, new: Any):
        self._recompute()

    def _recompute(self):
        old_value = self._value
        try:
            self._value = self._compute()
            if old_value != self._value:
                self.notify("value", old_value, self._value)
        except Exception as e:
            print(f"[ComputedState] Compute error: {e}")

    @property
    def value(self) -> T:
        return self._value

    def dispose(self):
        """Clean up subscriptions."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()


# =============================================================================
# Batch Updates (defer notifications)
# =============================================================================


class BatchContext:
    """
    Context manager for batching state updates.

    Defers all notifications until the batch completes.

    Example:
        with BatchContext(state):
            state.tempo = 140
            state.key = "Am"
            state.playing = True
        # Single notification after batch
    """

    def __init__(self, *states: ReactiveState):
        self._states = states
        self._pending: Dict[int, List[tuple]] = {}  # state_id -> [(key, old, new), ...]

    def __enter__(self):
        for state in self._states:
            state_id = id(state)
            self._pending[state_id] = []

            # Intercept notifications
            original_notify = state.notify

            def batched_notify(key, old, new, sid=state_id):
                self._pending[sid].append((key, old, new))

            state._batch_original_notify = original_notify
            state.notify = batched_notify

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for state in self._states:
            # Restore original notify
            state.notify = state._batch_original_notify
            del state._batch_original_notify

            # Fire batched notifications
            for key, old, new in self._pending.get(id(state), []):
                state.notify(key, old, new)

        self._pending.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Observable",
    "ReactiveState",
    "StateAggregator",
    "ComputedState",
    "BatchContext",
    "StateCallback",
    "AsyncStateCallback",
    "reactive_dataclass",
    "observe",
]

