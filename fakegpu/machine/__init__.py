"""A simulated GPU machine that keeps running in the background.

The daemon (``fakegpu daemon``) owns a device layout, watches every FakeCUDA
process that publishes state to it, records what happens as events and
answers control requests. ``fakegpu console`` is a terminal client; other
clients use :class:`fakegpu.machine.client.MachineClient` or the JSON
protocol documented in :mod:`fakegpu.machine.service`.

Submodules are imported directly so that ``fakegpu.init()``, which only
needs ``client``, does not load the daemon.
"""
