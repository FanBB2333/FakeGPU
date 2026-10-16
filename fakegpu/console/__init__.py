"""Terminal console for the background FakeGPU machine (``fakegpu console``).

The console is a client of the machine daemon: everything it shows and does
goes through the daemon's JSON protocol, see :mod:`fakegpu.machine.service`.
Requires the ``console`` extra (Textual).
"""
