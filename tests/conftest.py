import os

# A developer's running ``fakegpu daemon`` must not change the devices that
# FakeCUDA tests see. Tests of the machine itself opt back in explicitly.
os.environ["FAKEGPU_MACHINE"] = "off"
