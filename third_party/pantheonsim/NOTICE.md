# PantheonSim kernel execution core

FakeGPU links the PTX parser, cubin loader, PTX/SASS CPU interpreters and memory model from
[PantheonSim](https://github.com/pantheongpu/pantheonsim), under the
[Apache License 2.0](LICENSE).

The source is downloaded at build time from commit
`42919cd4a95b012b7e106094df059f41fa9c71de` and verified against SHA256
`7595f5570e650a54577a86fa0df0093a49d1380caf03a3d96bf540de83871f67`.
The upstream sources are compiled without modification. The upstream CUDA
shims, device catalog and third-party vendor components are not linked.

The integration in `src/cuda/kernel_executor.cpp` belongs to FakeGPU.
