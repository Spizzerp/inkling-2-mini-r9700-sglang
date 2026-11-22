"""Compile, at image build time, the aiter modules that SGLang otherwise recompiles at every start.

The base image ships aiter's kernel modules built for gfx950 (MI350X/MI355X). On another chip aiter
notices that a module carries no code for the running GPU and rebuilds it on the spot, into the
container's own file system, so the result is lost when the container is removed. On the R9700 that is
three modules and about 42 s of every start (measured 22 Nov 2026: custom_all_reduce 19.0 s,
rmsnorm_quant 17.9 s, sample 5.5 s; the same three for the standard and the long-context entry).

Building them here needs no GPU: GPU_ARCHS names the target. Run as
    GPU_ARCHS=gfx1201 python3 prebuild_aiter_gfx1201.py
To find modules this list misses, look for "rebuilding for gfx1201" in the server log.
"""
import inspect
import os
import sys

from aiter.jit import core

MODULES = ["module_custom_all_reduce", "module_rmsnorm_quant", "module_sample"]

target = os.environ.get("GPU_ARCHS", "")
if target != "gfx1201":
    sys.exit("set GPU_ARCHS=gfx1201 (it is %r)" % target)

accepted = set(inspect.signature(core.build_module).parameters)
for name in MODULES:
    so = os.path.join(core.get_user_jit_dir(), name + ".so")
    args = {k: v for k, v in core.get_args_of_build(name).items() if k in accepted}
    args["md_name"] = name
    core.build_module(**args)
    archs = core._so_offload_archs(so)
    if archs != {target}:
        sys.exit("%s: built for %s, wanted %s" % (name, sorted(archs), target))
    print("%s: built for %s, %d bytes" % (name, target, os.path.getsize(so)))
