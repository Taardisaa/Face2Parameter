# Early Doorstop/RenderDoc bootstrap

The official-launch native attempt showed actual D3D11 vanilla menus with zero
captures and no BepInEx/bridge initialization. It was rejected and its owned
processes stopped by main. Source explains the collision: Doorstop3.4.0.0 uses a
Unity GetProcAddress IAT redirect to enter Mono/BepInEx; RenderDoc1.46 also replaces
that slot and forwards unresolved Mono exports to Windows rather than the previous
Doorstop redirect. Primary upstream report/workaround:
[UnityDoorstop issue24](https://github.com/NeighTools/UnityDoorstop/issues/24).
Pinned implementations:
[Doorstop main](https://github.com/NeighTools/UnityDoorstop/blob/v3.4.0.0/Proxy/main.c),
[Doorstop IAT](https://github.com/NeighTools/UnityDoorstop/blob/v3.4.0.0/Proxy/hook.h),
[RenderDoc IAT](https://github.com/baldurk/renderdoc/blob/v1.46/renderdoc/os/win32/win32_hook.cpp).

The independent net472 DLL references **mscorlib only** and exposes exactly one
Doorstop wildcard entry: `public static void HS2_RenderDocBootstrap.Main()`.
It is loaded as Doorstop's temporary target **after Doorstop already entered its
Mono initializer**, then explicitly loads the pinned official RenderDoc before the
original BepInEx preloader assembly/entry is loaded. Subsequent IAT replacement
cannot erase the managed entry already executing. It never copies/skips the
original preloader logic or disables a shader/effect/plugin.

## Original installed entry oracle

Installed BepInEx.Preloader5.4.21.0 SHA256:
`9678c74f47353a11afde4ab05887388bcbb69dffc4e28db93af1d1ce0b6ee635`.
CPU Cecil IL audit includes original methods/exception ranges/references:

- `BepInEx.Preloader.Entrypoint.Main()` calls EnvVars.LoadVars, installs its own
  temporary ResolveCurrentDirectory handler and calls PreloaderPreMain by reflection.
- PreloaderPreMain sets platform, executable/core paths and original search dirs,
  loads critical assemblies, switches resolvers and enters PreloaderMain.
- PreloaderMain retains the configured runtime-fix branch and calls Preloader.Run.
- Main's original catch/logging/finally resolver removal remains entirely original.

The wrapper always calls **original Entrypoint.Main**, with unchanged zero-arg/
void/public/static signature. It verifies the pinned original file SHA before
Assembly.LoadFrom on enabled, disabled, absent and malformed-request paths.
`DOORSTOP_INVOKE_DLL_PATH` is restored to that actual original file before it
loads EnvVars; before/restored values and invocation/return UTC are recorded.
The ordinary-original path never attempts to load RenderDoc. Failed RenderDoc
validation/load still invokes the full original entry and records capture as
unaccepted. A replaced/mismatched original preloader is not an approved fallback.

## Main-task-only deployment and one explicit request

No configuration/preloader DLL was edited or installed by this child. Main backs
up/restores Doorstop config and keeps the original preloader intact. Place only
`HS2_RenderDocBootstrap.dll` alongside `BepInEx.Preloader.dll` under core. The
CPU `fixture_only/BepInEx.Preloader.dll` must **never** be deployed.

Main's temporary Doorstop target would be:

```ini
targetAssembly=BepInEx\core\HS2_RenderDocBootstrap.dll
```

Explicit sidecar beside the wrapper: `HS2_RenderDocBootstrap.request.ini`:

```ini
Enabled=true
RenderDocPath=C:/absolute/official/RenderDoc_1.46_64/renderdoc.dll
ExpectedRenderDocSha256=809da38e3867d9fd09cc5c30dd5310500dee75e999166d7a14ad1ef6e9eca65d
ExpectedPreloaderSha256=9678c74f47353a11afde4ab05887388bcbb69dffc4e28db93af1d1ce0b6ee635
ExpectedProcessPath=E:/HoneySelect2_ArcticFox/HoneySelect2.exe
ExpectedBackend=D3D11
ReceiptPath=C:/absolute/fresh_bootstrap_receipt.json
```

Native game command line must explicitly include `-force-d3d11`; contradictory
backend flags are rejected. This records requested backend; actual driver remains
a native acceptance requirement. RenderDoc must have the exact official hash and
AMD64/PE32+ header, with an absolute renderdoc.dll filename. Only that explicit
library is loaded; no search/download/other-tool fallback is implemented.

Late guards require x64 Doorstop-initialized context, no original BepInEx assembly
already loaded, no own visible/Unity window, no observed NVIDIA/AMD/Intel graphics
driver or Vulkan/D3D12 module and no already-loaded RenderDoc. Failure never silently
accepts late capture. These source/observational gates still require real-game
acceptance; they are not proof of all possible private engine state.

Receipts use FileMode.CreateNew. Invalid enabled validation before requested path
selection, absent/disabled/malformed requests and write failures retain an explicit
unique fallback under core:
`HS2_RenderDocBootstrap.receipt.ACTUAL_PID.GUID.json`.
The actual path, requested path when rejected and fallback are recorded. Existing
receipts are never overwritten. A final emergency write failure uses the same
unique prefix plus `.error.txt`. Timestamps include SDK load, original path
restoration and original entry invocation/return. SDK load success is not capture/
plugin acceptance: both markers remain false until main independently verifies
the full native plugin/Maker/SSS/capture pipeline.

## CPU verification and remaining runtime work

```text
python tools/offline_renderer/build_early_bootstrap.py --preloader E:/HoneySelect2_ArcticFox/BepInEx/core/BepInEx.Preloader.dll --cecil E:/HoneySelect2_ArcticFox/BepInEx/core/Mono.Cecil.dll --out outputs/offline_renderer_repair_20261012/NEW_FRESH_BOOTSTRAP_BUILD
```

Latest ignored build: `early_bootstrap_cpu_v2/HS2_RenderDocBootstrap.dll`, SHA256
`fa4927883805f3473205ccd54ecc4f5d80f5d05f02778133447013b8c4197391`.
CPU Cecil asserts exactly mscorlib reference and one Main; original installed
entry IL is read without execution. Nineteen sequence/entry fixture checks validate
disabled/failed/successful SDK ordering, original pinned hash rejection, exact
original-entry invocation/environment restoration, request and backend failures.
They never execute LoadLibrary, the installed preloader or graphical programs.

Still pending: main-controlled early native startup, original BepInEx/bridge/Maker/
SSS initialization, actual same-freeze scoped capture, all GPU geometry/dependency
correspondence and full candidate RGB renderer acceptance. This bootstrap is
source-derived startup preparation, not a completed renderer.
