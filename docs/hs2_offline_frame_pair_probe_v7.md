# HS2 offline renderer — frame-pair probe v7 (2026-10-25)

Probe v7 keeps everything from [v6.1](hs2_offline_frame_pair_probe_v6.md) §10 (fixed `smr_*`
fields, bridge pre_cull poses, all fail-closed checks). It adds the orchestrator's v7 inputs and
multi-yaw sessions. It was built and tested only: nothing was launched or deployed.

## 1 Build identity

| item | value |
|---|---|
| build | `outputs/offline_renderer_probe_v7_20261025/build_v7/HS2_FramePairProbe.dll`; a second `--no-incremental` build from a fresh `obj/` (`rebuild_check/build`) gave the same bytes |
| SHA-256 | `693e9a2fe0f6c29d7ea5262f0d1abe6947bc2abde728c026e3a332e0b2f5fe87` |
| plugin | `codex.hs2.framepairprobe` **7.0.0**. Scene, renders and receipt schema 7. |
| new reference | `HoneySelect2_Data/Managed/IL.dll` (needed for `Singleton<T>`) |
| session script | `tools/offline_renderer/frame_pair_probe/paired_capture_session.ps1` (see the table below) |
| superseded (moved, not deleted) | the 6.1.0 script → `frame_pair_probe/deprecated/paired_capture_session_v6_1.ps1`; intermediate v7 builds → `outputs/offline_renderer_probe_v7_20261025/deprecated/` |

The session script:
- is pinned to `693e9a2f…`;
- uses the fresh dirs `HS2Mod/artifacts/chenger_reference_fit_20261012/paired_capture_session_v7_20261025` and `E:/HS2_offline_renderer_data/paired_session_v7_20261025`, which do not exist yet;
- has its other pins unchanged and re-checked against the installed files: bridge `efa943cb…`, readback `b1cb390a…`, Doorstop config `4e9baa18…`, UnityPlayer `5034cdc2…`, bootstrap `fa492788…`, manifest `e403c5ad…`.

New files:
- `AnimationState.cs`
- `TransformChains.cs`
- `tools/offline_renderer/pose_clip_bindings.py`
- `tools/offline_renderer/test_pose_clip_bindings.py`
- `frame_pair_probe/tests/test_session_yaw.ps1`

Changed files:
- `ProbeCore.cs`, `NativeUnity.cs`, `RenderRecorder.cs`, `SceneState.cs`
- `FramePairProbe.cs`, `BridgeCapture.cs`, `HS2_FramePairProbe.csproj`
- `verify_probe_v5_session.py`, `paired_capture_session.ps1`
- `tests/ProbeCoreTests/Tests.cs`, `test_frame_pair_probe_v5.py`

UnityPlayer RVAs are for `5034cdc2…` (PDB `BFA91A8C…`), read with `tools/offline_renderer/unityplayer_disasm.py`.

## 2 New fields and their sources

### 2.1 RenderNode queue of every camera render (`post_render.data.render_nodes`)

The node queue is reachable from the Camera only during its render:

| source | what it does |
|---|---|
| `Camera::CustomRender` 0x661400 → `InitializeRenderLoopContext` 0x6ba530 (call at 0x66176c) | Allocates a RenderLoopContext (0x838 bytes) and stores it at `[[Camera+0x530]+0]`, with these fields: context+0x8 = Camera*; +0x7c8 = SharedRendererScene* (`[CullResults+0x800]`); +0x7d0 = scene+0x8; +0x7d8 = scene+0x10068; +0x830 = RenderLoop*. |
| SharedRendererScene ctor 0x63f330 | Constructs the two RenderNodeQueues at +0x8 and +0x10068. |
| static `onPostRender` (0x66181a) | Runs inside the render, with the context attached. |
| `CleanupAfterRendering` 0x65f720 → `CleanupAfterRenderLoop` 0x6b70a0 | Frees the context and zeroes `[RenderLoop+0]`. |
| RenderNodeQueue (`Reset` 0x692080) | Nodes at +0x0, count at +0x10, stride 0x1a0. |
| `Renderer::AddAsRenderNode` 0x739400 | node+0x190 = renderer instance id (`[BaseRenderer-0x30]` = Renderer+0x8). |
| `MeshRenderer::AddAsRenderNode` 0x77aad0 | node+0x100 = `[mesh+0x174]`, where the mesh is `[Renderer+0x38+0x188]`. |
| `SkinnedMeshRenderer::AddAsRenderNode` 0x77abe0 | node+0x100 = `[mesh+0x174]`, where the mesh is `[Renderer+0x38+0x1a0]`. |

The Mesh ctor 0x7a0d50 initializes +0x174 at 0x7a1145.

**Export.** For every post_render record:
- the RenderLoop, context and scene pointers;
- both queues, each with its count and one row per node: `[index, instance_id, mesh_id(node+0x100), check]`.

**Fail closed.**
- *Freshness:* `[[Camera+0x530]+0]` must be 0 at that render's pre_cull. An unreadable or unobserved pre_cull counts as not fresh.
- *Context links:* the context must name this camera and this RenderLoop, and the queue pointers must lie inside the scene.
- *Count:* the node count must be below 65536.
- *Mesh id:* for every node whose renderer is a scene MeshRenderer or SkinnedMeshRenderer, node+0x100 must equal `[cached mesh+0x174]`.

Other renderer types are recorded as `not_a_mesh_renderer:<type>`, and ids that are not scene renderers as `instance_not_a_scene_renderer`. Any violation makes the record `render_nodes_error`.

**Validation before the trigger:**
- the instance id offset is 8;
- the MeshRenderer cached mesh equals `MeshFilter.sharedMesh` for every MeshRenderer with a MeshFilter;
- Camera.main's RenderLoop has no context outside a render.

### 2.2 Skinned renderer bounds and prepared info

| field | source |
|---|---|
| `world_aabb_center_extent_0xc0`, `local_aabb_center_extent_0xd8` (every renderer row) | `SkinnedMeshRenderer::UpdateTransformInfo` 0x78adf0. It stores the bone-based world AABB at `[smr+0xc0]` (0x78b0e5), and `InverseTransformAABB(world, root pos, rot)` → `[smr+0xd8]`. This is the TransformInfo of `GetWorldAABB`/`GetLocalAABB`. Before the trigger, it must equal `Renderer.bounds` for ≥ 90 % of enabled Mesh/SkinnedMeshRenderers. |
| `smr_update_when_offscreen_0x28c` | Same function, branch at 0x78aeb1. It must equal `updateWhenOffscreen`, both before the trigger and per row. |
| `skinned_bounds_update_when_offscreen` (every post_render) | World/local AABB and L2W of each updateWhenOffscreen renderer, i.e. the bounds that render used. |
| `bone_bounds_*` (updateWhenOffscreen rows) | `Mesh::GetCachedBonesBounds` 0x7a4770: Mesh+0x120 `dynamic_array<MinMaxAABB>`, size at +0x130. The cache is current when size == `[[Mesh+0x38]+0xf0]` (bindpose count). Exported for completeness with a `bone_bounds_state`. It is not required, because the orchestrator computes these bounds exactly. |
| `smr_manager_handle_0x2f8`, `prepared_renderer_info.flag_bytes_0_4` (every SMR row) | `SkinnedMeshRendererManager` `s_Instance` (RVA 0x15615d0). `AddRenderer` 0xbd9fd0: handle → `[smr+0x2f8]`, `[[mgr+0x108]+h*8]` = renderer. `AddPreparedInfo` 0xbd9e70: `[[mgr+0x128]+h*8]` = PreparedRendererInfo*. `InitializePreparedInfo` 0xbdce90 sets the flag bytes: +1 = no mesh, bone weights or `smr+0x2c8`; +2 = bonesPerVertex ≠ 1; +4 = updateWhenOffscreen and (no mesh or bone weights); +0 and +3 are initialized to 0. `RemoveRenderer` 0xbddaf0 sets the handle to −1, which is exported as unregistered. A registered handle whose manager slot is not this renderer fails, both before the trigger and per row. |

### 2.3 Animation source of the PoseFreeze write set (scene export and every bridge pre_cull: `animation`)

**What the probe exports.**
- *Every Animator under the ChaControl:*
  - transform, controller, avatar and flags;
  - the controller's clips;
  - the layer-0 current/next state and clip infos.
- *The Maker pose source:*
  - `CustomBase.poseNo`, `backPoseNo` and `animeStateName`;
  - the `custom_pose_m/f` (500/501) list entry: MainManifest, MainAB, MainData, Clip, IKAB, IKData;
  - the bundle file `AssetBundleManager.BaseDownloadingURL + MainAB` and whether it exists.

  The source chain is `CustomBase.ChangeAnimation` (decompiled_acs/CharaCustom/CustomBase.cs 1464-1526) → `ChaControl.LoadAnimation`. The singletons are read only after `IsInstance()`.

**Why offline parsing.** The clip bindings are not reachable through the runtime managed API. `pose_clip_bindings.py` parses the bundle offline:
- the controller `m_TOS`;
- the state whose `m_FullPathID` equals the runtime fullPathHash, its blend-tree `m_ClipID`s and `m_WriteDefaultValues`;
- each clip's `genericBindings`, resolved by CRC-32 of the path below the Animator's GameObject. For Transform bindings (typeID 4), attribute 1/2/3/4 = position/rotation/scale/euler.

It fails closed unless:
- the runtime controller is the entry's MainData;
- the state matches exactly one controller state;
- the runtime current clips equal the state's clips.

Paths without a transform are listed as `unbound`.

**Installed result** (bundle `custom/00/anim_f_00.unity3d`, controller `edit_F`, state Idle_00):
- 1063 bindings, of which 1042 are bound. 7 Transform paths are absent from the character, and 7 typeID-95 rows bind the Animator root.
- 345 transforms are written.
- `m_WriteDefaultValues` is true.
- The union over all 26 controller clips is the same 345 transforms.

The bridge's PoseFreeze runs speed 0, `Play(fullPathHash, 0, 0)` and `Update(0)`, and writes exactly this set.

### 2.4 Camera and light chains (scene export and every bridge pre_cull: `transform_chains`)

The chains are root first, with local and world TRS for each transform:
- Camera.main's chain;
- every scene transform named `CharaCamera` or `Lights Custom`, with its path (both were assumed identity);
- every scene Light's chain.

### 2.5 Multi-yaw

- **Probe:** `BridgeBody.Parse` checks `yaw` (a finite number; omitted = 0, as in the bridge), and the receipt records `bridge.body_yaw_degrees`.
- **Bridge:** the installed bridge orbits Camera.main around `target` by `Quaternion.Euler(0, yaw, 0)` (MakerRenderStyleService).
- **Script:** with `-Yaw` or `-YawSet` the body file is a template without `path`. Per window the script:
  - writes a resolved body with that yaw and a fresh `bridge_capture_<nonce>_yaw<tag>.png`;
  - re-reads it and fails if any other field changed;
  - for `-YawSet`, runs the windows in the given order, waiting for each window's receipt and `done` line (stopping on an abort, a timeout or the game exiting), and keeps `yaw_set_<id>.json` (yaw → nonce, window, body, captured frame, RDC).

## 3 Verifier (`verify_probe_v5_session.py`, probe ≥ 7)

All v6.1 checks still apply. New checks:

| check | requirement |
|---|---|
| `render_nodes_v7` | Every post_render record in the captured frame has fresh `render_nodes` with 2 queues and no error; there are as many post_renders as pre_culls; at least one render has nodes; every node check is valid. |
| `skinned_bounds_v7` | Every SMR row has the updateWhenOffscreen byte and the prepared-info fields, and every per-render bounds row has no error. |
| `animation_source_v7` | All animator rows exported without error, a layer-0 current clip, and a Maker pose entry whose bundle exists. |
| `transform_chains_v7` | Camera.main's parent is `CharaCamera`, and both named transforms and the light chains are present. |
| `bridge_pre_cull_animation_v7`, `bridge_pre_cull_transform_chains_v7` | Renderstyle only: every bridge pre_cull carries both blocks. |

Observations (not acceptance):
- `render_nodes_unknown_instance_ids`
- `bone_bounds_state`
- `bridge_yaw_degrees`

The v6.1 baseline window `766d8184…` still verifies as accepted.

## 4 Runbook (main agent, user approval to launch)

```powershell
$S = 'C:\Users\13666\Workspace\Face2Parameter\tools\offline_renderer\frame_pair_probe\paired_capture_session.ps1'
```

1. Run `& $S launch`, then `& $S restore-config` at the menu. The BepInEx log must show `FramePairProbe 7.0.0 ready`.
2. **Maker and card:**
   1. Run `/game/maker/start` (female).
   2. Load the DINOv2 v1 baseline card `74d2625f…` read-only.
   3. Record snapshot, ABMX, expression and renderstyle.

   The baseline windows use the card with an empty patch: no ABMX or slider change.
3. **Yaw template:** write it with a file tool (never through a shell escape). Use the live `o_head` bounds-centre target, ortho 1.42995071, 1086×1448, `freeze_pose: true`, and no `path`. Example `bridge_body_yaw_template.json` in the bulk dir:
   `{"target":[x,y,z],"ortho_size":1.42995071,"width":1086,"height":1448,"freeze_pose":true}`.
   Keep the same target for every yaw.
4. **Baseline yaw windows, in this order:** 0, +30, +45, +60, +90, −90, then optionally −45. Either:
   - run all of them in one call: `& $S request -RequestMode renderstyle -BodyJson <template> -YawSet "0,30,45,60,90,-90"` (append `,-45` for the optional one); or
   - run one window per call, recording state between windows: `& $S request -RequestMode renderstyle -BodyJson <template> -Yaw 0`, then 30, 45, 60, 90, −90 (−45).

   Each window produces the full export, the RDC and `bridge_body_<nonce>.json` with its yaw. `DriverHooks` defaults to 1 and `Meshes` to ALL.
5. **Per window:**
   1. Convert the RDC: `renderdoc_replay_msvc.exe convert <rdc> rdc <parse>\capture.zip.xml zip.xml`, with the hash-pinned converter and RenderDoc 1.46.
   2. Run `verify_probe_v5_session.py`. It must print `accepted`.
   3. Run `python tools/offline_renderer/pose_clip_bindings.py --window <window> --out <fresh>.json` for the write set.
6. **After the set:** record the state again. It must be unchanged, with the card SHA unchanged.
7. **Quit:** `& $S request -RequestMode quit`. Wait for the process to exit, then run `& $S cleanup`.

## 5 Tests (CPU)

| suite | result |
|---|---|
| `tests/ProbeCoreTests` | 87 passed. New for v7: RenderNodes freshness, links and counts; mesh id; AABB/bone bounds; PreparedInfo handle and slot; yaw parsing. |
| `test_frame_pair_probe_v5.py` | 18 passed. New for v7: accept, plus rejections for node mismatch, not fresh, `render_nodes_error`, a missing post_render, missing prepared info, a missing pose bundle, a missing bridge animation, a missing `Lights Custom`, a wrong camera parent and missing bridge chains. |
| `test_pose_clip_bindings.py` (new) | 4 passed: CRC-32 path resolution, a missing animation block, the installed Idle_00 bundle (1063/1042/345), and state/clip/controller mismatches failing closed |
| `test_pair_skin_capture.py`, `test_paired_skin.py`, `test_gpu_skin_inputs.py` | 21 passed (unchanged) |
| `tests/test_session_yaw.ps1` (new) | 38 passed: resolved bodies for all yaws keep every other field, PNG tags, rejection of NaN/text yaw, a template with `path`, and unknown keys |
