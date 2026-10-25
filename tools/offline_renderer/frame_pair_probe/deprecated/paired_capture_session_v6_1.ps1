<#
One user-run paired-capture session (probe v6.1.0): deploy the FramePairProbe, launch
HS2 through the audited early RenderDoc bootstrap, restore Doorstop config, write
probe requests, and remove the probe afterwards. Every deployed/original file is
hash-pinned; any mismatch stops before changing anything further.

Bulky data (scene exports, raw texels, RDCs, bridge PNG) goes to the fresh
E:\HS2_offline_renderer_data\paired_session_v6_1_20261025 (outside the game root,
user decision 2026-10-20); small receipts go to the HS2Mod session directory.
Superseded versions are kept in deprecated\: paired_capture_session_v6_0.ps1 (probe 6.0.0)
and paired_capture_session_v5_1.ps1 (probe 5.0.1).

  .\paired_capture_session.ps1 launch          # deploy probe + early-bootstrap launch
  .\paired_capture_session.ps1 restore-config  # after the game reaches its menu
  .\paired_capture_session.ps1 request -RequestMode frame        [-Frames 8 -TriggerIndex 3]
  .\paired_capture_session.ps1 request -RequestMode renderstyle  -BodyJson <file>  [-Frames 8 -TriggerIndex 3]
       (both: [-DriverHooks 1|0] [-Meshes ALL|<comma list>]; defaults 1 and ALL)
  .\paired_capture_session.ps1 request -RequestMode quit         # clean exit via ExitDialog.GameEnd(false)
  .\paired_capture_session.ps1 cleanup         # after the game has exited
#>
param(
    [Parameter(Mandatory)][ValidateSet('launch', 'restore-config', 'request', 'cleanup')][string]$Mode,
    [ValidateSet('frame', 'renderstyle', 'quit')][string]$RequestMode,
    [ValidateSet('0', '1')][string]$DriverHooks = '1',
    [string]$Meshes = 'ALL',
    [string]$BodyJson,
    [int]$Frames = 8,
    [int]$TriggerIndex = 3
)
$ErrorActionPreference = 'Stop'

$Game      = 'E:\HoneySelect2_ArcticFox'
$F2P       = 'C:\Users\13666\Workspace\Face2Parameter'
$Scope     = 'C:\Users\13666\Workspace\HS2Mod\artifacts\chenger_reference_fit_20261012'
$Session   = "$Scope\paired_capture_session_v6_1_20261025"
$Bulk      = 'E:\HS2_offline_renderer_data\paired_session_v6_1_20261025'
$ProbeSrc  = "$F2P\outputs\offline_renderer_probe_v6_1_20261025\build_v6_1\HS2_FramePairProbe.dll"
$ManifestSrc = "$F2P\outputs\offline_renderer_probe_v5_20261020\shader_manifest_v5.tsv"
$ProbeDir  = "$Game\BepInEx\plugins\HS2_FramePairProbe"
$Bootstrap = "$F2P\outputs\offline_renderer_repair_20261012\early_bootstrap_cpu_v2\HS2_RenderDocBootstrap.dll"
$RenderDoc = "$F2P\outputs\offline_renderer_repair_20261012\renderdoc_portable_1_46\extracted\RenderDoc_1.46_64\renderdoc.dll"
$Bridge    = "$Game\BepInEx\Plugins\HS2_McpBridge\HS2_McpBridge.dll"
$Readback  = "$Game\BepInEx\Plugins\HS2_McpBridge\HS2_RenderReadback.dll"
$Dispatch  = "$Game\BepInEx\Plugins\HS2_McpBridge\HS2_RenderDocScope.request.json"
$UnityPlayer = "$Game\UnityPlayer.dll"
$MinFreeGB = 100

$Pinned = @{
    Probe       = '37a0b87cb006c10abddba6acbcecad894128cbbd4e86c5ec5fb5219407cee6ff'
    Manifest    = 'e403c5ad5524ece1d362329716777cf8f20d774a8d9d6e9e827459daf5ff2036'
    Bootstrap   = 'fa4927883805f3473205ccd54ecc4f5d80f5d05f02778133447013b8c4197391'
    RenderDoc   = '809da38e3867d9fd09cc5c30dd5310500dee75e999166d7a14ad1ef6e9eca65d'
    Bridge      = 'efa943cb78099fcfccc11a10d4947693216e58e9525006b74ab2ee20d4ac8c21'
    Config      = '4e9baa188898decca8d799efa3517ff0f0820c1a54ba4c827fd569fdbac5809c'
    Readback    = 'b1cb390aa8f927e749730ec0c1f1133a944d0d5bd1f6138c32480fdc50245cb3'
    UnityPlayer = '5034cdc2bff71dc8d9e038cb71059f8965d0d879f8098af0f3c726ea06e13923'
}

function Sha($path) { (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant() }
function Require($path, $expected, $label) {
    if (-not (Test-Path $path)) { throw "$label missing: $path" }
    $actual = Sha $path
    if ($actual -ne $expected) { throw "$label hash $actual differs from pinned $expected ($path)" }
}
function RequireNoDispatch() {
    # The installed bridge routes /maker/renderstyle/capture through HS2_RenderDocDispatch when this file exists.
    if (Test-Path $Dispatch) { throw "RenderDoc dispatch request exists; the bridge would not run its plain capture route: $Dispatch" }
}
function FreeGB($root) { [math]::Floor((Get-PSDrive -Name $root.Substring(0, 1)).Free / 1GB) }

switch ($Mode) {
    'launch' {
        Require $ProbeSrc $Pinned.Probe 'Probe v6.1.0 build'
        Require $ManifestSrc $Pinned.Manifest 'Shader manifest v5'
        Require $Bootstrap $Pinned.Bootstrap 'Bootstrap build'
        Require $RenderDoc $Pinned.RenderDoc 'RenderDoc 1.46'
        Require $Bridge $Pinned.Bridge 'Installed bridge'
        Require $Readback $Pinned.Readback 'Installed native readback'
        Require $UnityPlayer $Pinned.UnityPlayer 'UnityPlayer.dll (native layouts recovered from this binary)'
        Require "$Game\doorstop_config.ini" $Pinned.Config 'Original doorstop_config.ini'
        RequireNoDispatch
        if (Test-Path $ProbeDir) { throw "Probe directory already exists; review it first: $ProbeDir" }
        if (Test-Path $Session) { throw "Session directory already exists; choose a fresh one: $Session" }
        if (Test-Path $Bulk) { throw "Bulk data directory already exists; choose a fresh one: $Bulk" }
        $free = FreeGB $Bulk
        if ($free -lt $MinFreeGB) { throw "Only $free GB free on $($Bulk.Substring(0, 2)); $MinFreeGB GB required" }
        New-Item -ItemType Directory -Path $Bulk | Out-Null
        Copy-Item $ManifestSrc "$Bulk\shader_manifest_v5.tsv"
        Require "$Bulk\shader_manifest_v5.tsv" $Pinned.Manifest 'Copied shader manifest'
        New-Item -ItemType Directory -Path $ProbeDir | Out-Null
        Copy-Item $ProbeSrc $ProbeDir
        Require "$ProbeDir\HS2_FramePairProbe.dll" $Pinned.Probe 'Deployed probe'
        Write-Host "Probe v6.1.0 deployed: $ProbeDir; bulk data: $Bulk ($free GB free)"
        # early_scope_game.py re-checks processes/hashes, creates the session dir, backs up the config and launches.
        python "$Scope\early_scope_game.py" launch --session $Session `
            --bootstrap $Bootstrap --bootstrap-sha $Pinned.Bootstrap `
            --renderdoc $RenderDoc --bridge-sha $Pinned.Bridge
        if ($LASTEXITCODE -ne 0) {
            Remove-Item -Recurse -Force $ProbeDir
            throw "early_scope_game.py launch failed ($LASTEXITCODE); probe removed again"
        }
        Write-Host 'Launched. When the game reaches its menu run: .\paired_capture_session.ps1 restore-config'
    }
    'restore-config' {
        python "$Scope\early_scope_game.py" restore-config --session $Session
        if ($LASTEXITCODE -ne 0) { throw "restore-config failed ($LASTEXITCODE)" }
        Require "$Game\doorstop_config.ini" $Pinned.Config 'Restored doorstop_config.ini'
        Write-Host 'Doorstop config restored to original.'
    }
    'request' {
        if (-not $RequestMode) { throw '-RequestMode frame|renderstyle|quit is required' }
        Require "$ProbeDir\HS2_FramePairProbe.dll" $Pinned.Probe 'Deployed probe'
        Require "$Bulk\shader_manifest_v5.tsv" $Pinned.Manifest 'Session shader manifest'
        Require $Readback $Pinned.Readback 'Installed native readback'
        RequireNoDispatch
        if (-not (Get-Process -Name HoneySelect2 -ErrorAction SilentlyContinue)) { throw 'HoneySelect2 is not running' }
        if (Test-Path "$ProbeDir\HS2_FramePairProbe.request.ini") { throw 'A probe request is still pending' }
        $free = FreeGB $Bulk
        if ($free -lt $MinFreeGB) { throw "Only $free GB free on $($Bulk.Substring(0, 2)); $MinFreeGB GB required" }
        $nonce = [guid]::NewGuid().ToString('N')
        if ($RequestMode -eq 'quit') {
            # Small receipt only; the probe writes quit_request_receipt.json, then calls the game's exit.
            $text = "Mode=quit`nNonce=$nonce`nOutDir=$Session\quit_$nonce`n"
            [IO.File]::WriteAllText("$Session\request_$nonce.ini", $text, (New-Object Text.UTF8Encoding $false))
            [IO.File]::WriteAllText("$ProbeDir\HS2_FramePairProbe.request.tmp", $text, (New-Object Text.UTF8Encoding $false))
            Move-Item "$ProbeDir\HS2_FramePairProbe.request.tmp" "$ProbeDir\HS2_FramePairProbe.request.ini"
            Write-Host "Quit request nonce=$nonce receipt=$Session\quit_$nonce\quit_request_receipt.json; wait for the HoneySelect2 process to exit"
            break
        }
        $lines = @(
            "Mode=$RequestMode", "Nonce=$nonce",
            "OutDir=$Bulk\window_$nonce", "RawOutDir=$Bulk\raw_$nonce",
            "Frames=$Frames", "TriggerIndex=$TriggerIndex", 'PollSeconds=120',
            "CapturePrefix=$Bulk\rdc_$nonce",
            "ExpectedRenderDocSha256=$($Pinned.RenderDoc)", "ExpectedUnityPlayerSha256=$($Pinned.UnityPlayer)",
            "ReadbackDll=$Readback", "ExpectedReadbackSha256=$($Pinned.Readback)",
            'RawTimeoutSeconds=900', 'RawMinFreeGB=50',
            "ShaderManifest=$Bulk\shader_manifest_v5.tsv",
            "Meshes=$Meshes", "DriverHooks=$DriverHooks")
        if ($RequestMode -eq 'renderstyle') {
            if (-not $BodyJson -or -not (Test-Path $BodyJson)) { throw '-BodyJson <bridge capture body file> is required for renderstyle' }
            $body = Get-Content -Raw $BodyJson | ConvertFrom-Json
            $allowed = 'path', 'yaw', 'target', 'ortho_size', 'width', 'height', 'freeze_pose'
            foreach ($p in $body.PSObject.Properties.Name) { if ($allowed -notcontains $p) { throw "Installed bridge rejects body key $p" } }
            if (-not $body.width -or -not $body.height) { throw 'Bridge body must give width and height' }
            if ((Test-Path $body.path) -or -not $body.path.StartsWith($Bulk)) { throw "Bridge PNG path must be fresh and under $Bulk" }
            $bodyCopy = "$Bulk\bridge_body_$nonce.json"
            Copy-Item $BodyJson $bodyCopy
            $lines += @('BridgeUrl=http://127.0.0.1:43127/maker/renderstyle/capture', "BridgeBodyFile=$bodyCopy",
                        "ExpectedBridgeSha256=$($Pinned.Bridge)", "BridgeDispatchRequestFile=$Dispatch", 'BridgeTimeoutSeconds=180')
        }
        $text = ($lines -join "`n") + "`n"
        [IO.File]::WriteAllText("$Session\request_$nonce.ini", $text, (New-Object Text.UTF8Encoding $false))
        # Written last and atomically: the probe polls for this exact file name every 15 frames.
        [IO.File]::WriteAllText("$ProbeDir\HS2_FramePairProbe.request.tmp", $text, (New-Object Text.UTF8Encoding $false))
        Move-Item "$ProbeDir\HS2_FramePairProbe.request.tmp" "$ProbeDir\HS2_FramePairProbe.request.ini"
        Write-Host "Request $RequestMode nonce=$nonce window=$Bulk\window_$nonce (copy: $Session\request_$nonce.ini)"
    }
    'cleanup' {
        if (Get-Process -Name HoneySelect2 -ErrorAction SilentlyContinue) { throw 'HoneySelect2 is still running; quit it first' }
        if (Test-Path $ProbeDir) {
            Require "$ProbeDir\HS2_FramePairProbe.dll" $Pinned.Probe 'Deployed probe'
            $extra = Get-ChildItem $ProbeDir | Where-Object Name -ne 'HS2_FramePairProbe.dll'
            if ($extra) { throw "Probe directory has unexpected files; review: $($extra.Name -join ', ')" }
            Remove-Item -Recurse -Force $ProbeDir
            Write-Host 'Probe removed.'
        } else { Write-Host 'Probe directory already absent.' }
        Require "$Game\doorstop_config.ini" $Pinned.Config 'doorstop_config.ini'
        RequireNoDispatch
        $bulkFiles = Get-ChildItem -Recurse -File $Bulk -ErrorAction SilentlyContinue
        @{ probe_removed = -not (Test-Path $ProbeDir); doorstop_config_sha256 = (Sha "$Game\doorstop_config.ini");
           original_config_restored = $true; dispatch_request_absent = -not (Test-Path $Dispatch);
           bulk_dir = $Bulk; bulk_files = @($bulkFiles).Count; bulk_bytes = ($bulkFiles | Measure-Object Length -Sum).Sum;
           cleaned_at = (Get-Date).ToString('o') } |
            ConvertTo-Json | Set-Content -Encoding UTF8 "$Session\cleanup_receipt.json"
        Write-Host "Cleanup receipt: $Session\cleanup_receipt.json"
    }
}
