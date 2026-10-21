<#
One user-run paired-capture session: deploy the read-only FramePairProbe, launch
HS2 through the audited early RenderDoc bootstrap, restore Doorstop config, and
remove the probe afterwards. Every deployed/original file is hash-pinned; any
mismatch stops before changing anything further.

  .\paired_capture_session.ps1 launch          # deploy probe + early-bootstrap launch
  .\paired_capture_session.ps1 restore-config  # after the game reaches its menu
  .\paired_capture_session.ps1 cleanup         # after the game has exited
#>
param([Parameter(Mandatory)][ValidateSet('launch', 'restore-config', 'cleanup')][string]$Mode)
$ErrorActionPreference = 'Stop'

$Game      = 'E:\HoneySelect2_ArcticFox'
$F2P       = 'C:\Users\13666\Workspace\Face2Parameter'
$Scope     = 'C:\Users\13666\Workspace\HS2Mod\artifacts\chenger_reference_fit_20261012'
$Session   = "$Scope\paired_capture_session_20261020"
$ProbeSrc  = "$F2P\outputs\offline_renderer_skinning_20261020\probe_build_v4\HS2_FramePairProbe.dll"
$ProbeDir  = "$Game\BepInEx\plugins\HS2_FramePairProbe"
$Bootstrap = "$F2P\outputs\offline_renderer_repair_20261012\early_bootstrap_cpu_v2\HS2_RenderDocBootstrap.dll"
$RenderDoc = "$F2P\outputs\offline_renderer_repair_20261012\renderdoc_portable_1_46\extracted\RenderDoc_1.46_64\renderdoc.dll"
$Bridge    = "$Game\BepInEx\Plugins\HS2_McpBridge\HS2_McpBridge.dll"

$Pinned = @{
    Probe     = 'd7e87bd55ed3b88fa17354da4f900187580acc3a7db92746c765c54704e633e1'
    Bootstrap = 'fa4927883805f3473205ccd54ecc4f5d80f5d05f02778133447013b8c4197391'
    RenderDoc = '809da38e3867d9fd09cc5c30dd5310500dee75e999166d7a14ad1ef6e9eca65d'
    Bridge    = 'efa943cb78099fcfccc11a10d4947693216e58e9525006b74ab2ee20d4ac8c21'
    Config    = '4e9baa188898decca8d799efa3517ff0f0820c1a54ba4c827fd569fdbac5809c'
}

function Sha($path) { (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant() }
function Require($path, $expected, $label) {
    if (-not (Test-Path $path)) { throw "$label missing: $path" }
    $actual = Sha $path
    if ($actual -ne $expected) { throw "$label hash $actual differs from pinned $expected ($path)" }
}

switch ($Mode) {
    'launch' {
        Require $ProbeSrc $Pinned.Probe 'Probe build'
        Require $Bootstrap $Pinned.Bootstrap 'Bootstrap build'
        Require $RenderDoc $Pinned.RenderDoc 'RenderDoc 1.46'
        Require $Bridge $Pinned.Bridge 'Installed bridge'
        Require "$Game\doorstop_config.ini" $Pinned.Config 'Original doorstop_config.ini'
        if (Test-Path $ProbeDir) { throw "Probe directory already exists; review it first: $ProbeDir" }
        if (Test-Path $Session) { throw "Session directory already exists; choose a fresh one: $Session" }
        New-Item -ItemType Directory -Path $ProbeDir | Out-Null
        Copy-Item $ProbeSrc $ProbeDir
        Require "$ProbeDir\HS2_FramePairProbe.dll" $Pinned.Probe 'Deployed probe'
        Write-Host "Probe deployed: $ProbeDir"
        # early_scope_game.py re-checks processes/hashes, backs up the config and launches.
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
        @{ probe_removed = -not (Test-Path $ProbeDir); doorstop_config_sha256 = (Sha "$Game\doorstop_config.ini");
           original_config_restored = $true; cleaned_at = (Get-Date).ToString('o') } |
            ConvertTo-Json | Set-Content -Encoding UTF8 "$Session\cleanup_receipt.json"
        Write-Host "Cleanup receipt: $Session\cleanup_receipt.json"
    }
}
