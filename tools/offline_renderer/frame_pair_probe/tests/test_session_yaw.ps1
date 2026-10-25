# Offline test of the session script's multi-yaw body resolution (no game, no request written).
# Loads only the function definitions of paired_capture_session.ps1 and runs them against a
# temporary bulk directory.  pwsh -File tests\test_session_yaw.ps1
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot '..\paired_capture_session.ps1'
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $script), [ref]$null, [ref]$null)
$funcs = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)
foreach ($f in $funcs) { . ([scriptblock]::Create($f.Extent.Text)) }
$Inv = [Globalization.CultureInfo]::InvariantCulture
$AllowedBodyKeys = 'path', 'yaw', 'target', 'ortho_size', 'width', 'height', 'freeze_pose'
$Bulk = Join-Path ([IO.Path]::GetTempPath()) ("yaw_test_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $Bulk | Out-Null
$fail = 0; $pass = 0
function T($ok, $what) { if ($ok) { $script:pass++ } else { $script:fail++; Write-Host "FAIL: $what" } }
function Throws($sb, $what) { try { & $sb; $script:fail++; Write-Host "FAIL (no throw): $what" } catch { $script:pass++ } }
try {
    $tpl = Join-Path $Bulk 'template.json'
    WriteUtf8 $tpl '{"yaw":0.0,"target":[0.10687163,14.676028,-0.061020674],"ortho_size":1.42995071,"width":1086,"height":1448,"freeze_pose":true}'
    foreach ($y in @(0, 30, 45, 60, 90, -90, -45, 22.5)) {
        $n = [guid]::NewGuid().ToString('N')
        $f = ResolveYawBody $tpl ([double]$y) $n
        $b = Get-Content -Raw $f | ConvertFrom-Json
        T ([double]$b.yaw -eq [double]$y) "yaw $y written"
        T ($b.path -eq "$Bulk\bridge_capture_${n}_yaw$(YawTag ([double]$y)).png") "png path for $y"
        T ([double]$b.target[0] -eq 0.10687163 -and [double]$b.target[1] -eq 14.676028 -and [double]$b.target[2] -eq -0.061020674) "target unchanged for $y"
        T ([double]$b.ortho_size -eq 1.42995071 -and $b.width -eq 1086 -and $b.height -eq 1448 -and $b.freeze_pose -eq $true) "other fields for $y"
    }
    T ((YawTag -90.0) -eq 'm90' -and (YawTag 22.5) -eq '22p5') 'yaw tags'
    T ((ParseYaw ' -45 ') -eq -45) 'parse yaw'
    Throws { ParseYaw 'NaN' } 'NaN yaw'
    Throws { ParseYaw 'abc' } 'text yaw'
    $withPath = Join-Path $Bulk 'with_path.json'
    WriteUtf8 $withPath '{"path":"E:\\x.png","target":[0,0,0],"ortho_size":1,"width":64,"height":64}'
    Throws { ResolveYawBody $withPath 30.0 'n1' } 'template with path rejected'
    $badKey = Join-Path $Bulk 'bad_key.json'
    WriteUtf8 $badKey '{"target":[0,0,0],"ortho_size":1,"width":64,"height":64,"geometry_out_path":"x"}'
    Throws { ResolveYawBody $badKey 30.0 'n2' } 'unknown body key rejected'
} finally {
    $deprecated = Join-Path ([IO.Path]::GetTempPath()) 'yaw_test_done'
    if (-not (Test-Path $deprecated)) { New-Item -ItemType Directory $deprecated | Out-Null }
    Move-Item $Bulk $deprecated
}
Write-Host "test_session_yaw: $pass passed, $fail failed"
if ($fail) { exit 1 }
