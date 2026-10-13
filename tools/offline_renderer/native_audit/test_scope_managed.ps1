param([Parameter(Mandatory=$true)][string]$BuildDirectory)
$ErrorActionPreference='Stop'
$scopeRoot=[IO.Path]::GetFullPath($BuildDirectory)
$requestPath=Join-Path $scopeRoot 'HS2_RenderDocScope.request.json'
if (Test-Path -LiteralPath $requestPath) { throw 'Fresh managed CPU test requires no existing request' }
[void][Reflection.Assembly]::LoadFile((Join-Path $scopeRoot 'Newtonsoft.Json.dll'))
[void][Reflection.Assembly]::LoadFile((Join-Path $scopeRoot 'HS2_RenderDocScope.dll'))
# No request/disabled/mismatched path must not invoke/load our native DLL or RenderDoc.
[HS2_RenderDocScope]::BeginFromNativeTexture([IntPtr]1, (Join-Path $scopeRoot 'other.png'))
[HS2_RenderDocScope]::EndAfterSavedPng((Join-Path $scopeRoot 'other.png'))
[HS2_RenderDocScope]::AbortOwnedSafely()
if ([HS2_RenderDocScope]::LastReceipt -ne '{"status":"not_started"}') { throw 'No-request path changed state' }
try {
  @{Enabled=$false; GeometryBeforePath='invalid-relative-before.json'; GeometryAfterPath='invalid-relative-after.json'; ExpectedBridgeSha256='invalid'} | ConvertTo-Json | Set-Content -LiteralPath $requestPath -Encoding UTF8
  [HS2_RenderDocScope]::BeginFromNativeTexture([IntPtr]1, (Join-Path $scopeRoot 'disabled.png'))
  @{Enabled=$true; ExactPngPath=(Join-Path $scopeRoot 'requested.png'); GeometryBeforePath='invalid-relative-before.json'; GeometryAfterPath='invalid-relative-after.json'; ExpectedBridgeSha256='invalid'} | ConvertTo-Json | Set-Content -LiteralPath $requestPath -Encoding UTF8
  [HS2_RenderDocScope]::BeginFromNativeTexture([IntPtr]1, (Join-Path $scopeRoot 'mismatch.png'))
  if ([HS2_RenderDocScope]::LastReceipt -ne '{"status":"not_started"}') { throw 'Nonselected capture changed state' }
  @{Enabled=$true; ExpectedPid=0; ExactPngPath=(Join-Path $scopeRoot 'pid.png')} | ConvertTo-Json | Set-Content -LiteralPath $requestPath -Encoding UTF8
  [HS2_RenderDocScope]::BeginFromNativeTexture([IntPtr]1, (Join-Path $scopeRoot 'pid.png'))
  if ([HS2_RenderDocScope]::LastReceipt -notmatch 'Expected HS2 process mismatch') { throw 'Expected PID not enforced' }
  $loaded=@([Diagnostics.Process]::GetCurrentProcess().Modules | Where-Object { $_.ModuleName -in @('renderdoc.dll','hs2_renderdoc_scope_native.dll') })
  if ($loaded.Count -ne 0) { throw 'CPU disabled/rejected paths loaded a native graphics/helper module' }
  'Managed CPU opt-in checks passed: missing/disabled/mismatched request, wrong PID, no native loads.'
} finally { Remove-Item -LiteralPath $requestPath }
