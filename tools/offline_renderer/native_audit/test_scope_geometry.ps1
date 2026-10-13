param([Parameter(Mandatory=$true)][string]$BuildDirectory,[Parameter(Mandatory=$true)][string]$FixtureDirectory,[string]$Variant='good')
$ErrorActionPreference='Stop'
$scopeRoot=[IO.Path]::GetFullPath($BuildDirectory)
[void][Reflection.Assembly]::LoadFile((Join-Path $scopeRoot 'Newtonsoft.Json.dll'))
$scopeAssembly=[Reflection.Assembly]::LoadFile((Join-Path $scopeRoot 'HS2_RenderDocScope.dll'))
$fixture=[Reflection.Assembly]::LoadFile((Join-Path ([IO.Path]::GetFullPath($FixtureDirectory)) 'HS2_McpBridge.dll'))
$type=$scopeAssembly.GetType('GeometryCapture',$true)
$prepare=$type.GetMethod('Prepare',[Reflection.BindingFlags]'Static,NonPublic')
$temporary=Join-Path $scopeRoot ('geometry_cpu_'+$Variant+'_'+[Guid]::NewGuid().ToString('N'))
if(Test-Path -LiteralPath $temporary){throw 'Fresh CPU geometry test directory required'}
[void][IO.Directory]::CreateDirectory($temporary)
function File-Hash([string]$path) {
  $hasher=[Security.Cryptography.SHA256]::Create();$stream=[IO.File]::OpenRead($path)
  try {return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-','').ToLowerInvariant()}
  finally {$stream.Dispose();$hasher.Dispose()}
}
function New-Request {
  $r=New-Object 'HS2_RenderDocScope+Request'
  $r.GeometryBeforePath=Join-Path $temporary ([Guid]::NewGuid().ToString('N')+'_before.json')
  $r.GeometryAfterPath=Join-Path $temporary ([Guid]::NewGuid().ToString('N')+'_after.json')
  $r.ExpectedBridgeSha256=File-Hash $fixture.Location
  return $r
}
function Assert-Rejected($r,[string]$reason) {
  try {$result=$prepare.Invoke($null,[object[]]@($r.PSObject.BaseObject))} catch {
    if($_.Exception.ToString() -notmatch $reason){throw "Wrong rejection: $($_.Exception)"}; return
  }; throw 'Expected geometry request rejection did not occur'
}
$r=New-Request
if($Variant -ne 'good') {
  if($Variant -eq 'wrong_signature'){Assert-Rejected $r 'method signature differs'}
  elseif($Variant -eq 'wrong_ctor'){Assert-Rejected $r 'object-only initializer'}
  else{throw 'Unknown fixture variant'}
  "CPU geometry $Variant metadata rejection passed; fixture never invoked."
  exit 0
}
$none=New-Object 'HS2_RenderDocScope+Request'
if($null -ne $prepare.Invoke($null,[object[]]@($none.PSObject.BaseObject))){throw 'Missing options changed legacy behavior'}
$r=New-Request;$r.GeometryAfterPath=$null;Assert-Rejected $r 'fresh absolute JSON'
$r=New-Request;$r.GeometryAfterPath=$r.GeometryBeforePath;Assert-Rejected $r 'must be distinct'
$r=New-Request;[IO.File]::WriteAllText($r.GeometryBeforePath,'retained');Assert-Rejected $r 'fresh absolute JSON'
if([IO.File]::ReadAllText($r.GeometryBeforePath) -ne 'retained'){throw 'Old geometry overwritten'}
$r=New-Request;$r.GeometryBeforePath='relative.json';Assert-Rejected $r 'fresh absolute JSON'
$r=New-Request;$r.ExpectedBridgeSha256=('0'*64);Assert-Rejected $r 'SHA256 mismatch'
$r=New-Request;$r.ReceiptPath=$r.GeometryAfterPath;Assert-Rejected $r 'must not alias'
$r=New-Request;$session=$prepare.Invoke($null,[object[]]@($r.PSObject.BaseObject))
$session.Before();$session.After()
$info=$session.Describe()
if(!$info['complete'] -or $info['before']['frame_count'] -ne 42 -or $info['after']['frame_count_end'] -ne 42){throw 'Paired scalar metadata failed'}
foreach($boundary in @('before','after')){
  $record=$info[$boundary]
  if($record['sha256'] -ne (File-Hash $record['path']) -or $record['bytes'] -ne ([IO.FileInfo]$record['path']).Length){throw 'Saved byte/hash witness mismatch'}
  if(!$record['native_export_receipt']['fixture_only']){throw 'Fixture diagnostic identity lost'}
}
$r=New-Request;$session=$prepare.Invoke($null,[object[]]@($r.PSObject.BaseObject))
[Environment]::SetEnvironmentVariable('HS2_SCOPE_CPU_FIXTURE_FAILURE','throw')
try {$session.Before();throw 'Native export failure accepted'}catch {if($_.Exception.ToString() -notmatch 'CPU fixture native export failure'){throw}}
finally {[Environment]::SetEnvironmentVariable('HS2_SCOPE_CPU_FIXTURE_FAILURE',$null)}
if($session.Complete){throw 'Failed export marked complete'}
$r=New-Request;$session=$prepare.Invoke($null,[object[]]@($r.PSObject.BaseObject))
[Environment]::SetEnvironmentVariable('HS2_SCOPE_CPU_FIXTURE_FAILURE','hash')
try {$session.Before();throw 'Hash failure accepted'}catch {if($_.Exception.ToString() -notmatch 'provenance mismatch'){throw}}
finally {[Environment]::SetEnvironmentVariable('HS2_SCOPE_CPU_FIXTURE_FAILURE',$null)}
if(!(Test-Path -LiteralPath $r.GeometryBeforePath) -or $session.Complete){throw 'Failure evidence not retained'}
$r=New-Request;$session=$prepare.Invoke($null,[object[]]@($r.PSObject.BaseObject));$session.Before()
[Environment]::SetEnvironmentVariable('HS2_SCOPE_CPU_FIXTURE_FAILURE','throw')
try {$session.After();throw 'After export failure accepted'}catch {if($_.Exception.ToString() -notmatch 'CPU fixture native export failure'){throw}}
finally {[Environment]::SetEnvironmentVariable('HS2_SCOPE_CPU_FIXTURE_FAILURE',$null)}
if($session.Complete -or $null -eq $session.Describe()['before']){throw 'After failure lost before witness or claimed complete'}
$r=New-Request;$session=$prepare.Invoke($null,[object[]]@($r.PSObject.BaseObject));$session.Before()
[IO.File]::WriteAllText($r.GeometryAfterPath,'external geometry retained')
try {$session.After();throw 'Late nonfresh after path accepted'}catch {if($_.Exception.ToString() -notmatch 'fresh absolute JSON'){throw}}
if([IO.File]::ReadAllText($r.GeometryAfterPath) -ne 'external geometry retained' -or $session.Complete){throw 'Late external geometry overwritten'}
$loaded=@([Diagnostics.Process]::GetCurrentProcess().Modules | Where-Object { $_.ModuleName -in @('renderdoc.dll','hs2_renderdoc_scope_native.dll') })
if($loaded.Count){throw 'Geometry CPU fixture loaded native graphics/helper module'}
'CPU geometry fixture checks passed: optional legacy, exact signature/ctor/arguments, paths, loaded hash, paired metadata, before/after export/hash/late-freshness failures retained; no graphics/game.'
