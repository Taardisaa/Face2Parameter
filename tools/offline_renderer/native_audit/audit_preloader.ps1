param([Parameter(Mandatory=$true)][string]$Preloader,[Parameter(Mandatory=$true)][string]$Cecil,[Parameter(Mandatory=$true)][string]$Output,[string]$Bootstrap)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $Output){throw 'Fresh metadata receipt required'}
Add-Type -Path $Cecil
$module=[Mono.Cecil.ModuleDefinition]::ReadModule([IO.Path]::GetFullPath($Preloader))
$names=@('Main','ResolveCurrentDirectory','PreloaderPreMain','PreloaderMain','LoadVars')
$methods=@();foreach($type in $module.Types){foreach($method in $type.Methods){if($names -notcontains $method.Name){continue}
  $methods+=@{signature=$method.FullName;attributes=$method.Attributes.ToString();il=@($method.Body.Instructions | ForEach-Object {$_.ToString()});exception_handlers=@($method.Body.ExceptionHandlers | ForEach-Object {$_.ToString()})}
}}
$references=@($module.AssemblyReferences | ForEach-Object {$_.FullName});$mvid=$module.Mvid.ToString();$module.Dispose()
function Hash([string]$path){$sha=[Security.Cryptography.SHA256]::Create();$input=[IO.File]::OpenRead($path);try{return [BitConverter]::ToString($sha.ComputeHash($input)).Replace('-','').ToLowerInvariant()}finally{$input.Dispose();$sha.Dispose()}}
$receipt=@{scope='CPU installed IL/static metadata only; no assembly/game/graphics execution';preloader_path=[IO.Path]::GetFullPath($Preloader);preloader_sha256=(Hash $Preloader);mvid=$mvid;assembly_references=$references;entry_chain=$methods}
if($Bootstrap){
  $boot=[Mono.Cecil.ModuleDefinition]::ReadModule([IO.Path]::GetFullPath($Bootstrap));$refs=@($boot.AssemblyReferences | ForEach-Object {$_.Name});
  if($refs.Count -ne 1 -or $refs[0] -ne 'mscorlib'){throw 'Bootstrap must reference only mscorlib'}
  $receipt.bootstrap=@{path=[IO.Path]::GetFullPath($Bootstrap);sha256=(Hash $Bootstrap);assembly_references=$refs;main_count=@($boot.Types | ForEach-Object {$_.Methods} | Where-Object {$_.Name -eq 'Main'}).Count};
  if($receipt.bootstrap.main_count -ne 1){throw 'Doorstop wildcard Main must select only the intended wrapper'}
  $boot.Dispose()
}
$receipt | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Output -Encoding UTF8
'CPU original preloader entry-chain and bootstrap mscorlib-only metadata audit passed.'
