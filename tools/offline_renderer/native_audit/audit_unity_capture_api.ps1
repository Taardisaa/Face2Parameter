param([Parameter(Mandatory=$true)][string]$Core,[Parameter(Mandatory=$true)][string]$Cecil,[Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $Output){throw 'Fresh Unity API audit receipt required'}
Add-Type -Path $Cecil
$module=[Mono.Cecil.ModuleDefinition]::ReadModule([IO.Path]::GetFullPath($Core))
$types=@('UnityEngine.GL','UnityEngine.WaitForEndOfFrame','UnityEngine.Texture','UnityEngine.Rendering.CommandBuffer')
$names=@('IssuePluginEvent','IssuePluginEventInternal','GetNativeTexturePtr','.ctor')
$methods=@()
foreach($type in $module.Types){if($types -notcontains $type.FullName){continue};foreach($method in $type.Methods){if($names -notcontains $method.Name){continue};$methods+=@{signature=$method.FullName;attributes=$method.Attributes.ToString();implementation=$method.ImplAttributes.ToString();il=if($method.HasBody){@($method.Body.Instructions | ForEach-Object {$_.ToString()})}else{@()}}}}
$exact=@($methods | Where-Object {$_.signature -eq 'System.Void UnityEngine.GL::IssuePluginEvent(System.IntPtr,System.Int32)'})
if($exact.Count -ne 1){throw 'Exact installed public GL native callback API unavailable'}
$receipt=@{scope='Installed Unity2018 metadata/source only; GL event threading ordering comes from official documentation, persistent Map closure remains unproven';core_path=[IO.Path]::GetFullPath($Core);core_sha256=(Get-FileHash -LiteralPath $Core -Algorithm SHA256).Hash.ToLowerInvariant();mvid=$module.Mvid.ToString();assembly=$module.Assembly.FullName;methods=$methods;native_callback_accepted=$false;map_lifetime_repair_accepted=$false}
$module.Dispose()
$receipt | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Output -Encoding utf8
'Installed Unity GL/native resource/end-of-frame API metadata audited; no assembly/game/graphics execution.'
