param([Parameter(Mandatory=$true)][string]$Helper,[Parameter(Mandatory=$true)][string]$Cecil,[Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $Output){throw 'Fresh JSON IL audit required'}
Add-Type -Path $Cecil
$module=[Mono.Cecil.ModuleDefinition]::ReadModule([IO.Path]::GetFullPath($Helper))
$banned=@('Newtonsoft.Json.JsonConvert','Newtonsoft.Json.JsonSerializer','Newtonsoft.Json.Serialization.DefaultContractResolver')
$references=@($module.AssemblyReferences | ForEach-Object {$_.Name})
if($references -contains 'System.Data'){throw 'Helper references forbidden System.Data'}
$methods=@();$queue=New-Object 'Collections.Generic.Queue[Mono.Cecil.TypeDefinition]'
foreach($type in $module.Types){$queue.Enqueue($type)}
while($queue.Count){$type=$queue.Dequeue();foreach($nested in $type.NestedTypes){$queue.Enqueue($nested)}
 foreach($method in $type.Methods){if(!$method.HasBody){continue};foreach($instruction in $method.Body.Instructions){
  $operand=$instruction.Operand
  if($operand -is [Mono.Cecil.MethodReference]){
   if($banned -contains $operand.DeclaringType.FullName){throw ('Forbidden reflection JSON call '+$method.FullName+' -> '+$operand.FullName)}
   if($operand.DeclaringType.Namespace.StartsWith('Newtonsoft.Json')){$methods+=@{caller=$method.FullName;target=$operand.FullName}}
  }
 }}
}
@{scope='CPU IL dependency audit only';helper=[IO.Path]::GetFullPath($Helper);assembly_references=$references;forbidden_reflection_serializer_calls=0;newtonsoft_calls=$methods} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Output -Encoding UTF8
$module.Dispose();'CPU helper IL JSON dependency audit passed: no System.Data/reflection serializer calls.'
