param(
    [Parameter(Mandatory=$true)][string]$AssemblyPath,
    [Parameter(Mandatory=$true)][string]$ResourceName,
    [Parameter(Mandatory=$true)][string]$CecilPath,
    [Parameter(Mandatory=$true)][string]$OutPath
)
$ErrorActionPreference = 'Stop'
$taskAssemblyPath = (Resolve-Path -LiteralPath $AssemblyPath).Path
$taskCecilPath = (Resolve-Path -LiteralPath $CecilPath).Path
$taskOutPath = [IO.Path]::GetFullPath($OutPath)
if (Test-Path -LiteralPath $taskOutPath) { throw 'Fresh output file required' }
Add-Type -Path $taskCecilPath
$taskAssembly = [Mono.Cecil.AssemblyDefinition]::ReadAssembly($taskAssemblyPath)
try {
    $taskResources = @($taskAssembly.MainModule.Resources | Where-Object { $_.Name -ceq $ResourceName })
    if ($taskResources.Count -ne 1 -or $taskResources[0].ResourceType.ToString() -ne 'Embedded') {
        throw 'Exactly one embedded resource required'
    }
    $taskDirectory = [IO.Path]::GetDirectoryName($taskOutPath)
    [IO.Directory]::CreateDirectory($taskDirectory) | Out-Null
    [IO.File]::WriteAllBytes($taskOutPath, $taskResources[0].GetResourceData())
    [ordered]@{
        assembly = $taskAssemblyPath
        assembly_sha256 = (Get-FileHash -LiteralPath $taskAssemblyPath -Algorithm SHA256).Hash.ToLowerInvariant()
        resource = $ResourceName
        output = $taskOutPath
        output_sha256 = (Get-FileHash -LiteralPath $taskOutPath -Algorithm SHA256).Hash.ToLowerInvariant()
        scope = 'Read-only embedded asset extraction; no assembly load, game or shader replacement'
    } | ConvertTo-Json | Set-Content -LiteralPath ($taskOutPath + '.provenance.json') -Encoding utf8
} finally {
    $taskAssembly.Dispose()
}
