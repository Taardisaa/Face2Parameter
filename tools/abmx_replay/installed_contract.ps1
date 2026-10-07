param(
    [string]$OutPath = 'C:/Users/13666/Workspace/Face2Parameter/outputs/abmx_replay_20261005/installed_contract.json'
)
$ErrorActionPreference = 'Stop'
$taskDll = 'E:/HoneySelect2_ArcticFox/BepInEx/Plugins/HS2ABMX.dll'
$taskStream = [System.IO.File]::OpenRead($taskDll)
$taskPe = [System.Reflection.PortableExecutable.PEReader]::new($taskStream)
try {
    $taskReader = [System.Reflection.Metadata.PEReaderExtensions]::GetMetadataReader($taskPe)
    $taskModule = $taskReader.GetModuleDefinition()
    $taskMvid = $taskReader.GetGuid($taskModule.Mvid).ToString()
    $taskVersion = $taskReader.GetAssemblyDefinition().Version.ToString()
    $taskApply = $null
    foreach ($taskHandle in $taskReader.TypeDefinitions) {
        $taskType = $taskReader.GetTypeDefinition($taskHandle)
        if ($taskReader.GetString($taskType.Namespace) -eq 'KKABMX.Core' -and $taskReader.GetString($taskType.Name) -eq 'BoneModifier') {
            foreach ($taskMethodHandle in $taskType.GetMethods()) {
                $taskMethod = $taskReader.GetMethodDefinition($taskMethodHandle)
                if ($taskReader.GetString($taskMethod.Name) -eq 'Apply') {
                    if ($null -ne $taskApply) { throw 'Ambiguous Apply method' }
                    $taskApply = [System.Reflection.Metadata.PEReaderExtensions]::GetMethodBody($taskPe, $taskMethod.RelativeVirtualAddress)
                }
            }
        }
    }
    if ($null -eq $taskApply) { throw 'Installed BoneModifier.Apply missing' }
    if ($taskMvid -ne '5442c72a-f463-4bf9-831a-247be87146c8' -or $taskVersion -ne '4.4.6.0') { throw 'Unsupported ABMX source variant' }
    $taskSha = [System.Security.Cryptography.SHA256]::Create()
    try { $taskIlHash = [System.BitConverter]::ToString($taskSha.ComputeHash($taskApply.GetILBytes())).Replace('-', '').ToLowerInvariant() }
    finally { $taskSha.Dispose() }
    $taskSourcePaths = @('C:/Users/13666/Workspace/HS2Mod/tools/ABMX_BoneModifier.cs', 'C:/Users/13666/Workspace/HS2Mod/tools/ABMX_BoneModifierData.cs', 'C:/Users/13666/Workspace/HS2Mod/tools/ABMX_Core.cs')
    $taskSources = @($taskSourcePaths | ForEach-Object { @{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()} })
    $taskCoreText = [System.IO.File]::ReadAllText($taskSourcePaths[2])
    $taskExclusionStart = $taskCoreText.IndexOf('internal static HashSet<string> NoRotationBones')
    $taskExclusionEnd = $taskCoreText.IndexOf('public const string Version', $taskExclusionStart)
    if ($taskExclusionStart -lt 0 -or $taskExclusionEnd -le $taskExclusionStart) { throw 'Rotation exclusion source absent' }
    $taskExclusionText = $taskCoreText.Substring($taskExclusionStart, $taskExclusionEnd - $taskExclusionStart)
    $taskExclusions = @([System.Text.RegularExpressions.Regex]::Matches($taskExclusionText, '"([A-Za-z0-9_]+)"') | ForEach-Object { $_.Groups[1].Value })
    $taskContract = @{
        schema_version=1; plugin_mvid=$taskMvid; plugin_version=$taskVersion; apply_method_il_sha256=$taskIlHash;
        assembly_path=$taskDll; assembly_sha256=(Get-FileHash -LiteralPath $taskDll -Algorithm SHA256).Hash.ToLowerInvariant();
        unity_core_path='E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/UnityEngine.CoreModule.dll';
        unity_core_sha256=(Get-FileHash -LiteralPath 'E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/UnityEngine.CoreModule.dll' -Algorithm SHA256).Hash.ToLowerInvariant();
        sources=$taskSources; allowed_pre_existing_patch_owners=@();
        is_coordinate_specific=$false; vector3_equality_sqr_threshold=9.9999994E-11;
        no_rotation_bones=$taskExclusions;
        semantics_scope='Installed serialized method IL independently read from PE metadata; no assembly initialization, game contact or mutation'
    }
    $taskResolvedOut = [System.IO.Path]::GetFullPath($OutPath)
    if ([System.IO.File]::Exists($taskResolvedOut)) { throw 'Never overwrite old evidence' }
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($taskResolvedOut)) | Out-Null
    [System.IO.File]::WriteAllText($taskResolvedOut, ($taskContract | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
    Write-Output $taskResolvedOut
}
finally { $taskPe.Dispose(); $taskStream.Dispose() }
