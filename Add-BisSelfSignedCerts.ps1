# This script retrieves all certificates from the CurrentUser\CA store,
# converts them to Base64-encoded PEM format, and appends them to the
# Python `certifi` certificate bundle.

# Ensure there is an active Python virtual environment
if (-not (Test-Path -Path "$env:VIRTUAL_ENV")) {
    Write-Error "No active Python virtual environment detected. Please activate a virtual environment and try again."
    exit 1
}

# Retrieve all certificates from the CurrentUser\CA store
$certs = Get-ChildItem Cert:\CurrentUser\CA

# Process each certificate and convert it to PEM format
$result = foreach ($cert in $certs) {
    $out = [System.Text.StringBuilder]::new()
    [void]$out.AppendLine("-----BEGIN CERTIFICATE-----")
    [void]$out.AppendLine([System.Convert]::ToBase64String($cert.RawData, 1))
    [void]$out.AppendLine("-----END CERTIFICATE-----")
    $out.ToString()
}

# Append the PEM-formatted certificates to the certifi bundle
Add-Content $(python -m certifi) $result
# Write a message with the number of certificates added
$certCount = $certs.Count
Write-Host "$certCount certificate(s) were successfully added to the certifi bundle."

exit 0
