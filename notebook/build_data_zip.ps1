# Builds Downloads\nav3_bars_66.zip: verified TriFrame tier-A bars (final/) for the 50 study stocks + 16 sector reserves,
# plus SPY daily (NEXUS V1 ETF store), and a SHA-256 manifest. Entry names use forward slashes (Colab/Linux safe).
# Usage: pwsh -File tools\build_data_zip.ps1
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$final = 'C:\Users\harsh\Downloads\us_stocks_ohlcv\final'
$spy   = 'C:\Users\harsh\nexus-alpha\artifacts\etf\1day\SPY.parquet'
$out   = 'C:\Users\harsh\Downloads\nav3_bars_66.zip'

$study = @{
  'Information Technology' = 'AAPL MSFT NVDA TXN INTC CSCO ORCL QCOM AMD ADBE'
  'Financials'             = 'JPM BAC WFC C GS MS V MA BRK.B SCHW'
  'Health Care'            = 'UNH JNJ LLY PFE MRK ABT TMO AMGN GILD CVS'
  'Consumer Discretionary' = 'AMZN HD MCD NKE SBUX F LOW TJX MAR ROST'
  'Communication Services' = 'GOOG NFLX DIS T VZ META OMC TTWO NYT LYV'
}
$reserve = @{
  'Information Technology' = 'ADI MU IBM INTU'
  'Financials'             = 'AXP PNC USB BLK'
  'Health Care'            = 'MDT BMY DHR ISRG'
  'Consumer Discretionary' = 'EBAY YUM ORLY'
  'Communication Services' = 'IMAX'
}
$tickers = @()
foreach ($k in $study.Keys) { $tickers += ($study[$k] -split ' ') }
foreach ($k in $reserve.Keys) { $tickers += ($reserve[$k] -split ' ') }
$tickers = $tickers | Sort-Object -Unique
if ($tickers.Count -ne 66) { throw "expected 66 tickers, got $($tickers.Count)" }

$files = [ordered]@{}
foreach ($t in $tickers) {
  foreach ($iv in '1day', '1hour', '1week') {
    $p = Join-Path $final "$iv\$t.parquet"
    if (-not (Test-Path $p)) { throw "missing $p" }
    $files["$iv/$t.parquet"] = $p
  }
}
$files['etf/SPY_1day.parquet'] = $spy

$sha = [ordered]@{}
foreach ($k in $files.Keys) { $sha[$k] = (Get-FileHash -Algorithm SHA256 $files[$k]).Hash.ToLower() }

$rootSha = (Select-String -Path (Join-Path $final 'DATA_CARD.md') -Pattern 'root sha256 `([0-9a-f]+)' ).Matches[0].Groups[1].Value
$manifest = [ordered]@{
  source            = 'TriFrame tier-A final/ (frozen 2026-10-11, data through 2026-10-09) + NEXUS V1 ETF store (SPY daily)'
  source_root_sha256_prefix = $rootSha
  data_end          = '2026-10-09'
  built_utc         = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
  study             = $study
  reserves          = $reserve
  files             = $sha
}
$mjson = ($manifest | ConvertTo-Json -Depth 5)
$mtmp = Join-Path $env:TEMP 'nav3_manifest.json'
[System.IO.File]::WriteAllText($mtmp, $mjson.Replace("`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))

if (Test-Path $out) { Remove-Item $out -Force }
$fs = [System.IO.File]::Open($out, [System.IO.FileMode]::CreateNew)
$zip = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Create)
try {
  [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $mtmp, 'nav3_bars/manifest.json', [System.IO.Compression.CompressionLevel]::Optimal)
  foreach ($k in $files.Keys) {
    [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $files[$k], "nav3_bars/$k", [System.IO.Compression.CompressionLevel]::NoCompression)
  }
} finally { $zip.Dispose(); $fs.Dispose() }

$msha = (Get-FileHash -Algorithm SHA256 $mtmp).Hash.ToLower()
"wrote $out ($([Math]::Round((Get-Item $out).Length / 1MB, 1)) MB), $($files.Count) data files"
"manifest sha256: $msha"
