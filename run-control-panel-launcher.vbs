Dim WshShell, fso, deployDir
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Cartella dove si trova questo script = deployment root
deployDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Imposta la variabile d'ambiente per il processo figlio
WshShell.Environment("Process")("_MARKET_MONITOR_FROZEN_ROOT") = deployDir

' Lancia il control panel: WindowStyle=0 -> nessuna finestra, bWaitOnReturn=False
WshShell.Run """" & deployDir & "\run-control-panel.exe""", 0, False
