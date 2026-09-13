@echo off
REM Cozucu icin cekim kaydeder -- ham landmark'lari da yazar.
REM Blender'a gerek YOK; bu kayit dosyaya gider, UDP'ye degil.
cd /d "%~dp0"
echo.
echo   facecap - COZUCU CEKIMI
echo.
echo   1) Onizleme penceresi acilinca ILK 2 SANIYE IFADESIZ DUR.
echo      Notr oradan ogreniliyor; gulumsersen butun cekim kayar.
echo   2) Sonra sirayla: agzini ac, gulumse, goz kirp, kasini kaldir,
echo      dudak buzustur. Her ifadeyi SONUNA KADAR yap.
echo   3) Bitince onizleme penceresinde  q  tusuna bas.
echo.
pause
"detector\.venv\Scripts\python.exe" "detector\detect.py" live --out "takes\deneme.jsonl" --landmarks --preview %*
echo.
echo   ---- kayit bitti: takes\deneme.jsonl ----
pause
