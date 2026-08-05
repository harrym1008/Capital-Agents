@echo off

call "D:\Apps\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set "CC=D:\Apps\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
set "CXX=D:\Apps\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
set "CL=/Zc:preprocessor"

cd /d "S:\CM3070 Final Project\Capital-Agents"
call "venv\Scripts\activate.bat"

python finbert/finbert_quantise.py
pause