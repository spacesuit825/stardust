

@REM call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

cd C:/Users/lachl/Documents/stardust/out/build/x64-Debug

del CMakeCache.txt

cmake C:/Users/lachl/Documents/stardust -G "Visual Studio 17 2022" 

cmake --build . --clean-first

del C:\Users\lachl\Documents\stardust\sandbox\vtkFiles\*.vtk

del C:\Users\lachl\Documents\stardust\sandbox\archive.zip

"./sandbox/Debug/test_scale.exe"



"C:/Program Files/7-Zip/7z" a -tzip C:/Users/lachl/Documents/stardust/sandbox/archive.zip -r C:/Users/lachl/Documents/stardust/sandbox/vtkFiles/*.vtk

cd C:/Users/lachl/Documents/stardust

