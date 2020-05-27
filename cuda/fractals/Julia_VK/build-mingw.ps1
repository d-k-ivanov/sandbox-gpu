#requires -version 3
cmd /c 'gcc -c -o bin/fractal-julia.o src/fractal-julia.c -I"include/"'
#cmd /c 'gcc -o bin/fractal-julia.exe bin/fractal-julia.o -L"libs/x86/mingw" -lglfw3 -mwindows -lglu32 -lopengl32'
cmd /c 'gcc -o bin/fractal-julia.exe bin/fractal-julia.o -L"libs/x64/mingw" -lglfw3 -mwindows -lglu32 -lopengl32'
