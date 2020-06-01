/*
 * =====================================================================
 *      Project :  csc-gpu-calc
 *      File    :  main.cpp
 *      Created :  31/05/2020 22:55:21 +0300
 *      Author  :  Dmitriy Ivanov
 * =====================================================================
 */

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <CL/cl.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[], char* env[])
{
    // To turn off messages about unused variables.
    ((void)argc);
    ((void)argv);
    ((void)env );

    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    #ifdef _WIN32
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_BLUE);
    #elif
    std::cout << termcolor::bold << termcolor::yellow;
    #endif

    std::cout << "Echo 1111.\n";

    #ifdef _WIN32
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    #elif
    std::cout << termcolor::reset;
    #endif

    return 0;
}
