#ifndef UTILS_H
#define UTILS_H
/*
 * =====================================================================
 *      Project :  csc-gpu-calc
 *      File    :  utils.h
 *      Created :  6/3/2020 1:41:33 AM +0300
 *      Author  :  Dmitriy Ivanov
 * =====================================================================
 */

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#endif

inline void InitConsole()
{
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif
}

inline void SetConsoleRed()
{
    #ifdef _WIN32
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED);
    #elif
    std::cout << termcolor::bold << termcolor::red;
    #endif
}

inline void SetConsoleGreen()
{
    #ifdef _WIN32
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN);
    #elif
    std::cout << termcolor::bold << termcolor::green;
    #endif
}

inline void SetConsoleBlue()
{
    #ifdef _WIN32
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_BLUE);
    #elif
    std::cout << termcolor::bold << termcolor::blue;
    #endif
}


inline void ResetConsole()
{
    #ifdef _WIN32
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    #elif
    std::cout << termcolor::reset;
    #endif
}

inline std::vector<std::string> SplitString(const std::string& s, const char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

#endif //UTILS_H
