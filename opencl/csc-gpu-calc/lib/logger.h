#ifndef CGC_LOGGER_H
#define CGC_LOGGER_H
/*
 * =====================================================================
 *      Project :  csc-gpu-calc
 *      File    :  logger.h
 *      Created :  6/1/2020 10:52:54 PM +0300
 *      Author  :  Dmitry Ivanov
 * =====================================================================
 */

#include <ostream>

#include <iostream>
#include <string>

#include "utils.h"

#ifdef _WIN32
    #include <windows.h>
#endif

namespace cgc::logger
{
    enum LogLevel
    {
        Error   = 0,
        Normal  = 1,
        Verbose = 2,
        Debug   = 3

    };

    inline LogLevel logLevel = Normal;

    class NoStreamBuf final : public std::streambuf {};

    inline NoStreamBuf noStreamBuf;
    inline std::ostream noOut(&noStreamBuf);

    constexpr std::ostream& LogE(const LogLevel lvl)
    {
        if (lvl >= Error)
        {
            return std::cout << "Error: ";
        }
        return noOut;
    }

    constexpr std::ostream& LogN(const LogLevel lvl)
    {
        if (lvl >= Normal)
        {
            return std::cout;
        }
        return noOut;
    }

    constexpr std::ostream& LogV(const LogLevel lvl)
    {
        if (lvl >= Verbose)
        {
            return std::cout;
        }
        return noOut;
    }

    constexpr std::ostream& LogD(const LogLevel lvl)
    {
        if (lvl >= Debug)
        {
            return std::cout << "Debug: ";
        }
        return noOut;
    }

    inline void PrintError(const std::string &msg)
    {
        SetConsoleRed();
        LogE(logLevel) << msg << std::endl;
        ResetConsole();
    }

    // template <LogLevel L>
    // std::ostream& operator<<(std::ostream& out, const Log<L>& v) {
    //    return out;
    // }

    // class NoStreamBuf final : public std::streambuf {};
    // inline  LogLevel logLevel = LogLevel::Normal;
    // inline NoStreamBuf noStreamBuf;
    // inline std::ostream noOut(&noStreamBuf);
    // #define LOG_V(x) (((x) >= cgc::logger::Verbose) ? std::cout                 : cgc::logger::noOut)
    // #define LOG_D(x) (((x) >= cgc::logger::Debug)   ? std::cout << "\tDEBUG: "  : cgc::logger::noOut)
}

#endif //CGC_LOGGER_H

