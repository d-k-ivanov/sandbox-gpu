#ifndef CGC_LOGGER_H
#define CGC_LOGGER_H
/*
 * =====================================================================
 *      Project :  csc-gpu-calc
 *      File    :  logger.h
 *      Created :  6/1/2020 10:52:54 PM +0300
 *      Author  :  Dmitriy Ivanov
 * =====================================================================
 */

#include <ostream>

namespace cgc::logger
{
    enum LogLevel
    {
        Normal  = 0,
        Verbose = 1,
        Debug   = 2

    };

    inline LogLevel logLevel = Normal;

    class NoStreamBuf final : public std::streambuf {};

    inline NoStreamBuf noStreamBuf;
    inline std::ostream noOut(&noStreamBuf);

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
            return std::cout << "\tDEBUG: ";
        }
        return noOut;
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

