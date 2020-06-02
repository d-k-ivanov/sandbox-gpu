/*
 * =====================================================================
 *      Project :  csc-gpu-calc
 *      File    :  main.cpp
 *      Created :  31/05/2020 22:55:21 +0300
 *      Author  :  Dmitriy Ivanov
 * =====================================================================
 */

#include <cxxopts.h>
#include <logger.h>

#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#ifdef _WIN32
    #include <windows.h>
#endif

using namespace cgc;

std::map<std::string, std::function<void()>> Init(int argc, char* argv[])
{
    std::map<std::string, std::function<void()>> problems =
    {
        // { "task1", task1},
        // { "task2", task2},
        // { "task3", task3}
    };

    cxxopts::Options options("csc-gpu-calc", "Description");
    options
        .positional_help("[optional args]")
        .show_positional_help();

    options.add_options()
        ("h,help",      "Show help")
        ("d,debug",     "Show debug output")
        ("v,verbose",   "Show verbose output")
        ("s,separate",  "Show separated numbers in output")
        ("t,task",      "Task number", cxxopts::value<int>(), "N");

    options.custom_help("[-h] [-v] [-s]");

    try {
        options.parse_positional({ "help", "verbose", "debug", "separate" });
        const auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << '\n';
            // std::system("pause");
            exit(1);
        }

        if (result.count("verbose")) {
            logger::logLevel = logger::Verbose;
        }
        else if (result.count("debug"))
        {
            logger::logLevel = logger::Debug;
        }

        if (result.count("task"))
        {
            const auto taskNumber = result["task"].as<int>();
            if (taskNumber > 3)
            {
                std::cout << "There are only three problems. Exiting...\n";
                exit(2);
            }
            LogV(logger::logLevel) << std::string(100, '-') << '\n';
            problems["problem" + std::to_string(taskNumber)]();
            LogV(logger::logLevel) << std::string(100, '-') << '\n';

            exit(0);
        }
    }
    catch (const cxxopts::OptionException & e) {
        std::cout << "Error: " << e.what() << " Showing help message...\n";
        std::cout << options.help() << '\n';
        exit(99);
    }

    return problems;
}

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

    std::map<std::string, std::function<void()>> tasks = Init(argc, argv);
    for (std::pair<std::string, std::function<void()>> func: tasks)
    {
        LogV(logger::logLevel) << std::string(100, '-') << '\n';
        func.second();
    }
    LogV(logger::logLevel) << std::string(100, '-') << '\n';


    #ifdef _WIN32
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    #elif
    std::cout << termcolor::reset;
    #endif

    // std::system("pause")
    return 0;
}
