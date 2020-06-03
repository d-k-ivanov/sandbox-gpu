/*
 * =====================================================================
 *      Project :  csc-gpu-calc
 *      File    :  main.cpp
 *      Created :  31/05/2020 22:55:21 +0300
 *      Author  :  Dmitriy Ivanov
 * =====================================================================
 */

#include "main.h"

#include <cxxopts.h>
#include <logger.h>
#include "utils.h"

#include <functional>
#include <iostream>
#include <string>

using namespace cgc;

std::map<std::string, std::function<void()>> Init(int argc, char* argv[])
{
    std::map<std::string, std::function<void()>> tasks =
    {
        { "task1", Task1},
        // { "task2", Task2},
        // { "task3", Task3}
    };

    cxxopts::Options options("csc-gpu-calc", "Description");
    options
        .positional_help("[optional args]")
        .show_positional_help();

    options.add_options()
        ("h,help",      "Show help")
        ("d,debug",     "Show debug output")
        ("v,verbose",   "Show verbose output")
        ("t,task",      "Task number", cxxopts::value<int>(), "N");

    options.custom_help("[-h] [-v] [-d] [-t N]");

    try {
        options.parse_positional({ "help", "verbose", "debug", "task" });
        const auto result = options.parse(argc, argv);

        if (result.count("help")) {
            SetConsoleGreen();
            std::cout << options.help() << '\n';
            ResetConsole();
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
            if (taskNumber > 1)
            {
                std::cout << "There are only one task. Exiting...\n";
                exit(2);
            }
            LogV(logger::logLevel) << std::string(100, '-') << '\n';
            tasks["task" + std::to_string(taskNumber)]();
            LogV(logger::logLevel) << std::string(100, '-') << '\n';

            exit(0);
        }
    }
    catch (const cxxopts::OptionException & e) {
        SetConsoleRed();
        LogE(logger::logLevel) << e.what() << " Showing help message...\n";
        SetConsoleGreen();
        std::cout << options.help() << '\n';
        ResetConsole();
        exit(99);
    }

    return tasks;
}

int main(int argc, char* argv[], char* env[])
{
    // To turn off messages about unused variables.
    ((void)argc);
    ((void)argv);
    ((void)env );

    InitConsole();

    auto tasks = Init(argc, argv);
    for (auto&& func: tasks)
    {
        LogV(logger::logLevel) << std::string(100, '-') << '\n';
        func.second();
    }
    LogV(logger::logLevel) << std::string(100, '-') << '\n';


    ResetConsole();
    // std::system("pause")
    return 0;
}
