/*
 * =====================================================================
 *      Project :  csc-gpu-calc
 *      File    :  task1.cpp
 *      Created :  6/3/2020 1:15:24 AM +0300
 *      Author  :  Dmitry Ivanov
 * =====================================================================
 */

#include <logger.h>
#include <utils.h>

#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <string>


#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
    #include <OpenCL/cl.hpp>
#else
    #include <CL/opencl.h>
    #include <CL/cl.hpp>
#endif

namespace cgc
{
    void Task1() {
        std::vector< cl::Platform > platforms;
        std::vector< cl::Device > devices;
        auto exitCode = cl::Platform::get( &platforms );
        if (exitCode != CL_SUCCESS)
        {
            logger::PrintError("Unable to get OpenCL Platforms");
            exit(EXIT_FAILURE);
        }

        assert( CL_SUCCESS ==  exitCode );
        assert( platforms.size() > 0 );

        std::cout << "OpenCL Platforms:" << std::endl;
        for( size_t i=0 ; i<platforms.size() ; i++ ) {
            std::string buf;
            std::cout << "  Platform " << (i+1) << ": " << std::endl;
            platforms[i].getInfo(  CL_PLATFORM_NAME , &buf );
            std::cout << "    CL_PLATFORM_NAME: " << buf << std::endl;
            platforms[i].getInfo(  CL_PLATFORM_VERSION , &buf );
            std::cout << "    CL_PLATFORM_VERSION: " << buf << std::endl;
            platforms[i].getInfo(  CL_PLATFORM_VENDOR , &buf );
            std::cout << "    CL_PLATFORM_VENDOR: " << buf << std::endl;
            platforms[i].getInfo( CL_PLATFORM_PROFILE , &buf );
            std::cout << "    CL_PLATFORM_PROFILE: " << buf << std::endl;
            platforms[i].getInfo(  CL_PLATFORM_EXTENSIONS , &buf );
            LogV(logger::logLevel) << "    CL_PLATFORM_EXTENSIONS: " << std::endl;
            for(auto & str: SplitString(buf, ' '))
            {
                LogV(logger::logLevel) << "        - " << str << std::endl;
            }

            exitCode = platforms[i].getDevices(CL_DEVICE_TYPE_ALL,&devices);
            if (exitCode != CL_SUCCESS)
            {
                logger::PrintError("Unable to get OpenCL Devices");
                exit(EXIT_FAILURE);
            }

            if (devices.empty())
            {
                std::cout << "    Unable to find any OpenCL Devices for this platform." << std::endl;
            }

            std::cout << "    Platform Devices:" << std::endl;
            for( size_t j=0 ; j<devices.size() ; j++ ) {
                // Buffers
                size_t                      sizeBuf;
                size_t                      sizeBufArr[32];

                char                        charBuf[1024];

                cl_int                      iBuf;
                cl_bool                     bBuf;
                cl_uint                     uiBuf;
                cl_ulong                    ulBuf;
                cl_device_type              typeBuf;
                cl_platform_id              platformBuf;
                cl_device_fp_config         fpBuf;
                cl_device_local_mem_type    memTypeBuf;
                cl_device_mem_cache_type    cacheTypeBuf;
                cl_command_queue_properties qpBuf;

                std::cout << "      Device " << (j+1) << ": " << std::endl;

                devices[j].getInfo( CL_DEVICE_NAME , &charBuf );
                std::cout << "        CL_DEVICE_NAME: " << charBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_VENDOR , &charBuf );
                std::cout << "        CL_DEVICE_VENDOR: " << charBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_VENDOR_ID , &iBuf );
                std::cout << "        CL_DEVICE_VENDOR_ID: " << iBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_VERSION , &charBuf );
                std::cout << "        CL_DEVICE_VERSION: " << charBuf << std::endl;
                devices[j].getInfo( CL_DRIVER_VERSION , &charBuf );
                std::cout << "        CL_DRIVER_VERSION: " << charBuf << std::endl;

                devices[j].getInfo( CL_DEVICE_TYPE , &typeBuf );
                std::cout << "        CL_DEVICE_TYPE: ";
                switch (typeBuf)
                {
                    case CL_DEVICE_TYPE_DEFAULT:
                        std::cout << "CL_DEVICE_TYPE_DEFAULT" << std::endl;
                        break;
                    case CL_DEVICE_TYPE_CPU:
                        std::cout << "CL_DEVICE_TYPE_CPU" << std::endl;
                        break;
                    case CL_DEVICE_TYPE_GPU:
                        std::cout << "CL_DEVICE_TYPE_GPU" << std::endl;
                        break;
                    case CL_DEVICE_TYPE_ACCELERATOR:
                        std::cout << "CL_DEVICE_TYPE_ACCELERATOR" << std::endl;
                        break;
                    case CL_DEVICE_TYPE_CUSTOM:
                        std::cout << "CL_DEVICE_TYPE_CUSTOM" << std::endl;
                        break;
                    default:
                        std::cout << "Unknown" << std::endl;
                }

                // if( iBuf & CL_DEVICE_TYPE_DEFAULT )
                //     std::cout << "CL_DEVICE_TYPE_DEFAULT" << std::endl;
                // if( iBuf & CL_DEVICE_TYPE_CPU )
                //     std::cout << "CL_DEVICE_TYPE_CPU" << std::endl;
                // if( iBuf & CL_DEVICE_TYPE_GPU )
                //     std::cout << "CL_DEVICE_TYPE_GPU" << std::endl;
                // if( iBuf & CL_DEVICE_TYPE_ACCELERATOR )
                //     std::cout << "CL_DEVICE_TYPE_ACCELERATOR" << std::endl;
                // if( iBuf & CL_DEVICE_TYPE_CUSTOM )
                //     std::cout << "CL_DEVICE_TYPE_CUSTOM" << std::endl;
                // if( iBuf & CL_DEVICE_TYPE_ALL )
                //     std::cout << "        MASK ALL: CL_DEVICE_TYPE_ALL" << std::endl;


                devices[j].getInfo( CL_DEVICE_ADDRESS_BITS , &uiBuf );
                std::cout << "        CL_DEVICE_ADDRESS_BITS: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_AVAILABLE , &bBuf );
                std::cout << "        CL_DEVICE_AVAILABLE: " << bBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_COMPILER_AVAILABLE , &bBuf );
                std::cout << "        CL_DEVICE_COMPILER_AVAILABLE: " << bBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_ENDIAN_LITTLE , &iBuf );
                std::cout << "        CL_DEVICE_ENDIAN_LITTLE: " << iBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_ERROR_CORRECTION_SUPPORT , &bBuf );
                std::cout << "        CL_DEVICE_ERROR_CORRECTION_SUPPORT: " << bBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_EXECUTION_CAPABILITIES , &iBuf );
                std::cout << "        CL_DEVICE_EXECUTION_CAPABILITIES: " << iBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_EXTENSIONS , &charBuf );

                LogV(logger::logLevel) << "        CL_DEVICE_EXTENSIONS: " << std::endl;
                for(auto & str: SplitString(charBuf, ' '))
                {
                    LogV(logger::logLevel) << "            - " << str << std::endl;
                }

                devices[j].getInfo( CL_DEVICE_NAME , &charBuf );
                std::cout << "        CL_DEVICE_NAME: " << charBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_PLATFORM , &platformBuf );
                std::cout << "        CL_DEVICE_PLATFORM: " << platformBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_PROFILE , &charBuf );
                std::cout << "        CL_DEVICE_PROFILE: " << charBuf << std::endl;

                devices[j].getInfo( CL_DEVICE_GLOBAL_MEM_CACHE_SIZE , &ulBuf );
                std::cout << "        CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: " << ulBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_GLOBAL_MEM_CACHE_TYPE , &cacheTypeBuf );
                std::cout << "        CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: " << cacheTypeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE , &uiBuf );
                std::cout << "        CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_GLOBAL_MEM_SIZE , &ulBuf );
                std::cout << "        CL_DEVICE_GLOBAL_MEM_SIZE: " << ulBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_IMAGE_SUPPORT , &bBuf );
                std::cout << "        CL_DEVICE_IMAGE_SUPPORT: " << bBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_IMAGE2D_MAX_HEIGHT , &sizeBuf );
                std::cout << "        CL_DEVICE_IMAGE2D_MAX_HEIGHT: " << sizeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_IMAGE2D_MAX_WIDTH , &sizeBuf );
                std::cout << "        CL_DEVICE_IMAGE2D_MAX_WIDTH: " << sizeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_IMAGE3D_MAX_DEPTH , &sizeBuf );
                std::cout << "        CL_DEVICE_IMAGE3D_MAX_DEPTH: " << sizeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_IMAGE3D_MAX_HEIGHT , &sizeBuf );
                std::cout << "        CL_DEVICE_IMAGE3D_MAX_HEIGHT: " << sizeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_IMAGE3D_MAX_WIDTH , &sizeBuf );
                std::cout << "        CL_DEVICE_IMAGE3D_MAX_WIDTH: " << sizeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_LOCAL_MEM_SIZE , &ulBuf );
                std::cout << "        CL_DEVICE_LOCAL_MEM_SIZE: " << ulBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_LOCAL_MEM_TYPE , &memTypeBuf );
                std::cout << "        CL_DEVICE_LOCAL_MEM_TYPE: " << memTypeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_CLOCK_FREQUENCY , &uiBuf );
                std::cout << "        CL_DEVICE_MAX_CLOCK_FREQUENCY: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_COMPUTE_UNITS , &uiBuf );
                std::cout << "        CL_DEVICE_MAX_COMPUTE_UNITS: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_CONSTANT_ARGS , &uiBuf );
                std::cout << "        CL_DEVICE_MAX_CONSTANT_ARGS: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE , &ulBuf );
                std::cout << "        CL_DEVICE_MAX_CONSTANT_BUFFER_SIZEe: " << ulBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_MEM_ALLOC_SIZE , &ulBuf );
                std::cout << "        CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << ulBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_PARAMETER_SIZE , &sizeBuf );
                std::cout << "        CL_DEVICE_MAX_PARAMETER_SIZE: " << sizeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_READ_IMAGE_ARGS , &uiBuf );
                std::cout << "        CL_DEVICE_MAX_READ_IMAGE_ARGS: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_SAMPLERS , &uiBuf );
                std::cout << "        CL_DEVICE_MAX_SAMPLERS: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_WORK_GROUP_SIZE , &sizeBuf );
                std::cout << "        CL_DEVICE_MAX_WORK_GROUP_SIZE: " << sizeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS , &sizeBuf );
                std::cout << "        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << sizeBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_WORK_ITEM_SIZES , &sizeBufArr );
                std::cout << "        CL_DEVICE_MAX_WORK_ITEM_SIZES: " << sizeBufArr[0] << " " << sizeBufArr[1] << " " << sizeBufArr[2]<< std::endl;
                devices[j].getInfo( CL_DEVICE_MAX_WRITE_IMAGE_ARGS , &uiBuf );
                std::cout << "        CL_DEVICE_MAX_WRITE_IMAGE_ARGS: " << uiBuf << std::endl;

                devices[j].getInfo( CL_DEVICE_MEM_BASE_ADDR_ALIGN , &uiBuf );
                std::cout << "        CL_DEVICE_MEM_BASE_ADDR_ALIGN: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE , &uiBuf );
                std::cout << "        CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: " << uiBuf << std::endl;


                devices[j].getInfo( CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR , &uiBuf );
                std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT , &uiBuf );
                std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT , &uiBuf );
                std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG , &uiBuf );
                std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT , &uiBuf );
                std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE , &uiBuf );
                std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: " << uiBuf << std::endl;

                devices[j].getInfo( CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR , &uiBuf );
                std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT , &uiBuf );
                std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_NATIVE_VECTOR_WIDTH_INT , &uiBuf );
                std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG , &uiBuf );
                std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT , &uiBuf );
                std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: " << uiBuf << std::endl;
                devices[j].getInfo( CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE , &uiBuf );
                std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: " << uiBuf << std::endl;

                devices[j].getInfo( CL_DEVICE_PROFILING_TIMER_RESOLUTION , &ulBuf );
                std::cout << "        CL_DEVICE_PROFILING_TIMER_RESOLUTION: " << ulBuf << std::endl;

                devices[j].getInfo( CL_DEVICE_QUEUE_PROPERTIES , &qpBuf );
                std::cout << "        CL_DEVICE_QUEUE_PROPERTIES: ";
                if( qpBuf & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
                    std::cout << "QUEUE_OUT_OF_ORDER_EXEC_MODE";
                if( qpBuf & CL_QUEUE_PROFILING_ENABLE )
                    std::cout << " QUEUE_PROFILING";
                std::cout << std::endl;


                devices[j].getInfo( CL_DEVICE_SINGLE_FP_CONFIG , &fpBuf );
                std::cout << "        CL_DEVICE_SINGLE_FP_CONFIG: " << fpBuf << std::endl;

                std::cout << std::endl;
            }

            std::cout << std::endl;
        }
    }
}
