#pragma once
#ifndef __HANDLERS_H__
#define __HANDLERS_H__
#include "includes.h"

// Errors handler: -------------------------------------------------------------------------------------
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
// -----------------------------------------------------------------------------------------------------

#endif  // __HANDLERS_H__