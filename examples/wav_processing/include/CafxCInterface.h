#ifndef _WS_CAFXCINTERFACE_H
#define _WS_CAFXCINTERFACE_H

#include "LibAiExportOs.h"
#include <stddef.h>
/// Return values / error codes
#define TER_NO_ERROR (0)
#define TER_INVALID_INSTANCE (1)
#define TER_INVALID_PARAMETER (2)
typedef void WSLogger;

/// C interface
/// Borrowing the pointer from the user given logger.
/// The user is responsible to delete his logger. We guarantee not using the logger after the last "ProcessAudioModel" call
extern "C" LIB_AI_EXPORT int __stdcall InitializeAudioModel(WSLogger* wsLoggerInstance);
extern "C" LIB_AI_EXPORT int __stdcall GetValidationBuffer(char const* bufferName, void** values, size_t* fcnt, size_t* scnt);
extern "C" LIB_AI_EXPORT int __stdcall DestroyAudioModel();
extern "C" LIB_AI_EXPORT int __stdcall PrepareAudioModel(const char* h5FileName);
extern "C" LIB_AI_EXPORT int __stdcall ProcessAudioModel(const float* inputBuffer, float* output);
extern "C" LIB_AI_EXPORT int __stdcall setParamValueAt(const char* paramName, double value);
extern "C" LIB_AI_EXPORT int __stdcall setNewParam(const char* paramName);
extern "C" LIB_AI_EXPORT int __stdcall getNumberOfParams();

#endif
