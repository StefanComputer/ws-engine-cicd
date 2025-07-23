#ifndef _WS_SCRIPTABLE_H
#define _WS_SCRIPTABLE_H

#include "LibUtilExportOs.h"

namespace WS
{
namespace Util
{
class JsonReader;
class JsonWriter;

class LIBUTIL_EXPORT Scriptable
{
public:
    Scriptable() = default;
    virtual ~Scriptable() = default;
    virtual bool readFromStream(JsonReader&) = 0;
    virtual bool writeToStream(JsonWriter&) = 0;
};

} // namespace Util
} // namespace WS

#endif