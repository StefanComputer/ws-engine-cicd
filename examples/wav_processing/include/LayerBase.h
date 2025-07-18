#ifndef _WS_LAYERBASE_H
#define _WS_LAYERBASE_H

#include "LibAiExportOs.h"
#include "BasicTypes.h"
#include <string>

namespace WS
{

class LIB_AI_EXPORT LayerBase
{
public:
LayerBase(std::string const& name, u32 uniqueId)
        : mName(name), mId(uniqueId), mIsValid(false)
    {
    }
virtual ~LayerBase() = default;

//Accessors
inline std::string const& getName() const { return mName; }
inline u32 getId() const { return mId; }
inline bool isValid() const { return mIsValid; }
inline void setValid(bool valid) { mIsValid = valid; }

inline void clearValid()
{
    mIsValid = false;
}

private:
    /// @brief The name of this layer buffer.
    std::string mName;

    /// A unique identifier used to link the binary buffer file.
    u32 mId;

    /// Set if this is a valid buffer.
    bool mIsValid;
};
} // namespace WS

#endif
