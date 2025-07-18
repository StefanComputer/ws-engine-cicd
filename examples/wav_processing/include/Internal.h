#ifndef _WS_INTERNAL_H_
#define _WS_INTERNAL_H_

#include "RapidWrapper.h"
#include <rapidjson/document.h>

namespace WS
{
namespace Util
{

struct RapidWrapper::Internal
{
    rapidjson::Document* mDocument;
    rapidjson::Value* mValue;
    rapidjson::Document::AllocatorType* mAllocator;

    Internal()
        : mDocument{new rapidjson::Document()}, mValue{nullptr}, mAllocator{nullptr}
    {
        mDocument->SetObject();
        mAllocator = &mDocument->GetAllocator();
    }

    Internal(std::string const& jsonContent)
        : mDocument{new rapidjson::Document()},
          mValue{nullptr},
          mAllocator{nullptr}
    {
        if(!jsonContent.empty())
        {
            mDocument->Parse(jsonContent.c_str());
            mAllocator = &mDocument->GetAllocator();
        }
        else
        {
            std::string msg{"Can not create wrapper with empty json content."};
            throw ScriptException{ScriptException::ExcType::JsonEmptyContent, std::move(msg)};
        }
    }

    Internal(rapidjson::Document* document)
        : mDocument{document},
          mValue{nullptr},
          mAllocator{nullptr}
    {
    }

    Internal(rapidjson::Value* value, rapidjson::Document::AllocatorType* allocator)
        : mDocument{nullptr},
          mValue{value},
          mAllocator{allocator}
    {
    }

    Internal(Internal& upper, std::string const& tag)
        : mDocument{nullptr},
          mValue{nullptr},
          mAllocator{upper.mAllocator}
    {
        if(upper.mDocument != nullptr)
        {
            mValue = &(*upper.mDocument)[tag.c_str()];
            return;
        }
        mValue = &(*upper.mValue)[tag.c_str()];
    }

    Internal(Internal& upper, std::string const& tag, int index)
        : mDocument{nullptr},
          mValue{nullptr},
          mAllocator{upper.mAllocator}
    {
        if(upper.mDocument != nullptr)
        {
            mValue = &(*upper.mDocument)[index][tag.c_str()];
            return;
        }
        mValue = &(*upper.mValue)[index][tag.c_str()];
    }

    Internal(Internal& upper, int index)
        : mDocument{nullptr},
          mValue{nullptr},
          mAllocator{upper.mAllocator}
    {
        if(upper.mDocument != nullptr)
        {
            mValue = &(*upper.mDocument)[index];
            return;
        }
        mValue = &(*upper.mValue)[index];
    }

    ~Internal()
    {
        // this also deletes mAllocator and all values from this document
        delete mDocument;
    }

    bool hasMember(std::string const& tag)
    {
        if(mDocument != nullptr)
        {
            return mDocument->HasMember(tag.c_str());
        }
        return mValue->HasMember(tag.c_str());
    }

    bool hasMember(std::string const& tag, int index)
    {
        if(mDocument != nullptr)
        {
            return (*mDocument)[index].HasMember(tag.c_str());
        }
        return (*mValue)[index].HasMember(tag.c_str());
    }

    /// Convert the data to a rapidjson::Value
    template <class T>
    rapidjson::Value toValue(T const& data)
    {
        return rapidjson::Value(data);
    }
    rapidjson::Value toValue(std::string const& data)
    {
        rapidjson::Value rjValue(rapidjson::StringRef(data.c_str()));
        return rjValue;
    }
};

} // namespace Util
} // namespace WS

#endif