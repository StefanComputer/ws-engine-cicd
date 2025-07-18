#ifndef _WS_JSONREADER_H
#define _WS_JSONREADER_H

#include "LibUtilExportOs.h"
#include "RapidWrapper.h"
#include <functional>
#include <iostream>
#include <string>

namespace WS
{
namespace Util
{
class Scriptable;

class LIBUTIL_EXPORT JsonReader
{
protected:
    std::vector<RapidWrapper::RapidWrapperPair> mWrapperStack;

public:
    JsonReader(RapidWrapper* wrapper, std::string const& subValueTag = "")
    {
        mWrapperStack.push_back({wrapper, "TopOfStack"});
        if(!subValueTag.empty())
        {
            mWrapperStack.push_back({readWrapperObject(subValueTag), subValueTag});
        }
    }
    ~JsonReader();

    using ScriptFunc = std::function<bool(JsonReader&)>;

    template <class T>
    bool readValuePair(std::string const& tag, T& value)
    {
        std::unique_ptr<RapidWrapper> wrapper{readWrapperObject(tag)};
        if(!wrapper)
        {
            return false;
        }
        return wrapper->readValue(value);
    }

    template <class T>
    bool readNamedValue(T& value) = delete;

    /// Reads the next value in the wrapper at the back of the stack
    /// Throws ScriptException::ExcType::JsonNotAnArray if the ending wrapper is not an array
    /// Throws ScriptException::ExcType::JsonInvalidType if the type of
    /// the content in the array does not match tample class T
    template <class T>
    bool readValue(T& value)
    {
        std::unique_ptr<RapidWrapper> wrapper{readWrapperObject()};
        if(!wrapper)
        {
            return false;
        }
        return wrapper->readValue(value);
    }

    /// Fetch the wrapper with the array of the given tag and push to the stack.
    /// Then return the size of the array.
    u32 readArrayBegin(std::string const& tag);

    /// End the reading of the array.
    void readArrayEnd();

    /// Read-in an array of simple types.
    template <class T>
    bool readArray(std::string const& tag, std::vector<T>& vec)
    {
        u32 size{readArrayBegin(tag)};
        if(size == static_cast<u32>(-1))
        {
            return false;
        }
        for(u32 i = 0; i < size; ++i)
        {
            T val{};
            if(readValue(val))
            {
                vec.push_back(val);
            }
        }
        readArrayEnd();
        return size != 0;
    }

    /// Read-in an array of json object, providing the caller gives
    /// a valid scriptToReadObject in order to interpret the "json object" correctly.
    /// When reading an array of object, the vector "vec" is lceared() and resize() to the size of the
    /// Json array, as specified in Json formatted file..
    template <class T>
    bool readArrayOfObjects(std::string const& tag, std::vector<T>& vec, std::function<void(JsonReader&, T&)> const& scriptToReadObject)
    {
        u32 size{readArrayBegin(tag)};
        if(size == static_cast<u32>(-1))
        {
            return false;
        }
        vec.clear();
        vec.resize(size);
        for(auto& it : vec)
        {
            if(readObjectBegin(""))
            {
                // Call the method to continue parsing
                scriptToReadObject(*this, it);
                readObjectEnd();
            }
        }
        readArrayEnd();
        return size != 0;
    }

    /// Fetch the wrapper of the object of the given tag and push to the stack.
    /// End the reading of the array.
    /// Pop all stacks till the array with the given tag is poped
    bool readObjectBegin(std::string const& tag);
    void readObjectEnd();
    bool readObject(std::string const& tag, ScriptFunc const& script);
    bool readObject(std::string const& tag, Scriptable& script);
    void setSilentException(bool silent) { mSilentExceptions = silent; }
private:
    void popBack();

    /// calls wrapper.readObject(tag) and handle exceptions
    RapidWrapper* readWrapperObject(std::string const& tag = "");
    bool mSilentExceptions = false;
};

} // namespace Util
} // namespace WS

#endif