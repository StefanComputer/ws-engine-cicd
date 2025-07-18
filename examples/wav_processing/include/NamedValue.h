#ifndef _WS_NAMEDVALUE_H
#define _WS_NAMEDVALUE_H

#include "JsonReader.h"
#include "JsonWriter.h"
#include "NamedObject.h"
#include "ScriptException.h"
#include "StrManip.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>

namespace WS
{
namespace Util
{
template <class T>
class MinMaxValidator final
{
public:
    explicit MinMaxValidator(T const& minValue, T const& maxValue) : mMin{minValue}, mMax{maxValue} {}
    bool operator()(T const& value, T& min, T& max)
    {
        min = mMin;
        max = mMax;
        return (value >= mMin) && (value <= mMax);
    }

private:
    T mMin;
    T mMax;
};

template <class T>
class NamedValue final : public NamedObject
{
public:
    using Validator = std::function<bool(T const& value, T& min, T& max)>;

    NamedValue(std::string const& name, T const& defValue, std::string const& label = "", Validator const& validator = Validator{})
        : NamedObject{name, label}, mValue{defValue}, mDefValue{defValue}, mValidator{validator}
    {
    }
    ~NamedValue() final = default;
    NamedValue(NamedValue<T> const&) = default;
    NamedValue(NamedValue<T>&&) = default;
    NamedValue<T>& operator=(NamedValue<T> const&) = default;
    NamedValue<T>& operator=(NamedValue<T>&&) & = default;

    NamedValue<T>& operator=(T const& value) &
    {
        validate(value);
        mValue = value;
        return *this;
    }

    T const& operator*() const
    {
        return mValue;
    }

    // NamedObject overrides
    void resetToDefault() final
    {
        mValue = mDefValue;
    }
    void setValuePtr(void const* valuePtr) final
    {
        T const* const theVal{static_cast<T const*>(valuePtr)};
        validate(*theVal);
        mValue = *theVal;
    }
    void const* getValuePtr() const final
    {
        return &mValue;
    }
    std::type_info const& getTypeInfo() const final
    {
        /// This value type signature
        return typeid(T);
    }

    void visitObject(VisitorFunctor const& functor) final
    {
        functor(this);
    }

    /// Accessor
    T const& getValue()
    {
        return mValue;
    }

    void getMinMax(T& min, T& max)
    {
        if(mValidator != nullptr)
        {
            T value{};
            mValidator(value, min, max);
        }
    }

private:
    void validate(T const& value) const
    {
        T min{}, max{};
        if((mValidator != nullptr) && !mValidator(value, min, max))
        {
            throw ScriptException{ScriptException::ExcType::NamedValueOutOfRange, std::string{getName()}};
        }
    }

    // Data members
    T mValue;
    T mDefValue;
    Validator mValidator;
};

template <class T>
inline bool readNamedValueHelper(WS::Util::JsonReader& reader, WS::Util::NamedValue<T>& value)
{
    T val{};
    if(reader.readValuePair(value.getName(), val))
    {
        value = val;
        return true;
    }
    return false;
}

} // namespace Util
} // namespace WS

template <>
inline bool WS::Util::JsonReader::readNamedValue(WS::Util::NamedValue<bool>& value)
{
    return readNamedValueHelper<bool>(*this, value);
}

template <>
inline bool WS::Util::JsonReader::readNamedValue(WS::Util::NamedValue<s32>& value)
{
    return readNamedValueHelper<s32>(*this, value);
}

template <>
inline bool WS::Util::JsonReader::readNamedValue(WS::Util::NamedValue<u32>& value)
{
    return readNamedValueHelper<u32>(*this, value);
}

template <>
inline bool WS::Util::JsonReader::readNamedValue(WS::Util::NamedValue<std::string>& value)
{
    return readNamedValueHelper<std::string>(*this, value);
}

template <>
inline bool WS::Util::JsonReader::readNamedValue(WS::Util::NamedValue<float>& value)
{
    return readNamedValueHelper<float>(*this, value);
}

template <>
inline bool WS::Util::JsonReader::readNamedValue(WS::Util::NamedValue<double>& value)
{
    return readNamedValueHelper<double>(*this, value);
}

template <>
inline bool WS::Util::JsonReader::readNamedValue(WS::Util::NamedValue<s64>& value)
{
    return readNamedValueHelper<s64>(*this, value);
}

template <>
inline bool WS::Util::JsonReader::readNamedValue(WS::Util::NamedValue<u64>& value)
{
    return readNamedValueHelper<u64>(*this, value);
}

template <class T>
inline bool WS::Util::JsonWriter::writeNamedValue(T& namedValue)
{
    return writeValuePair(namedValue.getName(), namedValue.getValue());
}

#endif
