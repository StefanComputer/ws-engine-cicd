#ifndef _WS_COMPONENTEXCEPTION_H
#define _WS_COMPONENTEXCEPTION_H
#include "Exception.h"
#include "StrFormat.h"
#include <exception>
namespace WS
{
namespace Util
{
/// Class to handle exceptions in Util and all components libraries.
/// This class to be used as the base class for all the Exception per components.
class ComponentException : public Exception
{
protected:
    ComponentException() = default;
    ComponentException(ComponentException&&) = default;
    ComponentException(ComponentException const& rhs) = default;
    explicit ComponentException(std::string&& compName, std::string&& msg) noexcept
        : Exception{std::move(msg)}, mCompName{std::move(compName)}
    {
    }
    ~ComponentException() override = default;
    ComponentException& operator=(ComponentException const&) & = default;
    ComponentException& operator=(ComponentException&&) & = default;
    // Returns the name of the component to which this exception belongs.
    std::string const& getCompName() const
    {
        return mCompName;
    }

private:
    std::string mCompName;
};
} // namespace Util
} // namespace WS
#endif