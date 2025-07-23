#ifndef _WS_NAMEDOBJECT_H
#define _WS_NAMEDOBJECT_H

#include "LibUtilExportOs.h"
#include "BasicTypes.h"
#include <functional>
#include <iostream>
#include <string>
#include <typeinfo>

namespace WS
{
namespace Util
{
class LIBUTIL_EXPORT NamedObject
{
public:
    using VisitorFunctor = std::function<void(NamedObject*)>;

    explicit NamedObject(std::string const& name, std::string const& label) : mName{name}, mLabel{label} {}
    virtual ~NamedObject() = default;

    /// Retrieves/Sets the name as set in the ctor or by setName(). Derived classes
    /// baptise the object appropriately.
    inline std::string const& getName() const
    {
        return mName;
    }
    inline void setName(std::string const& name) { mName = name; }

    /// Retrieves the label text associated with this NamedObject. Derived classes
    /// baptise the object appropriately.
    inline std::string const& getLabel() const
    {
        return mLabel;
    }

    /// Retrieves the label text associated with this NamedObject. Derived classes
    /// baptise the object appropriately.
    inline bool canEdit() const
    {
        return !mLabel.empty();
    }

    /// Resets this NamedObject to its default value.
    virtual void resetToDefault() = 0;

    /// Returns the value pointer of the data held in derived classes.
    /// This is ugly, but to be Autosar compliant.
    virtual void setValuePtr(void const*) {}
    virtual void const* getValuePtr() const
    {
        return nullptr;
    }

    /// Returns the type_info associated with the value being
    /// monitored. If no value exist, returns typeid(*this).
    virtual std::type_info const& getTypeInfo() const = 0;

    /// Visitor interface. Each named object will call the visitor's
    /// "onNamedObject()" method for all objects in the hierarchy.
    virtual void visitObject(VisitorFunctor const& functor) = 0;

protected:
    NamedObject(NamedObject const&) = default;
    NamedObject(NamedObject&&) = default;
    NamedObject& operator=(NamedObject const&) = default;
    NamedObject& operator=(NamedObject&&) = default;

private:
    std::string mName;
    std::string mLabel;
};

} // namespace Util
} // namespace WS

#endif
