#ifndef _WS_NAMEDGROUP_H
#define _WS_NAMEDGROUP_H

#include "JsonWriter.h"
#include "NamedValue.h"

#include <functional>
#include <vector>

namespace WS
{
namespace Util
{
class LIBUTIL_EXPORT NamedGroup final : public NamedObject
{
    using ValueGroupCont = std::vector<NamedObject*>;
    using GroupGroupCont = std::vector<NamedGroup*>;
    using MemberCont = std::vector<NamedObject*>;

public:
    class VisitorHelper
    {
    public:
        template <class VISITOR>
        void dispatchVisitor(WS::Util::NamedObject* obj, VISITOR& visitor);
    };

    explicit NamedGroup(std::string const& name, std::string const& label = "");
    explicit NamedGroup(std::string const& name, std::string const& label, std::initializer_list<NamedGroup*> const& groupInitList);
    ~NamedGroup() final = default;
    NamedGroup(NamedGroup const&) = default;
    NamedGroup(NamedGroup&&) = default;
    NamedGroup& operator=(NamedGroup const&) = default;
    NamedGroup& operator=(NamedGroup&&) = default;

    /// Add group members, using instance. The object is only referenced
    /// and this group does not own its instance.
    bool addValue(NamedObject* const value);
    bool addGroup(NamedGroup* const grp);
    MemberCont::size_type getMemberCnt() const
    {
        return mMembers.size();
    }

    /// Clear all members in the group
    void clear();

    /// Returns a generic base class pointer using its name, const/non-const versions.
    NamedObject* getValueMember(std::string const& name);
    NamedObject const* getValueMember(std::string const& name) const;

    /// Returns an NamedGroup using its name, const/non-const versions.
    NamedGroup* getGroupMember(std::string const& name);
    NamedGroup const* getGroupMember(std::string const& name) const;

    /// Set a given value based on its name. Returns false if the name is not
    /// found in the values member container. Proper type must be known
    /// for a specific value name.
    // PRQA S 2427 150 #Direct use of fundamental float is not part of Autosar.
    template <class T>
    bool setValue(std::string const& name, T const& value)
    {
        NamedObject* const m{getValueMember(name)};
        if(m != nullptr)
        {
            m->setValuePtr(&value);
            return true;
        }
        return false;
    }

    /// Returns the value using a name. Returns false if the
    /// name is not found in the values container.
    /// Proper type must be given for a specific name.
    template <class T>
    bool getValue(std::string const& name, T& value) const
    {
        NamedObject const* const m{getValueMember(name)};
        if(m != nullptr)
        {
            T const* const val{static_cast<T const*>(m->getValuePtr())};
            value = *val;
            return true;
        }
        else
        {
            value = T{};
        }
        return false;
    }

    /// NamedObject overrides
    void resetToDefault() final
    {
        for(NamedObject* const m : mMembers)
        {
            m->resetToDefault();
        }
    }

    std::type_info const& getTypeInfo() const final
    {
        /// This NamedGroup signature
        return typeid(*this);
    }

    void visitObject(VisitorFunctor const& functor) final
    {
        functor(this);
        for(NamedObject* m : mMembers)
        {
            m->visitObject(functor);
        }
    }

    /// Accessors
    inline bool containsSameNames() const
    {
        return mContainsSameNames;
    }

private:
    /// Other groups within this group
    GroupGroupCont mGroups;

    /// Values within this group
    ValueGroupCont mValues;

    // Overall list of every polymorphic types
    MemberCont mMembers;

    /// Tells if this group contains members that have same names.
    bool mContainsSameNames;
};

template <class VISITOR>
void NamedGroup::VisitorHelper::dispatchVisitor(WS::Util::NamedObject* obj, VISITOR& visitor)
{
    std::type_info const& tid{obj->getTypeInfo()};
    if(tid == typeid(bool))
    {
        visitor.onNamedObject(*static_cast<WS::Util::NamedValue<bool>*>(obj));
    }
    else if(tid == typeid(u32))
    {
        visitor.onNamedObject(*static_cast<WS::Util::NamedValue<u32>*>(obj));
    }
    else if(tid == typeid(s32))
    {
        visitor.onNamedObject(*static_cast<WS::Util::NamedValue<s32>*>(obj));
    }
    else if(tid == typeid(float))
    {
        visitor.onNamedObject(*static_cast<WS::Util::NamedValue<float>*>(obj));
    }
    else if(tid == typeid(std::string))
    {
        visitor.onNamedObject(*static_cast<WS::Util::NamedValue<std::string>*>(obj));
    }
    else if(tid == typeid(double))
    {
        visitor.onNamedObject(*static_cast<WS::Util::NamedValue<double>*>(obj));
    }
    else if(tid == typeid(u64))
    {
        visitor.onNamedObject(*static_cast<WS::Util::NamedValue<u64>*>(obj));
    }
    else if(tid == typeid(s64))
    {
        visitor.onNamedObject(*static_cast<WS::Util::NamedValue<s64>*>(obj));
    }
    else if(tid == typeid(WS::Util::NamedGroup))
    {
        visitor.onNamedObject(*static_cast<WS::Util::NamedGroup*>(obj));
    }
}

} // namespace Util
} // namespace WS

template <>
inline bool WS::Util::JsonReader::readNamedValue(WS::Util::NamedGroup&)
{
    std::string name;
    return readValuePair("groupName", name);
}

template <>
inline bool WS::Util::JsonWriter::writeNamedValue(WS::Util::NamedGroup& grp)
{
    return writeValuePair("groupName", grp.getName());
}

#endif
