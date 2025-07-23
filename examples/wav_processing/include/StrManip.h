#ifndef _WS_STRMANIP_H
#define _WS_STRMANIP_H

#include "BasicTypes.h"
#include "LibUtilExportOs.h"

#include <locale>
#include <string>
#include <vector>

namespace WS
{
namespace Util
{
class LIBUTIL_EXPORT StrManip final
{
public:
    using StringVec = std::vector<std::string>;

    static inline void toLower(std::string& str)
    {
        for(std::string::value_type& c : str)
        {
            c = std::tolower(c, std::locale());
        }
    }

    static inline void toUpper(std::string& str)
    {
        for(std::string::value_type& c : str)
        {
            c = std::toupper(c, std::locale());
        }
    }

    static inline void trim(std::string& str)
    {
        size_t const first{str.find_first_not_of(' ')};
        if(std::string::npos == first)
        {
            str = {};
            return;
        }
        size_t const last{str.find_last_not_of(' ')};
        // we know first >= last so newSize will be > 0
        size_t const newSize{static_cast<size_t>(last - first + 1U)};
        str = str.substr(first, newSize);
    }

    static inline bool endsWith(std::string const& source, std::string const& ending)
    {
        if(ending.size() > source.size())
        {
            return false;
        }
        return std::equal(ending.rbegin(), ending.rend(), source.rbegin());
    }

    static StringVec splitString(std::string const& str, std::string const& separatorList, bool const skipEmpty);

    /// Simple function that finds all occurence of oldStr within the string and
    /// replaces it with the newStr.
    static std::string findAndReplaceOnce(std::string const& str, std::string const& oldStr, std::string const& newStr);

    /// Counterpart method of splitString - given "strings" in a vector, this method will
    /// concatenate all of them and seperate them with "delim". No additional space are inserted.
    static std::string concatString(StringVec const& strings, std::string const& delim);

    /// Parses the given str and replaces ALL the spaces by "_", into new string returned.
    static std::string replaceSpacesByUnderscores(std::string const& str);

    /// Perform case insensitive comparison.
    static bool caseInsensitiveCompare(std::string const& str1, std::string const& str2);

    /// Converts hexadecimal numbers into appropriate chosen type.
    static u32 hexaToU32(std::string const& str);
    static ushort hexaToU16(std::string const& str);
    static u8 hexaToU8(std::string const& str);
    static u64 hexaToU64(std::string const& str);

    /// Converts the given string "in place" and replaces any instances
    /// of environment variables with %env_var% to its actual value.
    static void substituteEnvironmentVariables(std::string& pathToFix);

}; // namespace StrManip
} // namespace Util
} // namespace WS

#endif // _IRYS_STR_MANIP_H__
