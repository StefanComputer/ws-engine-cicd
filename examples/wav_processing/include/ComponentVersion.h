#pragma once

#include <string>
#include <fstream>

namespace WS
{
/// @brief  Class ComponentVersion is used to read the version of the application in the current repository
class ComponentVersion
{
public:
    // Returns the version of the software in this repo
    static std::string getVersion()
    {
        std::string version{"0.0"};
        std::ifstream ifs{"version.txt", std::ios_base::in};
        if(ifs.is_open())
        {
            std::getline(ifs, version);
        }
        return version;
    }
};
}