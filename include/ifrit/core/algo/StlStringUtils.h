
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/core/base/IfritBase.h"

namespace Ifrit
{

    Vec<String> SplitString(const String& str, const String& delimiter)
    {
        Vec<String> result;
        size_t      start = 0;
        size_t      end   = str.find(delimiter);
        while (end != String::npos)
        {
            result.push_back(str.substr(start, end - start));
            start = end + delimiter.length();
            end   = str.find(delimiter, start);
        }
        result.push_back(str.substr(start, end));
        return result;
    }

    String JoinString(const Vec<String>& strings, const String& delimiter)
    {
        String result;
        for (size_t i = 0; i < strings.size(); ++i)
        {
            result += strings[i];
            if (i < strings.size() - 1)
            {
                result += delimiter;
            }
        }
        return result;
    }
} // namespace Ifrit