
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

    struct OrderedPair3
    {
        u32 a, b, c;
        u32 ra, rb, rc;

        OrderedPair3(u32 wa, u32 wb, u32 wc)
        {
            Vec<u32> px = { wa, wb, wc };
            std::sort(px.begin(), px.end());
            a  = px[0];
            b  = px[1];
            c  = px[2];
            ra = wa;
            rb = wb;
            rc = wc;
        }

        bool operator==(const OrderedPair3& other) const { return a == other.a && b == other.b && c == other.c; }
    };

} // namespace Ifrit

namespace std
{
    template <> struct hash<Ifrit::OrderedPair3>
    {
        size_t operator()(const Ifrit::OrderedPair3& pair) const
        {
            return hash<Ifrit::u32>()(pair.a) ^ hash<Ifrit::u32>()(pair.b) ^ hash<Ifrit::u32>()(pair.c);
        }
    };
} // namespace std