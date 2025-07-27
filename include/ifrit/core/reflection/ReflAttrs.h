#pragma once
#ifdef IF_REFLPARSER_PASS
    #ifndef __clang__
        #error "This module requires Clang to be used for parsing. Please ensure you are using Clang as your compiler."
    #endif
    #define IF_CLASS() __attribute__((annotate("ifrit.refl.class")))
    #define IF_PROPERTY() __attribute__((annotate("ifrit.refl.property")))
    #define IF_FUNCTION() __attribute__((annotate("ifrit.refl.function")))
#else
    #define IF_CLASS()
    #define IF_PROPERTY()
    #define IF_FUNCTION()
#endif