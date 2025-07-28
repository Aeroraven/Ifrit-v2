#pragma once
#ifdef IF_REFLPARSER_PASS
    #ifndef __clang__
        #error "This module requires Clang to be used for parsing. Please ensure you are using Clang as your compiler."
    #endif
    #define IF_CLASS(...) __attribute__((annotate("ifrit.refl.class:" #__VA_ARGS__)))
    #define IF_PROPERTY(...) __attribute__((annotate("ifrit.refl.property:" #__VA_ARGS__)))
    #define IF_FUNCTION(...) __attribute__((annotate("ifrit.refl.function:" #__VA_ARGS__)))
#else
    #ifndef _MSC_VER
    // #error "Compiler not supported. Please use MSVC or Clang for this project."
    #endif

    #ifndef __INTELLISENSE__
        #define IF_CLASS(...)
        #define IF_PROPERTY(...)
        #define IF_FUNCTION(...)
    #else
        #define IF_CLASS(...) [[annotate("ifrit.refl.class:" #__VA_ARGS__)]]
        #define IF_PROPERTY(...) [[annotate("ifrit.refl.property:" #__VA_ARGS__)]]
        #define IF_FUNCTION(...) [[annotate("ifrit.refl.function:" #__VA_ARGS__)]]
    #endif
#endif
