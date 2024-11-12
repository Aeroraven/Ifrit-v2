#extension GL_EXT_nonuniform_qualifier : enable

#define IFRIT_BINDLESS_BINDING_UNIFORM 0
#define IFRIT_BINDLESS_BINDING_STORAGE 1
#define IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER 2
#define IFRIT_BINDLESS_SET_ID 0

#define _ifrit_bindlessNaming(name) u##name##_bindless
#define _ifrit_bindlessType(name) u##name##_bindless_type

#define RegisterUniform(name, type) layout(binding = IFRIT_BINDLESS_BINDING_UNIFORM, set = IFRIT_BINDLESS_SET_ID) \
    uniform _ifrit_bindlessType(name) type _ifrit_bindlessNaming(name)[]

#define RegisterStorage(name, type) layout(binding = IFRIT_BINDLESS_BINDING_STORAGE, set = IFRIT_BINDLESS_SET_ID) \
    buffer _ifrit_bindlessType(name) type _ifrit_bindlessNaming(name)[]

#define IFRIT_BINDLESS_SAMPLER2D_NAME _ifrit_bindlessNaming(cbsampler_2d)
layout(binding = IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER, set = IFRIT_BINDLESS_SET_ID) uniform sampler2D IFRIT_BINDLESS_SAMPLER2D_NAME[];

#define IFRIT_BINDLESS_SAMPLER2DU_NAME _ifrit_bindlessNaming(cbsampler_2du)
layout(binding = IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER, set = IFRIT_BINDLESS_SET_ID) uniform usampler2D IFRIT_BINDLESS_SAMPLER2DU_NAME[];


#define GetResource(name,id) _ifrit_bindlessNaming(name)[(id)]
#define GetSampler2D(id) IFRIT_BINDLESS_SAMPLER2D_NAME[(id)]
#define GetSampler2DU(id) IFRIT_BINDLESS_SAMPLER2DU_NAME[(id)]