#include "engine/shadervm/spirv/SpvVMInterpreter.h"
#include "./dependency/spirv.h"

namespace Ifrit::Engine::ShaderVM::Spirv::Impl {
	enum SpvVMAnalysisPass {
		IFSP_VMA_PASS_FIRST,
		IFSP_VMA_PASS_SECOND,
	};

	typedef void (*SpvInstFunc)(SpvVMContext* spvContext, SpvVMIntermediateReprBlock* irContextLocal,
		SpvVMIntermediateRepresentation* irContextGlobal,int opWordCount, uint32_t* params, int instLine,
		SpvVMAnalysisPass pass);

#define DEFINE_OP(name) void name(SpvVMContext* spvContext, SpvVMIntermediateReprBlock* irContextLocal, \
	SpvVMIntermediateRepresentation* irContextGlobal,int opWordCount, uint32_t* params, int instLine, SpvVMAnalysisPass pass)
#define ERROR_PREFIX printf("[Line %d, Pass %d] ", irContextGlobal->currentInst,irContextGlobal->currentPass);
#define UNIMPLEMENTED_OP(name){ ERROR_PREFIX printf("Unimplemented SPIR-V instruction: "#name); printf("\n");}
#define DEPRECATED_OP(name){ ERROR_PREFIX printf("Deprecated SPIR-V instruction: "#name); printf("\n");}

	namespace InstructionImpl {
		/* Helper */
		std::string getTargetTypes(SpvVMIntermediateReprExpTarget* target, SpvVMIntermediateRepresentation* irContextGlobal);
		void registerTypes(SpvVMIntermediateRepresentation* irContext) {
			/*
			irContext->generatedIR << "declare <4 x float> @ifspvm_func_imageSampleImplicitLod(i8*,<2 x float>)\n";
			irContext->generatedIR << "declare i8* @ifspvm_func_sampledImage(i8*,i8*)\n";
			irContext->generatedIR << "declare float @ifspvm_func_callExtHandlerF1(i32,...)\n";
			irContext->generatedIR << "declare i32 @ifspvm_func_callExtHandlerI1(i32,...)\n";
			irContext->generatedIR << "declare <2 x i32> @ifspvm_func_callExtHandlerI2(i32,...)\n";
			irContext->generatedIR << "declare <3 x i32> @ifspvm_func_callExtHandlerI3(i32,...)\n";
			irContext->generatedIR << "declare <4 x i32> @ifspvm_func_callExtHandlerI4(i32,...)\n";
			irContext->generatedIR << "declare <2 x float> @ifspvm_func_callExtHandlerF2(i32,...)\n";
			irContext->generatedIR << "declare <3 x float> @ifspvm_func_callExtHandlerF3(i32,...)\n";
			irContext->generatedIR << "declare <4 x float> @ifspvm_func_callExtHandlerF4(i32,...)\n";
			irContext->generatedIR << "\n";*/

		}
		std::string getStructName(SpvVMIntermediateReprExpTarget* target, SpvVMIntermediateRepresentation* irContextGlobal) {
			return "%ifspvm_struct." + std::to_string(target->resultTypeRef);
		}
		
		std::string getVariableNamePrefix(SpvVMIntermediateReprExpTarget* target, SpvVMIntermediateRepresentation* irContextGlobal) {
			if (target->isVariable) {
				if (target->isGlobal) {
					if (target->named) {
						return "@ifspvm_global_" + target->name + "_" + std::to_string(target->id);
					}
					return "@ifspvm_global_" + target->name + "_" + std::to_string(target->id);
				}
				else {
					if (target->named) {
						return "%ifspvm_var_" + target->name + "_" + std::to_string(target->id);
					}
					return "%ifspvm_var_" + std::to_string(target->id);
				}
			}
			else if (target->isUniform) {
				return "@ifspvm_uniform_" + std::to_string(target->id);;
			}
			else if (target->isFunction) {
				return "@ifspvm_func_" + std::to_string(target->id);;
			}
			else if (target->isConstant) {
				auto tpTarget = &irContextGlobal->targets[target->componentTypeRef];
				if (tpTarget->declType == IFSP_IRTARGET_DECL_INT) {
					return std::to_string(target->data.intValue);
				}
				else if (tpTarget->declType == IFSP_IRTARGET_DECL_FLOAT) {
					return std::to_string(target->data.floatValue);
				}
				else if (tpTarget->declType == IFSP_IRTARGET_DECL_VECTOR) {
					std::string ret = "<";
					for (int i = 0; i < target->compositeDataRef.size(); i++) {
						auto tp = getTargetTypes(&irContextGlobal->targets[target->compositeDataRef[i]], irContextGlobal);
						auto name = getVariableNamePrefix(&irContextGlobal->targets[target->compositeDataRef[i]], irContextGlobal);
						if (i > 0) {
							ret += " , ";
						}
						ret += tp + " " + name;
					}
					ret += ">";
					return ret;
				}
				else {
					ERROR_PREFIX
					printf("Unknown constant type\n");
				}
			}
			else if (target->isLabel) {
				return "ifspvm_label_" + std::to_string(target->id);
			}
			else if (target->isInstance) {
				if (target->named) {
					return "%ifspvm_temp_" + target->name + "_" + std::to_string(target->id);
				}
				return "%ifspvm_temp_" + std::to_string(target->id);
			}
			else {
				ERROR_PREFIX
				printf("Unknown variable name\n");
				return "{error var}";
			}
		}
		std::string getTargetTypes(SpvVMIntermediateReprExpTarget* target, SpvVMIntermediateRepresentation* irContextGlobal) {
			if (target->isLabel) return "label";
			if (target->isInstance)return getTargetTypes(&irContextGlobal->targets[target->resultTypeRef], irContextGlobal);
			if (target->isConstant)return getTargetTypes(&irContextGlobal->targets[target->componentTypeRef], irContextGlobal);

			if (target->declType == IFSP_IRTARGET_DECL_BOOL) return "i1";
			else if (target->declType == IFSP_IRTARGET_DECL_INT) {
				if (target->intWidth == 32) {
					if (target->intSignedness == 0) return "i32";
					else return "i32";
				}
				else if (target->intWidth == 64) {
					if (target->intSignedness == 0) return "i64";
					else return "i64";
				}
			}
			else if (target->declType == IFSP_IRTARGET_DECL_FLOAT) {
				if (target->floatWidth == 32) return "float";
				else if (target->floatWidth == 64) return "f64";
			}
			else if (target->declType == IFSP_IRTARGET_DECL_VECTOR) {
				auto componentType = getTargetTypes(&irContextGlobal->targets[target->componentTypeRef], irContextGlobal);
				return "<" + std::to_string(target->componentCount) + " x " + componentType  + ">";
			}
			else if (target->declType == IFSP_IRTARGET_DECL_ARRAY) {
				auto componentType = getTargetTypes(&irContextGlobal->targets[target->componentTypeRef], irContextGlobal);
				return "[" + std::to_string(target->componentCount) + " x " + componentType + "]";
			}
			else if (target->declType == IFSP_IRTARGET_DECL_MATRIX) {
				auto columnTypeRef = irContextGlobal->targets[target->componentTypeRef];
				auto columnCount = target->componentCount;
				auto rowCount = irContextGlobal->targets[target->componentTypeRef].componentCount;
				auto elementType = getTargetTypes(&irContextGlobal->targets[columnTypeRef.componentTypeRef], irContextGlobal);
				auto totalCount = columnCount * rowCount;
				return "<" + std::to_string(totalCount) + " x " + elementType + ">";
			}
			else if (target->declType == IFSP_IRTARGET_DECL_VOID) {
				return "void";
			}
			else if (target->declType == IFSP_IRTARGET_DECL_STRUCT) {
				return getStructName(target, irContextGlobal);
			}
			else if (target->declType == IFSP_IRTARGET_DECL_POINTER) {
				auto baseType = getTargetTypes(&irContextGlobal->targets[target->componentTypeRef], irContextGlobal);
				return baseType;
			}
			else if (target->declType == IFSP_IRTARGET_DECL_IMAGE) {
				//Treat as pointer
				return "i8*";
			}
			else if (target->declType == IFSP_IRTARGET_DECL_SAMPLED_IMAGE) {
				//Treat as pointer
				return "i8*";
			}
			else if (target->declType == IFSP_IRTARGET_DECL_SAMPLER) {
				//Treat as pointer
				return "i8*";
			}
			else {
				ERROR_PREFIX
				printf("Unknown type: Target=%d\n",target->id);
				return "{error type}";
			}
		}
		int getVariableSize(SpvVMIntermediateReprExpTarget* target, SpvVMIntermediateRepresentation* irContextGlobal) {
			if (target->isInstance)return getVariableSize(&irContextGlobal->targets[target->resultTypeRef], irContextGlobal);
			if (target->declType == IFSP_IRTARGET_DECL_POINTER) {
				return getVariableSize(&irContextGlobal->targets[target->componentTypeRef], irContextGlobal);
			}
			if (target->declType == IFSP_IRTARGET_DECL_BOOL) return 1;
			else if (target->declType == IFSP_IRTARGET_DECL_INT) {
				if (target->intWidth == 32) {
					if (target->intSignedness == 0) return 4;
					else return 4;
				}
				else if (target->intWidth == 64) {
					if (target->intSignedness == 0) return 8;
					else return 8;
				}
			}
			else if (target->declType == IFSP_IRTARGET_DECL_FLOAT) {
				if (target->floatWidth == 32) return 4;
				else if (target->floatWidth == 64) return 8;
			}
			else if (target->declType == IFSP_IRTARGET_DECL_VECTOR) {
				auto componentType = getVariableSize(&irContextGlobal->targets[target->componentTypeRef], irContextGlobal);
				return target->componentCount * componentType;
			}
			else if (target->declType == IFSP_IRTARGET_DECL_ARRAY) {
				auto componentType = getVariableSize(&irContextGlobal->targets[target->componentTypeRef], irContextGlobal);
				return target->componentCount * componentType;
			}
			else if (target->declType == IFSP_IRTARGET_DECL_MATRIX) {
				auto columnTypeRef = irContextGlobal->targets[target->componentTypeRef];
				auto columnCount = target->componentCount;
				auto rowCount = irContextGlobal->targets[target->componentTypeRef].componentCount;
				auto elementType = getVariableSize(&irContextGlobal->targets[columnTypeRef.componentTypeRef], irContextGlobal);
				auto totalCount = columnCount * rowCount;
				return totalCount * elementType;
			}
			else if (target->declType == IFSP_IRTARGET_DECL_STRUCT) {
				auto ret = 0;
				for (int i = 0; i < target->memberTypeRef.size(); i++) {
					ret += getVariableSize(&irContextGlobal->targets[target->memberTypeRef[i]], irContextGlobal);
				}
				return ret;
			}
			else {
				ERROR_PREFIX
				printf("Unknown type: Target=%d\n", target->id);
				return 0;
			}

		}
		void elementwiseArithmeticIRGeneratorRaw(std::string instName, SpvVMIntermediateRepresentation* irContextGlobal,
			std::string resultName, std::string resultType, std::string opLName, std::string opRName) {
			irContextGlobal->generatedIR << resultName << " = " << instName << " " << resultType << " ";
			irContextGlobal->generatedIR << opLName << ", " << opRName << std::endl;
		}
		void elementwiseArithmeticIRGenerator(std::string instName, SpvVMIntermediateRepresentation* irContextGlobal,
			SpvVMIntermediateReprExpTarget* opL, SpvVMIntermediateReprExpTarget* opR, SpvVMIntermediateReprExpTarget* resultNode) {
			auto opLTypeRef = (opL->isConstant) ? opL->componentTypeRef : opL->resultTypeRef;
			auto opRTypeRef = (opR->isConstant) ? opR->componentTypeRef : opR->resultTypeRef;
			auto opLIsVector = irContextGlobal->targets[opLTypeRef].declType == IFSP_IRTARGET_DECL_VECTOR;
			auto opRIsVector = irContextGlobal->targets[opRTypeRef].declType == IFSP_IRTARGET_DECL_VECTOR;
			auto opLBaseType = getTargetTypes(&irContextGlobal->targets[opLTypeRef], irContextGlobal);
			auto opRBaseType = getTargetTypes(&irContextGlobal->targets[opRTypeRef], irContextGlobal);

			auto opLElementType = opLBaseType;
			auto opRElementType = opRBaseType;
			if (opLIsVector) {
				opLElementType = getTargetTypes(&irContextGlobal->targets[irContextGlobal->targets[opLTypeRef].componentTypeRef], irContextGlobal);
			}
			if (opRIsVector) {
				opRElementType = getTargetTypes(&irContextGlobal->targets[irContextGlobal->targets[opRTypeRef].componentTypeRef], irContextGlobal);
			}

			auto resultTypeRef = resultNode->resultTypeRef;
			auto resultType = getTargetTypes(&irContextGlobal->targets[resultTypeRef], irContextGlobal);
			auto resultName = getVariableNamePrefix(resultNode, irContextGlobal);

			// Generate IR
			if (opRElementType != opLElementType) {
				irContextGlobal->generatedIR << "{invalid operands}" << std::endl;
				ERROR_PREFIX
				printf("Invalid vector operation for different base types: %s %s\n", opLElementType.c_str(), opRElementType.c_str());
				return;
			}
			if (opLIsVector != opRIsVector || opLBaseType != opRBaseType) {
				irContextGlobal->generatedIR << "{invalid operands}" << std::endl;
				ERROR_PREFIX
				printf("Invalid vector operation for scalar-vector ops\n");
				return;
			}
			auto opLName = getVariableNamePrefix(opL, irContextGlobal);
			auto opRName = getVariableNamePrefix(opR, irContextGlobal);
			elementwiseArithmeticIRGeneratorRaw(instName, irContextGlobal, resultName, resultType, opLName, opRName);
		}
		int getAccessChainOffset(SpvVMIntermediateRepresentation* irContextGlobal,SpvVMIntermediateReprExpTarget* target) {
			auto ref = target->accessChainRef;
			int curOffset = 0;
			for (int i = 0; i < target->accessChain.size(); i++) {
				auto offset = target->accessChain[i];
				auto curTarget = irContextGlobal->targets[ref];
				if (curTarget.declType == IFSP_IRTARGET_DECL_POINTER) {
					curTarget = irContextGlobal->targets[curTarget.componentTypeRef];
				}
				if (curTarget.declType == IFSP_IRTARGET_DECL_STRUCT) {
					curOffset += curTarget.memberOffset[offset];
					ref = curTarget.memberTypeRef[offset];
				}
				else if (curTarget.declType == IFSP_IRTARGET_DECL_VECTOR) {
					auto baseType = irContextGlobal->targets[curTarget.componentTypeRef];
					if(baseType.declType == IFSP_IRTARGET_DECL_FLOAT) {
						curOffset += offset * baseType.floatWidth / 8;
					}
					else if(baseType.declType == IFSP_IRTARGET_DECL_INT) {
						curOffset += offset * baseType.intWidth / 8;
					}
				}
				else {
					ERROR_PREFIX
					printf("Invalid access chain\n");
				}
				
			}
			return curOffset;
		}

		/* 3.52.1. Miscellaneous Instructions */
		DEFINE_OP(spvOpNop) {
			
		}
		DEFINE_OP(spvOpUndef) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].exprType = IFSP_IRTARGET_INTERMEDIATE_UNDEF;
			irContextGlobal->targets[targetId].id = targetId;
		}

		/* 3.52.2. Debug Instructions */
		DEFINE_OP(spvOpSourceContinued) {
			UNIMPLEMENTED_OP(OpSourceContinued) 
		}
		DEFINE_OP(spvOpSource) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			irContextGlobal->attributes[IFSP_IR_SOURCE_TYPE] = params[0];
		}
		DEFINE_OP(spvOpSourceExtension) { 
			UNIMPLEMENTED_OP(OpSourceExtension) 
		}
		DEFINE_OP(spvOpName) { 
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto name = reinterpret_cast<const char*>(params + 1);
			irContextGlobal->targets[targetId].name = name;
			irContextGlobal->targets[targetId].name.erase(std::remove_if(
				irContextGlobal->targets[targetId].name.begin(), 
				irContextGlobal->targets[targetId].name.end(), 
				[](char p) {return p == '@'; }), irContextGlobal->targets[targetId].name.end());
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].id = targetId;
			irContextGlobal->targets[targetId].named = true;
		}
		DEFINE_OP(spvOpMemberName) { 
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto memberIdx = params[1];
			auto name = reinterpret_cast<const char*>(params + 2);
			if(irContextGlobal->targets[targetId].memberName.size() <= memberIdx) {
				irContextGlobal->targets[targetId].memberName.resize(memberIdx + 1);
			}
			irContextGlobal->targets[targetId].memberName[memberIdx] = name;
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpString) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto name = reinterpret_cast<const char*>(params + 1);
			irContextGlobal->targets[targetId].debugString = name;
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].exprType = IFSP_IRTARGET_STRING;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpLine) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			UNIMPLEMENTED_OP(OpLine)
		}
		DEFINE_OP(spvOpNoLine) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			UNIMPLEMENTED_OP(OpNoLine) 
		}
		DEFINE_OP(spvOpModuleProcessed) { 
			if (pass == IFSP_VMA_PASS_SECOND) return;
			UNIMPLEMENTED_OP(OpModuleProcessed)
		}

		/* 3.52.3. Annotation Instructions */
		DEFINE_OP(spvOpDecorate) { 
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto decoration = params[1];
			if (decoration == spv::Decoration::DecorationLocation) {
				irContextGlobal->targets[targetId].location = params[2];
			}
			if (decoration == spv::Decoration::DecorationDescriptorSet) {
				irContextGlobal->targets[targetId].descSet = params[2];
			}
			if (decoration == spv::Decoration::DecorationBinding) {
				irContextGlobal->targets[targetId].binding = params[2];
			}
			if (decoration == spv::Decoration::DecorationBuiltIn) {
				if (params[2] == spv::BuiltIn::BuiltInPosition) {
					irContextGlobal->targets[targetId].isBuiltinPos = true;
				}
			}
			irContextGlobal->targets[targetId].decoration = decoration;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpMemberDecorate) { 
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto memberIdx = params[1];
			auto decoration = params[2];
			if(irContextGlobal->targets[targetId].memberDecoration.size() <= memberIdx) {
				irContextGlobal->targets[targetId].memberDecoration.resize(memberIdx + 1);
				irContextGlobal->targets[targetId].memberOffset.resize(memberIdx + 1);
			}
			if (decoration == spv::Decoration::DecorationOffset) {
				irContextGlobal->targets[targetId].memberOffset[memberIdx] = params[3];
			}
			else {
				irContextGlobal->targets[targetId].memberDecoration[memberIdx] = decoration;
			}
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpGroupDecorate) { DEPRECATED_OP(OpGroupDecorate) }
		DEFINE_OP(spvOpGroupMemberDecorate) { DEPRECATED_OP(OpGroupMemberDecorate) }
		DEFINE_OP(spvOpDecorationGroup) { DEPRECATED_OP(OpDecorationGroup) }

		/* 3.52.4. Extension Instructions */
		DEFINE_OP(spvOpExtension) { UNIMPLEMENTED_OP(OpExtension) }
		DEFINE_OP(spvOpExtInstImport) { UNIMPLEMENTED_OP(OpExtInstImport) }
		DEFINE_OP(spvOpExtInst) { 
			ERROR_PREFIX;
			printf("External calls not supported\n");
			return;

			auto retType = params[0];
			auto resultId = params[1];
			auto extSet = params[2];
			auto extName = params[3];

			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[resultId].activated = true;
				irContextGlobal->targets[resultId].id = resultId;
				irContextGlobal->targets[resultId].resultTypeRef = retType;
				irContextGlobal->targets[resultId].isInstance = true;
			}

			if (pass == IFSP_VMA_PASS_SECOND) {
				auto retTypeRef = irContextGlobal->targets[retType];
				auto retTypeComps = 1;
				if (retTypeRef.declType == IFSP_IRTARGET_DECL_VECTOR) {
					retTypeComps = retTypeRef.componentCount;
					retTypeRef = irContextGlobal->targets[retTypeRef.componentTypeRef];
				}
				auto baseType = getTargetTypes(&retTypeRef, irContextGlobal);
				std::string callSig = "@ifspvm_func_callExtHandler";
				if (baseType == "i32") callSig += "I";
				else if (baseType == "float") callSig += "F";
				callSig += (retTypeComps + '0');

				
				//Call
				auto callName = getVariableNamePrefix(&irContextGlobal->targets[resultId], irContextGlobal);
				auto returnTypeName = getTargetTypes(&irContextGlobal->targets[retType], irContextGlobal);
				irContextGlobal->generatedIR << callName << " = call ";
				irContextGlobal->generatedIR << returnTypeName << " ";
				irContextGlobal->generatedIR << "(i32, ...) ";
				irContextGlobal->generatedIR << callSig << "(i32 " << extName;

				for (int i = 4; i < opWordCount - 1; i++) {
					auto paramTypeRef = params[i];
					auto paramType = getTargetTypes(&irContextGlobal->targets[paramTypeRef], irContextGlobal);
					auto paramName = getVariableNamePrefix(&irContextGlobal->targets[params[i]], irContextGlobal);
					irContextGlobal->generatedIR << ", " << paramType << " " << paramName;
				}
				irContextGlobal->generatedIR << ")" << std::endl;
			}

		}

		/* 3.52.5. Mode-Setting Instructions */
		DEFINE_OP(spvOpMemoryModel) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			irContextGlobal->memoryModel = params[0];
			irContextGlobal->addressingModel = params[1];
		}
		DEFINE_OP(spvOpEntryPoint) { 
			
			irContextGlobal->entryPointExecutionModel = params[0];
			irContextGlobal->entryPointId = params[1];
			irContextGlobal->entryPointName = reinterpret_cast<const char*>(params + 2);
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto entryPointRef = irContextGlobal->entryPointId;
				auto entryPoint = &irContextGlobal->targets[entryPointRef];
				auto entryPointName = getVariableNamePrefix(entryPoint, irContextGlobal);
				irContextGlobal->shaderMaps.mainFuncSymbol = entryPointName;
			}

			if (pass == IFSP_VMA_PASS_FIRST) {
				auto entryPointSize = irContextGlobal->entryPointName.size();
				auto entryPointSizeWordUsed = (entryPointSize + 3) / 4;
				irContextGlobal->entryPointInterfaces.resize(opWordCount - 3 - entryPointSizeWordUsed);
				for (int i = 0; i < irContextGlobal->entryPointInterfaces.size(); i++) {
					irContextGlobal->entryPointInterfaces[i] = params[2 + entryPointSizeWordUsed + i];
				}
			}
			
		}
		DEFINE_OP(spvOpExecutionMode) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			irContextGlobal->entryPointExecutionMode = params[1];
		}
		DEFINE_OP(spvOpCapability) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			irContextGlobal->capability = params[0];
		}

		/* 3.52.6. Type-Declaration Instructions */
		DEFINE_OP(spvOpTypeVoid) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_VOID;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeBool) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_BOOL;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeInt) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto width = params[1];
			auto signedness = params[2];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_INT;
			irContextGlobal->targets[targetId].intWidth = width;
			irContextGlobal->targets[targetId].intSignedness = signedness;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeFloat) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto width = params[1];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_FLOAT;
			irContextGlobal->targets[targetId].floatWidth = width;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeVector) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto componentTypeRef = params[1];
			auto componentCount = params[2];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_VECTOR;
			irContextGlobal->targets[targetId].componentTypeRef = componentTypeRef;
			irContextGlobal->targets[targetId].componentCount = componentCount;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeMatrix) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto columnTypeRef = params[1];
			auto columnCount = params[2];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_MATRIX;
			irContextGlobal->targets[targetId].componentTypeRef = columnTypeRef;
			irContextGlobal->targets[targetId].componentCount = columnCount;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeImage) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto imageTypeRef = params[1];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_IMAGE;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeSampler) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			UNIMPLEMENTED_OP(OpTypeSampler)
		}
		DEFINE_OP(spvOpTypeSampledImage) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto imageTypeRef = params[1];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_SAMPLED_IMAGE;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeArray) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto elementTypeRef = params[1];
			auto lengthId = params[2];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_ARRAY;
			irContextGlobal->targets[targetId].componentTypeRef = elementTypeRef;
			irContextGlobal->targets[targetId].componentCount = lengthId;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeRuntimeArray) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto elementTypeRef = params[1];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_RUNTIME_ARRAY;
			irContextGlobal->targets[targetId].componentTypeRef = elementTypeRef;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeStruct) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_STRUCT;
			irContextGlobal->targets[targetId].memberName.resize(opWordCount - 2);
			for(int i = 1; i < opWordCount - 1; i++) {
				irContextGlobal->targets[targetId].memberName[i - 1] = params[i];
			}
			irContextGlobal->targets[targetId].id = targetId;
			
			// Generate IR
			auto structName = getStructName(&irContextGlobal->targets[targetId], irContextGlobal);
			std::string structType = "type {";
			for(int i = 1; i < opWordCount - 1; i++) {
				auto memberTypeRef = params[i];
				auto memberType = getTargetTypes(&irContextGlobal->targets[memberTypeRef], irContextGlobal);
				irContextGlobal->targets[targetId].memberTypeRef.push_back(memberTypeRef);
				structType += memberType;
				if (i != opWordCount - 2) {
					structType += ", ";
				}
			}
			structType += "}";
			irContextGlobal->generatedIR << structName << " = " << structType << std::endl;
		}
		DEFINE_OP(spvOpTypeOpaque) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto name = reinterpret_cast<const char*>(params + 1);
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_OPAQUE;
			irContextGlobal->targets[targetId].debugString = name;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypePointer) {
			auto targetId = params[0];
			auto storageClass = params[1];
			auto typeRef = params[2];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_POINTER;
				irContextGlobal->targets[targetId].componentTypeRef = typeRef;
				irContextGlobal->targets[targetId].storageClass = storageClass;
				irContextGlobal->targets[targetId].id = targetId;
				irContextGlobal->targets[targetId].isUniform = (storageClass == spv::StorageClass::StorageClassUniform);
			}
		}
		DEFINE_OP(spvOpTypeFunction) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[0];
			auto returnTypeRef = params[1];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].declType = IFSP_IRTARGET_DECL_FUNCTION;
			irContextGlobal->targets[targetId].componentTypeRef = returnTypeRef;
			irContextGlobal->targets[targetId].componentCount = opWordCount - 3;
			irContextGlobal->targets[targetId].memberTypeRef.resize(opWordCount - 3);
			for(int i = 2; i < opWordCount - 1; i++) {
				irContextGlobal->targets[targetId].memberTypeRef[i - 2] = params[i];
			}
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpTypeEvent) { UNIMPLEMENTED_OP(OpTypeEvent) }
		DEFINE_OP(spvOpTypeDeviceEvent) { UNIMPLEMENTED_OP(OpTypeDeviceEvent) }
		DEFINE_OP(spvOpTypeReserveId) { UNIMPLEMENTED_OP(OpTypeReserveId) }
		DEFINE_OP(spvOpTypeQueue) { UNIMPLEMENTED_OP(OpTypeQueue) }
		DEFINE_OP(spvOpTypePipe) { UNIMPLEMENTED_OP(OpTypePipe) }

		/* 3.52.7. Constant Creation Instructions */
		DEFINE_OP(spvOpConstantTrue) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[1];
			auto typeRef = params[0];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].componentTypeRef = typeRef;
			irContextGlobal->targets[targetId].data.boolValue = true;
			irContextGlobal->targets[targetId].isConstant = true;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpConstantFalse) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[1];
			auto typeRef = params[0];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].componentTypeRef = typeRef;
			irContextGlobal->targets[targetId].data.boolValue = false;
			irContextGlobal->targets[targetId].isConstant = true;
			irContextGlobal->targets[targetId].id = targetId;
		}
		DEFINE_OP(spvOpConstant) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[1];
			auto typeRef = params[0];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].componentTypeRef = typeRef;
			
			auto typeDesc = irContextGlobal->targets[targetId].declType;
			irContextGlobal->targets[targetId].data.intValue = params[2];
			irContextGlobal->targets[targetId].isConstant = true;
			irContextGlobal->targets[targetId].id = targetId;
			irContextGlobal->targets[targetId].exprType = IFSP_IRTARGET_TYPE_CONSTANT;
		}
		DEFINE_OP(spvOpConstantComposite) {
			if (pass == IFSP_VMA_PASS_SECOND) return;
			auto targetId = params[1];
			auto typeRef = params[0];
			irContextGlobal->targets[targetId].activated = true;
			irContextGlobal->targets[targetId].componentTypeRef = typeRef;
			irContextGlobal->targets[targetId].compositeDataRef.resize(opWordCount - 3);
			for(int i = 2; i < opWordCount - 1; i++) {
				irContextGlobal->targets[targetId].compositeDataRef[i - 2] = params[i];
			}
			irContextGlobal->targets[targetId].isConstant = true;
			irContextGlobal->targets[targetId].id = targetId;
			irContextGlobal->targets[targetId].exprType = IFSP_IRTARGET_TYPE_CONSTANT_COMPOSITE;
		}
		DEFINE_OP(spvOpConstantSampler) { UNIMPLEMENTED_OP(OpConstantSampler) }
		DEFINE_OP(spvOpConstantNull) { UNIMPLEMENTED_OP(OpConstantNull) }

		/* 3.52.8 Memory Instructions */
		DEFINE_OP(spvOpVariable) {
			auto targetId = params[1];
			auto typeRef = params[0];
			auto storageClass = params[2];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].isVariable = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].isGlobal = (irContextGlobal->activatedFunctionRef == -1);
				irContextGlobal->targets[targetId].storageClass = storageClass;
				irContextGlobal->targets[targetId].id = targetId;
			}
			// Register Shader Params
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto varName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				if (storageClass == spv::StorageClass::StorageClassInput) {
					auto location = irContextGlobal->targets[targetId].location;
					auto byteReq = getVariableSize(&irContextGlobal->targets[targetId], irContextGlobal);
					if (location == -1) {
						ERROR_PREFIX
						printf("Invalid input location\n");
					}
					if (irContextGlobal->shaderMaps.inputVarSymbols.size() <= location) {
						irContextGlobal->shaderMaps.inputVarSymbols.resize(location + 1);
						irContextGlobal->shaderMaps.inputSize.resize(location + 1);
					}
					irContextGlobal->shaderMaps.inputVarSymbols[location] = varName;
					irContextGlobal->shaderMaps.inputSize[location] = byteReq;
				}else if(storageClass == spv::StorageClass::StorageClassOutput) {
					if (irContextGlobal->targets[targetId].isBuiltinPos) {
						irContextGlobal->shaderMaps.builtinPositionSymbol = varName;
					}
					else {
						auto location = irContextGlobal->targets[targetId].location;
						auto byteReq = getVariableSize(&irContextGlobal->targets[targetId], irContextGlobal);

						if (location == -1) {
							ERROR_PREFIX
								printf("Invalid output location\n");
						}
						if (irContextGlobal->shaderMaps.outputVarSymbols.size() <= location) {
							irContextGlobal->shaderMaps.outputVarSymbols.resize(location + 1);
							irContextGlobal->shaderMaps.outputSize.resize(location + 1);
						}
						irContextGlobal->shaderMaps.outputVarSymbols[location] = varName;
						irContextGlobal->shaderMaps.outputSize[location] = byteReq;
					}
				}
				else if (storageClass == spv::StorageClass::StorageClassUniform) {
					auto binding = irContextGlobal->targets[targetId].binding;
					auto descSet = irContextGlobal->targets[targetId].descSet;
					irContextGlobal->shaderMaps.uniformSize.push_back(getVariableSize(&irContextGlobal->targets[targetId], irContextGlobal));
					irContextGlobal->shaderMaps.uniformVarLoc.push_back({ binding,descSet });
					irContextGlobal->shaderMaps.uniformVarSymbols.push_back(varName);
				}
			}

			// Generate IR
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto variableName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto variableType = getTargetTypes(&irContextGlobal->targets[typeRef], irContextGlobal);
				if (irContextGlobal->activatedFunctionRef != -1) {
					irContextGlobal->generatedIR << variableName << " = alloca " << variableType << std::endl;
				}
				else {
					irContextGlobal->generatedIR << variableName << " = global " << variableType << " undef" << std::endl;
				}
			}
		}
		DEFINE_OP(spvOpLoad) {
			auto targetId = params[1];
			auto targetType = params[0];
			auto pointerId = params[2];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].resultTypeRef = targetType;
				irContextGlobal->targets[targetId].id = targetId;
			}

			//Generate IR
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto pointerIsAccessChain = irContextGlobal->targets[pointerId].isAccessChain;
				auto variableName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto pointerName = getVariableNamePrefix(&irContextGlobal->targets[pointerId], irContextGlobal);
				auto typeNameRet = getTargetTypes(&irContextGlobal->targets[irContextGlobal->targets[pointerId].resultTypeRef], irContextGlobal);
				auto typeName = getTargetTypes(&irContextGlobal->targets[targetType], irContextGlobal);
				
				irContextGlobal->generatedIR << variableName << " = load " << typeName << ", " << typeNameRet << "* " << pointerName << std::endl;
				
			}
		}
		DEFINE_OP(spvOpStore) {
			if(pass == IFSP_VMA_PASS_FIRST) return;
			//Generate IR
			if (opWordCount != 3) {
				ERROR_PREFIX
				printf("Invalid store instruction\n");
			}
			auto pointerId = params[0];
			auto pointerName = getVariableNamePrefix(&irContextGlobal->targets[pointerId], irContextGlobal);
			auto valueId = params[1];
			
			auto isConst = irContextGlobal->targets[valueId].isConstant;
			std::string valueType;
			if (isConst) {
				valueType = getTargetTypes(&irContextGlobal->targets[irContextGlobal->targets[valueId].componentTypeRef], irContextGlobal);
			}
			else {
				valueType = getTargetTypes(&irContextGlobal->targets[irContextGlobal->targets[valueId].resultTypeRef], irContextGlobal);
			}
			auto valueName = getVariableNamePrefix(&irContextGlobal->targets[valueId], irContextGlobal);
			irContextGlobal->generatedIR << "store " << valueType << " " << valueName << ", " << valueType << "* " << pointerName << std::endl;

		}
		DEFINE_OP(spvOpAccessChain) {
			auto targetId = params[1];
			auto resultType = params[0];
			auto base = params[2];

			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].resultTypeRef = resultType;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].isAccessChain = true;
				irContextGlobal->targets[targetId].isVariable = true;

				irContextGlobal->targets[targetId].accessChainRef = base;
				irContextGlobal->targets[targetId].accessChain.resize(opWordCount - 4);
				for (int i = 3; i < opWordCount - 1; i++) {
					irContextGlobal->targets[targetId].accessChain[i - 3] = params[i];
				}
				irContextGlobal->targets[targetId].id = targetId;

				auto curLocType = irContextGlobal->targets[base].resultTypeRef;
				if (irContextGlobal->targets[curLocType].declType == IFSP_IRTARGET_DECL_POINTER) {
					curLocType = irContextGlobal->targets[curLocType].componentTypeRef;
				}
				//irContextGlobal->targets[targetId].resultTypeRef = curLocType;
			}

			if (pass == IFSP_VMA_PASS_SECOND) {
				auto pointerId = base;
				auto curLoc = pointerId;
				auto curLocType = irContextGlobal->targets[curLoc].resultTypeRef;
				if (irContextGlobal->targets[curLocType].declType == IFSP_IRTARGET_DECL_POINTER) {
					curLocType = irContextGlobal->targets[curLocType].componentTypeRef;
				}
				auto ptrName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto tmpTpRet = getTargetTypes(&irContextGlobal->targets[resultType], irContextGlobal);
				auto tmpTp = getTargetTypes(&irContextGlobal->targets[curLocType], irContextGlobal);
				irContextGlobal->generatedIR << ptrName << " = ";
				
				irContextGlobal->generatedIR << " getelementptr  " << tmpTp << ", ";
				irContextGlobal->generatedIR << tmpTp << "* " << getVariableNamePrefix(&irContextGlobal->targets[pointerId], irContextGlobal) << ", ";
				irContextGlobal->generatedIR << "i32 0";
				for (int i = 0; i < irContextGlobal->targets[targetId].accessChain.size(); i++) {
					auto offset = irContextGlobal->targets[targetId].accessChain[i];
					auto offsetValue = irContextGlobal->targets[offset].data.intValue;
					irContextGlobal->generatedIR << ", i32 " << offsetValue;
				}
				irContextGlobal->generatedIR << std::endl;
			}
			
		}

		/* 3.52.9. Function Instructions */
		DEFINE_OP(spvOpFunction) {
			if (irContextGlobal->activatedFunctionRef != -1) {
				ERROR_PREFIX
				printf("Nested function definition is not allowed");
				printf("\n");
			}
			auto targetId = params[1];
			auto returnTypeRef = params[0];
			auto functionControl = params[2];
			auto functionTypeRef = params[3];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].resultTypeRef = returnTypeRef;
				irContextGlobal->targets[targetId].componentTypeRef = functionTypeRef;
				irContextGlobal->targets[targetId].functionControl = functionControl;
				irContextGlobal->targets[targetId].isFunction = true;

			}

			// Generate Func IR
			irContextGlobal->activatedFunctionRef = targetId;
			if (pass == IFSP_VMA_PASS_SECOND) {
				irContextGlobal->recordedFuncParams = 0;
				irContextGlobal->targets[targetId].id = targetId;
				auto funcName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto funcType = getTargetTypes(&irContextGlobal->targets[returnTypeRef], irContextGlobal);
				irContextGlobal->functionInstIR << "define " << funcType << " " << funcName << "(";

				if (irContextGlobal->recordedFuncParams == irContextGlobal->targets[targetId].componentCount) {
					irContextGlobal->functionInstIR << ") {\n";
					irContextGlobal->generatedIR << "\n" << irContextGlobal->functionInstIR.str();
					irContextGlobal->functionInstIR.clear();
				}
			}
		}
		DEFINE_OP(spvOpFunctionParameter) {
			if(irContextGlobal->activatedFunctionRef == -1) {
				ERROR_PREFIX
				printf("Function parameter must be defined within a function");
				printf("\n");
			}
			auto targetId = params[1];
			auto typeRef = params[0];
			auto parentId = irContextGlobal->activatedFunctionRef;
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[parentId].functionParamRef.push_back(targetId);
				irContextGlobal->targets[targetId].id = targetId;
			}
			// Generate Func IR
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto funcName = getVariableNamePrefix(&irContextGlobal->targets[parentId], irContextGlobal);
				auto paramName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto paramType = getTargetTypes(&irContextGlobal->targets[typeRef], irContextGlobal);
				if (irContextGlobal->recordedFuncParams > 0) {
					irContextGlobal->functionInstIR << ", ";
				}
				irContextGlobal->functionInstIR << paramType << " " << paramName;
				irContextGlobal->recordedFuncParams++;
				if (irContextGlobal->recordedFuncParams == irContextGlobal->targets[parentId].componentCount) {
					irContextGlobal->functionInstIR << ") {\n";
					irContextGlobal->generatedIR << "\n" << irContextGlobal->functionInstIR.str();
					irContextGlobal->functionInstIR.clear();
				}
			}
		}
		DEFINE_OP(spvOpFunctionEnd) {
			if(irContextGlobal->activatedFunctionRef == -1) {
				ERROR_PREFIX
				printf("Function end must be defined within a function");
				printf("\n");
			}
			irContextGlobal->activatedFunctionRef = -1;
			
			// Generate IR
			if (pass == IFSP_VMA_PASS_SECOND) {
				irContextGlobal->generatedIR << "}\n";
			}
		}
		DEFINE_OP(spvOpFunctionCall) {
			auto resultType = params[0];
			auto resultId = params[1];
			auto functionId = params[2];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[resultId].activated = true;
				irContextGlobal->targets[resultId].resultTypeRef = resultType;
				irContextGlobal->targets[resultId].isInstance = true;
				for (int i = 3; i < opWordCount - 1; i++) {
					auto paramId = params[i];
					irContextGlobal->targets[resultId].compositeDataRef.push_back(paramId);
				}
				irContextGlobal->targets[resultId].id = resultId;
			}

			// Generate IR
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto funcName = getVariableNamePrefix(&irContextGlobal->targets[functionId], irContextGlobal);
				auto resultName = getVariableNamePrefix(&irContextGlobal->targets[resultId], irContextGlobal);
				auto funcResultType = getTargetTypes(&irContextGlobal->targets[resultId], irContextGlobal);
				irContextGlobal->generatedIR << resultName << " = call " << funcResultType << " " << funcName << "(";
				for (int i = 0; i < irContextGlobal->targets[resultId].compositeDataRef.size(); i++) {
					auto paramName = getVariableNamePrefix(&irContextGlobal->targets[irContextGlobal->targets[resultId].compositeDataRef[i]], irContextGlobal);
					auto paramType = getTargetTypes(&irContextGlobal->targets[irContextGlobal->targets[resultId].compositeDataRef[i]], irContextGlobal);
					if (i > 0) {
						irContextGlobal->generatedIR << ", ";
					}
					irContextGlobal->generatedIR << paramType << " " << paramName;
				}
				irContextGlobal->generatedIR << ")\n";
			}
		}

		/* 3.52.10. Image Instructions */
		DEFINE_OP(spvOpSampledImage) { 
			auto targetId = params[1];
			auto resultType = params[0];
			auto imageId = params[2];
			auto samplerId = params[3];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].resultTypeRef = resultType;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto image = irContextGlobal->targets[imageId];
				auto sampler = irContextGlobal->targets[samplerId];
				auto imageType = getTargetTypes(&image, irContextGlobal);
				auto samplerType = getTargetTypes(&sampler, irContextGlobal);
				auto resultName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto resultTypeName = getTargetTypes(&irContextGlobal->targets[resultType], irContextGlobal);
				irContextGlobal->generatedIR << resultName << " = call " << resultTypeName << " @ifspvm_func_sampledImage(";
				irContextGlobal->generatedIR << imageType << " " << getVariableNamePrefix(&image, irContextGlobal) << ", ";
				irContextGlobal->generatedIR << samplerType << " " << getVariableNamePrefix(&sampler, irContextGlobal) << ")\n";
			}


		}
		DEFINE_OP(spvOpImageSampleImplicitLod) { 
			auto targetId = params[1];
			auto resultType = params[0];
			auto imageId = params[2];
			auto coordId = params[3];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].resultTypeRef = resultType;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto image = irContextGlobal->targets[imageId];
				auto coord = irContextGlobal->targets[coordId];
				auto imageType = getTargetTypes(&image, irContextGlobal);
				auto coordType = getTargetTypes(&coord, irContextGlobal);
				auto resultName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto resultTypeName = getTargetTypes(&irContextGlobal->targets[resultType], irContextGlobal);
				irContextGlobal->generatedIR << resultName << " = call " << resultTypeName << " @ifspvm_func_imageSampleImplicitLod(";
				irContextGlobal->generatedIR << imageType << " " << getVariableNamePrefix(&image, irContextGlobal) << ", ";
				irContextGlobal->generatedIR << coordType << " " << getVariableNamePrefix(&coord, irContextGlobal) << ")\n";
			}

		}
		DEFINE_OP(spvOpImageSampleExplicitLod) { 
			UNIMPLEMENTED_OP(OpImageSampleExplicitLod)
		}
		DEFINE_OP(spvOpImageSampleDrefImplicitLod) { 
			UNIMPLEMENTED_OP(OpImageSampleDrefImplicitLod)
		}

		/* 3.52.11 Conversion Instructions */
		DEFINE_OP(spvOpConvertFToU) { 
			UNIMPLEMENTED_OP(OpConvertFToU) 
		}
		DEFINE_OP(spvOpConvertFToS) { 
			UNIMPLEMENTED_OP(OpConvertFToS) 
		}
		DEFINE_OP(spvOpConvertSToF) { 
			UNIMPLEMENTED_OP(OpConvertSToF) 
		}
		DEFINE_OP(spvOpConvertUToF) { 
			UNIMPLEMENTED_OP(OpConvertUToF) 
		}
		DEFINE_OP(spvOpUConvert) { 
			UNIMPLEMENTED_OP(OpUConvert) 
		}
		DEFINE_OP(spvOpSConvert) { 
			UNIMPLEMENTED_OP(OpSConvert) 
		}
		DEFINE_OP(spvOpFConvert) { 
			UNIMPLEMENTED_OP(OpFConvert) 
		}

		/* 3.52.12 Composite Instructions */
		DEFINE_OP(spvOpVectorExtractDynamic) {
			UNIMPLEMENTED_OP(OpVectorExtractDynamic)
		}
		DEFINE_OP(spvOpVectorInsertDynamic) {
			UNIMPLEMENTED_OP(OpVectorInsertDynamic)
		}
		DEFINE_OP(spvOpVectorShuffle) {
			auto resultType = params[0];
			auto resultId = params[1];
			auto vec1Id = params[2];
			auto vec2Id = params[3];
			auto componentCount = opWordCount - 5;
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[resultId].activated = true;
				irContextGlobal->targets[resultId].resultTypeRef = resultType;
				irContextGlobal->targets[resultId].isInstance = true;
				irContextGlobal->targets[resultId].compositeDataRef.resize(componentCount);
				for (int i = 4; i < opWordCount - 1; i++) {
					irContextGlobal->targets[resultId].compositeDataRef[i - 4] = params[i];
				}
				irContextGlobal->targets[resultId].id = resultId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto vec1 = irContextGlobal->targets[vec1Id];
				auto vec2 = irContextGlobal->targets[vec2Id];
				auto vec1Type = getTargetTypes(&vec1, irContextGlobal);
				auto vec2Type = getTargetTypes(&vec2, irContextGlobal);
				auto resultName = getVariableNamePrefix(&irContextGlobal->targets[resultId], irContextGlobal);
				irContextGlobal->generatedIR << resultName << " = shufflevector " << vec1Type << " " << getVariableNamePrefix(&vec1, irContextGlobal) << ", " << vec2Type << " " << getVariableNamePrefix(&vec2, irContextGlobal);
				
				auto elementType = irContextGlobal->targets[irContextGlobal->targets[vec1.resultTypeRef].componentTypeRef];
				auto elementTypeName = getTargetTypes(&elementType, irContextGlobal);

				irContextGlobal->generatedIR << ", <" << componentCount << " x " << elementTypeName << "> <i32 ";
				for (int i = 0; i < componentCount; i++) {
					if (i > 0) {
						irContextGlobal->generatedIR << ", i32 ";
					}
					irContextGlobal->generatedIR << irContextGlobal->targets[resultId].compositeDataRef[i];
				}
				irContextGlobal->generatedIR << ">\n";
			}
		}
		DEFINE_OP(spvOpCompositeConstruct) {
			auto targetId = params[1];
			auto typeRef = params[0];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].compositeDataRef.resize(opWordCount - 2);
				for (int i = 2; i < opWordCount - 1; i++) {
					irContextGlobal->targets[targetId].compositeDataRef[i - 2] = params[i];
				}
				irContextGlobal->targets[targetId].id = targetId;
				irContextGlobal->targets[targetId].isInstance = true;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto targetType = irContextGlobal->targets[typeRef];
				if (targetType.declType != IFSP_IRTARGET_DECL_VECTOR) {
					ERROR_PREFIX
						printf("Cannot construct non-vector structure\n");
				}
				auto resultName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto resultType = getTargetTypes(&irContextGlobal->targets[targetType.componentTypeRef], irContextGlobal);
				std::string lastName = "undef";
				auto elementType = irContextGlobal->targets[targetType.componentTypeRef];
				auto elementTypeName = getTargetTypes(&elementType, irContextGlobal);
				for (int i = 2; i < opWordCount - 1; i++) {
					auto curName = "%ifspvm_compositeconstruct_" + std::to_string(instLine) + "_" + std::to_string(i-2);
					if (i == opWordCount - 2) {
						curName = resultName;
					}
					irContextGlobal->generatedIR << curName << " = insertelement <" << opWordCount - 3 << " x " << elementTypeName << "> " << lastName << ", " << elementTypeName << " " << getVariableNamePrefix(&irContextGlobal->targets[irContextGlobal->targets[targetId].compositeDataRef[i-2]], irContextGlobal) << ", i32 " << i-2;
					irContextGlobal->generatedIR << std::endl;
					lastName = curName;
				}
			}
		}
		DEFINE_OP(spvOpCompositeExtract) {
			auto targetId = params[1];
			auto typeRef = params[0];
			auto compositeId = params[2];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				//Generate IR
				auto composite = irContextGlobal->targets[compositeId];
				auto compositeType = getTargetTypes(&composite, irContextGlobal);
				auto resultName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto numComponents = opWordCount - 4;
				irContextGlobal->generatedIR << resultName << " = extractelement " << compositeType << " " << getVariableNamePrefix(&composite, irContextGlobal) << ", i32 ";
				for(int i=0;i<numComponents;i++) {
					if(i > 0) {
						irContextGlobal->generatedIR << ", i32 ";
					}
					irContextGlobal->generatedIR << params[i+3];
				}
				irContextGlobal->generatedIR << std::endl;
			}
		}
		DEFINE_OP(spvOpCompositeInsert) {
			UNIMPLEMENTED_OP(OpCompositeInsert)
		}

		/* 3.52.13 Arithmetic  Instructions */
		DEFINE_OP(spvOpSNegate) {
			UNIMPLEMENTED_OP(OpSNegate)
		}
		DEFINE_OP(spvOpFNegate) {
			UNIMPLEMENTED_OP(OpFNegate)
		}
		DEFINE_OP(spvOpIAdd) {
			auto targetId = params[1];
			auto typeRef = params[0];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto opL = irContextGlobal->targets[params[2]];
				auto opR = irContextGlobal->targets[params[3]];
				elementwiseArithmeticIRGenerator("add", irContextGlobal, &opL, &opR, &irContextGlobal->targets[targetId]);
			}
		}
		DEFINE_OP(spvOpFAdd) {
			auto targetId = params[1];
			auto typeRef = params[0];
			if(pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if(pass == IFSP_VMA_PASS_SECOND) {
				auto opL = irContextGlobal->targets[params[2]];
				auto opR = irContextGlobal->targets[params[3]];
				elementwiseArithmeticIRGenerator("fadd", irContextGlobal, &opL, &opR, &irContextGlobal->targets[targetId]);
			}
		}
		DEFINE_OP(spvOpISub) {
			UNIMPLEMENTED_OP(OpISub)
		}
		DEFINE_OP(spvOpFSub) {
			auto targetId = params[1];
			auto typeRef = params[0];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto opL = irContextGlobal->targets[params[2]];
				auto opR = irContextGlobal->targets[params[3]];
				elementwiseArithmeticIRGenerator("fsub", irContextGlobal, &opL, &opR, &irContextGlobal->targets[targetId]);
			}
		}
		DEFINE_OP(spvOpIMul) {
			UNIMPLEMENTED_OP(OpIMul)
		}
		DEFINE_OP(spvOpFMul) {
			auto targetId = params[1];
			auto typeRef = params[0];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto opL = irContextGlobal->targets[params[2]];
				auto opR = irContextGlobal->targets[params[3]];
				elementwiseArithmeticIRGenerator("fmul", irContextGlobal, &opL, &opR, &irContextGlobal->targets[targetId]);
			}
		}
		DEFINE_OP(spvOpUDiv) {
			UNIMPLEMENTED_OP(OpUDiv)
		}
		DEFINE_OP(spvOpSDiv) {
			UNIMPLEMENTED_OP(OpSDiv)
		}
		DEFINE_OP(spvOpFDiv) {
			auto targetId = params[1];
			auto typeRef = params[0];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto opL = irContextGlobal->targets[params[2]];
				auto opR = irContextGlobal->targets[params[3]];
				elementwiseArithmeticIRGenerator("fdiv", irContextGlobal, &opL, &opR, &irContextGlobal->targets[targetId]);
			}
		}
		DEFINE_OP(spvOpUMod) {
			UNIMPLEMENTED_OP(OpUMod)
		}
		DEFINE_OP(spvOpSRem) {
			UNIMPLEMENTED_OP(OpSRem)
		}
		DEFINE_OP(spvOpSMod) {
			UNIMPLEMENTED_OP(OpSMod)
		}
		DEFINE_OP(spvOpFRem) {
			UNIMPLEMENTED_OP(OpFRem)
		}
		DEFINE_OP(spvOpFMod) {
			UNIMPLEMENTED_OP(OpFMod)
		}
		DEFINE_OP(spvOpVectorTimesScalar) {
			auto retType = params[0];
			auto retId = params[1];
			auto vecId = params[2];
			auto scalarId = params[3];
			if(pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[retId].activated = true;
				irContextGlobal->targets[retId].isInstance = true;
				irContextGlobal->targets[retId].resultTypeRef = retType;
				irContextGlobal->targets[retId].id = retId;
			}
			if(pass == IFSP_VMA_PASS_SECOND) {
				auto vecInst = irContextGlobal->targets[vecId];
				auto vec = irContextGlobal->targets[irContextGlobal->targets[vecId].resultTypeRef];
				auto scalar = irContextGlobal->targets[scalarId];
				auto vecType = getTargetTypes(&vec, irContextGlobal);
				auto vecName = getVariableNamePrefix(&vecInst, irContextGlobal);
				auto scalarType = getTargetTypes(&scalar, irContextGlobal);
				auto retName = getVariableNamePrefix(&irContextGlobal->targets[retId], irContextGlobal);
				auto retType = getTargetTypes(&irContextGlobal->targets[retId], irContextGlobal);
				
				//Broadcast Scalar
				auto scalarName = getVariableNamePrefix(&scalar, irContextGlobal);
				auto scalarTypeName = getTargetTypes(&scalar, irContextGlobal);
				std::string lastName = "undef";
				for (int i = 0; i < vec.componentCount; i++) {
					auto curName = "%ifspvm_broadcast_" + std::to_string(instLine) + "_" + std::to_string(i);
					irContextGlobal->generatedIR << curName << " = insertelement <"<< vec.componentCount <<" x float> " << lastName << ", float " << scalarName << ", i32 " << i;
					irContextGlobal->generatedIR << "\n";
					lastName = curName;
				}
				elementwiseArithmeticIRGeneratorRaw("fmul", irContextGlobal, retName,retType, lastName, vecName);
				
			}
		}
		DEFINE_OP(spvOpMatrixTimesVector) {
			UNIMPLEMENTED_OP(OpMatrixTimesVector)
		}
		DEFINE_OP(spvOpDot) {
			UNIMPLEMENTED_OP(OpDot)
		}

		/* 3.52.15 Relational and Logical Instructions */
		DEFINE_OP(spvOpAny) {
			UNIMPLEMENTED_OP(OpAny) 
		}
		DEFINE_OP(spvOpAll) {
			UNIMPLEMENTED_OP(OpAll) 
		}
		DEFINE_OP(spvOpIsNan) {
			UNIMPLEMENTED_OP(OpIsNan) 
		}
		DEFINE_OP(spvOpIsInf) {
			UNIMPLEMENTED_OP(OpIsInf) 
		}
		DEFINE_OP(spvOpIsFinite) {
			UNIMPLEMENTED_OP(OpIsFinite) 
		}
		DEFINE_OP(spvOpIsNormal) {
			UNIMPLEMENTED_OP(OpIsNormal) 
		}
		DEFINE_OP(spvOpSignBitSet) {
			UNIMPLEMENTED_OP(OpSignBitSet) 
		}
		DEFINE_OP(spvOpLessOrGreater) {
			UNIMPLEMENTED_OP(OpLessOrGreater) 
		}
		DEFINE_OP(spvOpOrdered) {
			UNIMPLEMENTED_OP(OpOrdered) 
		}
		DEFINE_OP(spvOpUnordered) {
			UNIMPLEMENTED_OP(OpUnordered) 
		}
		DEFINE_OP(spvOpLogicalEqual) {
			UNIMPLEMENTED_OP(OpLogicalEqual) 
		}
		DEFINE_OP(spvOpLogicalNotEqual) {
			UNIMPLEMENTED_OP(OpLogicalNotEqual) 
		}
		DEFINE_OP(spvOpLogicalOr) {
			UNIMPLEMENTED_OP(OpLogicalOr) 
		}
		DEFINE_OP(spvOpLogicalAnd) {
			UNIMPLEMENTED_OP(OpLogicalAnd) 
		}
		DEFINE_OP(spvOpLogicalNot) {
			UNIMPLEMENTED_OP(OpLogicalNot) 
		}
		DEFINE_OP(spvOpSelect) {
			UNIMPLEMENTED_OP(OpSelect) 
		}
		DEFINE_OP(spvOpIEqual) {
			UNIMPLEMENTED_OP(OpIEqual) 
		}
		DEFINE_OP(spvOpINotEqual) {
			UNIMPLEMENTED_OP(OpINotEqual) 
		}
		DEFINE_OP(spvOpUGreaterThan) {
			UNIMPLEMENTED_OP(OpUGreaterThan) 
		}
		DEFINE_OP(spvOpSGreaterThan) {
			UNIMPLEMENTED_OP(OpSGreaterThan) 
		}
		DEFINE_OP(spvOpUGreaterThanEqual) {
			UNIMPLEMENTED_OP(OpUGreaterThanEqual) 
		}
		DEFINE_OP(spvOpSGreaterThanEqual) {
			UNIMPLEMENTED_OP(OpSGreaterThanEqual) 
		}
		DEFINE_OP(spvOpULessThan) {
			UNIMPLEMENTED_OP(OpULessThan) 
		}
		DEFINE_OP(spvOpSLessThan) {
			auto targetId = params[1];
			auto typeRef = params[0];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto opL = irContextGlobal->targets[params[2]];
				auto opR = irContextGlobal->targets[params[3]];
				elementwiseArithmeticIRGenerator("icmp sge", irContextGlobal, &opL, &opR, &irContextGlobal->targets[targetId]);
			}
		}
		DEFINE_OP(spvOpULessThanEqual) {
			UNIMPLEMENTED_OP(OpULessThanEqual) 
		}
		DEFINE_OP(spvOpSLessThanEqual) {
			UNIMPLEMENTED_OP(OpSLessThanEqual) 
		}
		DEFINE_OP(spvOpFOrdEqual) {
			UNIMPLEMENTED_OP(OpFOrdEqual) 
		}
		DEFINE_OP(spvOpFUnordEqual) {
			UNIMPLEMENTED_OP(OpFUnordEqual) 
		}
		DEFINE_OP(spvOpFOrdNotEqual) {
			UNIMPLEMENTED_OP(OpFOrdNotEqual) 
		}
		DEFINE_OP(spvOpFUnordNotEqual) {
			UNIMPLEMENTED_OP(OpFUnordNotEqual) 
		}
		DEFINE_OP(spvOpFOrdLessThan) {
			auto targetId = params[1];
			auto typeRef = params[0];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto opL = irContextGlobal->targets[params[2]];
				auto opR = irContextGlobal->targets[params[3]];
				elementwiseArithmeticIRGenerator("fcmp olt", irContextGlobal, &opL, &opR, &irContextGlobal->targets[targetId]);
			}
		}
		DEFINE_OP(spvOpFUnordLessThan) {
			UNIMPLEMENTED_OP(OpFUnordLessThan) 
		}
		DEFINE_OP(spvOpFOrdGreaterThan) {
			auto targetId = params[1];
			auto typeRef = params[0];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].resultTypeRef = typeRef;
				irContextGlobal->targets[targetId].id = targetId;
			}
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto opL = irContextGlobal->targets[params[2]];
				auto opR = irContextGlobal->targets[params[3]];
				elementwiseArithmeticIRGenerator("fcmp ogt", irContextGlobal, &opL, &opR, &irContextGlobal->targets[targetId]);
			}
		}
		DEFINE_OP(spvOpFUnordGreaterThan) {
			UNIMPLEMENTED_OP(OpFUnordGreaterThan) 
		}
		DEFINE_OP(spvOpFOrdLessThanEqual) {
			UNIMPLEMENTED_OP(OpFOrdLessThanEqual) 
		}
		DEFINE_OP(spvOpFUnordLessThanEqual) {
			UNIMPLEMENTED_OP(OpFUnordLessThanEqual) 
		}
		DEFINE_OP(spvOpFOrdGreaterThanEqual) {
			UNIMPLEMENTED_OP(OpFOrdGreaterThanEqual) 
		}
		DEFINE_OP(spvOpFUnordGreaterThanEqual) {
			UNIMPLEMENTED_OP(OpFUnordGreaterThanEqual) 
		}


		/* 3.52.17. Control-Flow Instructions */
		DEFINE_OP(spvOpLoopMerge) {
			UNIMPLEMENTED_OP(OpLoopMerge)
		}
		DEFINE_OP(spvOpSelectionMerge) {
			auto mergeId = params[0];
			irContextGlobal->targets[mergeId].activated = true;
			irContextGlobal->targets[mergeId].isMergedBlock = true;
		}
		DEFINE_OP(spvOpLabel) {
			auto targetId = params[0];
			if (pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].exprType = IFSP_IRTARGET_INTERMEDIATE_UNDEF;
				irContextGlobal->targets[targetId].id = targetId;
				irContextGlobal->targets[targetId].isLabel = true;
			}

			//Generate IR
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto labelName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				irContextGlobal->generatedIR << "\n" << labelName << ":" << std::endl;
			}
		}
		DEFINE_OP(spvOpBranch) {
			if (pass == IFSP_VMA_PASS_FIRST) return;
			auto destId = params[0];
			auto destName = getVariableNamePrefix(&irContextGlobal->targets[destId], irContextGlobal);
			
			//Generate IR
			irContextGlobal->generatedIR << "br label %" << destName << std::endl;
		}
		DEFINE_OP(spvOpBranchConditional) {
			if (pass == IFSP_VMA_PASS_FIRST) return;
			auto conditionId = params[0];
			auto trueLabelId = params[1];
			auto falseLabelId = params[2];
			auto conditionName = getVariableNamePrefix(&irContextGlobal->targets[conditionId], irContextGlobal);
			auto conditionType = getTargetTypes(&irContextGlobal->targets[conditionId], irContextGlobal);
			auto trueLabelName = getVariableNamePrefix(&irContextGlobal->targets[trueLabelId], irContextGlobal);
			auto falseLabelName = getVariableNamePrefix(&irContextGlobal->targets[falseLabelId], irContextGlobal);

			//Generate IR
			irContextGlobal->generatedIR << "br " << conditionType << " " << conditionName << ", label %" << trueLabelName << ", label %" << falseLabelName << std::endl;
		}
		DEFINE_OP(spvOpSwitch) {
			UNIMPLEMENTED_OP(OpSwitch)
		}
		DEFINE_OP(spvOpReturn) {
			if (pass == IFSP_VMA_PASS_FIRST) return;
			
			//Generate IR
			irContextGlobal->generatedIR << "ret void" << std::endl;
		}
		DEFINE_OP(spvOpReturnValue) {
			UNIMPLEMENTED_OP(OpReturnValue)
		}
		DEFINE_OP(spvOpPhi) {
			auto targetId = params[1];
			auto targetType = params[0];
			if(pass == IFSP_VMA_PASS_FIRST) {
				irContextGlobal->targets[targetId].activated = true;
				irContextGlobal->targets[targetId].resultTypeRef = targetType;
				irContextGlobal->targets[targetId].isInstance = true;
				irContextGlobal->targets[targetId].id = targetId;
			}

			//Generate IR
			if (pass == IFSP_VMA_PASS_SECOND) {
				auto phiName = getVariableNamePrefix(&irContextGlobal->targets[targetId], irContextGlobal);
				auto phiType = getTargetTypes(&irContextGlobal->targets[targetType], irContextGlobal);
				irContextGlobal->generatedIR << phiName << " = phi " << phiType << " ";
				for (int i = 2; i < opWordCount - 1; i += 2) {
					auto labelId = params[i];
					auto labelName = getVariableNamePrefix(&irContextGlobal->targets[labelId], irContextGlobal);
					auto valueId = params[i + 1];
					auto valueName = getVariableNamePrefix(&irContextGlobal->targets[valueId], irContextGlobal);
					if (i > 2) {
						irContextGlobal->generatedIR << ", ";
					}
					irContextGlobal->generatedIR << "[" << valueName << ", %" << labelName << "]";
				}
				irContextGlobal->generatedIR << std::endl;
			}
		}


	}
#undef DEPRECATED_OP
#undef UNIMPLEMENTED_OP
#undef DEFINE_OP

	void loadLookupTable(std::unordered_map<spv::Op, SpvInstFunc>& outMap) {
		outMap[spv::OpNop] = InstructionImpl::spvOpNop;
		outMap[spv::OpUndef] = InstructionImpl::spvOpUndef;
		outMap[spv::OpSourceContinued] = InstructionImpl::spvOpSourceContinued;
		outMap[spv::OpSource] = InstructionImpl::spvOpSource;
		outMap[spv::OpSourceExtension] = InstructionImpl::spvOpSourceExtension;
		outMap[spv::OpName] = InstructionImpl::spvOpName;
		outMap[spv::OpMemberName] = InstructionImpl::spvOpMemberName;
		outMap[spv::OpString] = InstructionImpl::spvOpString;
		outMap[spv::OpLine] = InstructionImpl::spvOpLine;
		outMap[spv::OpNoLine] = InstructionImpl::spvOpNoLine;
		outMap[spv::OpModuleProcessed] = InstructionImpl::spvOpModuleProcessed;
		outMap[spv::OpDecorate] = InstructionImpl::spvOpDecorate;
		outMap[spv::OpMemberDecorate] = InstructionImpl::spvOpMemberDecorate;
		outMap[spv::OpGroupDecorate] = InstructionImpl::spvOpGroupDecorate;
		outMap[spv::OpGroupMemberDecorate] = InstructionImpl::spvOpGroupMemberDecorate;
		outMap[spv::OpDecorationGroup] = InstructionImpl::spvOpDecorationGroup;
		outMap[spv::OpExtension] = InstructionImpl::spvOpExtension;
		outMap[spv::OpExtInstImport] = InstructionImpl::spvOpExtInstImport;
		outMap[spv::OpExtInst] = InstructionImpl::spvOpExtInst;
		outMap[spv::OpMemoryModel] = InstructionImpl::spvOpMemoryModel;
		outMap[spv::OpEntryPoint] = InstructionImpl::spvOpEntryPoint;
		outMap[spv::OpExecutionMode] = InstructionImpl::spvOpExecutionMode;
		outMap[spv::OpCapability] = InstructionImpl::spvOpCapability;

		outMap[spv::OpTypeVoid] = InstructionImpl::spvOpTypeVoid;
		outMap[spv::OpTypeBool] = InstructionImpl::spvOpTypeBool;
		outMap[spv::OpTypeInt] = InstructionImpl::spvOpTypeInt;
		outMap[spv::OpTypeFloat] = InstructionImpl::spvOpTypeFloat;
		outMap[spv::OpTypeVector] = InstructionImpl::spvOpTypeVector;
		outMap[spv::OpTypeMatrix] = InstructionImpl::spvOpTypeMatrix;
		outMap[spv::OpTypeImage] = InstructionImpl::spvOpTypeImage;
		outMap[spv::OpTypeSampler] = InstructionImpl::spvOpTypeSampler;
		outMap[spv::OpTypeSampledImage] = InstructionImpl::spvOpTypeSampledImage;
		outMap[spv::OpTypeArray] = InstructionImpl::spvOpTypeArray;
		outMap[spv::OpTypeRuntimeArray] = InstructionImpl::spvOpTypeRuntimeArray;
		outMap[spv::OpTypeStruct] = InstructionImpl::spvOpTypeStruct;
		outMap[spv::OpTypeOpaque] = InstructionImpl::spvOpTypeOpaque;
		outMap[spv::OpTypePointer] = InstructionImpl::spvOpTypePointer;
		outMap[spv::OpTypeFunction] = InstructionImpl::spvOpTypeFunction;
		outMap[spv::OpTypeEvent] = InstructionImpl::spvOpTypeEvent;
		outMap[spv::OpTypeDeviceEvent] = InstructionImpl::spvOpTypeDeviceEvent;
		outMap[spv::OpTypeReserveId] = InstructionImpl::spvOpTypeReserveId;
		outMap[spv::OpTypeQueue] = InstructionImpl::spvOpTypeQueue;
		outMap[spv::OpTypePipe] = InstructionImpl::spvOpTypePipe;

		outMap[spv::OpConstantTrue] = InstructionImpl::spvOpConstantTrue;
		outMap[spv::OpConstantFalse] = InstructionImpl::spvOpConstantFalse;
		outMap[spv::OpConstant] = InstructionImpl::spvOpConstant;
		outMap[spv::OpConstantComposite] = InstructionImpl::spvOpConstantComposite;
		outMap[spv::OpConstantSampler] = InstructionImpl::spvOpConstantSampler;
		outMap[spv::OpConstantNull] = InstructionImpl::spvOpConstantNull;

		outMap[spv::OpVariable] = InstructionImpl::spvOpVariable;
		outMap[spv::OpLoad] = InstructionImpl::spvOpLoad;
		outMap[spv::OpStore] = InstructionImpl::spvOpStore;
		outMap[spv::OpAccessChain] = InstructionImpl::spvOpAccessChain;

		outMap[spv::OpFunction] = InstructionImpl::spvOpFunction;
		outMap[spv::OpFunctionParameter] = InstructionImpl::spvOpFunctionParameter;
		outMap[spv::OpFunctionEnd] = InstructionImpl::spvOpFunctionEnd;
		outMap[spv::OpFunctionCall] = InstructionImpl::spvOpFunctionCall;

		outMap[spv::OpSampledImage] = InstructionImpl::spvOpSampledImage;
		outMap[spv::OpImageSampleImplicitLod] = InstructionImpl::spvOpImageSampleImplicitLod;
		outMap[spv::OpImageSampleExplicitLod] = InstructionImpl::spvOpImageSampleExplicitLod;
		outMap[spv::OpImageSampleDrefImplicitLod] = InstructionImpl::spvOpImageSampleDrefImplicitLod;

		outMap[spv::OpConvertFToU] = InstructionImpl::spvOpConvertFToU;
		outMap[spv::OpConvertFToS] = InstructionImpl::spvOpConvertFToS;
		outMap[spv::OpConvertSToF] = InstructionImpl::spvOpConvertSToF;
		outMap[spv::OpConvertUToF] = InstructionImpl::spvOpConvertUToF;
		outMap[spv::OpUConvert] = InstructionImpl::spvOpUConvert;
		outMap[spv::OpSConvert] = InstructionImpl::spvOpSConvert;
		outMap[spv::OpFConvert] = InstructionImpl::spvOpFConvert;
		outMap[spv::OpVectorExtractDynamic] = InstructionImpl::spvOpVectorExtractDynamic;
		outMap[spv::OpVectorInsertDynamic] = InstructionImpl::spvOpVectorInsertDynamic;
		outMap[spv::OpVectorShuffle] = InstructionImpl::spvOpVectorShuffle;
		outMap[spv::OpCompositeConstruct] = InstructionImpl::spvOpCompositeConstruct;
		outMap[spv::OpCompositeExtract] = InstructionImpl::spvOpCompositeExtract;
		outMap[spv::OpCompositeInsert] = InstructionImpl::spvOpCompositeInsert;

		outMap[spv::OpSNegate] = InstructionImpl::spvOpSNegate;
		outMap[spv::OpFNegate] = InstructionImpl::spvOpFNegate;
		outMap[spv::OpIAdd] = InstructionImpl::spvOpIAdd;
		outMap[spv::OpFAdd] = InstructionImpl::spvOpFAdd;
		outMap[spv::OpISub] = InstructionImpl::spvOpISub;
		outMap[spv::OpFSub] = InstructionImpl::spvOpFSub;
		outMap[spv::OpIMul] = InstructionImpl::spvOpIMul;
		outMap[spv::OpFMul] = InstructionImpl::spvOpFMul;
		outMap[spv::OpUDiv] = InstructionImpl::spvOpUDiv;
		outMap[spv::OpSDiv] = InstructionImpl::spvOpSDiv;
		outMap[spv::OpFDiv] = InstructionImpl::spvOpFDiv;
		outMap[spv::OpUMod] = InstructionImpl::spvOpUMod;
		outMap[spv::OpSRem] = InstructionImpl::spvOpSRem;
		outMap[spv::OpSMod] = InstructionImpl::spvOpSMod;
		outMap[spv::OpFRem] = InstructionImpl::spvOpFRem;
		outMap[spv::OpFMod] = InstructionImpl::spvOpFMod;
		outMap[spv::OpVectorTimesScalar] = InstructionImpl::spvOpVectorTimesScalar;
		outMap[spv::OpMatrixTimesVector] = InstructionImpl::spvOpMatrixTimesVector;
		outMap[spv::OpDot] = InstructionImpl::spvOpDot;

		outMap[spv::OpAny] = InstructionImpl::spvOpAny;
		outMap[spv::OpAll] = InstructionImpl::spvOpAll;
		outMap[spv::OpIsNan] = InstructionImpl::spvOpIsNan;
		outMap[spv::OpIsInf] = InstructionImpl::spvOpIsInf;
		outMap[spv::OpIsFinite] = InstructionImpl::spvOpIsFinite;
		outMap[spv::OpIsNormal] = InstructionImpl::spvOpIsNormal;
		outMap[spv::OpSignBitSet] = InstructionImpl::spvOpSignBitSet;
		outMap[spv::OpLessOrGreater] = InstructionImpl::spvOpLessOrGreater;
		outMap[spv::OpOrdered] = InstructionImpl::spvOpOrdered;
		outMap[spv::OpUnordered] = InstructionImpl::spvOpUnordered;
		outMap[spv::OpLogicalEqual] = InstructionImpl::spvOpLogicalEqual;
		outMap[spv::OpLogicalNotEqual] = InstructionImpl::spvOpLogicalNotEqual;
		outMap[spv::OpLogicalOr] = InstructionImpl::spvOpLogicalOr;
		outMap[spv::OpLogicalAnd] = InstructionImpl::spvOpLogicalAnd;
		outMap[spv::OpLogicalNot] = InstructionImpl::spvOpLogicalNot;
		outMap[spv::OpSelect] = InstructionImpl::spvOpSelect;
		outMap[spv::OpIEqual] = InstructionImpl::spvOpIEqual;
		outMap[spv::OpINotEqual] = InstructionImpl::spvOpINotEqual;
		outMap[spv::OpUGreaterThan] = InstructionImpl::spvOpUGreaterThan;
		outMap[spv::OpSGreaterThan] = InstructionImpl::spvOpSGreaterThan;
		outMap[spv::OpUGreaterThanEqual] = InstructionImpl::spvOpUGreaterThanEqual;
		outMap[spv::OpSGreaterThanEqual] = InstructionImpl::spvOpSGreaterThanEqual;
		outMap[spv::OpULessThan] = InstructionImpl::spvOpULessThan;
		outMap[spv::OpSLessThan] = InstructionImpl::spvOpSLessThan;
		outMap[spv::OpULessThanEqual] = InstructionImpl::spvOpULessThanEqual;
		outMap[spv::OpSLessThanEqual] = InstructionImpl::spvOpSLessThanEqual;
		outMap[spv::OpFOrdEqual] = InstructionImpl::spvOpFOrdEqual;
		outMap[spv::OpFUnordEqual] = InstructionImpl::spvOpFUnordEqual;
		outMap[spv::OpFOrdNotEqual] = InstructionImpl::spvOpFOrdNotEqual;
		outMap[spv::OpFUnordNotEqual] = InstructionImpl::spvOpFUnordNotEqual;
		outMap[spv::OpFOrdLessThan] = InstructionImpl::spvOpFOrdLessThan;
		outMap[spv::OpFUnordLessThan] = InstructionImpl::spvOpFUnordLessThan;
		outMap[spv::OpFOrdGreaterThan] = InstructionImpl::spvOpFOrdGreaterThan;
		outMap[spv::OpFUnordGreaterThan] = InstructionImpl::spvOpFUnordGreaterThan;
		outMap[spv::OpFOrdLessThanEqual] = InstructionImpl::spvOpFOrdLessThanEqual;
		outMap[spv::OpFUnordLessThanEqual] = InstructionImpl::spvOpFUnordLessThanEqual;
		outMap[spv::OpFOrdGreaterThanEqual] = InstructionImpl::spvOpFOrdGreaterThanEqual;
		outMap[spv::OpFUnordGreaterThanEqual] = InstructionImpl::spvOpFUnordGreaterThanEqual;

		outMap[spv::OpLoopMerge] = InstructionImpl::spvOpLoopMerge;
		outMap[spv::OpSelectionMerge] = InstructionImpl::spvOpSelectionMerge;
		outMap[spv::OpPhi] = InstructionImpl::spvOpPhi;
		outMap[spv::OpLabel] = InstructionImpl::spvOpLabel;
		outMap[spv::OpBranch] = InstructionImpl::spvOpBranch;
		outMap[spv::OpBranchConditional] = InstructionImpl::spvOpBranchConditional;
		outMap[spv::OpSwitch] = InstructionImpl::spvOpSwitch;
		outMap[spv::OpReturn] = InstructionImpl::spvOpReturn;
		outMap[spv::OpReturnValue] = InstructionImpl::spvOpReturnValue;
	}
}

namespace Ifrit::Engine::ShaderVM::Spirv {
	void SpvVMInterpreter::parseRawContext(SpvVMContext* context, SpvVMIntermediateRepresentation* outIr) {
		static std::unordered_map<spv::Op, Impl::SpvInstFunc> lookupTable;
		static int lookupTableInit = 0;
		if (lookupTableInit == 0) {
			Impl::loadLookupTable(lookupTable);
			lookupTableInit = 1;
		}
		// Analyze
		auto analyze = [&](Impl::SpvVMAnalysisPass pass) {
			printf(" ===== Analysis Pass %d =====\n", pass);
			for (int i = 0; i < context->instructions.size(); i++) {
				auto& inst = context->instructions[i];
				auto op = static_cast<spv::Op>(inst.opCode);
				auto it = lookupTable.find(op);
				if (it == lookupTable.end()) {
					printf("Unknown opcode %d\n", op);
					continue;
				}
				auto func = it->second;
				outIr->currentInst = i;
				outIr->currentPass = pass;
				func(context, nullptr, outIr, inst.opWordCounts, inst.opParams.data(), i,pass);
			}
		};
		Impl::InstructionImpl::registerTypes(outIr);
		analyze(Impl::IFSP_VMA_PASS_FIRST);
		analyze(Impl::IFSP_VMA_PASS_SECOND);
		
		// Postproc
		auto cleanUpSymbolPrefix = [&](std::string& p) {
			p.erase(std::remove_if(p.begin(), p.end(), [](char x) { return (x == '@'); }), p.end());
		};
		for (auto& x : outIr->shaderMaps.inputVarSymbols) cleanUpSymbolPrefix(x);
		for (auto& x : outIr->shaderMaps.outputVarSymbols) cleanUpSymbolPrefix(x);
		for (auto& x : outIr->shaderMaps.uniformVarSymbols) cleanUpSymbolPrefix(x);
		cleanUpSymbolPrefix(outIr->shaderMaps.mainFuncSymbol);
		cleanUpSymbolPrefix(outIr->shaderMaps.builtinPositionSymbol);
	}

	void SpvVMInterpreter::exportLlvmIR(SpvVMIntermediateRepresentation* ir, std::string* outLlvmIR) {
		std::string irStr = ir->generatedIR.str();
		*outLlvmIR = irStr;
		printf("===========\n%s\n", irStr.c_str());
		printf("Input Vars:\n");
		for (auto i = 0; auto& x : ir->shaderMaps.inputVarSymbols) {
			printf("[%d] %s (%d)\n", i++, x.c_str(), ir->shaderMaps.inputSize[i]);
		}
		printf("Output Vars:\n");
		for (auto i = 0; auto & x : ir->shaderMaps.outputVarSymbols) {
			printf("[%d] %s (%d)\n", i++, x.c_str(), ir->shaderMaps.outputSize[i]);
		}
		printf("Uniform Vars:\n");
		for (auto i = 0; auto & x : ir->shaderMaps.uniformVarSymbols) {
			printf("[%d,%d] %s (%d)\n", ir->shaderMaps.uniformVarLoc[i].first, ir->shaderMaps.uniformVarLoc[i].second
				, x.c_str(), ir->shaderMaps.uniformSize[i]);
			i++;
		}
		printf("Entry: %s\n", ir->shaderMaps.mainFuncSymbol.c_str());
	}
}