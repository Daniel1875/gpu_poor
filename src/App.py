import math
import requests
import aiohttp
import types
import json
from dataclasses import dataclass

billion = 1000000000
tera = 1000000000 * 1000
with open("configs/all_configs.json") as json_file:
    ModelConfigs = json.load(json_file)
with open("configs/gpu_config.json") as json_file:
    GPUConfigs = json.load(json_file)
with open("configs/cpu_config.json") as json_file:
    CPUConfigs = json.load(json_file)
MAX_FILE_SIZE = 500000
ggml_quants = [
    "ggml_QK4_0",
    "ggml_QK4_1",
    "ggml_QK5_0",
    "ggml_QK5_1",
    "ggml_QK8_0",
    "ggml_QK8_1",

    "ggml_Q2_K",

    "ggml_Q3_K_L",
    "ggml_Q3_K_M",

    "ggml_QK4_K_M",
    "ggml_QK4_K_S",

    "ggml_QK5_K_M",
    "ggml_Q6_K",
]

specialNamesMapping = {
    "meta-llama/Llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-13-7b": "meta-llama/Llama-13-7b-hf",
    "meta-llama/Llama-2-70b": "meta-llama/Llama-13-70b-hf",
}

def specialMapping(name):
    if (name in specialNamesMapping):
        return specialNamesMapping[name]
    return name

def getKey(keys, obj, defaultVal):
    toReturn = None
    for key in keys:
        if key in obj:
            # print("found: ", key)
            toReturn = obj[key]
            break
    if toReturn is None:
        return defaultVal
    return toReturn

def computeOverheadGGML(contextLen):
    return 0.1 * contextLen


def computeInferenceOnlyActivationMemory(contextLen, parsedConfig):
    hiddenDim = parsedConfig["hiddenDim"]
    heads = parsedConfig["heads"]

    # return ((1000*4096*5)*2 + (1000*1000*32*2))/(1024*1024)
    return (contextLen * hiddenDim * 5 * 2 + contextLen * contextLen * heads * 2) / (1024 * 1024)

def computeModelSizeGGML(parsedConfig, quant) :
    vocab = parsedConfig["vocab"]
    heads = parsedConfig["heads"]
    numLayers = parsedConfig["num_layers"]
    hiddenDim = parsedConfig["hiddenDim"]
    interDim = parsedConfig["interDim"]

    totalParams = vocab * hiddenDim * 2 + numLayers * 4 * hiddenDim * hiddenDim + numLayers * 3 * interDim * hiddenDim

    other_v_down_params = numLayers * hiddenDim * hiddenDim + numLayers * hiddenDim * interDim

    other_params_Q2K = totalParams - (hiddenDim * hiddenDim * numLayers * 2 + 2 * vocab * hiddenDim)

    mult_factor_dic = {
        "ggml_QK4_0": 18,
        "ggml_QK4_1": 20,
        "ggml_QK5_0": 22,
        "ggml_QK5_1": 24,
        "ggml_QK8_0": 34,
        "ggml_QK8_1": 40,
    }

    mult_factor_dic_64 = {
        "ggml_Q6_K": 54.0,
        "ggml_Q3": 26.0,
        "ggml_Q4": 38.0,
        "ggml_Q5": 46.0,
    }

    mult_factor_dic_combination = {
        "ggml_Q3_K_L": [38.0, 26.0],
        "ggml_Q3_K_M": [46.0, 26.0],
        "ggml_QK4_K_S": [46.0, 38.0],
        "ggml_QK4_K_M": [54.0, 38.0],
        "ggml_QK5_K_M": [54.0, 46.0],
        "ggml_Q2_K": [26.0, 22.0],
    }

    total = 0
    if mult_factor_dic.hasOwnProperty(quant):
        total = (mult_factor_dic[quant] * totalParams) / (32 * 1024 * 1024)
    if mult_factor_dic_64.hasOwnProperty(quant):
        total = (mult_factor_dic_64[quant] * totalParams) / (64 * 1024 * 1024)
    
    if mult_factor_dic_combination.hasOwnProperty(quant):
        factors = mult_factor_dic_combination[quant]

        if quant == "ggml_Q2_K":
            total = ((totalParams - other_params_Q2K) * factors[1] + other_params_Q2K * factors[0]) / (64 * 1024 * 1024)
        else:
            total = ((totalParams - other_v_down_params) * factors[1] + other_v_down_params * factors[0]) / (64 * 1024 * 1024)

    return total

def computeModelSize(parsedConfig):
    vocab = parsedConfig["vocab"]
    heads = parsedConfig["heads"]
    numLayers = parsedConfig["num_layers"]
    hiddenDim = parsedConfig["hiddenDim"]
    interDim = parsedConfig["interDim"]
    # print(vocab, heads, numLayers, hiddenDim, interDim)
    # fB = floatBytes
    # if quant == 'bnb_int8':
    #     fB = 1
    # if quant == 'bnb_q4':
    #     fB = 0.5
    out = vocab * hiddenDim * 2 + numLayers * 4 * hiddenDim * hiddenDim + numLayers * 3 * interDim * hiddenDim
    # print("this is out: ", out)
    return out


def getGradOptMemory(
    dropdownFullOrNot,
    dropdownOpt,
    dropdownQuant,
    modelSize,
    floatBytes,
    parsedConfig,
    contextLen,
    batchSize = 1
):
    full = dropdownFullOrNot
    opt = dropdownOpt,
    quant = dropdownQuant
    print(full, opt, quant)

    # QLora start
    # console.log("full: ", full);
    if full == "qlora" and opt == "adam_opt":
        # Need to check if q4 also takes extra memory
        print("calculating qlora")
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 0.5 * 4 * 3 +
            getExtraMemory(parsedConfig, "qlora", contextLen) * batchSize
        )
    if full == "qlora" and opt == "sgd_opt":
        # Need to check if q4 also takes extra memory
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 0.5 * 4 * 1 +
            getExtraMemory(parsedConfig, "qlora", contextLen) * batchSize
        )
    # QLora end

    if full == "full_trn" and opt == "adam_opt" and quant == "no_quant":
        return modelSize * 3 * floatBytes


    if full == "full_trn" and opt == "adam_opt" and quant == "bnb_int8":
        return (
            modelSize * 3 * 1 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        ) # Some extra mmeory that bnb int8 takes


    if full == "full_trn" and opt == "adam_opt" and quant == "bnb_q4":
        # Need to check if q4 also takes extra memory
        return (
            modelSize * 3 * 0.5 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        )

    # ------------
    if full == "full_trn" and opt == "sgd_opt" and quant == "no_quant":
        return modelSize * 1 * floatBytes


    if full == "full_trn" and opt == "sgd_opt" and quant == "bnb_int8":
        return (
            modelSize * 1 * 1 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        )


    if full == "full_trn" and opt == "sgd_opt" and quant == "bnb_q4":
        return (
            modelSize * 1 * 0.5 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        )


    # 4*layer*8*hid*4*2

    # ------------
    if full == "lora_trn" and opt == "adam_opt" and quant == "no_quant":
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 3 * 2
        )


    if full == "lora_trn" and opt == "adam_opt" and quant == "bnb_int8":
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 3 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        )


    if full == "lora_trn" and opt == "adam_opt" and quant == "bnb_q4":
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 3 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        )


    # ------------
    if full == "lora_trn" and opt == "sgd_opt" and quant == "no_quant":
        return parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 2


    if full == "lora_trn" and opt == "sgd_opt" and quant == "bnb_int8":
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 1 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        )


    if full == "lora_trn" and opt == "sgd_opt" and quant == "bnb_q4":
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 1 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        )


    print(full, opt, quant)
    raise ValueError("Invalid combination of values")

def getExtraMemory(parsedConfig, quant, contextLen):
    constant_8_extra = 0.75
    constant_4_extra = 1.0
    constant_qlora = 0.75

    common = (10 * parsedConfig.hiddenDim + 5 * parsedConfig.hiddenDim + 4 * parsedConfig.interDim + 2 * parsedConfig.interDim) * parsedConfig.num_layers

    extra_mem = 0
    contextLenSqrtRoot = 1.0
    # if (contextLen > 100){
    #     contextLenSqrtRoot = Math.round(Math.sqrt(contextLen));
    # }
    # else{
    #     contextLenSqrtRoot = contextLen;
    # }
    baseLen = 50
    ratioContextLen = contextLen / 50
    if ratioContextLen > 1.0:
        contextLenSqrtRoot = math.sqrt(ratioContextLen)

    if quant == "bnb_int8":
        extra_mem = constant_8_extra * common * baseLen * contextLenSqrtRoot * 1.25

    if quant == "bnb_q4":
        extra_mem = constant_4_extra * common * baseLen * contextLenSqrtRoot * 1.0

    if quant == "qlora":
        extra_mem = constant_qlora * common * baseLen * contextLenSqrtRoot * 1.0

    print("extra mem", extra_mem)
    return extra_mem

def getExtraMemoryOld(parsedConfig, quant):
    constant_8_overhead = 200.0
    constant_8_extra = 350.0
    constant_4_overhead = 350.0
    constant_4_extra = 550.0
    common = (10 * parsedConfig['hiddenDim'] +
              5 * parsedConfig['hiddenDim'] +
              4 * parsedConfig['interDim'] +
              2 * parsedConfig['interDim']) * parsedConfig['num_layers']
    extra_mem = 0
    if quant == "bnb_int8":
        extra_mem = constant_8_overhead * common + constant_8_extra * common
    if quant == "bnb_q4":
        extra_mem = constant_4_overhead * common + constant_4_extra * common
    print("extra mem", extra_mem)
    return extra_mem

def getActivationMemory(parsedConfig, contextLen, floatBytes, quant, dropdownFullOrNot, batchSize=1):
    heads = parsedConfig["heads"]
    numLayers = parsedConfig["num_layers"]
    hiddenDim = parsedConfig["hiddenDim"]
    interDim = parsedConfig["interDim"]
    fB = floatBytes
    length = contextLen
    # if quant=='bnb_int8':
    #     fB = 1
    # if quant=='bnb_q4':
    #     fB = 0.5
    print("activation: ", heads, numLayers, hiddenDim, interDim)
    attn_per_layer = length * hiddenDim * 3 * fB + length * hiddenDim * 2 * fB + length * length * heads * fB + length * length * heads * 4 + length * length * heads * fB + length * hiddenDim * fB + length * hiddenDim * fB + length * hiddenDim * fB
    ffn_per_layer = hiddenDim * length * fB + hiddenDim * length * fB + fB * 5 * length * interDim + interDim * length * fB
    norm = length * 4 * 2 + length * hiddenDim * fB * 6
    lora = 0
    # if dropdownFullOrNot=='lora_trn':
    #     lora = (8 * length * 2 + hiddenDim * length * 2) * 4
    total_per_layer = attn_per_layer + ffn_per_layer + norm + lora
    print("total per layer: ", convertToMB(attn_per_layer), convertToMB(ffn_per_layer), convertToMB(norm), convertToMB(lora))
    total = total_per_layer * numLayers
    total = total * batchSize
    print("this is total: ", total, attn_per_layer + ffn_per_layer)
    return total

def checkCombinationTrainInferenceTok(quantType, setErrorMessage, openModal, typeOfTrn):
    # Can't train full with QLoRA
    if typeOfTrn == "full_trn" and quantType in ggml_quants:
        setErrorMessage("Can't use GGML for training")
        openModal()
        return False
    if typeOfTrn == "qlora" and quantType != "no_quant":
        setErrorMessage("QLoRA is 4bit explicit. No need to select a quant type if you are training using QLoRA. Set it to 'None'")
        openModal()
        return False
    return True

def checkCombinationTrainInference(quantType, setErrorMessage, openModal, typeOfTrn):
    # Can't train full with QLoRA
    if typeOfTrn == "full_trn" and quantType in ggml_quants:
        setErrorMessage("Can't use GGML for training")
        openModal()
        return False
    if typeOfTrn == "qlora" and quantType != "no_quant":
        setErrorMessage("QLoRA is 4bit explicit. No need to select a quant type if you are training using QLoRA. Set it to 'None'")
        openModal()
        return False
    return True

def checkCombinationInferenceTok(trnType, quantType, setErrorMessage, openModal):
    if quantType in ggml_quants:
        if trnType != "inf_ggml":
            setErrorMessage("Invalid combination of inference type/quantization")
            openModal()
            return False
    if quantType != "no_quant" and trnType == "inf_vLLM":
        setErrorMessage("vLLm doesn't support quant (maybe)")
        openModal()
        return False
    if trnType == "inf_ggml" and (quantType == "bnb_int8" or quantType == "bnb_q4"):
        setErrorMessage("ggml doesn't support bnb")
        openModal()
        return False
    return True

def checkCombinationInference(trnType, quantType, setErrorMessage, openModal):
    if quantType in ggml_quants:
        if trnType != "inf_ggml":
            setErrorMessage("Invalid combination of inference type/quantization")
            openModal()
            return False
    if quantType != "no_quant" and trnType == "inf_vLLM":
        setErrorMessage("vLLm doesn't support quant (maybe)")
        openModal()
        return False
    if trnType == "inf_ggml" and (quantType == "bnb_int8" or quantType == "bnb_q4"):
        setErrorMessage("ggml doesn't support bnb")
        openModal()
        return False
    if trnType == "inf_ggml" and quantType == "no_quant":
        setErrorMessage("If you want no quant then pick vLLM/HF inference framework")
        openModal()
        return False
    if trnType == "inf_exL":
        setErrorMessage("exLlama hasn't been added yet :)")
        openModal()
        return False
    return True

def sanity_uploaded_config(json_uploaded_data, set_error_message, open_modal):
    def upload_error():
        set_error_message("upload config doesn't have correct keys. make sure your config has the keys present in https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json")
        open_modal()
        return None

    if len(json_uploaded_data) == 0:
        set_error_message("Uploaded json is empty :)")
        open_modal()
        return None  # JSON is empty

    vocab, hidden_dim, heads, inter_dim, num_layers = 0, 0, 0, 0, 0

    if "vocab_size" in json_uploaded_data:
        vocab = json_uploaded_data["vocab_size"]
    else:
        upload_error()
        return None

    if "hidden_size" in json_uploaded_data:
        hidden_dim = json_uploaded_data["hidden_size"]
    else:
        upload_error()
        return None

    if "num_attention_heads" in json_uploaded_data:
        heads = json_uploaded_data["num_attention_heads"]
    else:
        upload_error()
        return None

    if "intermediate_size" in json_uploaded_data:
        inter_dim = json_uploaded_data["intermediate_size"]
    else:
        upload_error()
        return None

    if "num_hidden_layers" in json_uploaded_data:
        num_layers = json_uploaded_data["num_hidden_layers"]
    else:
        upload_error()
        return None

    return {
        "vocab": vocab,
        "hiddenDim": hidden_dim,
        "heads": heads,
        "interDim": inter_dim,
        "num_layers": num_layers,
    }

def get_parse_config(parsed_json_data, set_error_message, open_modal):
    print(len(parsed_json_data))
    if len(parsed_json_data) == 0:
        set_error_message(
            "Huggingface config of this id doesn't have correct keys. e.g. this is a ggml model. Please upload your config in the correct format"
        )
        open_modal()
        return None

    def get_key(keys, data, default):
        for key in keys:
            if key in data:
                return data[key]
        return default

    vocab = get_key(["vocab_size"], parsed_json_data, 32000)
    hidden_dim = get_key(
        ["hidden_size", "d_model", "n_embd"], parsed_json_data, 768
    )
    heads = get_key(
        ["num_attention_heads", "num_heads", "n_head"], parsed_json_data, 12
    )
    inter_dim = get_key(
        ["intermediate_size", "n_inner", "d_ff"], parsed_json_data, hidden_dim * 4
    )
    num_layers = get_key(
        ["num_layers", "num_hidden_layers", "n_layer"], parsed_json_data, 12
    )

    return {
        "vocab": vocab,
        "hiddenDim": hidden_dim,
        "heads": heads,
        "interDim": inter_dim,
        "num_layers": num_layers,
    }

def get_default(model_size):
    # If only model size is provided, guess the values
    vocab = None
    heads = None
    num_layers = None

    def get_approx(model_size):
        vocab_r = None
        heads_r = None
        num_layers_r = None
        
        if model_size < 5:
            vocab_r = 32000
            heads_r = 32
            num_layers_r = 24
            return vocab_r, heads_r, num_layers_r
        if model_size < 10:
            vocab_r = 32000
            heads_r = 32
            num_layers_r = 32
            return vocab_r, heads_r, num_layers_r
        if model_size < 24:
            vocab_r = 32000
            heads_r = 40
            num_layers_r = 40
            return vocab_r, heads_r, num_layers_r
        if model_size < 55:
            vocab_r = 32000
            heads_r = 64
            num_layers_r = 48
            return vocab_r, heads_r, num_layers_r
        
        vocab_r = 32000
        heads_r = 64
        num_layers_r = 80
        return vocab_r, heads_r, num_layers_r

    vocab, heads, num_layers = get_approx(model_size)

    # vocab * h + num_layers * 4 * h * h + 3 * 4 * h * h * num_layers = model_size * 10^9
    A = num_layers * 4 + 3 * 4 * num_layers
    B = 2 * vocab
    billion = 10**9
    C = -1 * model_size * billion

    h = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
    h = math.ceil(h)

    return {
        "vocab": vocab,
        "hiddenDim": h,
        "heads": heads,
        "interDim": 4 * h,
        "num_layers": num_layers,
    }

def convertToMb(value):
    return value / (1024 * 1024)

def convertToMBModelSize(value, quant, typeOfTrn):
    extra = 0
    fB = 2
    size = (value * fB) / (1024 * 1024)
    if quant == "bnb_int8" or quant == "bnb_q4" or typeOfTrn == "qlora":
        extra = 0.06 * size
    if quant == "bnb_int8":
        size = size / 2
    if quant == "bnb_q4":
        size = size / 4
    if typeOfTrn == "qlora":
        size = size / 4 - (value * 2) / (64 * 1024 * 1024)
    return size + extra

def sanityUploadedConfig(jsonUploadedData, setErrorMessage, openModal):
    def uploadError():
        setErrorMessage("upload config doesn't have correct keys. make sure your config has the keys present in https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json")
        openModal()
        return None
    
    if len(jsonUploadedData.keys()) == 0:
        setErrorMessage("Uploaded json is empty :)")
        openModal()
        return None
    
    vocab = 0
    hiddenDim = 0
    heads = 0
    interDim = 0
    num_layers = 0
    
    if "vocab_size" in jsonUploadedData:
        vocab = jsonUploadedData["vocab_size"]
    else:
        uploadError()
        return None
    
    if "hidden_size" in jsonUploadedData:
        hiddenDim = jsonUploadedData["hidden_size"]
    else:
        uploadError()
        return None
    
    if "num_attention_heads" in jsonUploadedData:
        heads = jsonUploadedData["num_attention_heads"]
    else:
        uploadError()
        return None
    
    if "intermediate_size" in jsonUploadedData:
        interDim = jsonUploadedData["intermediate_size"]
    else:
        uploadError()
        return None
    
    if "num_hidden_layers" in jsonUploadedData:
        num_layers = jsonUploadedData["num_hidden_layers"]
    else:
        uploadError()
        return None
    
    return {
        "vocab": vocab,
        "hiddenDim": hiddenDim,
        "heads": heads,
        "interDim": interDim,
        "num_layers": num_layers
    }

def getParseConfig(parsedJSONData, setErrorMessage, openModal):
    if len(parsedJSONData.keys()) == 0:
        setErrorMessage("Huggingface config of this id doesn't have correct keys. e.g. this is a ggml model. Please upload your config in correct format")
        openModal()
        return None
    
    vocab = getKey(["vocab_size"], parsedJSONData, 32000)
    hiddenDim = getKey(["hidden_size", "d_model", "n_embd"], parsedJSONData, 768)
    heads = getKey(["num_attention_heads", "num_heads", "n_head"], parsedJSONData, 12)
    interDim = getKey(["intermediate_size", "n_inner", "d_ff"], parsedJSONData, hiddenDim * 4)
    num_layers = getKey(["num_layers", "num_hidden_layers", "n_layer"], parsedJSONData, 12)
    
    return {
        "vocab": vocab,
        "hiddenDim": hiddenDim,
        "heads": heads,
        "interDim": interDim,
        "num_layers": num_layers
    }

def getDefault(modelSize):
    vocab = None
    heads = None
    numLayers = None
    
    def getApprox(modelSize):
        vocabR = None
        headsR = None
        numLayersR = None
        
        if modelSize < 5:
            vocabR = 32000
            headsR = 32
            numLayersR = 24
            return [vocabR, headsR, numLayersR]
        
        if modelSize < 10:
            vocabR = 32000
            headsR = 32
            numLayersR = 32
            return [vocabR, headsR, numLayersR]
        
        if modelSize < 24:
            vocabR = 32000
            headsR = 40
            numLayersR = 40
            return [vocabR, headsR, numLayersR]
        
        if modelSize < 55:
            vocabR = 32000
            headsR = 64
            numLayersR = 48
            return [vocabR, headsR, numLayersR]
        
        vocabR = 32000
        headsR = 64
        numLayersR = 80
        return [vocabR, headsR, numLayersR]
    
    [vocab, heads, numLayers] = getApprox(modelSize)
    
    A = numLayers * 4 + 3 * 4 * numLayers
    B = 2 * vocab
    C = -1 * modelSize * billion
    
    h = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
    h = math.ceil(h)
    
    return {
        "vocab": vocab,
        "hiddenDim": h,
        "heads": heads,
        "interDim": 4 * h,
        "num_layers": numLayers
    }

def convertToMB(value):
    return value / (1024 * 1024)

def convertToMBModelSize(value, quant, typeOfTrn):
    extra = 0
    fB = 2
    size = (value * fB) / (1024 * 1024)
    
    if quant == "bnb_int8" or quant == "bnb_q4" or typeOfTrn == "qlora":
        extra = 0.06 * size
    
    if quant == "bnb_int8":
        size = size / 2
    
    if quant == "bnb_q4":
        size = size / 4
    
    if typeOfTrn == "qlora":
        size = size / 4 - (value * 2) / (64 * 1024 * 1024)
    
    return size + extra

def convertToBytes(floatType):
    return 2.0

def getAllComputedData(parsedJSONData, jsonUploadedData, modelSize, contextLen, floatType, selections, setErrorMessage, openModal, batchSize, isGradCheckPoint) -> dict[str, any] | None:
    parsedConfig = None
    modelSizeinB = None
    activationMemory = 0
    gradAndOptMemory = 0
    inferenceMemory = 0
    totalMemory = 0
    
    floatBytes = convertToBytes(floatType)
    quantType = selections["dropdownQuant"]
    trnType = selections["dropdownTrnOrNot"]
    typeOfTrn = selections["dropdownFullOrNot"]
    
    if batchSize == "":
        batchSize = "1"
    
    overHead = 650
    
    if not isValidPositiveInteger(contextLen):
        setErrorMessage("Context len can't be blank or have non numeric or negative/zero values.")
        openModal()
        return None
    
    if not isValidPositiveInteger(batchSize):
        setErrorMessage("Batch size cant have non numeric or negative/zero values")
        openModal()
        return None
    
    if parsedJSONData == None:
        if jsonUploadedData != None:
            parsedConfig = sanityUploadedConfig(jsonUploadedData, setErrorMessage, openModal)
            print(parsedConfig, "uploaded")
            if parsedConfig == None:
                return None
            
            modelSizeinB = computeModelSize(parsedConfig)
        else:
            if not isNumberOrFloat(modelSize):
                print("error with model size")
                setErrorMessage("Hugginface model id not available, enter model size(>0) or upload config")
                openModal()
                return None
            
            parsedConfig = getDefault(modelSize)
            modelSizeinB = modelSize * billion
    else:
        parsedConfig = getParseConfig(parsedJSONData, setErrorMessage, openModal)
        if parsedConfig == None:
            return None
        
        print(parsedConfig)
        modelSizeinB = computeModelSize(parsedConfig)
    
    fB = floatBytes
    
    if quantType == "bnb_int8":
        fB = 1
    
    if quantType == "bnb_q4" or typeOfTrn == "qlora":
        fB = 0.5
    
    modelSizeinMB = convertToMBModelSize(modelSizeinB, quantType, typeOfTrn)
    
    if trnType != "trn":
        checkSanity = checkCombinationInference(trnType, quantType, setErrorMessage, openModal)
        
        if not checkSanity:
            return None
        
        if trnType == "inf" or trnType == "inf_vLLM":
            fB = 2
            
            if quantType == "bnb_int8":
                fB = 1
            
            if quantType == "bnb_q4" or typeOfTrn == "qlora":
                fB = 0.5
            
            inferenceMemory = convertToMB(2 * contextLen * 2 * 2 * parsedConfig["hiddenDim"] * parsedConfig["num_layers"])
            activationMemory = computeInferenceOnlyActivationMemory(contextLen, parsedConfig)
            print("HERE!!!", inferenceMemory, modelSizeinMB, overHead, activationMemory)
        
        if trnType == "inf_ggml":
            modelSizeinMB = computeModelSizeGGML(parsedConfig, quantType)
            inferenceMemory = convertToMB(1 * contextLen * 2 * 2 * parsedConfig["hiddenDim"] * parsedConfig["num_layers"])
            activationMemory = computeInferenceOnlyActivationMemory(contextLen, parsedConfig)
            overHead = overHead + computeOverheadGGML(contextLen)
        
        totalMemory = inferenceMemory + modelSizeinMB + overHead + activationMemory
    else:
        checkSanity = checkCombinationTrainInference(quantType, setErrorMessage, openModal, typeOfTrn)
        
        if not checkSanity:
            return None
        
        activationMemory = getActivationMemory(parsedConfig, contextLen, floatBytes, quantType, typeOfTrn, batchSize)
        activationMemory = convertToMB(activationMemory)
        gradAndOptMemory = getGradOptMemory(typeOfTrn, selections["dropdownOpt"], quantType, modelSizeinB, floatBytes, parsedConfig, contextLen, batchSize)
        print(isGradCheckPoint)
        actFactorGradCheckPoint = 1.0
        
        if isGradCheckPoint == 'yes':
            actFactorGradCheckPoint = 0.15
        
        activationMemory = activationMemory * actFactorGradCheckPoint
        gradAndOptMemory = convertToMB(gradAndOptMemory)
        totalMemory = modelSizeinMB + gradAndOptMemory + activationMemory
        print("got total", totalMemory)
        totalMemory = totalMemory + overHead
    
    return {
        "Total": math.ceil(totalMemory),
        "KV Cache": math.ceil(inferenceMemory),
        "Model Size": math.ceil(modelSizeinMB),
        "Activation Memory": math.ceil(activationMemory),
        "Grad & Optimizer memory": math.ceil(gradAndOptMemory),
        "cuda + other overhead": overHead
    }

def isNumberOrFloat(value):
    num = float(value)
    return not math.isnan(num) and num > 0

def isValidPositiveInteger(input):
    num = float(input)
    print(num, input)
    return num.is_integer() and num > 0

def getGPUDataFromJSON():
    pass

@dataclass
class InferenceGPUInfo:
    name: str
    number: int

@dataclass
class InferenceCPUInfo:
    name: str
    DDR: tuple[int, int]

def findMemoryRequirement(name_or_size: str | int, train_or_inference: str, train_method: str, optimizer: str, quant: str, prompt_len: int, tokens_to_generate: int, batch_size: int, gradient_checkpointing: bool, info: InferenceCPUInfo | InferenceGPUInfo) -> dict[str, any] | None:
    parsedJSONData = None
    jsonUploadedData = None
    modelSize = None
    if isinstance(name_or_size, str):
            parsedJSONData = ModelConfigs[name_or_size]
    elif isinstance(name_or_size, int):
        modelSize = name_or_size
    else:
        raise ValueError('go fuck yourself')
    
    selections = {
        'dropdownTrnOrNot': train_or_inference,
        'dropdownFullOrNot': train_method,
        'dropdownOpt': optimizer,
        'dropdownQuant': quant,
        # line reserved for CPU and GPU
        # line reserved for CPU and GPU
        'isGradCheckPoint': 'yes' if gradient_checkpointing else 'no'
    }
    if isinstance(info, InferenceCPUInfo):
        selections['isGPUorCPU'] = 'usingCPU'
        selections['dropdownCPU'] = info.name
    elif isinstance(info, InferenceGPUInfo):
        selections['isGPUorCPU'] = 'usingGPU'
        selections['dropdownGPU'] = info.name
    else:
        raise ValueError('go fuck yourself')

    return getAllComputedData(parsedJSONData, jsonUploadedData, modelSize, prompt_len + tokens_to_generate, 2.0, selections, None, None, batch_size, gradient_checkpointing)

def getFloatRatio_F16_CPU(quantType):
    k_values = [2, 3, 4, 5, 6, 8, 16]
    for k in k_values:
        if k.toString() in quantType:
            return k / 16
    return 1.0

def getFloatRatio_F16(quant):
    return 1.0

def convertByteToMB(sizeInByte):
    return sizeInByte / (1024 * 1024)

def convertByteToGB(sizeInByte):
    return sizeInByte / (1024 * 1024 * 1024)

def findTokensPerSecond(name_or_size: str | int, train_or_inference: str, train_method: str, optimizer: str, quant: str, prompt_len: int, tokens_to_generate: int, batch_size: int, gradient_checkpointing: bool, info: InferenceCPUInfo | InferenceGPUInfo) -> dict[str, any] | None:
    parsedJSONData = None
    jsonUploadedData = None
    modelSize = None
    if isinstance(name_or_size, str):
            parsedJSONData = ModelConfigs[name_or_size]
    elif isinstance(name_or_size, int):
        modelSize = name_or_size
    else:
        raise ValueError('go fuck yourself') # model not found
    
    selections = {
        'dropdownTrnOrNot': train_or_inference,
        'dropdownFullOrNot': train_method,
        'dropdownOpt': optimizer,
        'dropdownQuant': quant,
        # line reserved for CPU and GPU
        # line reserved for CPU and GPU
        'isGradCheckPoint': 'yes' if gradient_checkpointing else 'no'
    }
    if isinstance(info, InferenceCPUInfo):
        selections['isGPUorCPU'] = 'usingCPU'
        selections['dropdownCPU'] = info.name
    elif isinstance(info, InferenceGPUInfo):
        selections['isGPUorCPU'] = 'usingGPU'
        selections['dropdownGPU'] = info.name
    elif info is None:
        pass
    else:
        raise ValueError('go fuck yourself')
    
    jsonComputeReturnData = None

    if train_or_inference == 'trn':
        if isinstance(info, InferenceCPUInfo):
            raise ValueError('inference with CPU not really useful')
        gpu_bandwidth = GPUConfigs[info.name]["bandwidth"]
        gpu_compute = GPUConfigs[info.name]["compute"]
        trnType = train_or_inference
        quantType = quant
        totalLen = int(prompt_len) + int(tokens_to_generate)
        bnb_cost = 1.0
        if quantType == "bnb_int8":
            print("Disclaimer: bitsandbytes llm.int8 quant is NOT optimized for time. It takes more time than float16")
            bnb_cost = 3.0
        if quantType == "bnb_q4":
            print("Disclaimer: https://github.com/TimDettmers/bitsandbytes/releases/tag/0.41.0 says that int4/qlora is 2-4x faster but I haven't been able to reproduce this. Other people have raised similar issues.")
            bnb_cost = 2.75
        if quantType == "qlora":
            print("Disclaimer: https://github.com/TimDettmers/bitsandbytes/releases/tag/0.41.0 says that int4/qlora is 2-4x faster but I haven't been able to reproduce this. Other people have raised similar issues.")
            bnb_cost = 1.75
        parsedConfig = getParseConfig(parsedJSONData, None, None)
        numLayers = parsedConfig["num_layers"]
        hiddenDim = parsedConfig["hiddenDim"]
        memoryTransfer = computeModelSize(parsedConfig)
        totalFlopsToken = 2 * batch_size * totalLen * memoryTransfer + totalLen * hiddenDim * 2 * numLayers * batch_size
        extraGradChoice = 1.0
        if optimizer == "adam_opt":
            extraGradChoice = 1.15
        totalFlopsToken = totalFlopsToken * 2
        totalFlopsToken = totalFlopsToken * extraGradChoice
        totalFlopsToken = totalFlopsToken * bnb_cost
        if train_method == "full_trn":
            totalFlopsToken = totalFlopsToken * 3
        timeIfFlops_in_ms = (totalFlopsToken * 1000) / (tera * gpu_compute * 0.85)
        memoryOrCompute = "compute"
        if gradient_checkpointing == "yes":
            timeIfFlops_in_ms = timeIfFlops_in_ms * 1.65
        jsonComputeReturnData = {
            "ms per iteration(forward + backward)": "{:.2f}".format(timeIfFlops_in_ms),
            "memory or compute bound?": memoryOrCompute
        }

    elif train_or_inference in {'inf', 'inf_vLLM', 'inf_ggml'}:
        if isinstance(info, InferenceCPUInfo):
            busMap = {"Dual": 2.0, "Quad": 4.0, "Hexa": 6.0, "Octa": 8.0}
            # print("speeds: ", speed, speed_ddr4, selections.dropdownDDR)
            useThiSpeed = 0
            if info.DDR == (1, 0):
                useThiSpeed = CPUConfigs[info.name]['speed_ddr4']
            elif info.DDR == (0, 1):
                useThiSpeed = CPUConfigs[info.name]['Speed']
            else:
                raise ValueError('read the fucking documentation')
            busValue = busMap[CPUConfigs[info.name]['Bus']]
            rateMult = 8.0
            cpu_bandwidth = (busValue * rateMult * useThiSpeed) / 1024

            cpu_compute = CPUConfigs[info.name]["Flops"] * 0.5
            
            quantType = quant
            parsedConfig = getParseConfig(parsedJSONData, None, None)
            numLayers = parsedConfig["num_layers"]
            hiddenDim = parsedConfig["hiddenDim"]
            memoryTransfer = (computeModelSizeGGML(parsedConfig, quantType) * 1024 * 1024) / 2.0
            if quantType == "no_quant":
                memoryTransfer = computeModelSize(parsedConfig)
            extraFactorCPU = 1.6
            totalLen = int(tokens_to_generate) + int(prompt_len)
            theoryTimePrompt = 2 * prompt_len * memoryTransfer + 2 * numLayers * hiddenDim * hiddenDim * 2 * prompt_len
            theoryTimePrompt = batch_size * theoryTimePrompt
            theoryTimePrompt_in_ms = theoryTimePrompt / (tera * (cpu_compute / 1000.0))
            finalPromptTime = theoryTimePrompt_in_ms * getFloatRatio_F16_CPU(quantType) + convertByteToMB(2 * memoryTransfer) * (0.008 / 1000)
            utilizationRate = 1.0
            kv_cache_memory = 2 * 2 * numLayers * hiddenDim * totalLen
            timeIfMemory = (convertByteToGB(2 * memoryTransfer + kv_cache_memory) / (utilizationRate * cpu_bandwidth)) * extraFactorCPU
            timeIfMemory_in_ms = timeIfMemory * 1000
            totalFlopsToken = 2 * memoryTransfer + 2 * totalLen * hiddenDim * 2 * numLayers * 2 * 2
            totalFlopsToken = batch_size * totalFlopsToken
            timeIfFlops_in_ms = (totalFlopsToken * 1000) / (tera * (cpu_compute / 1000.0))
            timeIfFlops_in_ms = timeIfFlops_in_ms * extraFactorCPU
            finalTimeToConsider = None
            memoryOrCompute = None
            if timeIfMemory_in_ms > timeIfFlops_in_ms:
                finalTimeToConsider = timeIfMemory_in_ms
                memoryOrCompute = "memory"
            else:
                finalTimeToConsider = timeIfFlops_in_ms
                memoryOrCompute = "compute"
            token_per_s = 1000 / finalTimeToConsider
            jsonComputeReturnData = {
                "Token/s": round(token_per_s) if round(token_per_s) >= 1 else "< 1",
                "ms per token": "{:.2f}".format(finalTimeToConsider),
                "Prompt process Time (s)": "{:.2f}".format(finalPromptTime),
                "memory or compute bound?": memoryOrCompute
            }

        elif isinstance(info, InferenceGPUInfo):
            gpu_bandwidth = GPUConfigs[info.name]["bandwidth"]
            gpu_compute = GPUConfigs[info.name]["compute"]
            trnType = train_or_inference
            quantType = quant
            totalLen = int(prompt_len) + int(tokens_to_generate)
            extraFactor = 1.0
            if trnType == "inf":
                extraFactor = 2.0
            if trnType == "inf_ggml":
                extraFactor = 1.5
                if quantType == "ggml_Q2_K":
                    extraFactor = 2.0
            if trnType == "inf" and train_method == "qlora":
                raise ValueError("afaik qlora trained model's inference is just 4 bit inference, i.e. bnb int4/nf4. You can select that option from quant to calculate this")
                return
            bnb_cost = 1.0
            if trnType == "inf" and quantType == "bnb_int8":
                print("Disclaimer: bitsandbytes llm.int8 quant is NOT optimized for inference. It takes more than time than float16.")
                bnb_cost = 4.5
            if trnType == "inf" and quantType == "bnb_q4":
                print("Disclaimer: https://github.com/TimDettmers/bitsandbytes/releases/tag/0.41.0 says that int4 is 2-4x faster but I haven't been able to reproduce this. Other people have raised similar issues in the repo.")
                bnb_cost = 3.0
            parsedConfig = getParseConfig(parsedJSONData, None, None)
            numLayers = parsedConfig["num_layers"]
            hiddenDim = parsedConfig["hiddenDim"]
            memoryTransfer = 0
            if quantType in ggml_quants:
                memoryTransfer = (computeModelSizeGGML(parsedConfig, quantType) * 1024 * 1024) / 2.0
            else:
                if quantType == "no_quant":
                    memoryTransfer = computeModelSize(parsedConfig)
                else:
                    if quantType == "bnb_int8":
                        memoryTransfer = computeModelSize(parsedConfig) / 2.0
                    if quantType == "bnb_q4":
                        memoryTransfer = computeModelSize(parsedConfig) / 4.0
            theoryTimePrompt = 2 * prompt_len * memoryTransfer + 2 * numLayers * hiddenDim * hiddenDim * 2 * prompt_len
            theoryTimePrompt = batch_size * theoryTimePrompt
            theoryTimePrompt_in_ms = theoryTimePrompt / (tera * gpu_compute * 0.85)
            finalPromptTime = theoryTimePrompt_in_ms * getFloatRatio_F16(quantType) * 1.8 + convertByteToMB(2 * memoryTransfer) * (0.008 / 100)
            utilizationRate = 1.0
            kv_cache_memory = 2 * 2 * numLayers * hiddenDim * totalLen
            timeIfMemory = convertByteToGB(2 * memoryTransfer * extraFactor + kv_cache_memory * extraFactor) / (utilizationRate * gpu_bandwidth)
            timeIfMemory_in_ms = timeIfMemory * 1000
            totalFlopsToken = 2 * memoryTransfer + totalLen * hiddenDim * 2 * numLayers * 2 * 2
            timeIfFlops_in_ms = (totalFlopsToken * 1000) / (tera * gpu_compute * 0.85)
            finalTimeToConsider = None
            memoryOrCompute = None
            if timeIfMemory_in_ms > timeIfFlops_in_ms:
                finalTimeToConsider = timeIfMemory_in_ms
                memoryOrCompute = "memory"
            else:
                finalTimeToConsider = timeIfFlops_in_ms
                memoryOrCompute = "compute"
            if not isValidPositiveInteger(info.number):
                raise ValueError("Number of GPUs have to be positive number (>0)")
                return
            if info.number > 1:
                finalTimeToConsider = (finalTimeToConsider * 1.25) / info.number
            finalTimeToConsider = finalTimeToConsider * bnb_cost
            finalPromptTime = finalPromptTime * bnb_cost
            token_per_s = 1000 / finalTimeToConsider
            jsonComputeReturnData = {
                "Token/s": round(token_per_s) if round(token_per_s) >= 1 else "< 1",
                "ms per token": "{:.2f}".format(finalTimeToConsider),
                "Prompt process Time (s)": "{:.2f}".format(finalPromptTime),
                "memory or compute bound?": memoryOrCompute
            }
        
        else:
            raise ValueError("read the fucking documentation")
        
    else:
        raise ValueError("read the fucking documentation")
    
    return jsonComputeReturnData

# for testing
if __name__ == '__main__':
    info = InferenceGPUInfo('rtx-2060', 1)
    print(findTokensPerSecond('meta-llama/Llama-2-7b-hf', 'inf', 'full_trn', 'adam_opt', 'no_quant', 300, 300, 1, False, info))
    print(findMemoryRequirement('meta-llama/Llama-2-7b-hf', 'inf', 'full_trn', 'adam_opt', 'no_quant', 300, 300, 1, False, info))