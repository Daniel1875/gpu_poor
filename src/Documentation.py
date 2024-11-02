import json

if __name__ == '__main__':
    print('Usage for memory requirement:')
    print('--------------------------------')
    print('Argument name_or_size: input either a string corresponding to the name of the model or an integer denoting the size of the model in billions. The models available are as follows:')
    with open("public/all_configs.json") as json_file:
        parsedJSONData = json.load(json_file)
        for item in parsedJSONData:
            print(item)
    print('--------------------------------')
    print('Argument train_or_inference:')
    print('Inference (Huggingface): \'inf\'')
    print('Inference (vLLM): \'inf_vLLM\'')
    print('Inference (GGML): \'inf_ggml\'')
    print('Training (Huggingface): \'trn\'')
    print('--------------------------------')
    print('Argument train_method:')
    print('Full: \'full_trn\'')
    print('LoRA: \'lora_trn\'')
    print('QLoRA: \'qlora\'')
    print('--------------------------------')
    print('Argument optimizer:')
    print('ADAM: \'adam_opt\'')
    print('SGD: \'sgd_opt\'')
    print('--------------------------------')
    print('Argument quant:')
    options_dict = {
        "None": "no_quant",
        "bnb int8": "bnb_int8",
        "bnb int4": "bnb_q4",
        "GGML Q2_K": "ggml_Q2_K",
        "GGML Q3_K_L": "ggml_Q3_K_L",
        "GGML Q3_K_M": "ggml_Q3_K_M",
        "GGML QK4_0": "ggml_QK4_0",
        "GGML QK4_1": "ggml_QK4_1",
        "GGML QK4_K_M": "ggml_QK4_K_M",
        "GGML QK4_K_S": "ggml_QK4_K_S",
        "GGML QK5_0": "ggml_QK5_0",
        "GGML QK5_1": "ggml_QK5_1",
        "GGML QK5_K_M": "ggml_QK5_K_M",
        "GGML Q6_K": "ggml_Q6_K",
        "GGML QK8_0": "ggml_QK8_0"
    }
    for item in options_dict:
        print(item + ': ' + options_dict[item])
    print('--------------------------------')
    print('Arguments prompt_len and tokens_to_generate are integers. Argument batch_size is integer. Argument gradient_checkpointing is boolean.')
    print('--------------------------------')
    print('Argument info is processor information. For GPU, create an InferenceGPUInfo object with the name of the GPU and the number of GPUs. For CPU, create an InferenceCPUInfo object with the name of the CPU and the RAM specs.')
    print('For CPU, see cpu_config.json. The "RAM specs" mentioned above shall be a tuple of to numbers, representing if DDR4 or DDR5 is used, respectively, with 0 and 1 respectively; exactly one of the two must be a 1. Refer to the CPU specs sheet:')
    print('If the CPU config file says either DDR4 or DDR5 is unavailable, it can only be set to 0, so the other must be 1.')