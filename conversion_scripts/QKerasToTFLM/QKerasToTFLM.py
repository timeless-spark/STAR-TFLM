import subprocess
import json
import numpy as np
import qkeras
from tensorflow import keras

def split_list(list, n):
    splitted_list = []
    for i in range(len(list)//n):
        splitted_list.append(list[n*i:n*i+n])
    return splitted_list

class Qkeras_to_TFLM:
    def __init__(self, qkeras_model, model_name, flatbuffer_version, script_out_dir, schema_file, flatbuffers_dir):
        super(Qkeras_to_TFLM, self).__init__()

        self.qkeras_model = qkeras_model
        self.model_name = model_name
        self.script_out_dir = script_out_dir
        self.schema_file = schema_file
        self.flatbuffers_dir = flatbuffers_dir

        self.flatbuffer_json_dict = {}
        self.flatbuffer_json_dict["version"] = flatbuffer_version
        self.flatbuffer_json_dict["operator_codes"] = []

        self.flatbuffer_json_dict["subgraphs"] = []
        self.flatbuffer_json_dict["subgraphs"].append({})
        self.flatbuffer_json_dict["subgraphs"][0]["tensors"] = []
        self.flatbuffer_json_dict["subgraphs"][0]["inputs"] = 0
        self.flatbuffer_json_dict["subgraphs"][0]["outputs"] = 0
        self.flatbuffer_json_dict["subgraphs"][0]["operators"] = []
        self.flatbuffer_json_dict["subgraphs"][0]["name"] = "main"

        self.flatbuffer_json_dict["description"] = "MLIR Converted"
        self.flatbuffer_json_dict["buffers"] = [{}, ]

        self.operator_codes_dict = {}
        self.tensors_dict = {}

        # Add operator codes for your layers here
        self.code_to_deprecated_code_dict = {"CONV_2D": 3, "MAX_POOL_2D": 17, "AVERAGE_POOL_2D": 1, "DEPTHWISE_CONV_2D": 4, "ADD": 0, "RESHAPE": 22, "FULLY_CONNECTED": 9, "SOFTMAX": 25}
        self.code_to_version_dict = {"CONV_2D": 3, "MAX_POOL_2D": 2, "AVERAGE_POOL_2D": 2, "DEPTHWISE_CONV_2D": 3, "ADD": 2, "RESHAPE": 1, "FULLY_CONNECTED": 4, "SOFTMAX": 2}
        self.code_to_builtin_options_type = {"CONV_2D": "Conv2DOptions", "MAX_POOL_2D": "Pool2DOptions", "AVERAGE_POOL_2D": "Pool2DOptions", "DEPTHWISE_CONV_2D": "DepthwiseConv2DOptions", "ADD": "AddOptions", "RESHAPE": "ReshapeOptions", "FULLY_CONNECTED": "FullyConnectedOptions", "SOFTMAX": "SoftmaxOptions"}

    def add_inputs_outputs(self):
        self.flatbuffer_json_dict["subgraphs"][0]["inputs"] = [0]
        self.flatbuffer_json_dict["subgraphs"][0]["outputs"] = [len(self.flatbuffer_json_dict["subgraphs"][0]["tensors"]) - 1]

    def ExportFlatbuffer(self):
        self.fill_flatbuffer()
        self.add_inputs_outputs()
        json.dump(self.flatbuffer_json_dict, open(f"{self.script_out_dir}/{self.model_name}.json", "w+"), indent = 2)
        subprocess.run(f"{self.flatbuffers_dir}/flatc -o {self.script_out_dir} --binary {self.schema_file} {self.script_out_dir}/{self.model_name}.json", shell=True, check=True)
        subprocess.run(f"xxd -i {self.script_out_dir}/{self.model_name}.tflite > {self.script_out_dir}/{self.model_name}_model_data.c", shell=True, check=True)
        c_name = f"{self.script_out_dir}/{self.model_name}.tflite"
        c_name = c_name.replace("/", "_").replace("-", "_").replace(".", "_").replace(" ", "_")
        f = open(f"{self.script_out_dir}/{self.model_name}_model_data.h", "w")
        f.write("#include <cstdint>\n\n")
        f.write(f"extern const unsigned char {c_name}[];\n")
        f.write(f"extern const unsigned int {c_name}_len;\n\n")
        f.write(f"#define {self.model_name}_model {c_name}\n")
        f.write(f"#define {self.model_name}_model_len {c_name}_len\n")
        f.close()

    def add_layer(self, code, inputs, outputs, builtin_options):
        if code not in self.operator_codes_dict:
            self.operator_codes_dict[code] = len(self.flatbuffer_json_dict["operator_codes"])
            self.flatbuffer_json_dict["operator_codes"].append({"deprecated_builtin_code": self.code_to_deprecated_code_dict[code], "version": self.code_to_version_dict[code], "builtin_code": code})
        for input in inputs:
            if input[0] not in self.tensors_dict:
                self.tensors_dict[input[0]] = len(self.flatbuffer_json_dict["subgraphs"][0]["tensors"])
                input[1]["buffer"] = len(self.flatbuffer_json_dict["buffers"])
                self.flatbuffer_json_dict["buffers"].append(input[2])
                self.flatbuffer_json_dict["subgraphs"][0]["tensors"].append(input[1])
            elif input[1] != None:
                input[1]["buffer"] = self.flatbuffer_json_dict["subgraphs"][0]["tensors"][self.tensors_dict[input[0]]]["buffer"]
                self.flatbuffer_json_dict["subgraphs"][0]["tensors"][self.tensors_dict[input[0]]] = input[1]
        for output in outputs:
            if output[0] not in self.tensors_dict:
                self.tensors_dict[output[0]] = len(self.flatbuffer_json_dict["subgraphs"][0]["tensors"])
                output[1]["buffer"] = len(self.flatbuffer_json_dict["buffers"])
                self.flatbuffer_json_dict["buffers"].append(output[2])
                self.flatbuffer_json_dict["subgraphs"][0]["tensors"].append(output[1])
        self.flatbuffer_json_dict["subgraphs"][0]["operators"].append({"opcode_index": self.operator_codes_dict[code],
                                                                       "inputs": [self.tensors_dict[input[0]] for input in inputs],
                                                                       "outputs": [self.tensors_dict[output[0]] for output in outputs],
                                                                       "builtin_options_type": self.code_to_builtin_options_type[code],
                                                                       "builtin_options": builtin_options,
                                                                       "custom_options_format": "FLEXBUFFERS"
                                                                       })
    def add_tensor(self, tensor):
        if tensor[0] not in self.tensors_dict:
                self.tensors_dict[tensor[0]] = len(self.flatbuffer_json_dict["subgraphs"][0]["tensors"])
                tensor[1]["buffer"] = len(self.flatbuffer_json_dict["buffers"])
                self.flatbuffer_json_dict["buffers"].append(tensor[2])
                self.flatbuffer_json_dict["subgraphs"][0]["tensors"].append(tensor[1])

    def get_parent_layers(self, layer):

        relevant_nodes = []
        for v in self.qkeras_model._nodes_by_depth.values():
            relevant_nodes += v

        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                continue

            for inbound_layer, _, _, _ in node.iterate_inbound():
                connections.append([inbound_layer.name, inbound_layer])

        return connections

    def net_family_tree(self):
        find_the_parent = {}
        find_the_child = {}
        for layer in self.qkeras_model.layers:
            parents = self.get_parent_layers(layer)
            find_the_parent[layer.name] = parents
            for parent in parents:
                if parent[0] in find_the_child:
                    find_the_child[parent[0]].append([layer.name, layer])
                else:
                    find_the_child[parent[0]] = [[layer.name, layer]]

        return (find_the_parent, find_the_child)

    def fill_flatbuffer(self):

        (find_the_parent, find_the_child) = self.net_family_tree()

        for layer in self.qkeras_model.layers:

            layer

            if layer.__class__.__name__ in ["QActivation"]:
                continue
            if layer.__class__.__name__ in ["ReLU"]:
                continue
            if layer.__class__.__name__ in ["Dropout"]: # assuming not to perform dropout rescaling
                continue

            if layer.__class__.__name__ in ["InputLayer"]:
            # child node
                child_layer = find_the_child[layer.name][0][1]
                out_act_layer = child_layer
                if out_act_layer.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly followed by a QAct layer, or non supported layer")

                out_shape = [dim if dim!=None else 1 for dim in out_act_layer.input_shape]

                out_quantizer = out_act_layer.quantizer

                out_nb = out_quantizer.bits
                out_n_parall = 32 // out_nb

                scale_out = out_quantizer.scale1.numpy().flatten()

                out_shape_signature = [-1] + [out_shape[i] for i in range(1, len(out_shape))]

                output_dict = {"shape": out_shape,
                            "type": "INT4" if out_nb==4 else ("INT8" if out_nb==8 else ("INT16" if out_nb==16 else "INT32")),
                            "buffer": None,
                            "name": layer.name,
                            "quantization": {"scale": scale_out.tolist(),
                                            "zero_point": out_quantizer.zeropoint.numpy().astype(int).flatten().tolist(),
                                            "details_type": "NONE",
                                            "quantized_dimension": 0
                                            },
                            "is_variable": False,
                            "shape_signature": out_shape_signature,
                            "has_rank": True
                            }

                output = (out_act_layer.name, output_dict, {})

                self.add_tensor(tensor=output)
                continue

            elif layer.__class__.__name__ in ["QConv2D", "QConv2DBatchnorm", "QDepthwiseConv2D", "QDepthwiseConv2DBatchnorm", "QDense", "QDenseBatchnorm"]:
            # parent node
                parent_layer = find_the_parent[layer.name][0][1]
            # child node
                child_layer = find_the_child[layer.name][0][1]

                if layer.__class__.__name__ in ["QConv2D", "QConv2DBatchnorm"]:
                    code = "CONV_2D"
                elif layer.__class__.__name__ in ["QDepthwiseConv2D", "QDepthwiseConv2DBatchnorm"]:
                    code = "DEPTHWISE_CONV_2D"
                else:
                    code = "FULLY_CONNECTED"

                in_act_layer = parent_layer
                in_act_name = in_act_layer.name
                if in_act_layer.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly preceded by a QAct layer, or non supported layer")

                fused_activation_function = "NONE"
                out_act_layer = child_layer
                if out_act_layer.__class__.__name__ in ["ReLU"]:
                    out_act_layer = find_the_child[out_act_layer.name][0][1]
                    fused_activation_function = "RELU"
                # if .... in ["Sigmoid"]:
                #   ... ... # Add your code for sigmoid
                if out_act_layer.__class__.__name__ in ["Dropout"]:
                    out_act_layer = find_the_child[out_act_layer.name][0][1]
                if out_act_layer.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly followed by a QAct layer, or non supported layer")

            # inputs
                in_shape = [dim if dim!=None else 1 for dim in in_act_layer.input_shape]

                in_quantizer = in_act_layer.quantizer
                weight_quantizer = layer.get_quantizers()[0]

                in_nb = in_quantizer.bits
                weight_nb = weight_quantizer.bits

                in_n_parall = 32 // (in_nb if in_nb > weight_nb else weight_nb)

                scale_in = in_quantizer.scale1.numpy().flatten()

                in_shape_signature = [-1] + [in_shape[i] for i in range(1, len(in_shape))]

                input_dict = {"shape": in_shape,
                            "type": "INT4" if in_nb==4 else ("INT8" if in_nb==8 else ("INT16" if in_nb==16 else "INT32")),
                            "buffer": None,
                            "name": in_act_name,
                            "quantization": {"scale": scale_in.tolist(),
                                            "zero_point": in_quantizer.zeropoint.numpy().astype(int).flatten().tolist(),
                                            "details_type": "NONE",
                                            "quantized_dimension": 0
                                            },
                            "is_variable": False,
                            "shape_signature": in_shape_signature,
                            "has_rank": True
                            }

                input = (in_act_name, input_dict, None)

            # weights
                weight_shape = [dim for dim in layer.weights[0].get_shape()]
                weight_shape = [weight_shape[-1]] + weight_shape[0:-1]

                alphaq_w = weight_quantizer.alphaq
                betaq_w = weight_quantizer.betaq

                scale_w = weight_quantizer.scale1.numpy().flatten() * weight_quantizer.m_i.numpy().flatten()

                if layer.__class__.__name__ in ["QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm", "QDenseBatchnorm"]:
                    weight_buffer = layer.get_folded_weights()[0]
                else:
                    weight_buffer = layer.get_weights()[0]

                tmp = np.divide(weight_buffer, scale_w, dtype=np.float64)
                tmp[np.isnan(tmp)] = 0
                tmp[np.isinf(tmp)] = 0
                weight_buffer = np.clip(np.trunc(tmp + np.sign(tmp)*0.5), alphaq_w, betaq_w).astype(int)

                if layer.__class__.__name__ in ["QConv2D", "QConv2DBatchnorm", "QDepthwiseConv2D", "QDepthwiseConv2DBatchnorm"]:
                    weight_buffer = np.moveaxis(weight_buffer, 3, 0).flatten().tolist()
                else:
                    weight_buffer = np.moveaxis(weight_buffer, 1, 0).flatten().tolist()

                tmp_weight_buffer = weight_buffer

                mod_weight_buffer = []

                if weight_nb == 4:
                    weight_type = "INT4"
                    for i, j in zip(tmp_weight_buffer[::2], tmp_weight_buffer[1::2]):
                        p1 = i.to_bytes(1, signed=True, byteorder="big")[0] & 15
                        p2 = j.to_bytes(1, signed=True, byteorder="big")[0] << 4 & 255
                        p = p1 | p2
                        mod_weight_buffer.append(p)
                elif weight_nb == 8:
                    weight_type = "INT8"
                    for i in tmp_weight_buffer:
                        p = i.to_bytes(1, signed=True, byteorder="big")[0] & 255
                        mod_weight_buffer.append(p)
                elif weight_nb == 16:
                    weight_type = "INT16"
                    for i in weight_buffer:
                        p = i.to_bytes(2, signed=True, byteorder="big")
                        mod_weight_buffer.append(p[1]) # 0
                        mod_weight_buffer.append(p[0]) # 1
                else:
                    weight_type = "INT32"
                    for i in weight_buffer:
                        p = i.to_bytes(4, signed=True, byteorder="big")
                        mod_weight_buffer.append(p[3]) # 0
                        mod_weight_buffer.append(p[2]) # 1
                        mod_weight_buffer.append(p[1]) # 2
                        mod_weight_buffer.append(p[0]) # 3

                weights_dict = {"shape": weight_shape,
                                "type": weight_type,
                                "buffer": None,
                                "name": str(layer.name + "_weights"),
                                "quantization": {"scale": scale_w.tolist(),
                                                "zero_point": [0 for _ in range(len(scale_w))],
                                                "details_type": "NONE",
                                                "quantized_dimension": 0 if layer.__class__.__name__ in ["QConv2D", "QConv2DBatchnorm", "QDense", "QDenseBatchnorm"] else 3
                                                },
                                "is_variable": False,
                                "has_rank": True}

                weights = (str(layer.name + "_weights"), weights_dict, {"data": mod_weight_buffer})

                bias_shape = [dim for dim in layer.weights[1].get_shape()]

                if layer.__class__.__name__ in ["QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm", "QDenseBatchnorm"]:
                    bias_buffer = layer.get_folded_weights()[1]
                else:
                    bias_buffer = layer.get_weights()[1]

                scale_b = scale_w * scale_in

                bias_quantizer = layer.get_quantizers()[1]

                bias_nb_real = bias_quantizer.bits
                bias_nb = 32

                alphaq_b = - (2**31 - 1)
                betaq_b = 2**31 - 1

                tmp = np.divide(bias_buffer, scale_b, dtype=np.float64)
                tmp[np.isnan(tmp)] = 0
                tmp[np.isinf(tmp)] = 0
                bias_buffer = np.clip(np.trunc(tmp + np.sign(tmp)*0.5), alphaq_b, betaq_b).astype(int).flatten().tolist()

                mod_bias_buffer = []

                if bias_nb == 4:
                    bias_type = "INT4"
                    for i, j in zip(bias_buffer[::2], bias_buffer[1::2]):
                        p1 = i.to_bytes(1, signed=True, byteorder="big")[0] & 15
                        p2 = j.to_bytes(1, signed=True, byteorder="big")[0] << 4 & 255
                        p = p1 | p2
                        mod_bias_buffer.append(p)
                elif bias_nb == 8:
                    bias_type = "INT8"
                    for i in bias_buffer:
                        p = i.to_bytes(1, signed=True, byteorder="big")[0] & 255
                        mod_bias_buffer.append(p)
                elif bias_nb == 16:
                    bias_type = "INT16"
                    for i in bias_buffer:
                        p = i.to_bytes(2, signed=True, byteorder="big")
                        mod_bias_buffer.append(p[1]) # 0
                        mod_bias_buffer.append(p[0]) # 1
                else:
                    bias_type = "INT32"
                    for i in bias_buffer:
                        p = i.to_bytes(4, signed=True, byteorder="big")
                        mod_bias_buffer.append(p[3]) # 0
                        mod_bias_buffer.append(p[2]) # 1
                        mod_bias_buffer.append(p[1]) # 2
                        mod_bias_buffer.append(p[0]) # 3

                bias_dict = {"shape": bias_shape,
                                "type": bias_type,
                                "buffer": None,
                                "name": str(layer.name + "_bias"),
                                "quantization": {"scale": scale_b.tolist(),
                                                "zero_point": [0 for _ in range(len(scale_b))],
                                                "details_type": "NONE",
                                                "quantized_dimension": 0
                                                },
                                "is_variable": False,
                                "has_rank": True}

                bias = (str(layer.name + "_bias"), bias_dict, {"data": mod_bias_buffer})

                inputs = [input, weights, bias]

            # outputs
                out_shape = [dim if dim!=None else 1 for dim in out_act_layer.input_shape]

                out_quantizer = out_act_layer.quantizer

                out_nb = out_quantizer.bits
                out_n_parall = 32 // out_nb

                scale_out = out_quantizer.scale1.numpy().flatten()

                out_shape_signature = [-1] + [out_shape[i] for i in range(1, len(out_shape))]

                output_dict = {"shape": out_shape,
                            "type": "INT4" if out_nb==4 else ("INT8" if out_nb==8 else ("INT16" if out_nb==16 else "INT32")),
                            "buffer": None,
                            "name": out_act_layer.name,
                            "quantization": {"scale": scale_out.tolist(),
                                            "zero_point": out_quantizer.zeropoint.numpy().astype(int).flatten().tolist(),
                                            "details_type": "NONE",
                                            "quantized_dimension": 0
                                            },
                            "is_variable": False,
                            "shape_signature": out_shape_signature,
                            "has_rank": True
                            }

                output = (out_act_layer.name, output_dict, {})

                outputs = [output]

            # builtin_options
                if layer.__class__.__name__ in ["QConv2D", "QConv2DBatchnorm"]:
                    builtin_options = {"padding": "VALID" if layer.padding == "valid" else "SAME",
                                    "stride_w": layer.strides[1],
                                    "stride_h": layer.strides[0],
                                    "fused_activation_function": fused_activation_function,
                                    "dilation_w_factor": layer.dilation_rate[1],
                                    "dilation_h_factor": layer.dilation_rate[0]
                                    }
                elif layer.__class__.__name__ in ["QDepthwiseConv2D", "QDepthwiseConv2DBatchnorm"]:
                    builtin_options = {"padding": "VALID" if layer.padding == "valid" else "SAME",
                                    "stride_w": layer.strides[1],
                                    "stride_h": layer.strides[0],
                                    "depth_multiplier": layer.depth_multiplier,
                                    "fused_activation_function": fused_activation_function,
                                    "dilation_w_factor": layer.dilation_rate[1],
                                    "dilation_h_factor": layer.dilation_rate[0]
                                    }
                else:
                    builtin_options = {"fused_activation_function": fused_activation_function,
                                    "weights_format": "DEFAULT",
                                    "keep_num_dims": False,
                                    "asymmetric_quantize_inputs": False 
                                    }

            elif layer.__class__.__name__ in ["Add"]:
            # parent nodes
                parent_layers_list = find_the_parent[layer.name]
                parent_layer_1 = parent_layers_list[0][1]
                parent_layer_2 = parent_layers_list[1][1]

            # child node
                child_layer = find_the_child[layer.name][0][1]

                code = "ADD"

                in_act_layer_1 = parent_layer_1
                in_act_name_1 = in_act_layer_1.name
                if in_act_layer_1.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly preceded by a QAct layer, or non supported layer")

                in_act_layer_2 = parent_layer_2
                in_act_name_2 = in_act_layer_2.name
                if in_act_layer_2.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly preceded by a QAct layer, or non supported layer")

                fused_activation_function = "NONE"
                out_act_layer = child_layer
                if out_act_layer.__class__.__name__ in ["ReLU"]:
                    out_act_layer = find_the_child[out_act_layer.name][0][1]
                    fused_activation_function = "RELU"
                # if .... in ["Sigmoid"]:
                #   ... ... # Add your code for sigmoid
                if out_act_layer.__class__.__name__ in ["Dropout"]:
                    out_act_layer = find_the_child[out_act_layer.name][0][1]
                if out_act_layer.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly followed by a QAct layer, or non supported layer")

            # inputs
                in_1_shape = [dim if dim!=None else 1 for dim in in_act_layer_1.input_shape]

                in_1_quantizer = in_act_layer_1.quantizer

                in_1_nb = in_1_quantizer.bits
                in_1_n_parall = 32 // in_1_nb

                scale_in_1 = in_1_quantizer.scale1.numpy()

                input_1 = (in_act_name_1, None, None)

                in_2_shape = [dim if dim!=None else 1 for dim in in_act_layer_2.input_shape]

                in_2_quantizer = in_act_layer_2.quantizer

                in_2_nb = in_2_quantizer.bits
                in_2_n_parall = 32 // in_2_nb

                scale_in_2 = in_2_quantizer.scale1.numpy()

                input_2 = (in_act_name_2, None, None)

                inputs = [input_1, input_2]

            # outputs
                out_shape = [dim if dim!=None else 1 for dim in out_act_layer.input_shape]

                out_quantizer = out_act_layer.quantizer

                out_nb = out_quantizer.bits
                out_n_parall = 32 // out_nb

                scale_out = out_quantizer.scale1.numpy().flatten()

                out_shape_signature = [-1] + [out_shape[i] for i in range(1, len(out_shape))]

                output_dict = {"shape": out_shape,
                            "type": "INT4" if out_nb==4 else ("INT8" if out_nb==8 else ("INT16" if out_nb==16 else "INT32")),
                            "buffer": None,
                            "name": out_act_layer.name,
                            "quantization": {"scale": scale_out.tolist(),
                                            "zero_point": out_quantizer.zeropoint.numpy().astype(int).flatten().tolist(),
                                            "details_type": "NONE",
                                            "quantized_dimension": 0
                                            },
                            "is_variable": False,
                            "shape_signature": out_shape_signature,
                            "has_rank": True
                            }

                output = (out_act_layer.name, output_dict, {})

                outputs = [output]

            # builtin_options
                builtin_options = {"fused_activation_function": fused_activation_function,
                                "pot_scale_int16": False
                                }

            elif layer.__class__.__name__ in ["Flatten"]:
            # parent node
                parent_layer = find_the_parent[layer.name][0][1]
            # child node -> non servono child
                child_layer = find_the_child[layer.name][0][1]

                code = "RESHAPE"

                in_act_layer = parent_layer
                if in_act_layer.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly preceded by a QAct layer, or non supported layer")
                
                out_act_layer = child_layer

            # inputs
                in_shape = [dim if dim!=None else 1 for dim in in_act_layer.input_shape]

                in_quantizer = in_act_layer.quantizer

                in_nb = in_quantizer.bits
                in_n_parall = 32 // in_nb

                scale_in = in_quantizer.scale1.numpy()

                input = (in_act_layer.name, None, None)

                reshape_dims = [-1 if dim==None else dim for dim in layer.output_shape]

                reshape_buffer = []
                for i in reshape_dims:
                    p = i.to_bytes(4, signed=True, byteorder="big")
                    reshape_buffer.append(p[3])
                    reshape_buffer.append(p[2])
                    reshape_buffer.append(p[1])
                    reshape_buffer.append(p[0])

                reshape_dict = {"shape": [len(reshape_dims)],
                                "type": "INT32",
                                "buffer": None,
                                "name": str(layer.name + "_reshape_info"),
                                "quantization": {"details_type": "NONE",
                                                "quantized_dimension": 0
                                                },
                                "is_variable": False,
                                "has_rank": True
                                }

                reshape_info = (layer.name, reshape_dict, {"data": reshape_buffer})

                inputs = [input, reshape_info]

            # outputs
                out_shape = [dim if dim!=None else 1 for dim in out_act_layer.input_shape]
                
                new_layer = qkeras.QActivation(f"quantized_bits_featuremap(bits=1,integer=1,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")
                new_layer.build(out_act_layer.input_shape)

                fake = keras.models.Sequential()
                fake.add(new_layer)
                fake.predict(np.ones([dim if dim!=None else 1 for dim in out_act_layer.input_shape]))

                new_layer._name = f"{layer.name}_out"

                new_layer.quantizer = in_act_layer.quantizer

                find_the_child[layer.name] = [[new_layer.name, new_layer]]
                find_the_child[new_layer.name] = [[out_act_layer.name, out_act_layer]]

                find_the_parent[out_act_layer.name] = [[new_layer.name, new_layer]]
                find_the_parent[new_layer.name] = [[layer.name, layer]]

                out_quantizer = new_layer.quantizer

                out_nb = out_quantizer.bits
                out_n_parall = 32 // out_nb

                scale_out = out_quantizer.scale1.numpy().flatten()

                out_shape_signature = [-1] + [out_shape[i] for i in range(1, len(out_shape))]

                output_dict = {"shape": out_shape,
                            "type": "INT4" if out_nb==4 else ("INT8" if out_nb==8 else ("INT16" if out_nb==16 else "INT32")),
                            "buffer": None,
                            "name": new_layer.name,
                            "quantization": {"scale": scale_out.tolist(),
                                            "zero_point": out_quantizer.zeropoint.numpy().astype(int).flatten().tolist(),
                                            "details_type": "NONE",
                                            "quantized_dimension": 0
                                            },
                            "is_variable": False,
                            "shape_signature": out_shape_signature,
                            "has_rank": True
                            }

                output = (new_layer.name, output_dict, {})

                outputs = [output]

            # builtin_options
                builtin_options = {"new_shape": out_shape
                                }

            elif layer.__class__.__name__ in ["MaxPooling2D", "AveragePooling2D"]:
            # parent node
                parent_layer = find_the_parent[layer.name][0][1]
            # child node
                child_layer = find_the_child[layer.name][0][1]

                if layer.__class__.__name__ in ["MaxPooling2D"]:
                    code = "MAX_POOL_2D"
                else:
                    code = "AVERAGE_POOL_2D"

                in_act_layer = parent_layer
                in_act_name = in_act_layer.name
                if in_act_layer.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly preceded by a QAct layer, or non supported layer")

                fused_activation_function = "NONE"
                out_act_layer = child_layer
                if out_act_layer.__class__.__name__ in ["ReLU"]:
                    out_act_layer = find_the_child[out_act_layer.name][0][1]
                    fused_activation_function = "RELU"
                # if .... in ["Sigmoid"]:
                #   ... ... # Add your code for sigmoid
                if out_act_layer.__class__.__name__ in ["Dropout"]:
                    out_act_layer = find_the_child[out_act_layer.name][0][1]
                if out_act_layer.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly followed by a QAct layer, or non supported layer")

            # inputs
                in_shape = [dim if dim!=None else 1 for dim in in_act_layer.input_shape]

                in_quantizer = in_act_layer.quantizer

                in_nb = in_quantizer.bits
                in_n_parall = 32 // in_nb

                scale_in = in_quantizer.scale1.numpy()

                input = (in_act_name, None, None)

                inputs = [input]

            # outputs
                out_shape = [dim if dim!=None else 1 for dim in out_act_layer.input_shape]
                
                out_quantizer = out_act_layer.quantizer

                out_nb = out_quantizer.bits
                out_n_parall = 32 // out_nb

                scale_out = out_quantizer.scale1.numpy().flatten()

                out_shape_signature = [-1] + [out_shape[i] for i in range(1, len(out_shape))]

                output_dict = {"shape": out_shape,
                            "type": "INT4" if out_nb==4 else ("INT8" if out_nb==8 else ("INT16" if out_nb==16 else "INT32")),
                            "buffer": None,
                            "name": out_act_layer.name,
                            "quantization": {"scale": scale_out.tolist(),
                                            "zero_point": out_quantizer.zeropoint.numpy().astype(int).flatten().tolist(),
                                            "details_type": "NONE",
                                            "quantized_dimension": 0
                                            },
                            "is_variable": False,
                            "shape_signature": out_shape_signature,
                            "has_rank": True
                            }

                output = (out_act_layer.name, output_dict, {})

                outputs = [output]

            # builtin_options
                builtin_options = {"padding": "VALID",
                                   "stride_w": layer.strides[1],
                                   "stride_h": layer.strides[0],
                                   "filter_width": layer.pool_size[1],
                                   "filter_height": layer.pool_size[0],
                                   "fused_activation_function": fused_activation_function
                                  }

            elif layer.__class__.__name__ in ["Softmax"]:
            # parent node
                parent_layer = find_the_parent[layer.name][0][1]
            # child node
                # softmax is supported ONLY as the last layer of a network - in this version

                code = "SOFTMAX"

                in_act_layer = parent_layer
                if in_act_layer.__class__.__name__ not in ["QActivation"]:
                    raise NotImplementedError(f"Layer {layer.__class__.__name__} not directly preceded by a QAct layer, or non supported layer")

            # inputs
                in_shape = [dim if dim!=None else 1 for dim in in_act_layer.input_shape]

                in_quantizer = in_act_layer.quantizer

                in_nb = 16  # Precision is fixed at 16 bit for this layer

                actual_in_nb = in_quantizer.bits

                scale_in = in_quantizer.scale1.numpy().flatten()
                zero_point_in = in_quantizer.zeropoint.numpy().astype(int).flatten()

                scale_in = scale_in / (2**(in_nb-actual_in_nb))
                zero_point_in = zero_point_in * (2**(in_nb-actual_in_nb))
                
                in_shape_signature = [-1] + [in_shape[i] for i in range(1, len(in_shape))]

                input_dict = {"shape": in_shape,
                            "type": "INT4" if in_nb==4 else ("INT8" if in_nb==8 else ("INT16" if in_nb==16 else "INT32")),
                            "buffer": None,
                            "name": in_act_layer.name,
                            "quantization": {"scale": scale_in.tolist(),
                                            "zero_point": zero_point_in.tolist(),
                                            "details_type": "NONE",
                                            "quantized_dimension": 0
                                            },
                            "is_variable": False,
                            "shape_signature": in_shape_signature,
                            "has_rank": True
                            }

                input = (in_act_layer.name, input_dict, None)

                inputs = [input]

            # outputs
                out_nb = 16  # Precision is fixed at 16 bit for this layer

                scale_out = [1./32768] # TFLM defines it as asymmetric on 31 bit
                zero_point_out = [0]

                output_dict = {"shape": in_shape,
                               "type": "INT16",
                               "buffer": None,
                               "name": "net_output",
                               "quantization": {"scale": scale_out,
                                               "zero_point": zero_point_out,
                                               "details_type": "NONE",
                                               "quantized_dimension": 0
                                               },
                               "is_variable": False,
                               "shape_signature": in_shape_signature,
                               "has_rank": True
                               }

                output = ("net_output", output_dict, {})

                outputs = [output]

            # builtin_options
                builtin_options = {"beta": 1.0
                                }

            else:
                print(layer.__class__.__name__)
                raise NotImplementedError("Non supported layer")
            
            print(layer.__class__.__name__)

            self.add_layer(code=code, inputs=inputs, outputs=outputs, builtin_options=builtin_options)