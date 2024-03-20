package main

import (
	"errors"
	"fmt"
	"mlgo/common"
	"mlgo/ml"
)

type simple_hparams struct{
	n_input  float32;
	n_output float32;
}

type simple_model struct {
	hparams simple_hparams

	weight *ml.Tensor
	bias   *ml.Tensor
}

const (
	READ_FROM_BIDENDIAN = true
	OUTPUT_TO_BIDENDIAN = true
)

func SimpleModelLoad(model *simple_model) error {
	fmt.Println("start SimpleModelLoad")
	model_bytes := common.ReadBytes(common.MODEL_ADDR, READ_FROM_BIDENDIAN)
	index := 0
	fmt.Println("model_bytes len: ", len(model_bytes))

	// verify magic
	magic := common.ReadInt32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)
	fmt.Printf("magic: %x\n", magic)
	if magic != 0x67676d6c {
		return errors.New("invalid model file (bad magic)")
	}

	// Reading model parameters (weight and bias)
	n_dims := int32(common.ReadInt32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN))
	model.hparams.n_input = n_dims
	model.hparams.n_output = 1 // Since it's a linear regression model

	if READ_FROM_BIDENDIAN {
		weight_data_size := model.hparams.n_input
		weight_data := common.DecodeFloat32List(model_bytes[index : index+4*int(weight_data_size)])
		index += 4 * int(weight_data_size)
		model.weight = ml.NewTensor1DWithData(nil, ml.TYPE_F32, uint32(model.hparams.n_input), weight_data)
	} else {
		model.weight = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_input))
		for i := 0; i < len(model.weight.Data); i++ {
			model.weight.Data[i] = common.ReadFP32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)
		}
	}

	if READ_FROM_BIDENDIAN {
		bias_data := common.DecodeFloat32List(model_bytes[index : index+4])
		index += 4
		model.bias = ml.NewTensor0DWithData(nil, ml.TYPE_F32, bias_data)
	} else {
		model.bias = ml.NewTensor0D(nil, ml.TYPE_F32)
		model.bias.Data[0] = common.ReadFP32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)
	}

	return nil
}


func SimpleModelEval(model *simple_model, input float32) float32 {
	fmt.Println("start SimpleModelEval")
	ctx0 := &ml.Context{}
	graph := ml.Graph{ThreadsCount: 1}

	inputTensor := ml.NewTensor0D(ctx0, ml.TYPE_F32)
	inputTensor.Data[0] = input

	// Linear regression computation: y = wx + b
	output := ml.Add(ctx0, ml.Mul(ctx0, model.weight, inputTensor), model.bias)

	// Run the computation
	ml.BuildForwardExpand(&graph, output)
	ml.GraphCompute(ctx0, &graph)

	return output.Data[0]
}

func SimpleStoreInMemory(output float32) {
	outputBytes := common.Float32ToBytes(output, OUTPUT_TO_BIDENDIAN)
	common.Output(outputBytes, OUTPUT_TO_BIDENDIAN)
}

func SimpleLinearRegression() {
	fmt.Println("Start Simple Linear Regression")
	input := float32(1.0) // Example input
	model := new(simple_model)
	err := SimpleModelLoad(model)
	if err != nil {
		fmt.Println(err)
		common.Halt()
	}
	output := SimpleModelEval(model, input)
	fmt.Printf("Output for input %.2f is %.2f\n", input, output)
	SimpleStoreInMemory(output)
}
