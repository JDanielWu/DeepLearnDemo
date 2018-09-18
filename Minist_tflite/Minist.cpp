//============================================================================
// Name        : Minist.cpp
// Author      : WuDan
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstdio>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "opencv2/opencv.hpp"



using namespace tflite;
using namespace std;
using namespace cv;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[])
{
	Mat img;
	if(2 == argc)
	{
		img = imread(argv[1], 0);
	}
	else
	{
		img = imread("1.jpg", 0);
	}

	cout<<"img channels:"<<img.channels()<<endl;

	std::unique_ptr<tflite::FlatBufferModel> model;
	std::unique_ptr<Interpreter> interpreter;

	model = tflite::FlatBufferModel::BuildFromFile("minist.tflite");

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;

	tflite::InterpreterBuilder(*model, resolver)(&interpreter);

	int input = interpreter->inputs()[0];

	// Allocate tensor buffers.
	interpreter->AllocateTensors();

	//tflite::PrintInterpreterState(interpreter.get());



	auto inputdata = interpreter->typed_input_tensor<float>(0);

	/*switch(interpreter->tensor(input)->type)
	{
	case kTfLiteFloat32:
		cout<<"float32."<<endl;
		break;
	}*/


	for (int i = 0; i < 28; ++i)
	{
		for(int j=0; j<28; ++j)
		{
			//inputdata[i*28 +j] = 0.25;
			inputdata[i*28 +j] = ((float)img.at<uchar>(i,j)) / 255.0;
		}

	}

	// Run inference
	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
	cout<<"Predict invoke--------:"<<endl;
	//printf("\n\n=== Post-invoke Interpreter State ===\n");
	//tflite::PrintInterpreterState(interpreter.get());

	float* output = interpreter->typed_output_tensor<float>(0);

	int iMaxIndex = 0;
	float maxValue = 0.0;
	for(int i=0; i<10; ++i)
	{
		cout<<"i:"<<i<<" value:"<<output[i]<<endl;
		if(output[i]>maxValue)
		{
			maxValue = output[i];
			iMaxIndex = i;
		}
	}

	cout<<"You input num img is:"<<iMaxIndex<<endl;
	return 0;
}
