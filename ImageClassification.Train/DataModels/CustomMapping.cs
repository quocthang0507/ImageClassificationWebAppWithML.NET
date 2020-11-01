using Microsoft.ML.Data;
using System;
using System.Drawing;
using System.Numerics;

namespace ImageClassification.Train.DataModels
{
    class InputData
    {
        [VectorType(6)] // attribute specifies vector type of known length
        public VBuffer<Byte> Image1; // the VBuffer<> type actually represents the data
    }

    // Define a class to represent the output column type for the custom transform
    class OutputData
    {
        // THE MAGICAL FIX: attribute specifies vector type of unknown length (i.e. VarVector)
        [VectorType()]
        public VBuffer<Byte> Image; // the VBuffer<> type actually represents the data
    }
}
