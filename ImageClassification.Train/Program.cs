using Common;
using ImageClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace ImageClassification.Train
{
    internal class Program
    {
        private static string outputMlNetModelFilePath, imagesFolderPathForPredictions, fullImagesetFolderPath;
        private static IDataView trainDataView, testDataView;
        private static MLContext mlContext;

        static void Main()
        {
            Console.OutputEncoding = Encoding.UTF8;

            //DownloadDataset(out outputMlNetModelFilePath, out imagesFolderPathForPredictions, out fullImagesetFolderPath);
            UseLocalDataset(out outputMlNetModelFilePath, out imagesFolderPathForPredictions, out fullImagesetFolderPath);

            mlContext = new MLContext(seed: null);

            // Specify MLContext Filter to only show feedback log/traces about ImageClassification
            // This is not needed for feedback output if using the explicit MetricsCallback parameter
            mlContext.Log += FilterMLContextLog;

            //PrepareDataset();
            TryApplyGrayScale();

            var pipeline = CreatePipeline();

            // 6. Train/create the ML model
            Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");

            // Measuring training time
            Stopwatch watch = Stopwatch.StartNew();

            //Train
            ITransformer trainedModel = pipeline.Fit(trainDataView);

            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"Training with transfer learning took: {elapsedMs / 1000} seconds");

            // 7. Get the quality metrics (accuracy, etc.)
            EvaluateModel(mlContext, testDataView, trainedModel);

            // 8. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
            mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);
            Console.WriteLine($"Model saved to: {outputMlNetModelFilePath}");

            // 9. Try a single prediction simulating an end-user app
            TrySinglePrediction(imagesFolderPathForPredictions, mlContext, trainedModel);

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        private static void PrepareDataset()
        {
            // 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: fullImagesetFolderPath, useFolderNameAsLabel: true);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            // 3. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)

            //Action<InputData, OutputData> mapping =
            //    (input, output) => output.Image = input.Image1;

            // 3.1. Convert dataset to grayscale
            //IDataView grayScaleDataset = mlContext.Transforms.LoadImages(
            //                                    outputColumnName: "RawImage",
            //                                    imageFolder: fullImagesetFolderPath,
            //                                    inputColumnName: "ImagePath")
            //    .Append(mlContext.Transforms.ConvertToGrayscale("GrayImage", "RawImage"))
            //    .Append(mlContext.Transforms.ResizeImages(outputColumnName: "ResizedImage", inputColumnName: "GrayImage", imageHeight: 500, imageWidth: 500))
            //    .Append(mlContext.Transforms.ExtractPixels("Image", "ResizedImage"))
            //    .Fit(shuffledFullImageFilePathsDataset)
            //    .Transform(shuffledFullImageFilePathsDataset);

            //var _1 = mlContext.Data.CreateEnumerable<IDataViewClass2>(grayScaleDataset, false);
            //PrintEnumerable(_1);
            //SaveToFiles(_1, "_");

            // 3.2. Load original dataset
            IDataView dataset = mlContext.Transforms.LoadRawImageBytes(
                                                outputColumnName: "Image",
                                                imageFolder: fullImagesetFolderPath,
                                                inputColumnName: "ImagePath")
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);

            //var _2 = mlContext.Data.CreateEnumerable<IDataViewClass>(dataset, false);
            //PrintEnumerable(_2);

            var shuffledFullImagesDataset = mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Fit(dataset)
                .Transform(dataset);

            //int size = GetSizeIDataView(shuffledFullImagesDataset);

            // 4. Split the data 80:20 into train and test sets, train and evaluate.
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            trainDataView = trainTestData.TrainSet;
            testDataView = trainTestData.TestSet;
        }

        private static EstimatorChain<KeyToValueMappingTransformer> CreatePipeline()
        {
            // 5. Define the model's training pipeline using DNN default values
            //
            EstimatorChain<KeyToValueMappingTransformer> pipeline = mlContext.MulticlassClassification.Trainers
                .ImageClassification(featureColumnName: "Image",
                                         labelColumnName: "LabelAsKey",
                                         validationSet: testDataView)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                      inputColumnName: "PredictedLabel"));

            // 5.1 (OPTIONAL) Define the model's training pipeline by using explicit hyper-parameters
            //
            //var options = new ImageClassificationTrainer.Options()
            //{
            //    FeatureColumnName = "Image",
            //    LabelColumnName = "LabelAsKey",
            //    // Just by changing/selecting InceptionV3/MobilenetV2/ResnetV250  
            //    // you can try a different DNN architecture (TensorFlow pre-trained model). 
            //    Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
            //    Epoch = 50,       //100
            //    BatchSize = 10,
            //    LearningRate = 0.01f,
            //    MetricsCallback = (metrics) => Console.WriteLine(metrics),
            //    ValidationSet = testDataView
            //};

            //var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
            //        .Append(mlContext.Transforms.Conversion.MapKeyToValue(
            //            outputColumnName: "PredictedLabel",
            //            inputColumnName: "PredictedLabel"));
            return pipeline;
        }

        private static int GetSizeIDataView(IDataView idv)
        {
            var schema = idv.Schema;
            int rows = 0;
            using (var cursor = idv.GetRowCursor(schema))
            {
                while (cursor.MoveNext())
                {
                    rows++;
                }
            }
            return rows;
        }

        private static bool ByteArrayToFile(string fileName, byte[] byteArray)
        {
            try
            {
                using (var fs = new FileStream(fileName, FileMode.Create, FileAccess.Write))
                {
                    fs.Write(byteArray, 0, byteArray.Length);
                    return true;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Exception caught in process: {0}", ex);
                return false;
            }
        }

        private static IEnumerable<Vector<Byte>> UpdateImageColumnToVectorColumn(MLContext mlContext, IDataView idv, string imageColName)
        {
            var images = idv.GetColumn<Bitmap>(imageColName);
            IEnumerable<Vector<Byte>> column = images.Select(image => new Vector<Byte>(ConvertImageToByteArray(image)));
            return column;
        }

        private static byte[] ConvertImageToByteArray(Bitmap image)
        {
            using (MemoryStream mStream = new MemoryStream())
            {
                image.Save(mStream, image.RawFormat);
                return mStream.ToArray();
            }
        }

        private static void DownloadDataset(out string outputMlNetModelFilePath, out string imagesFolderPathForPredictions, out string fullImagesetFolderPath)
        {
            const string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");
            imagesFolderPathForPredictions = Path.Combine(assetsPath, "inputs", "images-for-predictions", "FlowersForPredictions");
            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs", "images");

            // 1. Download the image set and unzip
            string finalImagesFolderName = DownloadImageSet(imagesDownloadFolderPath);
            fullImagesetFolderPath = Path.Combine(imagesDownloadFolderPath, finalImagesFolderName);
        }

        private static void UseLocalDataset(out string outputMlNetModelFilePath, out string imagesFolderPathForPredictions, out string fullImagesetFolderPath)
        {
            const string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs", "imageClassifier_New.pb");
            imagesFolderPathForPredictions = Path.Combine(assetsPath, "inputs", "predictions");
            fullImagesetFolderPath = Path.Combine(assetsPath, "inputs", "img");
        }

        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

            // Measuring time
            Stopwatch watch = Stopwatch.StartNew();

            IDataView predictionsDataView = trainedModel.Transform(testDataset);

            Microsoft.ML.Data.MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");
            ConsoleHelper.PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);

            watch.Stop();
            long elapsed2Ms = watch.ElapsedMilliseconds;

            Console.WriteLine($"Predicting and Evaluation took: {elapsed2Ms / 1000} seconds");
        }

        private static void TrySinglePrediction(string imagesFolderPathForPredictions, MLContext mlContext, ITransformer trainedModel)
        {
            // Create prediction function to try one prediction
            PredictionEngine<InMemoryImageData, ImagePrediction> predictionEngine = mlContext.Model
                .CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);

            IEnumerable<InMemoryImageData> testImages = FileUtils.LoadInMemoryImagesFromDirectory(
                imagesFolderPathForPredictions, false);

            foreach (InMemoryImageData imageToPredict in testImages)
            {
                ImagePrediction prediction = predictionEngine.Predict(imageToPredict);

                Console.WriteLine(
                    $"Image Filename : [{imageToPredict.ImageFileName}], " +
                    $"Scores : [{string.Join(",", prediction.Score)}], " +
                    $"Predicted Label : {prediction.PredictedLabel}");
            }
        }

        private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
            => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new ImageData(x.imagePath, x.label));

        private static IEnumerable<ImageData2> LoadImagesFromDirectory2(string folder, bool useFolderNameAsLabel = true)
            => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new ImageData2(x.imagePath, x.label));

        /// <summary>
        /// Download dataset and return file name
        /// </summary>
        /// <param name="imagesDownloadFolder"></param>
        /// <returns></returns>
        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            // get a set of images to teach the network about the new classes

            //SINGLE SMALL FLOWERS IMAGESET (200 files)
            const string fileName = "flower_photos_small_set.zip";
            string url = $"https://mlnetfilestorage.file.core.windows.net/imagesets/flower_images/flower_photos_small_set.zip?st=2019-08-07T21%3A27%3A44Z&se=2030-08-08T21%3A27%3A00Z&sp=rl&sv=2018-03-28&sr=f&sig=SZ0UBX47pXD0F1rmrOM%2BfcwbPVob8hlgFtIlN89micM%3D";
            Web.Download(url, imagesDownloadFolder, fileName);
            Compress.UnZip(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);

            //SINGLE FULL FLOWERS IMAGESET (3,600 files)
            //string fileName = "flower_photos.tgz";
            //string url = $"http://download.tensorflow.org/example_images/{fileName}";
            //Web.Download(url, imagesDownloadFolder, fileName);
            //Compress.ExtractTGZ(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(fileName);
        }

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);

        public static void ConsoleWriteImagePrediction(string ImagePath, string Label, string PredictedLabel, float Probability)
        {
            ConsoleColor defaultForeground = Console.ForegroundColor;
            ConsoleColor labelColor = ConsoleColor.Magenta;
            ConsoleColor probColor = ConsoleColor.Blue;

            Console.Write("Image File: ");
            Console.ForegroundColor = labelColor;
            Console.Write($"{Path.GetFileName(ImagePath)}");
            Console.ForegroundColor = defaultForeground;
            Console.Write(" original labeled as ");
            Console.ForegroundColor = labelColor;
            Console.Write(Label);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" predicted as ");
            Console.ForegroundColor = labelColor;
            Console.Write(PredictedLabel);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" with score ");
            Console.ForegroundColor = probColor;
            Console.Write(Probability);
            Console.ForegroundColor = defaultForeground;
            Console.WriteLine("");
        }

        private static void FilterMLContextLog(object sender, LoggingEventArgs e)
        {
            if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
            {
                Console.WriteLine(e.Message);
            }
        }

        private static void TryApplyGrayScale()
        {
            IEnumerable<ImageData2> images = LoadImagesFromDirectory2(folder: fullImagesetFolderPath, useFolderNameAsLabel: true);
            IDataView data = mlContext.Data.LoadFromEnumerable(images);
            // Image loading pipeline. 
            var pipeline = mlContext.Transforms.ConvertToGrayscale("GrayImage", "Image");

            var transformedData = pipeline.Fit(data).Transform(data);

            var shuffledFullImagesDataset = mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Fit(transformedData)
                .Transform(transformedData);

            // 4. Split the data 80:20 into train and test sets, train and evaluate.
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            trainDataView = trainTestData.TrainSet;
            testDataView = trainTestData.TestSet;
        }
    }
}

