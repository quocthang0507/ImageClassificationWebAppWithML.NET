using ImageClassification;
using ImageClassification.DataModels;
using ImageClassification.WebApp;
using ImageClassification.WebApp.ImageHelpers;
using ImageClassification.WebApp.ML.DataModels;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.ML;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace TensorFlowImageClassification.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ImageClassificationController : ControllerBase
    {
        public IConfiguration Configuration { get; }
        private readonly PredictionEnginePool<InMemoryImageData, ImagePrediction> _predictionEnginePool;
        private readonly ILogger<ImageClassificationController> _logger;

        public ImageClassificationController(PredictionEnginePool<InMemoryImageData, ImagePrediction> predictionEnginePool, IConfiguration configuration, ILogger<ImageClassificationController> logger) //When using DI/IoC
        {
            // Get the ML Model Engine injected, for scoring.
            _predictionEnginePool = predictionEnginePool;

            Configuration = configuration;

            // Get other injected dependencies.
            _logger = logger;
        }

        [HttpPost]
        [ProducesResponseType(200)]
        [ProducesResponseType(400)]
        [Route("classifyImage")]
        public async Task<IActionResult> ClassifyImage(IFormFile imageFile)
        {
            if (imageFile.Length == 0)
                return BadRequest();

            MemoryStream imageMemoryStream = new MemoryStream();
            await imageFile.CopyToAsync(imageMemoryStream);

            // Check that the image is valid.
            byte[] imageData = imageMemoryStream.ToArray();
            if (!imageData.IsValidImage())
                return StatusCode(StatusCodes.Status415UnsupportedMediaType);

            string ext = ImageValidationExtensions.GetImageFormat(imageData) == ImageValidationExtensions.ImageFormat.jpeg ? ".jpg" : ".png";
            string filePath = Path.Combine(Directory.GetCurrentDirectory(), @"wwwroot\img", Path.GetRandomFileName().Split('.')[0] + ext);
            /*
            using (var stream = System.IO.File.Create(filePath))
            {
                await imageFile.CopyToAsync(stream);
            }
            */

            using (FileStream stream = new FileStream(filePath, FileMode.Create))
            {
                await stream.WriteAsync(imageData);
            }

            _logger.LogInformation("Start processing image...");

            // Measure execution time.
            System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();

            // Set the specific image data into the ImageInputData type used in the DataView.
            InMemoryImageData imageInputData = new InMemoryImageData(image: imageData, label: null, imageFileName: null);

            // Predict code for provided image.
            ImagePrediction prediction = _predictionEnginePool.Predict(imageInputData);

            // Stop measuring time.
            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;
            _logger.LogInformation($"Image processed in {elapsedMs} miliseconds");

            // Predict the image's label (The one with highest probability).
            ImagePredictedLabelWithProbability imageBestLabelPrediction =
                new ImagePredictedLabelWithProbability
                {
                    PredictedLabel = prediction.PredictedLabel,
                    Probability = prediction.Score.Max(),
                    PredictionExecutionTime = elapsedMs,
                    ImageId = imageFile.FileName,
                };

            return Ok(imageBestLabelPrediction);
        }

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);

        // GET api/ImageClassification
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
            => new string[] { "ACK Heart beat 1", "ACK Heart beat 2" };
    }
}