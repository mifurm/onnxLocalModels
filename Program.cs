using Microsoft.ML.OnnxRuntimeGenAI;

var modelPath = @"C:\Dev\Phi-3-mini-4k-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32";
var model = new Model(modelPath);
var tokenizer = new Tokenizer(model);
int i = 0;
var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";

// chat start
//Console.WriteLine(@"Ask your question. Type an empty string to Exit.");

string[] questions =
[
    "What is the capital of Poland?", 
    "Who wrote 'Romeo and Juliet'?", 
    "What is the boiling point of water in degrees Celsius?", 
    "What is the largest planet in our solar system?", 
    "What is the atomic number of oxygen?", 
    "What is the square root of 144?", 
    "Who painted the Mona Lisa?", 
    "What is the currency of Japan?", 
    "How many continents are there?", 
    "What is the speed of light in a vacuum?", 
    "Who discovered penicillin?", 
    "What is the capital city of Australia?", 
    "What is the longest river in the world?", 
    "What is the smallest prime number?", 
    "Who invented the telephone?", 
    "What is the largest ocean on Earth?", 
    "What is the periodic table symbol for gold?", 
    "What is the tallest mountain in the world?", 
    "Who was the first person to walk on the moon?", 
    "What is the freezing point of water in degrees Celsius?"
];

var _wholeTaskStartTime = DateTime.Now;
var _wholeTaskTotalTokens = 0;
// chat loop
foreach (string question in questions)
{
    // Get user question
    //Console.WriteLine();
    //Console.Write(@"Q: ");
    //foreach (string question in questions)
    //{
    var userQ = question;
    if (string.IsNullOrEmpty(userQ))
    {
        break;
    }

    // show phi3 response
    Console.WriteLine("Question {0}: {1}", i.ToString(), question);
    Console.Write("Phi3: ");
    var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{userQ}<|end|><|assistant|>";
    var tokens = tokenizer.Encode(fullPrompt);

    var generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 2048);
    generatorParams.SetSearchOption("past_present_share_buffer", false);
    generatorParams.SetInputSequences(tokens);
    var _startTime = DateTime.Now;
    var generator = new Generator(model, generatorParams);
    var _totalOutputTokens = 0;
    while (!generator.IsDone())
    {
        generator.ComputeLogits();
        generator.GenerateNextToken();
        var outputTokens = generator.GetSequence(0);
        var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
        _totalOutputTokens = _totalOutputTokens + outputTokens.Length;
        var output = tokenizer.Decode(newToken);
        Console.Write(output);
    }
    var _endTime = DateTime.Now;
    Console.WriteLine();
    Console.WriteLine("Total Tokens: {0}, Time: {1} sec.", _totalOutputTokens, _endTime.Subtract(_startTime).TotalSeconds.ToString());
    Console.WriteLine();
    _wholeTaskTotalTokens = _wholeTaskTotalTokens + _totalOutputTokens;
    i++;
}

var _wholeTaskEndTime = DateTime.Now;

Console.WriteLine();
Console.WriteLine("Total Task Token Generated: {0}", _wholeTaskTotalTokens);
Console.WriteLine("Total Task Execution Time: {0} sec.", _wholeTaskEndTime.Subtract(_wholeTaskStartTime).TotalSeconds.ToString());
Console.WriteLine();
